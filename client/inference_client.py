"""SimpleInference portable client.

This file is designed to be copied as-is to any project.
Dependencies: grpcio, opencv-python, numpy, supervision, httpx
The generated detections_pb2.py and detections_pb2_grpc.py files
must be accessible (same directory or on sys.path).
"""

import logging
import time
from typing import Iterator, Optional

import cv2
import grpc
import httpx
import numpy as np
import supervision as sv

from server.generated import detections_pb2, detections_pb2_grpc

logger = logging.getLogger(__name__)


class InferenceClient:
    """Portable gRPC client for SimpleInference server.

    Usage:
        client = InferenceClient(host="localhost", port=50051)
        client.connect()
        client.fetch_config()

        frame = cv2.imread("image.jpg")
        detections = client.predict(frame)  # sv.Detections
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        http_port: int = 8080,
        jpeg_quality: int = 85,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._host = host
        self._port = port
        self._http_port = http_port
        self._jpeg_quality = jpeg_quality
        self._timeout_seconds = timeout_seconds

        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[detections_pb2_grpc.InferenceServiceStub] = None
        self._input_width: int = 640
        self._input_height: int = 640

    def connect(self) -> None:
        """Wait for the server to become ready.

        Polls /health every 2 seconds until the server responds
        with {"status": "ready"} or timeout is reached.

        Raises:
            ConnectionError: If server doesn't become ready within timeout.
        """
        health_url = f"http://{self._host}:{self._http_port}/health"
        deadline = time.time() + self._timeout_seconds
        poll_interval = 2.0

        logger.info("Connecting to %s:%d ...", self._host, self._port)

        while time.time() < deadline:
            try:
                resp = httpx.get(health_url, timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ready":
                        # Establish gRPC channel
                        self._channel = grpc.insecure_channel(
                            f"{self._host}:{self._port}"
                        )
                        self._stub = detections_pb2_grpc.InferenceServiceStub(
                            self._channel
                        )
                        logger.info("Connected to server.")
                        return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            time.sleep(poll_interval)

        raise ConnectionError(
            f"Server at {self._host} not ready after {self._timeout_seconds}s"
        )

    def fetch_config(self) -> dict:
        """Fetch server configuration via HTTP /config.

        Updates internal input dimensions from the server's active model.

        Returns:
            Server configuration dict.
        """
        config_url = f"http://{self._host}:{self._http_port}/config"
        resp = httpx.get(config_url, timeout=5.0)
        resp.raise_for_status()
        config = resp.json()

        self._input_width = config.get("input_width", self._input_width)
        self._input_height = config.get("input_height", self._input_height)

        logger.info(
            "Config synced: %s %dx%d",
            config.get("model_name", "unknown"),
            self._input_width,
            self._input_height,
        )
        return config

    def predict(self, image: np.ndarray) -> sv.Detections:
        """Run inference on a single image.

        Args:
            image: BGR image as numpy array (any size).

        Returns:
            sv.Detections from the server.

        Raises:
            RuntimeError: If not connected.
        """
        if self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")

        image_bytes = self._prepare_image(image)
        request = detections_pb2.InferenceRequest(
            image_data=image_bytes,
            width=self._input_width,
            height=self._input_height,
        )

        response = self._stub.Predict(request)
        return self._deserialize_response(response)

    def _prepare_image(self, image: np.ndarray) -> bytes:
        """Resize and JPEG-encode an image for transmission.

        Args:
            image: BGR image as numpy array.

        Returns:
            JPEG-encoded bytes.
        """
        resized = cv2.resize(image, (self._input_width, self._input_height))
        _, encoded = cv2.imencode(
            ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        return encoded.tobytes()

    def _deserialize_response(self, response) -> sv.Detections:
        """Convert gRPC InferenceResponse to sv.Detections.

        Args:
            response: InferenceResponse proto message.

        Returns:
            sv.Detections.
        """
        if not response.detections:
            return sv.Detections.empty()

        n = len(response.detections)
        xyxy = np.zeros((n, 4), dtype=np.float32)
        confidence = np.zeros(n, dtype=np.float32)
        class_id = np.zeros(n, dtype=np.int32)

        for i, det in enumerate(response.detections):
            xyxy[i] = det.bbox
            confidence[i] = det.confidence
            class_id[i] = det.class_id

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

    def stream_predict(
        self,
        frame_generator: Iterator[np.ndarray],
    ) -> Iterator[sv.Detections]:
        """Run streaming inference over a frame generator.

        Args:
            frame_generator: Iterator yielding BGR images as numpy arrays.

        Yields:
            sv.Detections for each frame.

        Raises:
            RuntimeError: If not connected.
        """
        if self._stub is None:
            raise RuntimeError("Not connected. Call connect() first.")

        def request_generator():
            for frame in frame_generator:
                image_bytes = self._prepare_image(frame)
                yield detections_pb2.InferenceRequest(
                    image_data=image_bytes,
                    width=self._input_width,
                    height=self._input_height,
                )

        for response in self._stub.StreamPredict(request_generator()):
            yield self._deserialize_response(response)

    def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
