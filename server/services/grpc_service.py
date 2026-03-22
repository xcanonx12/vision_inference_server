"""gRPC InferenceService implementation."""

import asyncio
import logging
import time
from typing import Optional

import cv2
import grpc
import numpy as np

from server.core.batch_manager import BatchManager
from server.core.config_loader import ServerConfig
from server.core.metrics_collector import MetricsCollector
from server.core.model_manager import ModelManager
from server.generated import detections_pb2, detections_pb2_grpc

logger = logging.getLogger(__name__)


class InferenceServicer(detections_pb2_grpc.InferenceServiceServicer):
    """gRPC servicer for inference requests.

    Handles Predict (unary), StreamPredict (bidirectional streaming),
    and GetServerConfig RPCs.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        metrics_collector: MetricsCollector,
        batch_manager: Optional[BatchManager] = None,
        batch_loop: Optional[asyncio.AbstractEventLoop] = None,
        server_config: Optional[ServerConfig] = None,
    ) -> None:
        self._model_manager = model_manager
        self._metrics = metrics_collector
        self._batch_manager = batch_manager
        self._batch_loop = batch_loop
        self._server_config = server_config or ServerConfig()

    # ── Helper methods ──────────────────────────────────────────────

    def _try_decode_image(self, image_data: bytes) -> np.ndarray | None:
        """Decode JPEG bytes, returning None on failure (no gRPC abort)."""
        if not image_data:
            return None
        buf = np.frombuffer(image_data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _build_detection_response(
        self, image: np.ndarray, detections, inference_time_ms: float
    ) -> detections_pb2.InferenceResponse:
        """Build an InferenceResponse proto from detections."""
        response = detections_pb2.InferenceResponse(
            image_width=image.shape[1],
            image_height=image.shape[0],
            inference_time_ms=inference_time_ms,
        )
        for i in range(len(detections)):
            det = detections_pb2.Detection(
                bbox=detections.xyxy[i].tolist(),
                confidence=float(detections.confidence[i]),
                class_id=int(detections.class_id[i]),
            )
            response.detections.append(det)
        return response

    # ── Unary RPC ───────────────────────────────────────────────────

    def Predict(
        self,
        request: detections_pb2.InferenceRequest,
        context: grpc.ServicerContext,
    ) -> detections_pb2.InferenceResponse:
        """Run single-image inference.

        Decodes JPEG from request, runs detection, returns serialized detections.
        """
        # Input validation — size check before decode
        max_bytes = self._server_config.max_image_bytes
        if len(request.image_data) > max_bytes:
            self._metrics.record_error("oversized_image")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Image too large ({len(request.image_data)} bytes, max {max_bytes}).",
            )
            return detections_pb2.InferenceResponse()

        # Decode image — abort on failure for unary RPCs
        image = self._try_decode_image(request.image_data)
        if image is None:
            self._metrics.record_error("decode_error")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Empty or invalid image data. Ensure it is valid JPEG.",
            )
            return detections_pb2.InferenceResponse()

        # Run inference (via BatchManager if available, direct otherwise)
        start = time.perf_counter()
        try:
            if self._batch_manager is not None and self._batch_loop is not None:
                future = asyncio.run_coroutine_threadsafe(
                    self._batch_manager.submit(image), self._batch_loop
                )
                detections = future.result(timeout=self._server_config.request_timeout_seconds)
            else:
                detector = self._model_manager.get_model(request.model_name)
                detections = detector.predict(image)
        except Exception as e:
            logger.error("Inference failed: %s", e)
            self._metrics.record_error("inference_error")
            context.abort(grpc.StatusCode.INTERNAL, f"Inference failed: {e}")
            return detections_pb2.InferenceResponse()

        inference_time_ms = (time.perf_counter() - start) * 1000
        self._metrics.record_inference(inference_time_ms, batch_size=1)
        response = self._build_detection_response(image, detections, inference_time_ms)

        logger.debug(
            "Predict: %d detections in %.1fms",
            len(detections), inference_time_ms,
        )
        return response

    # ── Bidirectional streaming RPC ─────────────────────────────────

    def StreamPredict(self, request_iterator, context):
        """Bidirectional streaming inference.

        Processes each frame as it arrives and yields the response immediately.
        Bad frames yield an empty response rather than killing the stream.
        """
        for request in request_iterator:
            image = self._try_decode_image(request.image_data)
            if image is None:
                logger.warning("StreamPredict: failed to decode frame, skipping")
                self._metrics.record_error("decode_error")
                yield detections_pb2.InferenceResponse()
                continue

            start = time.perf_counter()
            try:
                detector = self._model_manager.get_model(request.model_name)
                detections = detector.predict(image)
            except Exception as e:
                logger.error("StreamPredict inference failed: %s", e)
                self._metrics.record_error("inference_error")
                yield detections_pb2.InferenceResponse()
                continue

            inference_time_ms = (time.perf_counter() - start) * 1000
            self._metrics.record_inference(inference_time_ms, batch_size=1)
            response = self._build_detection_response(
                image, detections, inference_time_ms
            )

            logger.debug(
                "StreamPredict: %d detections in %.1fms",
                len(detections), inference_time_ms,
            )
            yield response

    def GetServerConfig(
        self,
        request: detections_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> detections_pb2.ServerConfigResponse:
        """Return current server/model configuration."""
        info = self._model_manager.get_model_info()
        return detections_pb2.ServerConfigResponse(
            model_name=info.get("model_name", ""),
            model_type=info.get("model_type", ""),
            backend=info.get("backend", ""),
            input_width=info.get("input_width", 0),
            input_height=info.get("input_height", 0),
            version=info.get("version", ""),
            device=info.get("device", ""),
            num_classes=info.get("num_classes", 0),
        )

