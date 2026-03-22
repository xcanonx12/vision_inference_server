"""gRPC InferenceService implementation."""

import logging
import time

import cv2
import grpc
import numpy as np

from server.core.model_manager import ModelManager
from server.generated import detections_pb2, detections_pb2_grpc

logger = logging.getLogger(__name__)


class InferenceServicer(detections_pb2_grpc.InferenceServiceServicer):
    """gRPC servicer for inference requests.

    Handles Predict (unary) and GetServerConfig RPCs.
    StreamPredict is stubbed for Phase 2.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    def Predict(
        self,
        request: detections_pb2.InferenceRequest,
        context: grpc.ServicerContext,
    ) -> detections_pb2.InferenceResponse:
        """Run single-image inference.

        Decodes JPEG from request, runs detection, returns serialized detections.
        """
        # Decode image
        image = self._decode_image(request.image_data, context)
        if image is None:
            return detections_pb2.InferenceResponse()

        # Run inference
        start = time.perf_counter()
        try:
            detector = self._model_manager.get_active_model()
            detections = detector.predict(image)
        except Exception as e:
            logger.error("Inference failed: %s", e)
            context.abort(grpc.StatusCode.INTERNAL, f"Inference failed: {e}")
            return detections_pb2.InferenceResponse()

        inference_time_ms = (time.perf_counter() - start) * 1000

        # Build response
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

        logger.debug(
            "Predict: %d detections in %.1fms",
            len(detections), inference_time_ms,
        )
        return response

    def StreamPredict(self, request_iterator, context):
        """Bidirectional streaming — stub for Phase 2."""
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "StreamPredict will be available in Phase 2",
        )

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

    def _decode_image(
        self,
        image_data: bytes,
        context: grpc.ServicerContext,
    ) -> np.ndarray | None:
        """Decode JPEG bytes to BGR numpy array."""
        if not image_data:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Empty image data",
            )
            return None

        buf = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        if image is None:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Failed to decode image. Ensure it is valid JPEG.",
            )
            return None

        return image
