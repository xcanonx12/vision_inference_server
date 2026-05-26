import tempfile

import cv2
import grpc
import numpy as np
import pytest
from unittest.mock import MagicMock

from server.generated import detections_pb2


@pytest.fixture(scope="module")
def servicer():
    """Create a real InferenceServicer with YOLO11n loaded."""
    from server.services.grpc_service import InferenceServicer
    from server.core.model_manager import ModelManager
    from server.core.config_loader import load_config
    import tempfile

    config_content = """
model:
  name: "yolo11n"
  type: "yolo11"
  backend: "pytorch"
  source: "local"
  path: "models/yolo11n.pt"
  input_width: 640
  input_height: 640
  confidence_threshold: 0.25
  iou_threshold: 0.45
warmup:
  enabled: false
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        f.flush()
        config = load_config(f.name)

    from server.core.metrics_collector import MetricsCollector

    mm = ModelManager(config)
    mm.load_model()
    return InferenceServicer(mm, MetricsCollector())


class TestGRPCPredict:
    def test_predict_returns_response(self, servicer, synthetic_image):
        _, encoded = cv2.imencode(".jpg", synthetic_image)
        request = detections_pb2.InferenceRequest(
            image_data=encoded.tobytes(),
            width=640,
            height=480,
        )
        context = MagicMock()
        response = servicer.Predict(request, context)
        assert isinstance(response, detections_pb2.InferenceResponse)
        assert response.inference_time_ms >= 0

    def test_predict_corrupted_image_aborts(self, servicer):
        request = detections_pb2.InferenceRequest(
            image_data=b"not_an_image",
            width=640,
            height=480,
        )
        context = MagicMock()
        servicer.Predict(request, context)
        context.abort.assert_called_once()

    def test_predict_empty_image_aborts(self, servicer):
        request = detections_pb2.InferenceRequest(
            image_data=b"",
            width=0,
            height=0,
        )
        context = MagicMock()
        servicer.Predict(request, context)
        context.abort.assert_called_once()


class TestGRPCGetServerConfig:
    def test_returns_config(self, servicer):
        request = detections_pb2.Empty()
        context = MagicMock()
        response = servicer.GetServerConfig(request, context)
        assert isinstance(response, detections_pb2.ServerConfigResponse)
        assert response.model_name == "yolo11n"
        assert response.model_type == "yolo11"
        assert response.backend == "pytorch"
        assert response.input_width == 640
        assert response.input_height == 640


class TestGRPCValidation:
    """Tests for input validation (oversized/empty images)."""

    @pytest.fixture(scope="class")
    def strict_servicer(self):
        """InferenceServicer with a very small max_image_bytes limit."""
        from server.services.grpc_service import InferenceServicer
        from server.core.model_manager import ModelManager
        from server.core.metrics_collector import MetricsCollector
        from server.core.config_loader import load_config, ServerConfig

        config_content = """
model:
  name: "yolo11n"
  type: "yolo11"
  backend: "pytorch"
  source: "local"
  path: "models/yolo11n.pt"
  input_width: 640
  input_height: 640
warmup:
  enabled: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()
            config = load_config(f.name)

        mm = ModelManager(config)
        mm.load_model()
        server_cfg = ServerConfig(max_image_bytes=100)  # 100-byte limit
        return InferenceServicer(mm, MetricsCollector(), server_config=server_cfg)

    def test_predict_rejects_oversized_image(self, strict_servicer, synthetic_image):
        """Image exceeding max_image_bytes returns INVALID_ARGUMENT."""
        _, encoded = cv2.imencode(".jpg", synthetic_image)
        image_bytes = encoded.tobytes()
        assert len(image_bytes) > 100, "Encoded image must exceed 100-byte limit for this test"

        request = detections_pb2.InferenceRequest(
            image_data=image_bytes,
            width=640,
            height=480,
        )
        context = MagicMock()
        strict_servicer.Predict(request, context)
        context.abort.assert_called_once()
        assert context.abort.call_args[0][0] == grpc.StatusCode.INVALID_ARGUMENT

    def test_predict_rejects_empty_image(self, servicer):
        """Empty image data returns INVALID_ARGUMENT."""
        request = detections_pb2.InferenceRequest(image_data=b"")
        context = MagicMock()
        servicer.Predict(request, context)
        context.abort.assert_called_once()
        assert context.abort.call_args[0][0] == grpc.StatusCode.INVALID_ARGUMENT
