import pytest
import numpy as np
import cv2
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
  path: "yolo11n.pt"
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

    mm = ModelManager(config)
    mm.load_model()
    return InferenceServicer(mm)


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
