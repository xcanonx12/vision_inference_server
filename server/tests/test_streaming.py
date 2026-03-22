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

    from server.core.metrics_collector import MetricsCollector

    mm = ModelManager(config)
    mm.load_model()
    return InferenceServicer(mm, MetricsCollector())


def _make_request(image: np.ndarray) -> detections_pb2.InferenceRequest:
    _, encoded = cv2.imencode(".jpg", image)
    return detections_pb2.InferenceRequest(
        image_data=encoded.tobytes(),
        width=640,
        height=480,
    )


class TestStreamPredict:
    def test_stream_processes_multiple_frames(self, servicer, synthetic_image):
        """StreamPredict processes 5 frames and yields 5 responses."""
        requests = [_make_request(synthetic_image) for _ in range(5)]
        context = MagicMock()

        responses = list(servicer.StreamPredict(iter(requests), context))
        assert len(responses) == 5
        for resp in responses:
            assert isinstance(resp, detections_pb2.InferenceResponse)
            assert resp.inference_time_ms >= 0

    def test_stream_empty_iterator(self, servicer):
        """Empty request iterator yields no responses."""
        context = MagicMock()
        responses = list(servicer.StreamPredict(iter([]), context))
        assert len(responses) == 0

    def test_stream_corrupted_frame_skipped(self, servicer, synthetic_image):
        """A corrupted frame in the stream doesn't kill the whole stream."""
        good = _make_request(synthetic_image)
        bad = detections_pb2.InferenceRequest(
            image_data=b"not_an_image", width=640, height=480,
        )
        requests = [good, bad, good]
        context = MagicMock()

        responses = list(servicer.StreamPredict(iter(requests), context))
        # Should get responses for both good frames; bad frame yields empty response
        assert len(responses) == 3

    def test_stream_30_frames_no_degradation(self, servicer, synthetic_image):
        """30 frames process without significant latency degradation."""
        requests = [_make_request(synthetic_image) for _ in range(30)]
        context = MagicMock()

        responses = list(servicer.StreamPredict(iter(requests), context))
        assert len(responses) == 30

        times = [r.inference_time_ms for r in responses]
        # Verify no extreme degradation: last frame shouldn't be >3x first frame
        assert times[-1] < times[0] * 3 + 50  # allow 50ms margin
