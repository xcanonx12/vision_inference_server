import pytest
import numpy as np
import supervision as sv

from server.detectors.base_detector import BaseDetector


class TestBaseDetector:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseDetector(config=None, backend=None)


class TestYOLO11Detector:
    @pytest.fixture(scope="class")
    def detector(self):
        """Load YOLO11n (nano) once for all tests in this class."""
        from server.detectors.yolo11_detector import YOLO11Detector
        from server.core.config_loader import ModelConfig
        from server.backends.pytorch_backend import PyTorchBackend
        from server.core.device_manager import DeviceManager

        config = ModelConfig(
            name="yolo11n",
            type="yolo11",
            backend="pytorch",
            source="local",
            path="models/yolo11n.pt",
            input_width=640,
            input_height=640,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )
        dm = DeviceManager()
        backend = PyTorchBackend()
        backend.load(config.path, dm.get_device_for_backend(config.backend))
        return YOLO11Detector(config, backend)

    def test_predict_returns_sv_detections(self, detector, synthetic_image):
        result = detector.predict(synthetic_image)
        assert isinstance(result, sv.Detections)

    def test_predict_detections_have_correct_fields(self, detector, synthetic_image):
        result = detector.predict(synthetic_image)
        assert hasattr(result, "xyxy")
        assert hasattr(result, "confidence")
        assert hasattr(result, "class_id")

    def test_warmup_completes_without_error(self, detector):
        detector.warmup(n_iterations=2)

    def test_predict_with_different_size_image(self, detector):
        """Non-square image should still work."""
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        result = detector.predict(img)
        assert isinstance(result, sv.Detections)

    def test_predict_detections_shape(self, detector, synthetic_image_640):
        result = detector.predict(synthetic_image_640)
        assert isinstance(result, sv.Detections)
        if len(result) > 0:
            assert result.xyxy.shape[1] == 4
            assert result.xyxy.dtype == np.float32
            assert result.confidence is not None
            assert result.class_id is not None
