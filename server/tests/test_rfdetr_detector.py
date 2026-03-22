import pytest
import numpy as np
import supervision as sv


class TestRFDETRDetector:
    @pytest.fixture(scope="class")
    def detector(self):
        """Load RF-DETR Nano (lightest) once for all tests."""
        from server.detectors.rfdetr_detector import RFDETRDetector
        from server.core.config_loader import ModelConfig

        config = ModelConfig(
            name="rfdetr-nano",
            type="rfdetr",
            backend="pytorch",
            source="local",
            path=None,  # rfdetr auto-downloads pretrained COCO weights
            input_width=384,
            input_height=384,
            confidence_threshold=0.5,
            iou_threshold=0.45,
        )
        return RFDETRDetector(config)

    def test_predict_returns_sv_detections(self, detector, synthetic_image):
        result = detector.predict(synthetic_image)
        assert isinstance(result, sv.Detections)

    def test_predict_detections_have_correct_fields(self, detector, synthetic_image):
        result = detector.predict(synthetic_image)
        assert hasattr(result, "xyxy")
        assert hasattr(result, "confidence")
        assert hasattr(result, "class_id")

    def test_predict_with_different_size_image(self, detector):
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        result = detector.predict(img)
        assert isinstance(result, sv.Detections)

    def test_warmup_completes_without_error(self, detector):
        detector.warmup(n_iterations=1)

    def test_num_classes_from_model(self, detector):
        assert detector.num_classes == 90  # COCO pretrained
