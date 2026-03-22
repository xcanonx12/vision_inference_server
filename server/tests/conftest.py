import numpy as np
import pytest
import supervision as sv


@pytest.fixture
def synthetic_image():
    """480x640 random BGR image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_image_640():
    """640x640 random BGR image matching default model input size."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_detections():
    """Sample sv.Detections with 2 boxes for testing."""
    return sv.Detections(
        xyxy=np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 1], dtype=np.int32),
    )
