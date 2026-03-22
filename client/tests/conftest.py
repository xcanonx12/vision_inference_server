import numpy as np
import pytest


@pytest.fixture
def synthetic_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
