"""YOLO11 detector implementation."""

import logging

import numpy as np
import supervision as sv

from server.backends.base_backend import BaseBackend
from server.core.config_loader import ModelConfig
from server.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class YOLO11Detector(BaseDetector):
    """YOLO11 detector using ultralytics via the backend layer.

    The backend (PyTorchBackend or ONNXBackend) handles model loading
    and raw inference. This detector handles the conversion of
    ultralytics Results to sv.Detections with confidence/IOU filtering.
    """

    def __init__(self, config: ModelConfig, backend: BaseBackend) -> None:
        super().__init__(config, backend)
        logger.info("YOLO11Detector initialized: %s", config.name)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """YOLO11 preprocessing is handled by ultralytics internally.

        The backend passes the raw image to ultralytics which handles
        resize, BGR→RGB, normalization, and HWC→CHW internally.
        """
        return image

    def postprocess(self, raw_output: list, original_shape: tuple[int, int]) -> sv.Detections:
        """Convert ultralytics Results to sv.Detections.

        Args:
            raw_output: List of ultralytics Results objects.
            original_shape: (height, width) of the original image.

        Returns:
            Filtered sv.Detections.
        """
        results = raw_output[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0:
            return detections

        # Filter by confidence threshold
        mask = detections.confidence >= self._config.confidence_threshold
        detections = detections[mask]

        return detections
