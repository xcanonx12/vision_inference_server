"""Abstract base class for all detectors."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import supervision as sv

from server.backends.base_backend import BaseBackend
from server.core.config_loader import ModelConfig


class BaseDetector(ABC):
    """Abstract detector interface.

    All detectors must implement preprocess and postprocess.
    The predict method provides the full pipeline: backend.infer → postprocess.
    Output is always sv.Detections.
    """

    def __init__(self, config: ModelConfig, backend: BaseBackend | None = None) -> None:
        self._config = config
        self._backend = backend

    @property
    def config(self) -> ModelConfig:
        return self._config

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for inference."""
        ...

    @abstractmethod
    def postprocess(self, raw_output: Any, original_shape: tuple[int, int]) -> sv.Detections:
        """Convert raw backend output to sv.Detections.

        Args:
            raw_output: Raw output from backend.infer().
            original_shape: (height, width) of the original image.

        Returns:
            sv.Detections with xyxy, confidence, and class_id.
        """
        ...

    def predict(self, image: np.ndarray) -> sv.Detections:
        """Full inference pipeline: infer → postprocess.

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            sv.Detections.
        """
        if self._backend is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no backend. Override predict() directly."
            )
        original_shape = image.shape[:2]
        raw_output = self._backend.infer(image)
        return self.postprocess(raw_output, original_shape)

    def warmup(self, n_iterations: int = 5) -> None:
        """Warm up the model with synthetic data.

        Args:
            n_iterations: Number of warmup inference passes.
        """
        synthetic = np.random.randint(
            0, 255,
            (self._config.input_height, self._config.input_width, 3),
            dtype=np.uint8,
        )
        for _ in range(n_iterations):
            self.predict(synthetic)
