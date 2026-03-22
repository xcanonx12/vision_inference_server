"""PyTorch backend using ultralytics for YOLO model inference."""

import logging

import numpy as np

from server.backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)


class PyTorchBackend(BaseBackend):
    """Backend that uses ultralytics YOLO model with PyTorch runtime.

    This backend loads .pt model files and runs inference via
    the ultralytics library, which handles PyTorch internally.
    """

    def __init__(self) -> None:
        self._model = None

    def load(self, model_path: str, device: str) -> None:
        """Load a PyTorch model via ultralytics.

        Args:
            model_path: Path to .pt model file (or model name for auto-download).
            device: Device string ('cpu' or 'cuda').
        """
        from ultralytics import YOLO

        logger.info("Loading PyTorch model: %s on %s", model_path, device)
        self._model = YOLO(model_path)
        self._model.to(device)
        self._device = device

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on a BGR image.

        Args:
            input_data: BGR image as numpy array (H, W, 3).

        Returns:
            Raw ultralytics Results object stored as object array for
            downstream postprocessing by the detector.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self._model(input_data, verbose=False)
        return np.array(results, dtype=object)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False
