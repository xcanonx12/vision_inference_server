"""TensorRT backend using ultralytics for .engine model inference."""

import logging

import numpy as np

from server.backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)


class TensorRTBackend(BaseBackend):
    """Backend that uses ultralytics YOLO with TensorRT .engine files.

    TensorRT engines are hardware-specific and must be compiled on the
    target GPU. The ultralytics library handles TensorRT inference
    transparently when given a .engine file.

    Requires CUDA — will not work on CPU.
    """

    def __init__(self) -> None:
        self._model = None

    def load(self, model_path: str, device: str) -> None:
        """Load a TensorRT engine via ultralytics.

        Args:
            model_path: Path to .engine model file.
            device: Device string (must be 'cuda').

        Raises:
            RuntimeError: If device is not 'cuda'.
        """
        if device != "cuda":
            raise RuntimeError("TensorRT backend requires CUDA device")

        from ultralytics import YOLO

        logger.info("Loading TensorRT model: %s on %s", model_path, device)
        self._model = YOLO(model_path)
        self._device = device

    def infer(self, input_data: np.ndarray) -> list:
        """Run inference on a BGR image.

        Args:
            input_data: BGR image as numpy array (H, W, 3).

        Returns:
            List of ultralytics Results objects.

        Raises:
            RuntimeError: If model has not been loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model(input_data, verbose=False)

    @classmethod
    def is_available(cls) -> bool:
        """Check if TensorRT and CUDA are available."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            import tensorrt  # noqa: F401
            return True
        except ImportError:
            return False
