"""ONNX Runtime backend for model inference."""

import logging

import numpy as np

from server.backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)


class ONNXBackend(BaseBackend):
    """Backend that uses ONNX Runtime for inference.

    Supports CPU and CUDA execution providers. For YOLO models,
    this backend loads .onnx files exported from ultralytics via
    the ultralytics YOLO class which handles ONNX transparently.
    """

    def __init__(self) -> None:
        self._model = None

    def load(self, model_path: str, device: str) -> None:
        """Load an ONNX model via ultralytics.

        Args:
            model_path: Path to .onnx model file.
            device: Device string ('cpu' or 'cuda').
        """
        from ultralytics import YOLO

        logger.info("Loading ONNX model: %s on %s", model_path, device)
        self._model = YOLO(model_path)
        if device != "cpu":
            self._model.to(device)
        self._device = device

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on a BGR image.

        Args:
            input_data: BGR image as numpy array (H, W, 3).

        Returns:
            Raw ultralytics Results object stored as object array.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self._model(input_data, verbose=False)
        return np.array(results, dtype=object)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import onnxruntime
            return True
        except ImportError:
            return False
