"""Model lifecycle manager: load, warmup, and expose active detector."""

import logging

from server.backends.base_backend import BaseBackend
from server.backends.onnx_backend import ONNXBackend
from server.backends.pytorch_backend import PyTorchBackend
from server.core.config_loader import AppConfig
from server.core.device_manager import DeviceManager
from server.detectors.base_detector import BaseDetector
from server.detectors.yolo11_detector import YOLO11Detector

logger = logging.getLogger(__name__)

BACKEND_REGISTRY: dict[str, type[BaseBackend]] = {
    "pytorch": PyTorchBackend,
    "onnx": ONNXBackend,
}

DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "yolo11": YOLO11Detector,
}


class ModelManager:
    """Manages the active detector lifecycle.

    Handles model loading, warmup, and provides access to the
    active detector for inference. Supports one active model at a time
    (hot-swap planned for Phase 2).
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._device_manager = DeviceManager()
        self._active_detector: BaseDetector | None = None

    def load_model(self) -> None:
        """Load the model specified in config.

        Instantiates the correct backend and detector based on
        config.model.type and config.model.backend.
        """
        model_cfg = self._config.model

        # Resolve backend
        backend_cls = BACKEND_REGISTRY.get(model_cfg.backend)
        if backend_cls is None:
            raise ValueError(f"Unknown backend: {model_cfg.backend}")

        if not backend_cls.is_available():
            raise RuntimeError(
                f"Backend '{model_cfg.backend}' is not available. "
                "Check that the required packages are installed."
            )

        backend = backend_cls()
        device = self._device_manager.get_device_for_backend(model_cfg.backend)
        backend.load(model_cfg.path, device)

        # Resolve detector
        detector_cls = DETECTOR_REGISTRY.get(model_cfg.type)
        if detector_cls is None:
            raise ValueError(f"Unknown model type: {model_cfg.type}")

        self._active_detector = detector_cls(model_cfg, backend)
        self._device = device

        logger.info(
            "Model loaded: %s (%s/%s) on %s",
            model_cfg.name, model_cfg.type, model_cfg.backend, device,
        )

    def warmup(self) -> None:
        """Run warmup on the active detector."""
        if self._active_detector is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        iterations = self._config.warmup.iterations
        logger.info("Running warmup: %d iterations", iterations)
        self._active_detector.warmup(n_iterations=iterations)
        logger.info("Warmup complete.")

    def get_active_model(self) -> BaseDetector:
        """Return the active detector.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._active_detector is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return self._active_detector

    def is_ready(self) -> bool:
        """Return True if a model is loaded and ready for inference."""
        return self._active_detector is not None

    def _get_num_classes(self) -> int:
        """Derive num_classes from the loaded model's metadata."""
        if self._active_detector is None:
            return 0
        backend = self._active_detector._backend
        model = getattr(backend, "_model", None)
        if model is not None and hasattr(model, "names"):
            return len(model.names)
        return 0

    def get_model_info(self) -> dict:
        """Return model metadata for /config and GetServerConfig."""
        if self._active_detector is None:
            return {"status": "loading"}

        model_cfg = self._config.model
        return {
            "model_name": model_cfg.name,
            "model_type": model_cfg.type,
            "backend": model_cfg.backend,
            "input_width": model_cfg.input_width,
            "input_height": model_cfg.input_height,
            "version": model_cfg.version,
            "device": getattr(self, "_device", "unknown"),
            "confidence_threshold": model_cfg.confidence_threshold,
            "iou_threshold": model_cfg.iou_threshold,
            "num_classes": self._get_num_classes(),
        }
