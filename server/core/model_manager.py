"""Model lifecycle manager: load, warmup, and expose model pool."""

import logging
import threading

from server.backends.base_backend import BaseBackend
from server.backends.onnx_backend import ONNXBackend
from server.backends.pytorch_backend import PyTorchBackend
from server.backends.tensorrt_backend import TensorRTBackend
from server.core.config_loader import AppConfig, ModelConfig, _validate_model_config
from server.core.device_manager import DeviceManager
from server.detectors.base_detector import BaseDetector
from server.detectors.rfdetr_detector import RFDETRDetector
from server.detectors.yolo11_detector import YOLO11Detector

logger = logging.getLogger(__name__)

BACKEND_REGISTRY: dict[str, type[BaseBackend]] = {
    "pytorch": PyTorchBackend,
    "onnx": ONNXBackend,
    "tensorrt": TensorRTBackend,
}

DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "yolo11": YOLO11Detector,
    "rfdetr": RFDETRDetector,
}


class ModelManager:
    """Manages a pool of named detectors.

    Supports single-model (backward compat) and multi-model modes.
    Models are keyed by their config name. The first loaded model
    becomes the default.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._device_manager = DeviceManager()
        self._models: dict[str, BaseDetector] = {}
        self._default_name: str = ""
        self._swap_lock = threading.RLock()

    def _load_single(self, model_cfg: ModelConfig) -> BaseDetector:
        """Load one model and return the detector instance.

        Args:
            model_cfg: Configuration for the model to load.

        Returns:
            Loaded BaseDetector instance.

        Raises:
            ValueError: If model type or backend is unknown.
            RuntimeError: If backend is not available.
        """
        detector_cls = DETECTOR_REGISTRY.get(model_cfg.type)
        if detector_cls is None:
            raise ValueError(f"Unknown model type: {model_cfg.type}")

        device = self._device_manager.get_device_for_backend(model_cfg.backend)

        if model_cfg.type == "rfdetr":
            # RF-DETR is self-contained -- no backend needed
            detector = detector_cls(model_cfg)
        else:
            # Standard backend flow (YOLO11, etc.)
            backend_cls = BACKEND_REGISTRY.get(model_cfg.backend)
            if backend_cls is None:
                raise ValueError(f"Unknown backend: {model_cfg.backend}")
            if not backend_cls.is_available():
                raise RuntimeError(
                    f"Backend '{model_cfg.backend}' is not available. "
                    "Check that the required packages are installed."
                )
            backend = backend_cls()
            backend.load(model_cfg.path, device)
            detector = detector_cls(model_cfg, backend)

        self._device = device
        logger.info(
            "Model loaded: %s (%s/%s) on %s",
            model_cfg.name, model_cfg.type, model_cfg.backend, device,
        )
        return detector

    def load_model(self) -> None:
        """Load models from config.

        If config.models is set, loads all models in the list.
        Otherwise, loads the single config.model entry.
        """
        if self._config.models:
            # Multi-model mode
            for model_cfg in self._config.models:
                detector = self._load_single(model_cfg)
                self._models[model_cfg.name] = detector
            self._default_name = self._config.models[0].name
        else:
            # Single-model mode (backward compat)
            model_cfg = self._config.model
            detector = self._load_single(model_cfg)
            self._models[model_cfg.name] = detector
            self._default_name = model_cfg.name

    def get_model(self, name: str = "") -> BaseDetector:
        """Lookup a model by name.

        Args:
            name: Model name. Empty string returns the default model.

        Returns:
            The requested BaseDetector.

        Raises:
            RuntimeError: If no models loaded or name not found.
        """
        if not self._models:
            raise RuntimeError("No model loaded. Call load_model() first.")

        lookup = name if name else self._default_name
        detector = self._models.get(lookup)
        if detector is None:
            available = list(self._models.keys())
            raise RuntimeError(
                f"Model '{lookup}' not found. Available: {available}"
            )
        return detector

    def get_active_model(self) -> BaseDetector:
        """Return the default detector (backward compat wrapper).

        Raises:
            RuntimeError: If no model is loaded.
        """
        return self.get_model("")

    def is_ready(self) -> bool:
        """Return True if at least one model is loaded and ready."""
        return len(self._models) > 0

    def hot_swap(self, new_model_cfg: ModelConfig) -> None:
        """Replace a model in the pool without server restart.

        Validates config, loads the new model (slow, outside lock),
        then atomically swaps the detector under the lock.
        Thread-safe -- concurrent hot_swap calls are serialized.

        Args:
            new_model_cfg: Configuration for the replacement model.
        """
        # Validate before doing expensive work
        _validate_model_config(new_model_cfg)

        logger.info(
            "Hot-swap starting: %s (%s/%s)",
            new_model_cfg.name, new_model_cfg.type, new_model_cfg.backend,
        )

        # Load new model outside the lock (slow operation)
        new_detector = self._load_single(new_model_cfg)

        # Atomic swap under lock -- only protects the reference update
        with self._swap_lock:
            self._models[new_model_cfg.name] = new_detector
            # Update config to reflect the swap
            if new_model_cfg.name == self._default_name:
                self._config = AppConfig(
                    server=self._config.server,
                    model=new_model_cfg,
                    inference=self._config.inference,
                    warmup=self._config.warmup,
                    models=self._config.models,
                )
            else:
                # Update the models list entry if it exists
                if self._config.models:
                    updated_models = []
                    found = False
                    for m in self._config.models:
                        if m.name == new_model_cfg.name:
                            updated_models.append(new_model_cfg)
                            found = True
                        else:
                            updated_models.append(m)
                    if not found:
                        updated_models.append(new_model_cfg)
                    self._config = AppConfig(
                        server=self._config.server,
                        model=self._config.model,
                        inference=self._config.inference,
                        warmup=self._config.warmup,
                        models=updated_models,
                    )
                else:
                    self._config = AppConfig(
                        server=self._config.server,
                        model=new_model_cfg,
                        inference=self._config.inference,
                        warmup=self._config.warmup,
                        models=self._config.models,
                    )

        logger.info("Hot-swap complete: %s", new_model_cfg.name)

    def warmup(self) -> None:
        """Run warmup on all models in the pool."""
        if not self._models:
            raise RuntimeError("No model loaded. Call load_model() first.")

        iterations = self._config.warmup.iterations
        for name, detector in self._models.items():
            logger.info("Running warmup for '%s': %d iterations", name, iterations)
            detector.warmup(n_iterations=iterations)
        logger.info("Warmup complete for all models.")

    def _get_num_classes(self, model_name: str = "") -> int:
        """Derive num_classes from a loaded model's metadata.

        Args:
            model_name: Model to query. Empty = default.
        """
        try:
            detector = self.get_model(model_name)
        except RuntimeError:
            return 0

        # RF-DETR: detector exposes num_classes directly
        if hasattr(detector, "num_classes"):
            return detector.num_classes
        # YOLO: backend model has .names dict
        backend = detector._backend
        model = getattr(backend, "_model", None)
        if model is not None and hasattr(model, "names"):
            return len(model.names)
        return 0

    def get_model_info(self, model_name: str = "") -> dict:
        """Return model metadata for /config and GetServerConfig.

        Args:
            model_name: Model to query. Empty = default.
        """
        try:
            detector = self.get_model(model_name)
        except RuntimeError:
            return {"status": "loading"}

        model_cfg = detector.config
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
            "num_classes": self._get_num_classes(model_name),
        }

    def get_inference_config(self):
        """Return the inference configuration.

        Returns:
            InferenceConfig from the current AppConfig.
        """
        return self._config.inference
