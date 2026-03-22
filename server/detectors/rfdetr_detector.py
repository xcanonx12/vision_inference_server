"""RF-DETR detector implementation using the rfdetr library."""

import logging
from typing import Any

import numpy as np
import supervision as sv

from server.core.config_loader import ModelConfig
from server.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)

RFDETR_VARIANT_MAP: dict[str, str] = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-base": "RFDETRBase",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}


class RFDETRDetector(BaseDetector):
    """RF-DETR detector using the rfdetr library.

    Unlike YOLO11, RF-DETR is self-contained: the rfdetr library handles
    model loading, preprocessing (ImageNet normalization, resize),
    inference, and postprocessing internally. Its predict() returns
    sv.Detections directly, so this detector bypasses the backend layer.
    """

    def __init__(self, config: ModelConfig, backend=None) -> None:
        super().__init__(config, backend)
        self._model = self._load_rfdetr_model()
        logger.info("RFDETRDetector initialized: %s", config.name)

    def _load_rfdetr_model(self):
        """Instantiate the correct rfdetr model class based on config name."""
        import rfdetr

        class_name = RFDETR_VARIANT_MAP.get(self._config.name, "RFDETRBase")
        model_cls = getattr(rfdetr, class_name, None)
        if model_cls is None:
            raise ValueError(
                f"RF-DETR variant '{class_name}' not found in rfdetr library. "
                f"Available: {[k for k in dir(rfdetr) if k.startswith('RFDETR')]}"
            )

        kwargs = {}
        if self._config.path:
            kwargs["pretrain_weights"] = self._config.path

        model = model_cls(**kwargs)
        logger.info(
            "RF-DETR model loaded: %s (resolution=%d)",
            class_name, model.model_config.resolution,
        )
        return model

    @property
    def num_classes(self) -> int:
        return self._model.model_config.num_classes

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Not used — rfdetr handles preprocessing internally."""
        return image

    def postprocess(self, raw_output: Any, original_shape: tuple[int, int]) -> sv.Detections:
        """Not used — rfdetr returns sv.Detections directly."""
        return sv.Detections.empty()

    def predict(self, image: np.ndarray) -> sv.Detections:
        """Run inference using rfdetr's native predict API.

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            sv.Detections.
        """
        return self._model.predict(image, threshold=self._config.confidence_threshold)
