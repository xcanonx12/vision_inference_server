"""Abstract base class for inference backends."""

from abc import ABC, abstractmethod

import numpy as np


class BaseBackend(ABC):
    """Abstract interface for model inference backends.

    All backends must implement load, infer, and is_available.
    """

    @abstractmethod
    def load(self, model_path: str, device: str) -> None:
        """Load model from path onto specified device."""
        ...

    @abstractmethod
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data. Returns raw output."""
        ...

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this backend's runtime is installed and functional."""
        ...
