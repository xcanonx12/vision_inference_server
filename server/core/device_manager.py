"""Device detection with automatic fallback chain."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    device: str
    cuda_available: bool
    onnx_gpu_available: bool
    cuda_device_name: str = ""


class DeviceManager:
    """Detects available compute devices and provides fallback logic.

    Fallback chain: CUDA → ONNX+CUDA → ONNX+CPU → PyTorch+CPU
    """

    def __init__(self) -> None:
        self._cuda_available = self._check_cuda()
        self._onnx_gpu_available = self._check_onnx_gpu()
        self._optimal_device = self._resolve_optimal_device()

    @staticmethod
    def _check_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        except Exception:
            return False

    @staticmethod
    def _check_onnx_gpu() -> bool:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            return "CUDAExecutionProvider" in providers
        except ImportError:
            return False
        except Exception:
            return False

    def _resolve_optimal_device(self) -> str:
        if self._cuda_available:
            return "cuda"
        return "cpu"

    def is_cuda_available(self) -> bool:
        return self._cuda_available

    def is_onnx_gpu_available(self) -> bool:
        return self._onnx_gpu_available

    def available_devices(self) -> list[str]:
        devices = ["cpu"]
        if self._cuda_available:
            devices.append("cuda")
        return devices

    def get_optimal_device(self) -> str:
        return self._optimal_device

    def get_device_for_backend(self, backend: str) -> str:
        """Return the appropriate device string for a given backend.

        Args:
            backend: One of 'pytorch', 'onnx', 'tensorrt'.

        Returns:
            Device string suitable for the backend.
        """
        if backend == "tensorrt":
            if not self._cuda_available:
                logger.warning("TensorRT requires CUDA but none found, falling back to ONNX+CPU")
                return "cpu"
            return "cuda"

        if backend == "onnx":
            if self._onnx_gpu_available:
                return "cuda"
            return "cpu"

        # pytorch
        return self._optimal_device

    def get_gpu_memory(self) -> dict:
        """Return GPU memory usage in MB, or empty dict if no GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024 / 1024
                total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                return {"gpu_used_mb": round(used, 1), "gpu_total_mb": round(total, 1)}
        except ImportError:
            pass
        return {}

    def get_device_info(self) -> DeviceInfo:
        cuda_name = ""
        if self._cuda_available:
            try:
                import torch
                cuda_name = torch.cuda.get_device_name(0)
            except Exception:
                cuda_name = "unknown"

        return DeviceInfo(
            device=self._optimal_device,
            cuda_available=self._cuda_available,
            onnx_gpu_available=self._onnx_gpu_available,
            cuda_device_name=cuda_name,
        )
