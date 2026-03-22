import pytest

from server.core.device_manager import DeviceManager, DeviceInfo


class TestDeviceManager:
    def test_cpu_always_available(self):
        dm = DeviceManager()
        assert "cpu" in dm.available_devices()

    def test_get_optimal_device_returns_string(self):
        dm = DeviceManager()
        device = dm.get_optimal_device()
        assert isinstance(device, str)
        assert device in ("cpu", "cuda", "cuda:0")

    def test_get_device_for_backend_pytorch(self):
        dm = DeviceManager()
        device = dm.get_device_for_backend("pytorch")
        assert isinstance(device, str)

    def test_get_device_for_backend_onnx(self):
        dm = DeviceManager()
        device = dm.get_device_for_backend("onnx")
        assert isinstance(device, str)

    def test_cuda_detection_does_not_raise(self):
        dm = DeviceManager()
        result = dm.is_cuda_available()
        assert isinstance(result, bool)

    def test_onnx_gpu_detection_does_not_raise(self):
        dm = DeviceManager()
        result = dm.is_onnx_gpu_available()
        assert isinstance(result, bool)

    def test_device_info_returns_typed_object(self):
        dm = DeviceManager()
        info = dm.get_device_info()
        assert isinstance(info, DeviceInfo)
        assert isinstance(info.device, str)
        assert isinstance(info.cuda_available, bool)
        assert isinstance(info.onnx_gpu_available, bool)
