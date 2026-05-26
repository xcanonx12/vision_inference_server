import pytest
import numpy as np
import torch

from server.backends.base_backend import BaseBackend
from server.backends.pytorch_backend import PyTorchBackend
from server.backends.onnx_backend import ONNXBackend
from server.backends.tensorrt_backend import TensorRTBackend


class TestBaseBackend:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseBackend()


class TestPyTorchBackend:
    def test_is_available_returns_bool(self):
        assert isinstance(PyTorchBackend.is_available(), bool)

    def test_pytorch_is_available(self):
        assert PyTorchBackend.is_available() is True

    def test_load_and_infer_yolo11n(self, synthetic_image):
        backend = PyTorchBackend()
        backend.load("models/yolo11n.pt", "cpu")
        result = backend.infer(synthetic_image)
        assert isinstance(result, list)
        assert len(result) > 0


class TestONNXBackend:
    def test_is_available_returns_bool(self):
        result = ONNXBackend.is_available()
        assert isinstance(result, bool)

    def test_is_available_does_not_raise(self):
        try:
            ONNXBackend.is_available()
        except Exception:
            pytest.fail("is_available() raised an exception")


class TestTensorRTBackend:
    def test_is_available_returns_bool(self):
        result = TensorRTBackend.is_available()
        assert isinstance(result, bool)

    def test_is_available_does_not_raise(self):
        try:
            TensorRTBackend.is_available()
        except Exception:
            pytest.fail("is_available() raised an exception")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_load_requires_cuda(self):
        backend = TensorRTBackend()
        assert hasattr(backend, 'load')
        assert hasattr(backend, 'infer')

    def test_load_rejects_cpu(self):
        backend = TensorRTBackend()
        with pytest.raises(RuntimeError, match="TensorRT backend requires CUDA device"):
            backend.load("dummy.engine", "cpu")

    def test_infer_without_load_raises(self):
        backend = TensorRTBackend()
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.infer(dummy)
