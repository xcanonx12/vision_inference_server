import pytest
import numpy as np

from server.backends.base_backend import BaseBackend
from server.backends.pytorch_backend import PyTorchBackend
from server.backends.onnx_backend import ONNXBackend


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
        backend.load("yolo11n.pt", "cpu")
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
