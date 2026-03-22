import pytest

from server.core.model_manager import ModelManager
from server.core.config_loader import load_config


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    config_content = """
model:
  name: "yolo11n"
  type: "yolo11"
  backend: "pytorch"
  source: "local"
  path: "yolo11n.pt"
  input_width: 640
  input_height: 640
  confidence_threshold: 0.25
  iou_threshold: 0.45
warmup:
  enabled: false
  iterations: 1
"""
    tmp = tmp_path_factory.mktemp("config")
    f = tmp / "config.yaml"
    f.write_text(config_content)
    return load_config(str(f))


class TestModelManager:
    def test_is_ready_false_before_load(self, config):
        mm = ModelManager(config)
        assert mm.is_ready() is False

    def test_load_model_succeeds(self, config):
        mm = ModelManager(config)
        mm.load_model()
        assert mm.is_ready() is True

    def test_get_active_model_not_none(self, config):
        mm = ModelManager(config)
        mm.load_model()
        assert mm.get_active_model() is not None

    def test_get_active_model_raises_before_load(self, config):
        mm = ModelManager(config)
        with pytest.raises(RuntimeError):
            mm.get_active_model()

    def test_get_model_info_has_required_fields(self, config):
        mm = ModelManager(config)
        mm.load_model()
        info = mm.get_model_info()
        assert info["model_name"] == "yolo11n"
        assert info["model_type"] == "yolo11"
        assert info["backend"] == "pytorch"
        assert info["input_width"] == 640
        assert info["input_height"] == 640
        assert "version" in info
        assert "device" in info

    def test_warmup_runs_without_error(self, config):
        mm = ModelManager(config)
        mm.load_model()
        mm.warmup()
