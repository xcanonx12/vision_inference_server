import pytest

from server.core.config_loader import load_config


class TestConfigLoaderValid:
    def test_valid_config_loads_correctly(self, tmp_path):
        config_content = """
server:
  host: "0.0.0.0"
  grpc_port: 50051
  http_port: 8080
  max_workers: 4

model:
  name: "yolo11n"
  version: "1.0.0"
  type: "yolo11"
  backend: "pytorch"
  source: "local"
  path: "/models/yolo11n.pt"
  input_width: 640
  input_height: 640
  confidence_threshold: 0.5
  iou_threshold: 0.45

inference:
  batch_size: 1

warmup:
  enabled: true
  iterations: 5
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        config = load_config(str(config_file))

        assert config.server.host == "0.0.0.0"
        assert config.server.grpc_port == 50051
        assert config.server.http_port == 8080
        assert config.model.name == "yolo11n"
        assert config.model.type == "yolo11"
        assert config.model.backend == "pytorch"
        assert config.model.input_width == 640
        assert config.model.confidence_threshold == 0.5
        assert config.warmup.enabled is True
        assert config.warmup.iterations == 5

    def test_defaults_applied_when_fields_omitted(self, tmp_path):
        config_content = """
model:
  name: "yolo11n"
  type: "yolo11"
  backend: "pytorch"
  source: "local"
  path: "/models/yolo11n.pt"
  input_width: 640
  input_height: 640
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        config = load_config(str(config_file))

        assert config.server.grpc_port == 50051
        assert config.server.http_port == 8080
        assert config.server.max_workers == 4
        assert config.model.confidence_threshold == 0.5
        assert config.model.iou_threshold == 0.45
        assert config.warmup.enabled is True
        assert config.warmup.iterations == 5
        assert config.inference.batch_size == 1

    def test_custom_model_path_accepted(self, tmp_path):
        config_content = """
model:
  name: "forklift-detector-v3"
  type: "yolo11"
  backend: "onnx"
  source: "local"
  path: "/models/custom/forklift-v3.onnx"
  input_width: 640
  input_height: 640
  confidence_threshold: 0.6
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        config = load_config(str(config_file))

        assert config.model.name == "forklift-detector-v3"
        assert config.model.path == "/models/custom/forklift-v3.onnx"
        assert config.model.confidence_threshold == 0.6


class TestConfigLoaderValidation:
    def test_missing_model_type_raises_error(self, tmp_path):
        config_content = """
server:
  grpc_port: 50051
model:
  name: "test"
  backend: "pytorch"
  source: "local"
  path: "/models/test.pt"
  input_width: 640
  input_height: 640
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        with pytest.raises((ValueError, Exception)):
            load_config(str(config_file))

    def test_invalid_backend_raises_error(self, tmp_path):
        config_content = """
model:
  name: "test"
  type: "yolo11"
  backend: "invalid_backend"
  source: "local"
  path: "/models/test.pt"
  input_width: 640
  input_height: 640
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        with pytest.raises((ValueError, Exception)):
            load_config(str(config_file))

    def test_invalid_model_type_raises_error(self, tmp_path):
        config_content = """
model:
  name: "test"
  type: "invalid_type"
  backend: "pytorch"
  source: "local"
  path: "/models/test.pt"
  input_width: 640
  input_height: 640
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        with pytest.raises((ValueError, Exception)):
            load_config(str(config_file))

    def test_invalid_source_raises(self, tmp_path):
        config_content = """
model:
  name: "test"
  type: "yolo11"
  backend: "pytorch"
  source: "unknown"
  path: "models/test.pt"
  input_width: 640
  input_height: 640
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        with pytest.raises((ValueError, Exception)):
            load_config(str(config_file))

    def test_local_source_requires_path(self, tmp_path):
        config_content = """
model:
  name: "test"
  type: "yolo11"
  backend: "pytorch"
  source: "local"
  input_width: 640
  input_height: 640
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        with pytest.raises((ValueError, Exception)):
            load_config(str(config_file))

    def test_missing_input_dimensions_raises_error(self, tmp_path):
        config_content = """
model:
  name: "test"
  type: "yolo11"
  backend: "pytorch"
  source: "local"
  path: "/models/test.pt"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        with pytest.raises((ValueError, Exception)):
            load_config(str(config_file))
