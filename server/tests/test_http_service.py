import pytest
from fastapi.testclient import TestClient

from server.core.config_loader import load_config
from server.core.model_manager import ModelManager
from server.services.http_service import create_app


@pytest.fixture(scope="module")
def loaded_manager(tmp_path_factory):
    config_content = """
model:
  name: "yolo11n"
  type: "yolo11"
  backend: "pytorch"
  source: "local"
  path: "yolo11n.pt"
  input_width: 640
  input_height: 640
warmup:
  enabled: false
"""
    tmp = tmp_path_factory.mktemp("config")
    f = tmp / "config.yaml"
    f.write_text(config_content)
    mm = ModelManager(load_config(str(f)))
    mm.load_model()
    return mm


@pytest.fixture(scope="module")
def client_ready(loaded_manager):
    app = create_app(loaded_manager)
    return TestClient(app)


@pytest.fixture
def client_loading(tmp_path):
    """Client with model NOT loaded — server in loading state."""
    config_content = """
model:
  name: "yolo11n"
  type: "yolo11"
  backend: "pytorch"
  source: "local"
  path: "yolo11n.pt"
  input_width: 640
  input_height: 640
warmup:
  enabled: false
"""
    f = tmp_path / "config.yaml"
    f.write_text(config_content)
    mm = ModelManager(load_config(str(f)))
    app = create_app(mm)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_ready(self, client_ready):
        response = client_ready.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "model" in data
        assert "uptime_seconds" in data

    def test_health_loading(self, client_loading):
        response = client_loading.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "loading"


class TestConfigEndpoint:
    def test_config_returns_model_info(self, client_ready):
        response = client_ready.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "yolo11n"
        assert data["model_type"] == "yolo11"
        assert data["backend"] == "pytorch"
        assert data["input_width"] == 640
        assert data["input_height"] == 640
