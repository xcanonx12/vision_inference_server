import pytest
import threading

from server.core.model_manager import ModelManager
from server.core.config_loader import load_config, ModelConfig, _validate_model_config


@pytest.fixture(scope="module")
def base_config(tmp_path_factory):
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
"""
    tmp = tmp_path_factory.mktemp("config")
    f = tmp / "config.yaml"
    f.write_text(config_content)
    return load_config(str(f))


@pytest.fixture(scope="module")
def loaded_manager(base_config):
    """Single ModelManager with yolo11n loaded, shared across tests."""
    mm = ModelManager(base_config)
    mm.load_model()
    return mm


class TestHotSwap:
    def test_hot_swap_same_model_type(self, loaded_manager):
        """Hot-swap with the same model type succeeds."""
        new_model_cfg = ModelConfig(
            name="yolo11n",
            type="yolo11",
            backend="pytorch",
            source="local",
            path="yolo11n.pt",
            input_width=640,
            input_height=640,
            confidence_threshold=0.3,
        )
        loaded_manager.hot_swap(new_model_cfg)
        assert loaded_manager.is_ready()
        info = loaded_manager.get_model_info()
        assert info["confidence_threshold"] == 0.3

    def test_hot_swap_updates_model_info(self, loaded_manager):
        new_model_cfg = ModelConfig(
            name="yolo11n",
            type="yolo11",
            backend="pytorch",
            source="local",
            path="yolo11n.pt",
            input_width=640,
            input_height=640,
            confidence_threshold=0.8,
        )
        loaded_manager.hot_swap(new_model_cfg)
        info = loaded_manager.get_model_info()
        assert info["confidence_threshold"] == 0.8

    def test_hot_swap_invalid_type_raises(self, loaded_manager):
        bad_cfg = ModelConfig(
            name="bad", type="invalid_type", backend="pytorch",
            source="local", path="x.pt", input_width=640, input_height=640,
        )
        with pytest.raises(ValueError):
            loaded_manager.hot_swap(bad_cfg)
        # Manager should still be ready with previous model
        assert loaded_manager.is_ready()

    def test_hot_swap_thread_safety(self, base_config):
        """Concurrent hot_swap calls don't corrupt state."""
        mm = ModelManager(base_config)
        mm.load_model()
        errors = []

        def swap():
            try:
                new_cfg = ModelConfig(
                    name="yolo11n", type="yolo11", backend="pytorch",
                    source="local", path="yolo11n.pt",
                    input_width=640, input_height=640,
                )
                mm.hot_swap(new_cfg)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=swap) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert mm.is_ready()


class TestHotSwapHTTPEndpoint:
    @pytest.fixture(scope="module")
    def client_ready(self, base_config):
        from fastapi.testclient import TestClient
        from server.services.http_service import create_app

        mm = ModelManager(base_config)
        mm.load_model()
        app = create_app(mm)
        return TestClient(app)

    def test_hot_swap_endpoint_returns_200(self, client_ready):
        payload = {
            "name": "yolo11n",
            "type": "yolo11",
            "backend": "pytorch",
            "source": "local",
            "path": "yolo11n.pt",
            "input_width": 640,
            "input_height": 640,
        }
        response = client_ready.post("/hot-swap", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ready", "swapped")

    def test_hot_swap_endpoint_invalid_model_returns_error(self, client_ready):
        payload = {
            "name": "invalid",
            "type": "invalid_type",
            "backend": "pytorch",
            "source": "local",
            "path": "nonexistent.pt",
            "input_width": 640,
            "input_height": 640,
        }
        response = client_ready.post("/hot-swap", json=payload)
        assert response.status_code == 500
