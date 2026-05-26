"""Tests for multi-model concurrent execution."""

import numpy as np
import pytest
import supervision as sv

from server.core.config_loader import (
    AppConfig, InferenceConfig, ModelConfig, ServerConfig, WarmupConfig,
)
from server.core.model_manager import ModelManager


@pytest.fixture(scope="module")
def multi_model_config():
    """Config with two models (both yolo11n but different names/sizes)."""
    return AppConfig(
        server=ServerConfig(),
        model=ModelConfig(
            name="detector", type="yolo11", backend="pytorch",
            path="models/yolo11n.pt", input_width=640, input_height=640,
        ),
        inference=InferenceConfig(),
        warmup=WarmupConfig(enabled=False),
        models=[
            ModelConfig(
                name="detector", type="yolo11", backend="pytorch",
                path="models/yolo11n.pt", input_width=640, input_height=640,
            ),
            ModelConfig(
                name="small-det", type="yolo11", backend="pytorch",
                path="models/yolo11n.pt", input_width=320, input_height=320,
            ),
        ],
    )


@pytest.fixture(scope="module")
def loaded_multi_manager(multi_model_config):
    mm = ModelManager(multi_model_config)
    mm.load_model()
    return mm


class TestConcurrentModels:

    def test_get_model_by_name(self, loaded_multi_manager):
        det = loaded_multi_manager.get_model("detector")
        assert det is not None
        small = loaded_multi_manager.get_model("small-det")
        assert small is not None
        assert det is not small

    def test_empty_name_returns_default(self, loaded_multi_manager):
        default = loaded_multi_manager.get_model("")
        by_name = loaded_multi_manager.get_model("detector")
        assert default is by_name

    def test_unknown_model_raises(self, loaded_multi_manager):
        with pytest.raises(RuntimeError, match="not found"):
            loaded_multi_manager.get_model("nonexistent")

    def test_hot_swap_one_model_preserves_other(self, loaded_multi_manager):
        original_small = loaded_multi_manager.get_model("small-det")
        new_cfg = ModelConfig(
            name="detector", type="yolo11", backend="pytorch",
            path="models/yolo11n.pt", input_width=640, input_height=640,
        )
        loaded_multi_manager.hot_swap(new_cfg)
        assert loaded_multi_manager.get_model("small-det") is original_small

    def test_get_model_info_for_specific_model(self, loaded_multi_manager):
        info = loaded_multi_manager.get_model_info("small-det")
        assert info["model_name"] == "small-det"
        assert info["input_width"] == 320

    def test_predict_on_both_models(self, loaded_multi_manager, synthetic_image):
        det1 = loaded_multi_manager.get_model("detector")
        det2 = loaded_multi_manager.get_model("small-det")
        r1 = det1.predict(synthetic_image)
        r2 = det2.predict(synthetic_image)
        # Both return sv.Detections (may be empty for random image)
        assert isinstance(r1, sv.Detections)
        assert isinstance(r2, sv.Detections)
