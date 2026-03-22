"""YAML configuration loader with typed dataclasses and validation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

VALID_BACKENDS = ("pytorch", "onnx", "tensorrt")
VALID_MODEL_TYPES = ("yolo11", "rfdetr")
VALID_SOURCES = ("local", "roboflow")


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    grpc_port: int = 50051
    http_port: int = 8080
    max_workers: int = 4


@dataclass
class ModelConfig:
    name: str
    type: str
    backend: str
    source: str = "local"
    path: Optional[str] = None
    version: str = "1.0.0"
    roboflow_project: Optional[str] = None
    roboflow_version: Optional[int] = None
    input_width: int = 640
    input_height: int = 640
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    class_names: list[str] = field(default_factory=list)


@dataclass
class InferenceConfig:
    batch_size: int = 1
    max_batch_size: int = 8
    dynamic_batching: bool = False


@dataclass
class WarmupConfig:
    enabled: bool = True
    iterations: int = 5
    synthetic_data: bool = True


@dataclass
class AppConfig:
    server: ServerConfig
    model: ModelConfig
    inference: InferenceConfig
    warmup: WarmupConfig


def _validate_model_config(model: ModelConfig) -> None:
    """Validate model configuration fields."""
    if model.type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Invalid model type '{model.type}'. Must be one of: {VALID_MODEL_TYPES}"
        )

    if model.backend not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend '{model.backend}'. Must be one of: {VALID_BACKENDS}"
        )

    if model.source not in VALID_SOURCES:
        raise ValueError(
            f"Invalid source '{model.source}'. Must be one of: {VALID_SOURCES}"
        )

    if model.source == "local" and not model.path:
        raise ValueError("model.path is required when source is 'local'")

    if model.source == "roboflow":
        if not model.roboflow_project:
            raise ValueError(
                "model.roboflow_project is required when source is 'roboflow'"
            )
        if model.roboflow_version is None:
            raise ValueError(
                "model.roboflow_version is required when source is 'roboflow'"
            )


def load_config(config_path: str) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If configuration is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw or "model" not in raw:
        raise ValueError("Config must contain a 'model' section")

    raw_server = raw.get("server", {}) or {}
    raw_model = raw.get("model", {})
    raw_inference = raw.get("inference", {}) or {}
    raw_warmup = raw.get("warmup", {}) or {}

    # Build model config — require type and input dimensions
    if "type" not in raw_model:
        raise ValueError("model.type is required")
    if "input_width" not in raw_model or "input_height" not in raw_model:
        raise ValueError("model.input_width and model.input_height are required")

    model_config = ModelConfig(
        name=raw_model.get("name", "unnamed"),
        type=raw_model["type"],
        backend=raw_model.get("backend", "pytorch"),
        source=raw_model.get("source", "local"),
        path=raw_model.get("path"),
        version=raw_model.get("version", "1.0.0"),
        roboflow_project=raw_model.get("roboflow_project"),
        roboflow_version=raw_model.get("roboflow_version"),
        input_width=raw_model["input_width"],
        input_height=raw_model["input_height"],
        confidence_threshold=raw_model.get("confidence_threshold", 0.5),
        iou_threshold=raw_model.get("iou_threshold", 0.45),
        class_names=raw_model.get("class_names", []) or [],
    )

    _validate_model_config(model_config)

    server_config = ServerConfig(
        host=raw_server.get("host", "0.0.0.0"),
        grpc_port=raw_server.get("grpc_port", 50051),
        http_port=raw_server.get("http_port", 8080),
        max_workers=raw_server.get("max_workers", 4),
    )

    inference_config = InferenceConfig(
        batch_size=raw_inference.get("batch_size", 1),
        max_batch_size=raw_inference.get("max_batch_size", 8),
        dynamic_batching=raw_inference.get("dynamic_batching", False),
    )

    warmup_config = WarmupConfig(
        enabled=raw_warmup.get("enabled", True),
        iterations=raw_warmup.get("iterations", 5),
        synthetic_data=raw_warmup.get("synthetic_data", True),
    )

    config = AppConfig(
        server=server_config,
        model=model_config,
        inference=inference_config,
        warmup=warmup_config,
    )

    logger.info(
        "Config loaded: model=%s type=%s backend=%s %dx%d",
        config.model.name,
        config.model.type,
        config.model.backend,
        config.model.input_width,
        config.model.input_height,
    )

    return config
