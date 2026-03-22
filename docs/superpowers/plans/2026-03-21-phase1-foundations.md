# Phase 1: Foundations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a functional end-to-end inference server with YOLO11 detector, gRPC+HTTP APIs, portable client, and Docker deployment.

**Architecture:** gRPC server (bidirectional streaming proto, unary Predict in Phase 1) + FastAPI HTTP (/health, /config) running concurrently. ModelManager loads detectors via backend abstraction. Client is a single portable .py file. All detection output is `sv.Detections`.

**Tech Stack:** Python 3.11, grpcio, grpcio-tools, ultralytics, supervision, fastapi, uvicorn, opencv-python, numpy, pyyaml, onnxruntime, pytest, Docker

---

## File Structure

```
simpleinference/
├── server/
│   ├── proto/
│   │   └── detections.proto
│   ├── generated/
│   │   ├── __init__.py
│   │   ├── detections_pb2.py          (auto-generated)
│   │   └── detections_pb2_grpc.py     (auto-generated)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config_loader.py           — YAML parsing + typed dataclasses
│   │   └── device_manager.py          — CPU/CUDA detection + fallback
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base_backend.py            — ABC for inference backends
│   │   ├── pytorch_backend.py         — PyTorch runtime
│   │   └── onnx_backend.py            — ONNX Runtime
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base_detector.py           — ABC: preprocess/predict/postprocess
│   │   └── yolo11_detector.py         — YOLO11 via ultralytics
│   ├── services/
│   │   ├── __init__.py
│   │   ├── grpc_service.py            — InferenceServicer (Predict + GetServerConfig)
│   │   └── http_service.py            — FastAPI app (/health, /config)
│   ├── model_manager.py               — Load, warmup, expose active model
│   ├── main.py                        — Entrypoint: gRPC + FastAPI concurrent
│   ├── config.yaml                    — Default configuration
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
│       ├── conftest.py                — Shared fixtures
│       ├── test_config_loader.py
│       ├── test_device_manager.py
│       ├── test_backends.py
│       ├── test_detectors.py
│       └── test_grpc_service.py
├── client/
│   ├── inference_client.py            — Single portable file
│   └── tests/
│       ├── conftest.py
│       └── test_client.py
├── models/
│   └── .gitkeep
├── docker-compose.yml
├── .gitignore
└── JOURNAL.md
```

---

## Task 1: Project Scaffolding + Proto Definition (T1.1)

**Files:**
- Create: `server/proto/detections.proto`
- Create: `server/generated/__init__.py`
- Create: `server/requirements.txt`
- Create: `client/tests/__init__.py` (empty)
- Create: `server/tests/__init__.py` (empty)
- Create: `models/.gitkeep`
- Create: `.gitignore`
- Create: all `__init__.py` files for packages

**No TDD for this task** — it's scaffolding and proto compilation.

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p server/{proto,generated,core,backends,detectors,services,tests}
mkdir -p client/tests
mkdir -p models
```

- [ ] **Step 2: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.pyo
.env
models/*.pt
models/*.onnx
models/*.trt
models/*.engine
!models/.gitkeep
*.egg-info/
dist/
build/
.pytest_cache/
htmlcov/
.coverage
```

- [ ] **Step 3: Create all __init__.py files**

Empty `__init__.py` in: `server/core/`, `server/backends/`, `server/detectors/`, `server/services/`, `server/generated/`, `server/tests/`, `client/tests/`

- [ ] **Step 4: Create detections.proto**

```protobuf
syntax = "proto3";

package simpleinference;

service InferenceService {
  rpc Predict (InferenceRequest) returns (InferenceResponse) {}
  rpc StreamPredict (stream InferenceRequest) returns (stream InferenceResponse) {}
  rpc GetServerConfig (Empty) returns (ServerConfigResponse) {}
}

message Empty {}

message InferenceRequest {
  bytes image_data = 1;
  int32 width = 2;
  int32 height = 3;
}

message InferenceResponse {
  repeated Detection detections = 1;
  int32 image_width = 2;
  int32 image_height = 3;
  float inference_time_ms = 4;
}

message Detection {
  float x1 = 1;
  float y1 = 2;
  float x2 = 3;
  float y2 = 4;
  float confidence = 5;
  int32 class_id = 6;
  string class_name = 7;
}

message ServerConfigResponse {
  string model_name = 1;
  string model_type = 2;
  string backend = 3;
  int32 input_width = 4;
  int32 input_height = 5;
  string version = 6;
  string device = 7;
}
```

- [ ] **Step 5: Create requirements.txt**

```
grpcio>=1.62.0
grpcio-tools>=1.62.0
supervision>=0.25.0
ultralytics>=8.3.0
fastapi>=0.115.0
uvicorn>=0.34.0
pyyaml>=6.0
opencv-python>=4.9.0
numpy>=1.26.0
onnxruntime>=1.17.0
pydantic>=2.0.0
```

- [ ] **Step 6: Compile proto**

```bash
python -m grpc_tools.protoc \
  -I./server/proto \
  --python_out=./server/generated \
  --grpc_python_out=./server/generated \
  ./server/proto/detections.proto
```

Fix the generated import in `detections_pb2_grpc.py`: change `import detections_pb2` to `from server.generated import detections_pb2` (or use relative import).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "[PHASE1] feat: project scaffolding and proto definition (T1.1)"
```

---

## Task 2: Config Loader (T1.2)

**Files:**
- Create: `server/core/config_loader.py`
- Create: `server/config.yaml`
- Test: `server/tests/test_config_loader.py`
- Create: `server/tests/conftest.py`

- [ ] **Step 1: Write test file for config loader**

```python
# server/tests/test_config_loader.py
import pytest
import tempfile
import os
from pathlib import Path


class TestConfigLoader:
    """Tests for YAML config loading and validation."""

    def test_valid_config_loads_correctly(self, tmp_path):
        """A well-formed config.yaml should parse into typed dataclasses."""
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

        from server.core.config_loader import load_config
        config = load_config(str(config_file))

        assert config.server.host == "0.0.0.0"
        assert config.server.grpc_port == 50051
        assert config.model.name == "yolo11n"
        assert config.model.type == "yolo11"
        assert config.model.backend == "pytorch"
        assert config.model.input_width == 640
        assert config.model.confidence_threshold == 0.5

    def test_missing_model_type_raises_error(self, tmp_path):
        """Config missing required model.type should raise ValueError."""
        config_content = """
server:
  grpc_port: 50051
model:
  name: "test"
  backend: "pytorch"
  source: "local"
  path: "/models/test.pt"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        from server.core.config_loader import load_config
        with pytest.raises((ValueError, Exception)):
            load_config(str(config_file))

    def test_invalid_backend_raises_error(self, tmp_path):
        """Backend not in [pytorch, onnx, tensorrt] should raise ValueError."""
        config_content = """
server:
  grpc_port: 50051
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

        from server.core.config_loader import load_config
        with pytest.raises((ValueError, Exception)):
            load_config(str(config_file))

    def test_defaults_applied_when_fields_omitted(self, tmp_path):
        """Omitted optional fields should get sensible defaults."""
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

        from server.core.config_loader import load_config
        config = load_config(str(config_file))

        assert config.server.grpc_port == 50051
        assert config.server.http_port == 8080
        assert config.server.max_workers == 4
        assert config.model.confidence_threshold == 0.5
        assert config.model.iou_threshold == 0.45
        assert config.warmup.enabled is True
        assert config.warmup.iterations == 5

    def test_roboflow_source_requires_project_fields(self, tmp_path):
        """source=roboflow without project/version should raise."""
        config_content = """
model:
  name: "test"
  type: "yolo11"
  backend: "pytorch"
  source: "roboflow"
  input_width: 640
  input_height: 640
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        from server.core.config_loader import load_config
        with pytest.raises((ValueError, Exception)):
            load_config(str(config_file))

    def test_local_source_requires_path(self, tmp_path):
        """source=local without path should raise."""
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

        from server.core.config_loader import load_config
        with pytest.raises((ValueError, Exception)):
            load_config(str(config_file))

    def test_custom_model_path_accepted(self, tmp_path):
        """Fine-tuned model with custom path and optional class_names should load."""
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
  class_names:
    - "forklift"
    - "person"
    - "pallet"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        from server.core.config_loader import load_config
        config = load_config(str(config_file))

        assert config.model.name == "forklift-detector-v3"
        assert config.model.path == "/models/custom/forklift-v3.onnx"
        assert config.model.class_names == ["forklift", "person", "pallet"]

    def test_class_names_optional(self, tmp_path):
        """class_names should default to empty list when omitted."""
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

        from server.core.config_loader import load_config
        config = load_config(str(config_file))
        assert config.model.class_names == []
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest server/tests/test_config_loader.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: Implement config_loader.py**

Create `server/core/config_loader.py` with:
- `ServerConfig` dataclass: host, grpc_port, http_port, max_workers (all with defaults)
- `ModelConfig` dataclass: name, version, type, backend, source, path (optional), roboflow_project (optional), roboflow_version (optional), input_width, input_height, confidence_threshold, iou_threshold, class_names (optional list — default empty)
- `InferenceConfig` dataclass: batch_size, max_batch_size, dynamic_batching
- `WarmupConfig` dataclass: enabled, iterations, synthetic_data
- `AppConfig` top-level dataclass composing all above
- `load_config(path: str) -> AppConfig` function: load YAML, validate, return typed config
- Validation: backend in {pytorch, onnx, tensorrt}, type in {yolo11, rfdetr}, source-specific field checks

Key: `class_names` is `list[str]` defaulting to `[]` — no validation against pretrained names. `name` is freeform.

- [ ] **Step 4: Create default config.yaml**

```yaml
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
  # class_names: []  # optional — leave empty for pretrained COCO classes

inference:
  batch_size: 1
  max_batch_size: 8
  dynamic_batching: false

warmup:
  enabled: true
  iterations: 5
  synthetic_data: true
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest server/tests/test_config_loader.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add server/core/config_loader.py server/config.yaml server/tests/test_config_loader.py server/tests/conftest.py
git commit -m "[PHASE1] feat: config loader with typed dataclasses and YAML validation (T1.2)"
```

---

## Task 3: Device Manager (T1.3)

**Files:**
- Create: `server/core/device_manager.py`
- Test: `server/tests/test_device_manager.py`

- [ ] **Step 1: Write tests**

```python
# server/tests/test_device_manager.py
import pytest
from server.core.device_manager import DeviceManager


class TestDeviceManager:
    def test_cpu_always_available(self):
        dm = DeviceManager()
        assert "cpu" in dm.available_devices()

    def test_get_optimal_device_returns_string(self):
        dm = DeviceManager()
        device = dm.get_optimal_device()
        assert isinstance(device, str)
        assert device in ("cpu", "cuda", "cuda:0")

    def test_get_device_for_backend_cpu(self):
        dm = DeviceManager()
        device = dm.get_device_for_backend("pytorch")
        assert isinstance(device, str)

    def test_cuda_detection_does_not_raise(self):
        """CUDA check should never raise, just return bool."""
        dm = DeviceManager()
        result = dm.is_cuda_available()
        assert isinstance(result, bool)

    def test_onnx_gpu_detection_does_not_raise(self):
        dm = DeviceManager()
        result = dm.is_onnx_gpu_available()
        assert isinstance(result, bool)

    def test_device_info_returns_dict(self):
        dm = DeviceManager()
        info = dm.get_device_info()
        assert "device" in info
        assert "cuda_available" in info
```

- [ ] **Step 2: Run tests — verify fail**

```bash
pytest server/tests/test_device_manager.py -v
```

- [ ] **Step 3: Implement device_manager.py**

`DeviceManager` class with:
- `is_cuda_available()` — wraps `torch.cuda.is_available()` in try/except
- `is_onnx_gpu_available()` — checks if `onnxruntime` has CUDA provider
- `available_devices()` — returns list of available device strings
- `get_optimal_device()` — returns best device with fallback chain
- `get_device_for_backend(backend: str)` — returns appropriate device for given backend
- `get_device_info()` — returns dict with device details for /config

- [ ] **Step 4: Run tests — verify pass**

```bash
pytest server/tests/test_device_manager.py -v
```

- [ ] **Step 5: Commit**

```bash
git add server/core/device_manager.py server/tests/test_device_manager.py
git commit -m "[PHASE1] feat: device manager with CPU/CUDA detection and fallback (T1.3)"
```

---

## Task 4: Backend Layer — PyTorch + ONNX (T1.4)

**Files:**
- Create: `server/backends/base_backend.py`
- Create: `server/backends/pytorch_backend.py`
- Create: `server/backends/onnx_backend.py`
- Test: `server/tests/test_backends.py`

- [ ] **Step 1: Write tests**

```python
# server/tests/test_backends.py
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
        """PyTorch should be available in dev environment."""
        assert PyTorchBackend.is_available() is True


class TestONNXBackend:
    def test_is_available_returns_bool(self):
        result = ONNXBackend.is_available()
        assert isinstance(result, bool)

    def test_is_available_does_not_raise(self):
        """is_available should never throw, even if onnxruntime is missing."""
        try:
            ONNXBackend.is_available()
        except Exception:
            pytest.fail("is_available() raised an exception")
```

- [ ] **Step 2: Run tests — verify fail**

```bash
pytest server/tests/test_backends.py -v
```

- [ ] **Step 3: Implement base_backend.py**

```python
from abc import ABC, abstractmethod
import numpy as np


class BaseBackend(ABC):
    @abstractmethod
    def load(self, model_path: str, device: str) -> None:
        """Load model from path onto specified device."""
        ...

    @abstractmethod
    def infer(self, tensor: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed tensor. Returns raw output."""
        ...

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this backend's runtime is installed and functional."""
        ...
```

- [ ] **Step 4: Implement pytorch_backend.py**

Uses `ultralytics.YOLO` model's internal predict — but the backend abstraction wraps the raw model load and forward pass. For YOLO11 specifically, we'll leverage ultralytics' own model loading.

Key: PyTorchBackend stores the model reference and device. `infer()` accepts a preprocessed numpy array and returns raw predictions.

- [ ] **Step 5: Implement onnx_backend.py**

Uses `onnxruntime.InferenceSession`. Provider selection: `['CUDAExecutionProvider', 'CPUExecutionProvider']`. `infer()` runs the session and returns output arrays.

- [ ] **Step 6: Run tests — verify pass**

```bash
pytest server/tests/test_backends.py -v
```

- [ ] **Step 7: Commit**

```bash
git add server/backends/
git commit -m "[PHASE1] feat: backend abstraction with PyTorch and ONNX implementations (T1.4)"
```

---

## Task 5: BaseDetector + YOLO11Detector (T1.5)

**Files:**
- Create: `server/detectors/base_detector.py`
- Create: `server/detectors/yolo11_detector.py`
- Test: `server/tests/test_detectors.py`

- [ ] **Step 1: Write tests**

```python
# server/tests/test_detectors.py
import pytest
import numpy as np
import supervision as sv
from server.detectors.base_detector import BaseDetector


class TestBaseDetector:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseDetector(config=None)


class TestYOLO11Detector:
    @pytest.fixture
    def detector(self):
        """Load YOLO11n (nano) for testing."""
        from server.detectors.yolo11_detector import YOLO11Detector
        from server.core.config_loader import ModelConfig
        config = ModelConfig(
            name="yolo11n",
            type="yolo11",
            backend="pytorch",
            source="local",
            path="yolo11n.pt",  # ultralytics auto-downloads nano
            input_width=640,
            input_height=640,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )
        return YOLO11Detector(config)

    @pytest.fixture
    def synthetic_image(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_predict_returns_sv_detections(self, detector, synthetic_image):
        result = detector.predict(synthetic_image)
        assert isinstance(result, sv.Detections)

    def test_predict_detections_have_correct_fields(self, detector, synthetic_image):
        result = detector.predict(synthetic_image)
        assert hasattr(result, 'xyxy')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'class_id')

    def test_warmup_completes_without_error(self, detector):
        detector.warmup(n_iterations=2)

    def test_predict_with_real_image(self, detector):
        """Test with a real image that should produce detections."""
        import cv2
        # Use the test image from .claude/resources if available
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = detector.predict(img)
        assert isinstance(result, sv.Detections)
        # xyxy should be float32 array with shape (N, 4)
        if len(result) > 0:
            assert result.xyxy.shape[1] == 4
            assert result.confidence is not None
            assert result.class_id is not None
```

- [ ] **Step 2: Run tests — verify fail**

```bash
pytest server/tests/test_detectors.py -v
```

- [ ] **Step 3: Implement base_detector.py**

ABC with:
- `__init__(self, config: ModelConfig)` — stores config
- `preprocess(image) -> Any` — abstract
- `postprocess(raw_output, original_shape) -> sv.Detections` — abstract
- `predict(image) -> sv.Detections` — concrete: preprocess → infer → postprocess
- `warmup(n_iterations=5)` — concrete: generates synthetic data, runs predict N times

- [ ] **Step 4: Implement yolo11_detector.py**

`YOLO11Detector(BaseDetector)`:
- `__init__`: loads `ultralytics.YOLO(config.path)` — this handles both .pt and .onnx
- `predict`: uses `self.model(image, conf=threshold, iou=iou_threshold, verbose=False)` — ultralytics handles pre/post internally
- Converts ultralytics `Results` → `sv.Detections` using `sv.Detections.from_ultralytics()`
- This approach leverages ultralytics' optimized pipeline rather than reimplementing preprocessing

For ONNX backend: `ultralytics.YOLO("model.onnx")` handles ONNX transparently.

- [ ] **Step 5: Run tests — verify pass**

```bash
pytest server/tests/test_detectors.py -v
```

- [ ] **Step 6: Commit**

```bash
git add server/detectors/ server/tests/test_detectors.py
git commit -m "[PHASE1] feat: BaseDetector abstraction and YOLO11Detector implementation (T1.5)"
```

---

## Task 6: Model Manager (T1.6)

**Files:**
- Create: `server/model_manager.py`
- Test (add to): `server/tests/test_detectors.py` or create `server/tests/test_model_manager.py`

- [ ] **Step 1: Write tests**

```python
# server/tests/test_model_manager.py
import pytest
from server.model_manager import ModelManager
from server.core.config_loader import load_config


class TestModelManager:
    @pytest.fixture
    def config(self, tmp_path):
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
        f = tmp_path / "config.yaml"
        f.write_text(config_content)
        return load_config(str(f))

    def test_load_model_succeeds(self, config):
        mm = ModelManager(config)
        mm.load_model()
        assert mm.get_active_model() is not None

    def test_get_model_info_returns_dict(self, config):
        mm = ModelManager(config)
        mm.load_model()
        info = mm.get_model_info()
        assert "model_name" in info
        assert "model_type" in info
        assert "backend" in info
        assert "input_width" in info

    def test_is_ready_false_before_load(self, config):
        mm = ModelManager(config)
        assert mm.is_ready() is False

    def test_is_ready_true_after_load(self, config):
        mm = ModelManager(config)
        mm.load_model()
        assert mm.is_ready() is True
```

- [ ] **Step 2: Run tests — verify fail**

```bash
pytest server/tests/test_model_manager.py -v
```

- [ ] **Step 3: Implement model_manager.py**

`ModelManager`:
- `__init__(config: AppConfig)` — stores config, sets `_active_detector = None`
- `load_model()` — instantiates correct detector based on `config.model.type`, calls `warmup()` if configured
- `warmup()` — delegates to `detector.warmup(config.warmup.iterations)`
- `get_active_model() -> BaseDetector` — returns active detector, raises if not loaded
- `is_ready() -> bool` — returns whether model is loaded
- `get_model_info() -> dict` — returns model metadata for /config endpoint
- Detector registry: `{"yolo11": YOLO11Detector}` — extensible for RF-DETR in Phase 2

- [ ] **Step 4: Run tests — verify pass**

```bash
pytest server/tests/test_model_manager.py -v
```

- [ ] **Step 5: Commit**

```bash
git add server/model_manager.py server/tests/test_model_manager.py
git commit -m "[PHASE1] feat: ModelManager with load, warmup, and model info (T1.6)"
```

---

## Task 7: gRPC Service (T1.7)

**Files:**
- Create: `server/services/grpc_service.py`
- Test: `server/tests/test_grpc_service.py`

- [ ] **Step 1: Write tests**

```python
# server/tests/test_grpc_service.py
import pytest
import numpy as np
import cv2
import grpc
from unittest.mock import MagicMock
from server.generated import detections_pb2, detections_pb2_grpc


class TestGRPCService:
    @pytest.fixture
    def servicer(self):
        from server.services.grpc_service import InferenceServicer
        from server.model_manager import ModelManager
        from server.core.config_loader import load_config
        import tempfile, os

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
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with open(path, 'w') as f:
            f.write(config_content)
        config = load_config(path)
        os.unlink(path)

        mm = ModelManager(config)
        mm.load_model()
        return InferenceServicer(mm)

    def test_predict_returns_response(self, servicer):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', img)
        request = detections_pb2.InferenceRequest(
            image_data=encoded.tobytes(),
            width=640,
            height=480
        )
        context = MagicMock()
        response = servicer.Predict(request, context)
        assert isinstance(response, detections_pb2.InferenceResponse)
        assert response.inference_time_ms >= 0

    def test_predict_corrupted_image_aborts(self, servicer):
        request = detections_pb2.InferenceRequest(
            image_data=b"not_an_image",
            width=640,
            height=480
        )
        context = MagicMock()
        servicer.Predict(request, context)
        context.abort.assert_called_once()

    def test_get_server_config_returns_config(self, servicer):
        request = detections_pb2.Empty()
        context = MagicMock()
        response = servicer.GetServerConfig(request, context)
        assert isinstance(response, detections_pb2.ServerConfigResponse)
        assert response.model_name == "yolo11n"
        assert response.model_type == "yolo11"
        assert response.input_width == 640
```

- [ ] **Step 2: Run tests — verify fail**

```bash
pytest server/tests/test_grpc_service.py -v
```

- [ ] **Step 3: Implement grpc_service.py**

`InferenceServicer(detections_pb2_grpc.InferenceServiceServicer)`:
- `__init__(model_manager: ModelManager)`
- `Predict(request, context)`:
  1. Decode JPEG from `request.image_data` via `cv2.imdecode`
  2. If decode fails → `context.abort(INVALID_ARGUMENT, "...")`
  3. `model_manager.get_active_model().predict(image)`
  4. Convert `sv.Detections` → repeated `Detection` messages
  5. Return `InferenceResponse` with detections + inference_time_ms
- `GetServerConfig(request, context)`:
  1. `model_manager.get_model_info()` → populate `ServerConfigResponse`
- `StreamPredict` — stub for Phase 2 (returns UNIMPLEMENTED)

- [ ] **Step 4: Run tests — verify pass**

```bash
pytest server/tests/test_grpc_service.py -v
```

- [ ] **Step 5: Commit**

```bash
git add server/services/grpc_service.py server/tests/test_grpc_service.py
git commit -m "[PHASE1] feat: gRPC InferenceServicer with Predict and GetServerConfig (T1.7)"
```

---

## Task 8: HTTP Service — FastAPI (T1.8)

**Files:**
- Create: `server/services/http_service.py`
- Test (optional inline): test via `TestClient` from fastapi

- [ ] **Step 1: Write tests**

```python
# Add to server/tests/test_http_service.py
import pytest
from fastapi.testclient import TestClient


class TestHTTPService:
    @pytest.fixture
    def client(self):
        from server.services.http_service import create_app
        from server.model_manager import ModelManager
        from server.core.config_loader import load_config
        import tempfile, os

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
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with open(path, 'w') as f:
            f.write(config_content)
        config = load_config(path)
        os.unlink(path)

        mm = ModelManager(config)
        mm.load_model()
        app = create_app(mm)
        return TestClient(app)

    def test_health_ready(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_config_endpoint(self, client):
        response = client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "input_width" in data
        assert "backend" in data
```

- [ ] **Step 2: Run tests — verify fail**

```bash
pytest server/tests/test_http_service.py -v
```

- [ ] **Step 3: Implement http_service.py**

`create_app(model_manager: ModelManager) -> FastAPI`:
- `GET /health` — returns `{"status": "ready"|"loading", "model": name, "uptime_seconds": float}`
- `GET /config` — returns model_manager.get_model_info() as JSON

Track server start time for uptime calculation.

- [ ] **Step 4: Run tests — verify pass**

```bash
pytest server/tests/test_http_service.py -v
```

- [ ] **Step 5: Commit**

```bash
git add server/services/http_service.py server/tests/test_http_service.py
git commit -m "[PHASE1] feat: FastAPI HTTP service with /health and /config (T1.8)"
```

---

## Task 9: Main Entrypoint (T1.9)

**Files:**
- Create: `server/main.py`

**No separate test file** — main.py is tested via integration tests. Focus on correct signal handling and concurrent startup.

- [ ] **Step 1: Implement main.py**

```python
# server/main.py
import logging
import signal
import sys
import threading
from concurrent import futures

import grpc
import uvicorn

from server.core.config_loader import load_config
from server.core.device_manager import DeviceManager
from server.model_manager import ModelManager
from server.services.grpc_service import InferenceServicer
from server.services.http_service import create_app
from server.generated import detections_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def serve(config_path: str = "server/config.yaml"):
    config = load_config(config_path)

    # Device info
    device_manager = DeviceManager()
    logger.info(f"Device: {device_manager.get_optimal_device()}")

    # Load model
    model_manager = ModelManager(config)
    logger.info(f"Loading model: {config.model.name} ({config.model.type}/{config.model.backend})")
    model_manager.load_model()

    if config.warmup.enabled:
        logger.info(f"Warming up ({config.warmup.iterations} iterations)...")
        model_manager.warmup()

    logger.info("Model ready.")

    # gRPC server
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.server.max_workers))
    servicer = InferenceServicer(model_manager)
    detections_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, grpc_server)
    grpc_server.add_insecure_port(f"{config.server.host}:{config.server.grpc_port}")

    # FastAPI app
    app = create_app(model_manager)

    # Graceful shutdown
    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info("Shutdown signal received.")
        shutdown_event.set()
        grpc_server.stop(grace=5)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Start gRPC in thread
    grpc_server.start()
    logger.info(f"gRPC server listening on {config.server.host}:{config.server.grpc_port}")

    # Start FastAPI in main thread
    logger.info(f"HTTP server listening on {config.server.host}:{config.server.http_port}")
    uvicorn.run(app, host=config.server.host, port=config.server.http_port, log_level="info")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "server/config.yaml"
    serve(config_path)
```

- [ ] **Step 2: Verify all existing tests still pass**

```bash
pytest server/tests/ -v
```

- [ ] **Step 3: Commit**

```bash
git add server/main.py
git commit -m "[PHASE1] feat: main entrypoint with concurrent gRPC + FastAPI (T1.9)"
```

---

## Task 10: InferenceClient (T1.10)

**Files:**
- Create: `client/inference_client.py`
- Test: `client/tests/test_client.py`
- Create: `client/tests/conftest.py`

- [ ] **Step 1: Write tests**

```python
# client/tests/test_client.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestInferenceClient:
    def test_init_defaults(self):
        from client.inference_client import InferenceClient
        c = InferenceClient()
        assert c._host == "localhost"
        assert c._port == 50051
        assert c._http_port == 8080
        assert c._jpeg_quality == 85

    def test_connect_timeout_raises(self):
        from client.inference_client import InferenceClient
        c = InferenceClient(host="localhost", port=59999, timeout_seconds=2)
        with pytest.raises(ConnectionError):
            c.connect()

    def test_resize_image_to_model_dims(self):
        from client.inference_client import InferenceClient
        c = InferenceClient()
        c._input_width = 640
        c._input_height = 640
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        resized = c._prepare_image(img)
        # Should be JPEG bytes
        assert isinstance(resized, bytes)
        assert len(resized) > 0
```

- [ ] **Step 2: Run tests — verify fail**

```bash
pytest client/tests/test_client.py -v
```

- [ ] **Step 3: Implement inference_client.py**

Single-file portable client. Dependencies: grpcio, opencv-python, numpy, supervision + generated pb2 files.

`InferenceClient`:
- `__init__(host, port, http_port, jpeg_quality, timeout_seconds)`
- `connect()` — polls `http://{host}:{http_port}/health` every 2s until `{"status": "ready"}` or timeout
- `fetch_config()` — GET `/config`, updates `_input_width`, `_input_height`
- `predict(image: np.ndarray) -> sv.Detections` — resize, JPEG encode, gRPC Predict, deserialize response
- `_prepare_image(image) -> bytes` — resize + JPEG encode
- `_deserialize_response(response) -> sv.Detections` — proto → sv.Detections

Include the generated proto imports at top with a note that `detections_pb2.py` and `detections_pb2_grpc.py` must be in the same directory.

- [ ] **Step 4: Run tests — verify pass**

```bash
pytest client/tests/test_client.py -v
```

- [ ] **Step 5: Commit**

```bash
git add client/
git commit -m "[PHASE1] feat: portable InferenceClient with connect, fetch_config, predict (T1.10)"
```

---

## Task 11: Docker (T1.11)

**Files:**
- Create: `server/Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create Dockerfile**

Multi-stage build:
- Stage 1 (`builder`): install grpc-tools, compile proto
- Stage 2 (`production`): python:3.11-slim, copy compiled proto + server code, install runtime deps

```dockerfile
FROM python:3.11-slim-bookworm AS builder
WORKDIR /app
COPY server/proto/ server/proto/
COPY server/requirements.txt .
RUN pip install grpcio-tools && \
    python -m grpc_tools.protoc \
      -I./server/proto \
      --python_out=./server/generated \
      --grpc_python_out=./server/generated \
      ./server/proto/detections.proto

FROM python:3.11-slim-bookworm AS production
WORKDIR /app
COPY --from=builder /app/server/generated/ server/generated/
COPY server/ server/
COPY client/ client/
RUN pip install --no-cache-dir -r server/requirements.txt
EXPOSE 50051 8080
CMD ["python", "-m", "server.main"]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
services:
  inference-server:
    build:
      context: .
      dockerfile: server/Dockerfile
    ports:
      - "50051:50051"
      - "8080:8080"
    volumes:
      - ./models:/models
      - ./server/config.yaml:/app/server/config.yaml
    environment:
      - ROBOFLOW_API_KEY=${ROBOFLOW_API_KEY:-}
    # Uncomment for GPU:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

- [ ] **Step 3: Create models/.gitkeep**

```bash
touch models/.gitkeep
```

- [ ] **Step 4: Verify all tests pass**

```bash
pytest server/tests/ client/tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add server/Dockerfile docker-compose.yml models/.gitkeep
git commit -m "[PHASE1] feat: Dockerfile and docker-compose for containerized deployment (T1.11)"
```

---

## Task 12: Integration Verification + JOURNAL Update

- [ ] **Step 1: Run full test suite with coverage**

```bash
pytest server/tests/ client/tests/ -v --cov=server --cov=client --cov-report=term-missing
```

Verify >= 80% coverage on new code.

- [ ] **Step 2: Update JOURNAL.md**

Add Phase 1 completion entry with: architecture decisions, components built, test results.

- [ ] **Step 3: Final commit**

```bash
git add JOURNAL.md
git commit -m "[PHASE1] docs: journal entry for Phase 1 completion"
```
