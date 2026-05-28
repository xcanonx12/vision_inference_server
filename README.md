# SimpleInference

SimpleInference is a real-time computer vision inference server built for applications that need low-latency object detection over a network. It is designed for live camera feeds, drone payloads, edge devices, and any system that requires streaming detection results without embedding a model directly in the application. The server exposes both a bidirectional gRPC streaming interface and a lightweight HTTP API, and returns results as `sv.Detections` objects from the [Roboflow Supervision](https://github.com/roboflow/supervision) library — making downstream annotation, filtering, and tracking straightforward.

---

## Features

- **Bidirectional gRPC streaming** — persistent connection, one frame in, one detection response out, no per-request handshake overhead
- **Multi-model routing** — load multiple models simultaneously and select the target model per request by name
- **Dynamic batching** — configurable batch window to maximize GPU utilization under load
- **Runtime hot-swap** — replace the active model via HTTP with no server restart
- **Latency metrics** — sliding-window P50/P95/P99 latency percentiles, throughput, and error rate via `GET /metrics`
- **Production Docker image** — non-root user, healthcheck, resource limits
- **Portable single-file client** — `client/inference_client.py` with no framework dependencies beyond `grpcio` and `supervision`

---

## Installation and Setup

### Option A: Docker (recommended)

Prerequisites: Docker, Docker Compose, and (for GPU) the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

**1. Place model weights in `./models/`.**

```bash
mkdir -p models
cp yolo11n.pt models/
```

**2. Configure the environment.**

```bash
cp .env.example .env
# Edit .env if you need a custom CONFIG_PATH or LOG_LEVEL
```

**3. Start the server.**

```bash
docker compose up
```

For GPU support, uncomment the `deploy.resources.reservations.devices` block in `docker-compose.yml` before starting:

```yaml
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

**4. Verify the server is ready.**

```bash
curl http://localhost:8080/health
# {"status": "ready", "model": "yolo11n", "uptime_seconds": 5.2}
```

Wait until `"status": "ready"` before sending inference requests.

---

### Option B: Local development install

Prerequisites: Python 3.11+, and optionally an NVIDIA GPU with CUDA 12.x drivers.

**1. Create and activate a virtual environment.**

```bash
python -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies.**

```bash
pip install -r server/requirements.txt
# Installs CPU-only PyTorch by default.
```

**3. (Optional) Install GPU-accelerated PyTorch for NVIDIA GPUs (CUDA 12.x).**

```bash
pip install --reinstall torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

**4. Compile the protobuf definition (once, or after proto changes).**

```bash
python -m grpc_tools.protoc \
  -I./server/proto \
  --python_out=./server/generated \
  --grpc_python_out=./server/generated \
  ./server/proto/detections.proto
```

**5. Place model weights in `./models/` and start the server.**

```bash
mkdir -p models
cp yolo11n.pt models/

python -m server.main
# or with a custom config:
python -m server.main path/to/config.yaml
```

**6. Verify the server is ready.**

```bash
curl http://localhost:8080/health
# {"status": "ready", "model": "yolo11n", "uptime_seconds": 3.1}
```

The server listens on:
- gRPC: `0.0.0.0:50051`
- HTTP: `0.0.0.0:8080`

---

## Client Usage

The client is `client/inference_client.py`. To use it in another project, copy that file along with the generated `detections_pb2.py` and `detections_pb2_grpc.py` files.

**Dependencies:** `grpcio`, `opencv-python`, `numpy`, `supervision`, `httpx`

### Single image

```python
from client.inference_client import InferenceClient
import cv2

client = InferenceClient(host="localhost", port=50051)
client.connect()       # polls /health until "ready"
client.fetch_config()  # syncs input dimensions from the server

frame = cv2.imread("image.jpg")
detections = client.predict(frame)  # returns sv.Detections

print(f"{len(detections)} objects detected")
print(detections.xyxy)        # [[x1, y1, x2, y2], ...]
print(detections.confidence)  # [0.93, 0.87, ...]
print(detections.class_id)    # [0, 2, ...]
```

### Bidirectional streaming

```python
def camera_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

for detections in client.stream_predict(camera_feed()):
    print(f"{len(detections)} objects detected")
```

For complete runnable examples — including annotated image output, CLI flags, and multi-model usage — see [`examples/`](examples/) and start with:

```bash
python examples/detect_image.py
```

---

## Environment Variables

Copy `.env.example` to `.env` to set defaults for `docker compose`. These variables are also read directly by the server process.

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_PATH` | `server/config.yaml` | Path to the YAML config file |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

A CLI argument (`python -m server.main path/to/config.yaml`) takes precedence over `CONFIG_PATH`.

---

## Configuration Reference

Edit `server/config.yaml` before starting the server.

### `server`

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `"0.0.0.0"` | Bind address for both gRPC and HTTP servers |
| `grpc_port` | `50051` | gRPC listen port |
| `http_port` | `8080` | HTTP listen port |
| `max_workers` | `4` | gRPC thread pool size |
| `request_timeout_seconds` | `30.0` | Per-request deadline |
| `max_image_bytes` | `10485760` | Maximum payload size (10 MB) |

### `model`

| Field | Default | Description |
|-------|---------|-------------|
| `name` | `"yolo11n"` | Model identifier used for routing |
| `version` | `"1.0.0"` | Informational version string |
| `type` | `"yolo11"` | Model architecture: `yolo11` or `rfdetr` |
| `backend` | `"pytorch"` | Inference backend: `pytorch`, `onnx`, or `tensorrt` |
| `source` | `"local"` | Weight source. Only `local` is supported. |
| `path` | `"models/yolo11n.pt"` | Path to weights file (`.pt`, `.onnx`, `.engine`, or `.pth`). Relative to repo root. Known pretrained names are auto-downloaded if the file is missing. |
| `input_width` | `640` | Model input width in pixels |
| `input_height` | `640` | Model input height in pixels |
| `confidence_threshold` | `0.5` | Minimum detection confidence |
| `iou_threshold` | `0.45` | NMS IoU threshold |
| `trt_fp16` | `false` | Enable FP16 precision for TensorRT backend |

### `inference`

| Field | Default | Description |
|-------|---------|-------------|
| `batch_size` | `1` | Default batch size |
| `max_batch_size` | `8` | Maximum batch size for dynamic batching |
| `dynamic_batching` | `false` | Enable dynamic batching |
| `batch_window_ms` | `10.0` | Maximum wait time to fill a batch (milliseconds) |

### `warmup`

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Run warmup passes on server start |
| `iterations` | `5` | Number of warmup inference passes |
| `synthetic_data` | `true` | Use synthetic frames for warmup |

### Multi-model mode

Replace the single `model:` block with a `models:` list. Each entry accepts the same fields:

```yaml
models:
  - name: "detector"
    type: "yolo11"
    backend: "pytorch"
    path: "models/yolo11n.pt"
    input_width: 640
    input_height: 640
    confidence_threshold: 0.5
    iou_threshold: 0.45

  - name: "classifier"
    type: "yolo11"
    backend: "onnx"
    path: "models/yolo11n-cls.onnx"
    input_width: 224
    input_height: 224
    confidence_threshold: 0.5
    iou_threshold: 0.45
```

Route to a specific model per request: `client.predict(frame, model_name="detector")`.

---

## API Reference

### gRPC (port 50051)

Defined in `server/proto/detections.proto`.

| RPC | Type | Description |
|-----|------|-------------|
| `Predict` | Unary | Send one frame, receive one response |
| `StreamPredict` | Bidirectional streaming | Persistent frame stream with per-frame responses |
| `GetServerConfig` | Unary | Fetch active model metadata |

#### Message schemas

```protobuf
message InferenceRequest {
  bytes  image_data = 1;  // JPEG-encoded image bytes
  int32  width      = 2;  // image width (pixels)
  int32  height     = 3;  // image height (pixels)
  string model_name = 4;  // target model; empty = server default
}

message InferenceResponse {
  repeated Detection detections        = 1;
  int32              image_width       = 2;
  int32              image_height      = 3;
  float              inference_time_ms = 4;
}

message Detection {
  repeated float bbox       = 1;  // [x1, y1, x2, y2]
  float          confidence = 2;
  int32          class_id   = 3;
}

message ServerConfigResponse {
  string model_name   = 1;
  string model_type   = 2;
  string backend      = 3;
  int32  input_width  = 4;
  int32  input_height = 5;
  string version      = 6;
  string device       = 7;
  int32  num_classes  = 8;
}
```

---

### HTTP (port 8080)

#### `GET /health`

```json
{"status": "ready", "model": "yolo11n", "uptime_seconds": 42.3}
```

`status` is `"loading"` until the model finishes initializing.

#### `GET /config`

```json
{
  "model_name": "yolo11n",
  "model_type": "yolo11",
  "backend": "pytorch",
  "device": "cuda:0",
  "input_width": 640,
  "input_height": 640,
  "version": "1.0.0",
  "num_classes": 80
}
```

#### `GET /metrics`

```json
{
  "model": "yolo11n",
  "backend": "pytorch",
  "device": "cuda:0",
  "uptime_seconds": 120.5,
  "inference": {
    "total_requests": 1500,
    "total_errors": 0,
    "error_rate_percent": 0.0,
    "throughput_fps": 45.2,
    "latency_ms": {
      "p50": 18.3,
      "p95": 24.1,
      "p99": 31.7,
      "min": 8.2,
      "max": 45.6
    }
  },
  "batching": {
    "enabled": false,
    "avg_batch_size": 1.0
  },
  "memory": {
    "gpu_used_mb": 512.0,
    "gpu_total_mb": 8192.0
  }
}
```

#### `POST /hot-swap`

Replace the active model without restarting the server.

Request body:

```json
{
  "name": "yolo11s",
  "type": "yolo11",
  "backend": "pytorch",
  "source": "local",
  "path": "models/yolo11s.pt",
  "input_width": 640,
  "input_height": 640,
  "confidence_threshold": 0.5,
  "iou_threshold": 0.45,
  "trt_fp16": false
}
```

Response:

```json
{"status": "swapped", "model": "yolo11s"}
```

---

## Supported Models and Backends

### Models

| Architecture | Config `type` | Supported weight formats |
|--------------|---------------|--------------------------|
| YOLO11 | `yolo11` | `.pt` (PyTorch), `.onnx`, `.engine` (TensorRT) |
| RF-DETR | `rfdetr` | `.pth` (PyTorch), `.onnx` |

### Backends

| Backend | Config `backend` | Notes |
|---------|-----------------|-------|
| PyTorch | `pytorch` | Default; runs on CPU or CUDA automatically |
| ONNX Runtime | `onnx` | Cross-platform; well-suited for CPU deployment |
| TensorRT | `tensorrt` | Maximum GPU throughput; requires a compiled `.engine` file |

#### Building a TensorRT engine

```bash
python -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='engine', half=True)  # half=True for FP16
"
# Then update config.yaml:
#   backend: tensorrt
#   path: yolo11n.engine
#   trt_fp16: true
```

#### Custom and fine-tuned models

Set `source: local` and point `path` at your weights file. The `type` and `backend` fields control loading — custom weights work identically to pretrained ones.

```yaml
model:
  name: "my-detector"
  type: "yolo11"
  backend: "pytorch"
  source: "local"
  path: "models/my-detector.pt"
```

#### RF-DETR

RF-DETR loading and pre/post-processing is handled by the `rfdetr` library. Point `path` at a `.pth` checkpoint; known pretrained names are auto-downloaded if missing.

```yaml
model:
  name: "rfdetr-nano"
  type: "rfdetr"
  backend: "pytorch"
  source: "local"
  path: "models/rf-detr-nano.pth"
  input_width: 384
  input_height: 384
  confidence_threshold: 0.5
  iou_threshold: 0.45
```

---

## Running Tests

```bash
source .venv/bin/activate

# Unit and integration tests
pytest server/tests/ client/tests/ -v -m "not slow"

# With coverage
pytest server/tests/ client/tests/ --cov=server --cov=client --cov-report=term-missing

# Load tests (sustained throughput, concurrent clients, memory stability)
pytest server/tests/test_load.py -m slow -v
```

---

## Project Structure

```
vision_inference/
├── server/
│   ├── main.py               # Server entrypoint
│   ├── config.yaml           # Default configuration
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── proto/
│   │   └── detections.proto
│   ├── generated/            # Auto-generated pb2 files (do not edit)
│   ├── core/
│   │   ├── config_loader.py
│   │   ├── device_manager.py
│   │   ├── model_manager.py
│   │   ├── metrics_collector.py
│   │   └── batch_manager.py
│   ├── backends/             # PyTorch / ONNX / TensorRT loaders
│   ├── detectors/            # YOLO11 and RF-DETR detectors
│   ├── services/
│   │   ├── grpc_service.py   # gRPC InferenceServicer
│   │   └── http_service.py   # FastAPI HTTP app
│   └── tests/
├── client/
│   ├── inference_client.py   # Portable client (copy this file)
│   └── tests/
├── examples/                 # Runnable client scripts (see examples/README.md)
├── models/                   # Mount point for weight files
├── .env.example              # Environment variable template
└── docker-compose.yml
```
