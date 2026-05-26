# SimpleInference

A gRPC-based real-time vision inference server designed for live cameras, drones, and edge devices. Returns universal `sv.Detections` (Roboflow Supervision) from YOLO11 and RF-DETR models, with PyTorch, ONNX, and TensorRT backends.

---

## Features

- **Bidirectional gRPC streaming** — zero-copy frame pipeline for real-time throughput
- **Multi-model routing** — run multiple models concurrently, select by `model_name` per request
- **Dynamic batching** — configurable batch window for GPU utilization
- **Hot-swap** — replace the active model at runtime with no downtime
- **Metrics** — sliding-window P50/P95/P99 latency percentiles, throughput, error rate
- **Production Docker** — healthcheck, non-root user, resource limits
- **Portable client** — single `.py` file, copy it anywhere

---

## Quick Start

### Docker (recommended)

```bash
# Place model weights in ./models/
mkdir -p models
cp yolo11n.pt models/

# Start the server
docker compose up

# Verify it's ready
curl http://localhost:8080/health
# {"status": "ready", "model": "yolo11n", "uptime_seconds": 5.2}
```

### Local (development)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (installs CPU torch by default)
pip install -r server/requirements.txt

# GPU (optional): reinstall the CUDA torch build matching your driver.
# For CUDA 12.x drivers (e.g. RTX laptops):
pip install --reinstall torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

# Compile proto (only needed once, or after proto changes)
python -m grpc_tools.protoc \
  -I./server/proto \
  --python_out=./server/generated \
  --grpc_python_out=./server/generated \
  ./server/proto/detections.proto

# Start the server
python -m server.main
# or with a custom config:
python -m server.main path/to/config.yaml
```

The server exposes:
- gRPC on `0.0.0.0:50051`
- HTTP on `0.0.0.0:8080`

---

## Client Usage

The client lives at `client/inference_client.py`. Copy it alongside the generated `detections_pb2.py` / `detections_pb2_grpc.py` files into any project.

**Dependencies:** `grpcio`, `opencv-python`, `numpy`, `supervision`, `httpx`

### Single image inference

```python
from client.inference_client import InferenceClient
import cv2

client = InferenceClient(host="localhost", port=50051)
client.connect()       # polls /health until "ready"
client.fetch_config()  # syncs input dimensions from server

frame = cv2.imread("image.jpg")
detections = client.predict(frame)  # returns sv.Detections

print(f"{len(detections)} objects detected")
print(detections.xyxy)       # [[x1, y1, x2, y2], ...]
print(detections.confidence) # [0.93, 0.87, ...]
print(detections.class_id)   # [0, 2, ...]
```

### Model routing (multi-model)

```python
# Route to a specific model by name
detections = client.predict(frame, model_name="detector")
detections = client.predict(frame, model_name="classifier")
```

### Streaming (bidirectional)

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

# With model routing:
for detections in client.stream_predict(camera_feed(), model_name="detector"):
    ...
```

### Context manager

```python
with InferenceClient(host="localhost", port=50051) as client:
    client.connect()
    client.fetch_config()
    frame = cv2.imread("image.jpg")
    detections = client.predict(frame)
```

### Constructor options

```python
InferenceClient(
    host="localhost",
    port=50051,
    http_port=8080,
    jpeg_quality=85,      # JPEG compression for transmission (1–100)
    timeout_seconds=30.0, # connect() timeout
)
```

---

## Configuration Reference

Edit `server/config.yaml` before starting the server.

### `server`

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `"0.0.0.0"` | Bind address for both gRPC and HTTP servers |
| `grpc_port` | `50051` | gRPC listen port |
| `http_port` | `8080` | HTTP (FastAPI) listen port |
| `max_workers` | `4` | gRPC thread pool size |
| `request_timeout_seconds` | `30.0` | Per-request deadline |
| `max_image_bytes` | `10485760` | Max payload size (10 MB) |

### `model`

| Field | Default | Description |
|-------|---------|-------------|
| `name` | `"yolo11n"` | Model identifier, used for routing |
| `version` | `"1.0.0"` | Informational version string |
| `type` | `"yolo11"` | Model architecture: `yolo11` or `rfdetr` |
| `backend` | `"pytorch"` | Inference backend: `pytorch`, `onnx`, or `tensorrt` |
| `source` | `"local"` | Weight source: `local` or `roboflow` |
| `path` | `"models/yolo11n.pt"` | Path to weights file (`.pt`, `.onnx`, `.engine`, or `.pth`). Relative to repo root. If the file is a known pretrained name (e.g. `yolo11n.pt`, `rf-detr-nano.pth`) and missing, it is auto-downloaded to this path. |
| `input_width` | `640` | Model input width (pixels) |
| `input_height` | `640` | Model input height (pixels) |
| `confidence_threshold` | `0.5` | Minimum detection confidence |
| `iou_threshold` | `0.45` | NMS IoU threshold |
| `trt_fp16` | `false` | Enable FP16 precision for TensorRT backend |
| `roboflow_project` | `null` | Roboflow project ID (required when `source: roboflow`) |
| `roboflow_version` | `null` | Roboflow model version (required when `source: roboflow`) |

### `inference`

| Field | Default | Description |
|-------|---------|-------------|
| `batch_size` | `1` | Default batch size |
| `max_batch_size` | `8` | Maximum batch size for dynamic batching |
| `dynamic_batching` | `false` | Enable dynamic batching |
| `batch_window_ms` | `10.0` | Max wait time to fill a batch (milliseconds) |

### `warmup`

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Run warmup on server start |
| `iterations` | `5` | Number of warmup inference passes |
| `synthetic_data` | `true` | Use synthetic frames for warmup |

### Multi-model mode

Replace the single `model:` block with a `models:` list. Each entry accepts the same fields as `model`:

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

Clients route to a model by name: `client.predict(frame, model_name="detector")`.

---

## API Reference

### gRPC (port 50051)

Defined in `server/proto/detections.proto`.

#### `Predict` — unary

Send one frame, receive one response.

```
InferenceRequest  →  InferenceResponse
```

#### `StreamPredict` — bidirectional streaming

Send a stream of frames, receive a stream of responses. Maintains one persistent connection for the full session.

```
stream InferenceRequest  →  stream InferenceResponse
```

#### `GetServerConfig` — unary

Fetch active model metadata.

```
Empty  →  ServerConfigResponse
```

#### Message types

```protobuf
message InferenceRequest {
  bytes  image_data = 1;  // JPEG-encoded image bytes
  int32  width      = 2;  // image width (pixels)
  int32  height     = 3;  // image height (pixels)
  string model_name = 4;  // target model; empty = server default
}

message InferenceResponse {
  repeated Detection detections      = 1;
  int32              image_width     = 2;
  int32              image_height    = 3;
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

Returns server readiness.

```json
{"status": "ready", "model": "yolo11n", "uptime_seconds": 42.3}
```

`status` is `"loading"` until the model is fully initialized.

#### `GET /config`

Returns active model configuration.

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

Returns inference statistics with sliding-window latency percentiles.

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

Replace the active model at runtime without restarting the server.

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

| Type | Config `type` | Weights format |
|------|---------------|----------------|
| YOLO11 | `yolo11` | `.pt` (PyTorch), `.onnx`, `.engine` (TensorRT) |
| RF-DETR | `rfdetr` | `.pth` (PyTorch), `.onnx` |

### Backends

| Backend | Config `backend` | Notes |
|---------|-----------------|-------|
| PyTorch | `pytorch` | Default; runs on CPU or CUDA automatically |
| ONNX Runtime | `onnx` | Cross-platform; good for CPU deployment |
| TensorRT | `tensorrt` | Fastest GPU inference; requires `.engine` file |

#### TensorRT: building an engine

```bash
# Export from YOLO11 .pt to TensorRT .engine
python -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='engine', half=True)  # half=True for FP16
"
# Update config.yaml:
#   backend: tensorrt
#   path: yolo11n.engine
#   trt_fp16: true
```

#### Custom / fine-tuned models

Set `source: local` and point `path` to your weights file. The `type` and `backend` fields control how they are loaded — custom weights work the same as pretrained ones.

```yaml
model:
  name: "my-detector"
  type: "yolo11"
  backend: "pytorch"
  source: "local"
  path: "models/my-detector.pt"
```

#### RF-DETR

RF-DETR is self-contained (the `rfdetr` library handles loading and pre/post-processing). Point `path` at a `.pth` checkpoint; a known pretrained name is auto-downloaded to that path if missing.

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

#### Roboflow-hosted models

```yaml
model:
  name: "my-detector"
  type: "yolo11"
  source: "roboflow"
  roboflow_project: "my-project"
  roboflow_version: 3
  input_width: 640
  input_height: 640
```

Set `ROBOFLOW_API_KEY` in the environment (or in a `.env` file at the project root).

---

## GPU Support (Docker)

Uncomment the GPU reservation block in `docker-compose.yml`:

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

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## Running Tests

```bash
source .venv/bin/activate

# Unit tests
pytest server/tests/ client/tests/ -v -m "not slow"

# With coverage
pytest server/tests/ client/tests/ --cov=server --cov=client --cov-report=term-missing

# Load / slow tests
pytest server/tests/test_load.py -m slow -v
```

---

## Project Structure

```
inference_server/
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
│   │   └── http_service.py   # FastAPI app
│   └── tests/
├── client/
│   ├── inference_client.py   # Portable client (copy this file)
│   └── tests/
├── models/                   # Mount point for weight files
└── docker-compose.yml
```
