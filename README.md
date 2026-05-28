# SimpleInference

A real-time computer vision inference server. Stream camera frames over gRPC, get object detections back with low latency. Supports YOLO11 and RF-DETR on PyTorch, ONNX, and TensorRT backends. Designed for live video pipelines, drones, edge devices, and any application that wants detection-as-a-service instead of embedding a model directly.

**Highlights**

- Bidirectional gRPC streaming + lightweight HTTP API
- Multi-model routing and runtime hot-swap
- Optional dynamic batching for GPU throughput
- P50/P95/P99 latency, throughput, and error metrics
- Portable single-file Python client
- Production Docker image with healthcheck

---

## Requirements

- Linux (tested on Ubuntu 22.04+)
- Python 3.11+
- Optional: NVIDIA GPU with CUDA 12.x for accelerated inference

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/xcanonx12/vision_inference_server.git
cd vision_inference_server
pip install -r requirements.txt
```

The default `requirements.txt` installs CPU-only PyTorch. For NVIDIA GPU support, reinstall with the matching CUDA build:

```bash
pip install --force-reinstall torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### Compile the gRPC proto files

The generated stubs are not checked in. Run once after installing dependencies:

```bash
mkdir -p server/generated
python -m grpc_tools.protoc \
    -I./server/proto \
    --python_out=./server/generated \
    --grpc_python_out=./server/generated \
    ./server/proto/detections.proto
sed -i 's/^import detections_pb2/from . import detections_pb2/' \
    server/generated/detections_pb2_grpc.py
touch server/generated/__init__.py
```

### Add a model

Place a model file under `models/`. YOLO11n is downloaded automatically on first launch if missing:

```bash
mkdir -p models
```

---

## Running the server

```bash
python -m server.main
```

Wait for the log line `Model ready.` The server listens on:

- `:50051` — gRPC inference
- `:8080` — HTTP control plane

---

## Running an example

With the server running, in a second terminal:

```bash
python examples/detect_image.py --image path/to/image.jpg
```

Or with no input file (uses a synthetic frame):

```bash
python examples/detect_image.py
```

Produces `annotated.jpg` in the working directory. See `examples/README.md` for streaming and webcam examples.

---

## Docker

Build and start the server with Docker Compose:

```bash
docker compose up --build
```

Ports `50051` (gRPC) and `8080` (HTTP) are exposed. Models are mounted read-only from `./models`. The image runs as a non-root user and includes a healthcheck on `/health`.

Stop:

```bash
docker compose down
```

---

## Configuration

### Environment (`.env`)

Copy the example and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `CONFIG_PATH` | `server/config.yaml` | Path to the YAML config file |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

`docker compose` reads `.env` automatically.

### Server config (`server/config.yaml`)

```yaml
server:
  host: "0.0.0.0"
  grpc_port: 50051
  http_port: 8080
  max_workers: 4

model:
  name: "yolo11n"
  type: "yolo11"          # yolo11 | rfdetr
  backend: "pytorch"      # pytorch | onnx | tensorrt
  source: "local"
  path: "models/yolo11n.pt"
  input_width: 640
  input_height: 640
  confidence_threshold: 0.5
  iou_threshold: 0.45

inference:
  batch_size: 1
  max_batch_size: 8
  dynamic_batching: false

warmup:
  enabled: true
  iterations: 5
  synthetic_data: true
```

---

## Using the client from another project

The client is a single file with no framework dependencies beyond `grpcio`, `opencv-python`, `numpy`, `supervision`, and `httpx`. Copy these into your project:

```
client/inference_client.py
server/generated/detections_pb2.py
server/generated/detections_pb2_grpc.py
server/generated/__init__.py
```

Then:

```python
import cv2
from client.inference_client import InferenceClient

client = InferenceClient(host="localhost", port=50051)
client.connect()         # blocks until /health reports "ready"
client.fetch_config()    # syncs input dimensions

frame = cv2.imread("image.jpg")
detections = client.predict(frame)   # returns sv.Detections
print(len(detections), "objects")
```

Streaming:

```python
def frames():
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame

for detections in client.stream_predict(frames()):
    # process detections per frame
    ...
```

---

## API reference

### gRPC service (`:50051`)

Service: `simpleinference.InferenceService`

| RPC | Type | Input | Output |
|---|---|---|---|
| `Predict` | unary | `InferenceRequest` | `InferenceResponse` |
| `StreamPredict` | bidi stream | stream of `InferenceRequest` | stream of `InferenceResponse` |
| `GetServerConfig` | unary | `Empty` | `ServerConfigResponse` |

**`InferenceRequest`**

| Field | Type | Description |
|---|---|---|
| `image_data` | `bytes` | JPEG-encoded frame |
| `width` | `int32` | Original frame width |
| `height` | `int32` | Original frame height |
| `model_name` | `string` | Target model (empty = default) |

**`InferenceResponse`**

| Field | Type | Description |
|---|---|---|
| `detections` | `repeated Detection` | List of detections |
| `image_width` | `int32` | Width used for inference |
| `image_height` | `int32` | Height used for inference |
| `inference_time_ms` | `float` | Server-side inference latency |

**`Detection`**

| Field | Type | Description |
|---|---|---|
| `bbox` | `repeated float` | `[x1, y1, x2, y2]` in input-image pixels |
| `confidence` | `float` | Score in `[0, 1]` |
| `class_id` | `int32` | Class index |

**`ServerConfigResponse`**

| Field | Type |
|---|---|
| `model_name` | `string` |
| `model_type` | `string` |
| `backend` | `string` |
| `input_width` | `int32` |
| `input_height` | `int32` |
| `version` | `string` |
| `device` | `string` |
| `num_classes` | `int32` |

### HTTP control plane (`:8080`)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness and readiness probe |
| `GET` | `/config` | Active model metadata |
| `GET` | `/metrics` | Latency, throughput, error rate |
| `POST` | `/hot-swap` | Replace active model at runtime |

**`GET /health`** — `200 OK`

```json
{
  "status": "ready",
  "model": "yolo11n",
  "uptime_seconds": 12.4
}
```

`status` is `"loading"` while the model warms up.

**`GET /metrics`** — `200 OK`

```json
{
  "model": "yolo11n",
  "backend": "pytorch",
  "device": "cuda:0",
  "uptime_seconds": 312.0,
  "inference": {
    "total_requests": 9001,
    "total_errors": 0,
    "error_rate_percent": 0.0,
    "throughput_fps": 29.8,
    "latency_ms": {"p50": 14.1, "p95": 22.6, "p99": 31.0}
  },
  "batching": {"enabled": false, "avg_batch_size": 1.0},
  "memory": {}
}
```

**`POST /hot-swap`** — request body:

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
  "iou_threshold": 0.45
}
```

Response:

```json
{"status": "swapped", "model": "yolo11s"}
```

---

## Tests

```bash
pytest server/tests/                       # unit + integration (fast)
pytest -m slow server/tests/test_load.py   # load tests (require running server)
```

---

## Project layout

```
.
├── client/                     # portable Python client
├── examples/                   # runnable example scripts
├── models/                     # model weights (gitignored)
├── server/
│   ├── core/                   # config, device, batching, model manager
│   ├── services/               # gRPC + HTTP services
│   ├── proto/                  # .proto definitions
│   ├── generated/              # generated gRPC stubs (compiled locally)
│   ├── tests/
│   ├── config.yaml
│   ├── Dockerfile
│   └── main.py
├── requirements.txt
├── docker-compose.yml
└── .env.example
```
