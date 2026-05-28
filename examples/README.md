# Examples

Runnable scripts demonstrating the SimpleInference client. Run them from the
repository root with the virtual environment active and a server running.

## 1. Start the server

```bash
python -m server.main
```

Wait for the log line `Model ready.` before running an example.

## 2. Detect on an image

```bash
# Use your own image:
python examples/detect_image.py --image path/to/image.jpg

# Or run with a synthetic frame (no input file needed):
python examples/detect_image.py
```

The script connects to the server, runs detection, prints the object count,
and writes an annotated image to `annotated.jpg` (override with `--output`).

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | synthetic frame | Input image path |
| `--host` | `localhost` | Server host |
| `--port` | `50051` | gRPC port |
| `--model` | server default | Model name for multi-model routing |
| `--output` | `annotated.jpg` | Annotated output path |

## Streaming from a webcam

The client also supports bidirectional streaming. Minimal loop:

```python
import cv2
from client.inference_client import InferenceClient

def frames():
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame

client = InferenceClient()
client.connect()
client.fetch_config()

for detections in client.stream_predict(frames()):
    print(f"{len(detections)} objects")
```
