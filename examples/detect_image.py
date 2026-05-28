"""Minimal end-to-end example: run detection on a single image.

Start the server first (see the project README), then run:

    python examples/detect_image.py --image path/to/image.jpg

If no image is given, a synthetic test frame is generated so the script
runs with zero setup. The annotated result is written to --output.
"""

import argparse
import os
import sys

# Allow running directly (`python examples/detect_image.py`) from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import supervision as sv

from client.inference_client import InferenceClient


def load_image(path: str | None) -> np.ndarray:
    """Load an image from disk, or synthesize one if no path is given."""
    if path:
        image = cv2.imread(path)
        if image is None:
            sys.exit(f"Could not read image: {path}")
        return image

    # Synthetic 640x640 frame with a few shapes so the script is runnable
    # without any input file. Real models may detect nothing here -- that
    # is expected; the point is to exercise the full client/server path.
    frame = np.full((640, 640, 3), 64, dtype=np.uint8)
    cv2.rectangle(frame, (120, 120), (300, 380), (200, 200, 200), -1)
    cv2.circle(frame, (460, 300), 90, (180, 180, 180), -1)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="SimpleInference image demo")
    parser.add_argument("--image", help="Path to an input image")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port")
    parser.add_argument("--model", default="", help="Model name (multi-model routing)")
    parser.add_argument("--output", default="annotated.jpg", help="Output image path")
    args = parser.parse_args()

    image = load_image(args.image)

    client = InferenceClient(host=args.host, port=args.port)
    client.connect()        # waits until the server reports "ready"
    client.fetch_config()   # syncs model input dimensions

    detections = client.predict(image, model_name=args.model)
    print(f"Detected {len(detections)} object(s).")

    labels = [
        f"{class_id} {conf:.2f}"
        for class_id, conf in zip(detections.class_id, detections.confidence)
    ]
    annotated = sv.BoxAnnotator().annotate(image.copy(), detections)
    annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)

    cv2.imwrite(args.output, annotated)
    print(f"Wrote annotated image to {args.output}")


if __name__ == "__main__":
    main()
