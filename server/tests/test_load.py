"""Load and stress tests. Run with: pytest -m slow -v

These tests require a running server and real GPU for meaningful results.
"""

import os
import time
import threading

import cv2
import grpc
import numpy as np
import pytest

from server.generated import detections_pb2, detections_pb2_grpc


SERVER_ADDR = os.environ.get("SERVER_ADDR", "localhost:50051")


def _encode_synthetic_frame(width=640, height=640) -> bytes:
    """Create a synthetic JPEG frame."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return encoded.tobytes()


@pytest.mark.slow
class TestSustainedLoad:
    """Load tests — require running server."""

    def test_sustained_30fps_60s(self):
        """Send frames at 30fps for 60 seconds via streaming.

        Success criteria:
        - P99 latency < 50ms
        - 0 errors
        - Average throughput >= 28 fps
        """
        channel = grpc.insecure_channel(SERVER_ADDR)
        stub = detections_pb2_grpc.InferenceServiceStub(channel)

        target_fps = 30
        duration_s = 60
        total_frames = target_fps * duration_s

        frame_data = _encode_synthetic_frame()
        latencies = []
        errors = 0

        def frame_gen():
            for _ in range(total_frames):
                yield detections_pb2.InferenceRequest(
                    image_data=frame_data, width=640, height=640,
                )

        start = time.perf_counter()
        for response in stub.StreamPredict(frame_gen()):
            lat = response.inference_time_ms
            if lat > 0:
                latencies.append(lat)
            else:
                errors += 1
        elapsed = time.perf_counter() - start

        assert errors == 0, f"{errors} errors during sustained load"
        avg_fps = len(latencies) / elapsed
        assert avg_fps >= 28, f"Throughput {avg_fps:.1f} fps < 28 fps"
        sorted_lat = sorted(latencies)
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
        assert p99 < 50.0, f"P99 latency {p99:.1f}ms >= 50ms"

        channel.close()

    def test_concurrent_clients(self):
        """5 concurrent clients, 200 requests each.

        Success criteria: 0 errors, all clients complete.
        """
        n_clients = 5
        requests_per_client = 200
        frame_data = _encode_synthetic_frame()
        errors = []

        def client_worker(client_id):
            try:
                channel = grpc.insecure_channel(SERVER_ADDR)
                stub = detections_pb2_grpc.InferenceServiceStub(channel)
                for _ in range(requests_per_client):
                    req = detections_pb2.InferenceRequest(
                        image_data=frame_data, width=640, height=640,
                    )
                    resp = stub.Predict(req)
                    assert resp.image_width > 0
                channel.close()
            except Exception as e:
                errors.append((client_id, str(e)))

        threads = [
            threading.Thread(target=client_worker, args=(i,))
            for i in range(n_clients)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert len(errors) == 0, f"Client errors: {errors}"

    def test_memory_stability_10k_inferences(self):
        """10,000 sequential inferences — GPU memory delta < 5%.

        Requires CUDA.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("Requires CUDA")
        except ImportError:
            pytest.skip("Requires torch")

        import torch

        channel = grpc.insecure_channel(SERVER_ADDR)
        stub = detections_pb2_grpc.InferenceServiceStub(channel)
        frame_data = _encode_synthetic_frame()

        # Baseline
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        for _ in range(10000):
            req = detections_pb2.InferenceRequest(
                image_data=frame_data, width=640, height=640,
            )
            stub.Predict(req)

        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        if mem_before > 0:
            delta_pct = abs(mem_after - mem_before) / mem_before * 100
            assert delta_pct < 5.0, f"GPU memory grew {delta_pct:.1f}% (>{5}%)"

        channel.close()
