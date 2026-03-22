"""In-memory metrics collector with sliding window percentiles."""

import threading
import time
from collections import deque

import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Thread-safe metrics collector using a sliding time window.

    Records inference latencies, batch sizes, and errors.
    Computes percentiles (P50/P95/P99) over the configured window.
    """

    def __init__(self, window_seconds: int = 60) -> None:
        self._window_seconds = window_seconds
        self._lock = threading.Lock()

        # Sliding window stores (timestamp, latency_ms) tuples
        self._latencies: deque[tuple[float, float]] = deque()
        self._batch_sizes: deque[tuple[float, int]] = deque()

        # Cumulative counters (not windowed)
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._start_time: float = time.monotonic()

    def record_inference(self, latency_ms: float, batch_size: int) -> None:
        """Record a successful inference."""
        now = time.monotonic()
        with self._lock:
            self._latencies.append((now, latency_ms))
            self._batch_sizes.append((now, batch_size))
            self._total_requests += 1

    def record_error(self, error_type: str) -> None:
        """Record an inference error."""
        with self._lock:
            self._total_errors += 1
        logger.debug("Metrics: error recorded — %s", error_type)

    def reset(self) -> None:
        """Clear all metrics (e.g. after hot-swap)."""
        with self._lock:
            self._latencies.clear()
            self._batch_sizes.clear()
            self._total_requests = 0
            self._total_errors = 0
            self._start_time = time.monotonic()

    def get_metrics(self) -> dict:
        """Return current metrics snapshot."""
        with self._lock:
            now = time.monotonic()
            self._prune(now)

            window_latencies = [lat for _, lat in self._latencies]
            window_batches = [bs for _, bs in self._batch_sizes]

            total_req = self._total_requests
            total_err = self._total_errors
            uptime = now - self._start_time

        # Compute outside lock
        total_events = total_req + total_err
        error_rate = (total_err / total_events * 100) if total_events > 0 else 0.0

        if window_latencies:
            sorted_lat = sorted(window_latencies)
            n = len(sorted_lat)
            latency_stats = {
                "p50": sorted_lat[int(n * 0.50)],
                "p95": sorted_lat[min(int(n * 0.95), n - 1)],
                "p99": sorted_lat[min(int(n * 0.99), n - 1)],
                "min": sorted_lat[0],
                "max": sorted_lat[-1],
            }
        else:
            latency_stats = {"p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}

        # Throughput: requests in window / window duration
        if window_latencies and uptime > 0:
            throughput = len(window_latencies) / min(uptime, self._window_seconds)
        else:
            throughput = 0.0

        avg_batch = (
            sum(window_batches) / len(window_batches) if window_batches else 0.0
        )

        return {
            "total_requests": total_req,
            "total_errors": total_err,
            "error_rate_percent": round(error_rate, 3),
            "throughput_fps": round(throughput, 1),
            "latency_ms": latency_stats,
            "avg_batch_size": round(avg_batch, 1),
        }

    def _prune(self, now: float) -> None:
        """Remove entries older than the window. Must hold lock."""
        cutoff = now - self._window_seconds
        while self._latencies and self._latencies[0][0] < cutoff:
            self._latencies.popleft()
        while self._batch_sizes and self._batch_sizes[0][0] < cutoff:
            self._batch_sizes.popleft()
