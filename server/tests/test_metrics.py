"""Tests for MetricsCollector."""

import time

import pytest


from server.core.metrics_collector import MetricsCollector


class TestMetricsCollector:
    """Unit tests for in-memory metrics collection."""

    def test_initial_state_has_zero_counts(self):
        mc = MetricsCollector(window_seconds=60)
        metrics = mc.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["total_errors"] == 0
        assert metrics["throughput_fps"] == 0.0

    def test_record_inference_updates_counts(self):
        mc = MetricsCollector(window_seconds=60)
        mc.record_inference(latency_ms=10.0, batch_size=1)
        mc.record_inference(latency_ms=15.0, batch_size=1)
        metrics = mc.get_metrics()
        assert metrics["total_requests"] == 2

    def test_percentiles_with_known_values(self):
        mc = MetricsCollector(window_seconds=60)
        # Insert 100 values: 1.0, 2.0, ..., 100.0
        for i in range(1, 101):
            mc.record_inference(latency_ms=float(i), batch_size=1)
        metrics = mc.get_metrics()
        latency = metrics["latency_ms"]
        assert 49.0 <= latency["p50"] <= 51.0
        assert 94.0 <= latency["p95"] <= 96.0
        assert 99.0 <= latency["p99"] <= 100.0
        assert latency["min"] == 1.0
        assert latency["max"] == 100.0

    def test_record_error_increments_count(self):
        mc = MetricsCollector(window_seconds=60)
        mc.record_error("decode_error")
        mc.record_error("inference_error")
        metrics = mc.get_metrics()
        assert metrics["total_errors"] == 2

    def test_error_rate_calculation(self):
        mc = MetricsCollector(window_seconds=60)
        for _ in range(97):
            mc.record_inference(latency_ms=10.0, batch_size=1)
        for _ in range(3):
            mc.record_error("test_error")
        metrics = mc.get_metrics()
        # 3 errors out of 100 total events = 3.0%
        assert 2.9 <= metrics["error_rate_percent"] <= 3.1

    def test_batch_size_tracking(self):
        mc = MetricsCollector(window_seconds=60)
        mc.record_inference(latency_ms=10.0, batch_size=4)
        mc.record_inference(latency_ms=10.0, batch_size=8)
        metrics = mc.get_metrics()
        assert metrics["avg_batch_size"] == 6.0

    def test_reset_clears_all(self):
        mc = MetricsCollector(window_seconds=60)
        mc.record_inference(latency_ms=10.0, batch_size=1)
        mc.record_error("test")
        mc.reset()
        metrics = mc.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["total_errors"] == 0

    def test_thread_safety_concurrent_writes(self):
        """Multiple threads writing concurrently should not crash."""
        import threading

        mc = MetricsCollector(window_seconds=60)
        errors = []

        def writer():
            try:
                for _ in range(100):
                    mc.record_inference(latency_ms=5.0, batch_size=1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert mc.get_metrics()["total_requests"] == 400
