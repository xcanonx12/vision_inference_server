"""Tests for BatchManager."""

import asyncio

import numpy as np
import pytest
import supervision as sv

from server.core.batch_manager import BatchManager


def _make_fake_detector(delay_ms: float = 0):
    """Returns a callable that simulates batch inference."""

    def batch_predict(images: list[np.ndarray]) -> list[sv.Detections]:
        if delay_ms > 0:
            import time
            time.sleep(delay_ms / 1000)
        return [sv.Detections.empty() for _ in images]

    return batch_predict


class TestBatchManager:
    """Unit tests for dynamic batching."""

    @pytest.mark.asyncio
    async def test_single_image_returns_detections(self):
        bm = BatchManager(
            batch_fn=_make_fake_detector(),
            max_batch_size=4,
            window_timeout_ms=50.0,
        )
        bm.start()
        try:
            image = np.zeros((640, 640, 3), dtype=np.uint8)
            result = await bm.submit(image)
            assert isinstance(result, sv.Detections)
        finally:
            await bm.stop()

    @pytest.mark.asyncio
    async def test_batches_accumulate_up_to_max(self):
        call_sizes = []

        def tracking_predict(images):
            call_sizes.append(len(images))
            return [sv.Detections.empty() for _ in images]

        bm = BatchManager(
            batch_fn=tracking_predict,
            max_batch_size=4,
            window_timeout_ms=200.0,  # long window so batch fills by count
        )
        bm.start()
        try:
            image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = await asyncio.gather(
                *[bm.submit(image) for _ in range(4)]
            )
            assert len(results) == 4
            assert sum(call_sizes) == 4  # all 4 images processed
            assert max(call_sizes) <= 4  # no batch exceeds max
        finally:
            await bm.stop()

    @pytest.mark.asyncio
    async def test_window_timeout_flushes_partial_batch(self):
        call_sizes = []

        def tracking_predict(images):
            call_sizes.append(len(images))
            return [sv.Detections.empty() for _ in images]

        bm = BatchManager(
            batch_fn=tracking_predict,
            max_batch_size=8,
            window_timeout_ms=20.0,  # short timeout
        )
        bm.start()
        try:
            image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = await asyncio.gather(
                bm.submit(image), bm.submit(image)
            )
            assert len(results) == 2
            assert any(s <= 2 for s in call_sizes)
        finally:
            await bm.stop()

    @pytest.mark.asyncio
    async def test_batch_size_one_behaves_like_no_batching(self):
        bm = BatchManager(
            batch_fn=_make_fake_detector(),
            max_batch_size=1,
            window_timeout_ms=10.0,
        )
        bm.start()
        try:
            image = np.zeros((640, 640, 3), dtype=np.uint8)
            result = await bm.submit(image)
            assert isinstance(result, sv.Detections)
        finally:
            await bm.stop()
