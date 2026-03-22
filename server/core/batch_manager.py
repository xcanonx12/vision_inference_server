"""Async dynamic batch manager for inference requests."""

import asyncio
import logging
from typing import Callable

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


class BatchManager:
    """Accumulates inference requests into batches.

    Flushes when either max_batch_size is reached or window_timeout_ms
    elapses since the first item in the current batch, whichever comes first.

    Args:
        batch_fn: Callable that takes list[np.ndarray] and returns list[sv.Detections].
        max_batch_size: Maximum images per batch.
        window_timeout_ms: Maximum wait time before flushing a partial batch.
    """

    def __init__(
        self,
        batch_fn: Callable[[list[np.ndarray]], list[sv.Detections]],
        max_batch_size: int = 8,
        window_timeout_ms: float = 10.0,
    ) -> None:
        self._batch_fn = batch_fn
        self._max_batch_size = max_batch_size
        self._window_timeout = window_timeout_ms / 1000.0
        self._queue: asyncio.Queue[tuple[np.ndarray, asyncio.Future]] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Start the batch processing loop."""
        self._running = True
        self._task = asyncio.ensure_future(self._process_loop())

    async def stop(self) -> None:
        """Stop the batch processing loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, image: np.ndarray) -> sv.Detections:
        """Submit an image for batched inference.

        Returns when the batch containing this image is processed.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put((image, future))
        return await future

    async def _process_loop(self) -> None:
        """Main loop: accumulate items, flush on size or timeout."""
        while self._running:
            batch: list[tuple[np.ndarray, asyncio.Future]] = []

            # Wait for the first item (with a periodic check so we can stop)
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                batch.append(item)
            except asyncio.TimeoutError:
                continue

            # Collect more items until max_batch_size or window timeout
            deadline = asyncio.get_event_loop().time() + self._window_timeout
            while len(batch) < self._max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # Execute the batch
            images = [img for img, _ in batch]
            futures = [fut for _, fut in batch]

            try:
                results = await asyncio.get_event_loop().run_in_executor(
                    None, self._batch_fn, images
                )
                for fut, result in zip(futures, results):
                    if not fut.done():
                        fut.set_result(result)
            except Exception as e:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)
