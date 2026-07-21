# event_queue.py
import asyncio
import logging
from typing import Callable, Awaitable, Any

logger = logging.getLogger(__name__)


class AsyncEventQueue:
    """🚀 通用高性能异步事件总线队列"""

    def __init__(
        self,
        consumer_handler: Callable[[Any], Awaitable[None]],
        maxsize: int = 10_000,
    ):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._consumer_handler = consumer_handler
        self._closed = False
        self._consumer_task = asyncio.create_task(self._consume_loop())

    def push(self, item: Any) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("EventQueue full, dropping event")

    async def _consume_loop(self) -> None:
        while True:
            try:
                item = await self._queue.get()
                await self._consumer_handler(item)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("EventQueue consumer error")

    async def wait_flush(self) -> None:
        await self._queue.join()

    async def join_and_close(self) -> None:
        self._closed = True
        await self.wait_flush()
        self._consumer_task.cancel()
        try:
            await self._consumer_task
        except asyncio.CancelledError:
            pass
