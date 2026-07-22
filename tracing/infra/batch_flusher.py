import asyncio
import logging
from typing import Any, List, Optional, Callable, Awaitable

logger = logging.getLogger(__name__)
_POISON_PILL = object()


class AsyncBatchProcessor:
    """
    【企业级高并发高可用版·最终完美防御闭环版】

    核心加固：
    1. 采用动态生命周期探测，在 `finally` 块中精准强杀真正残留的悬空任务。
    2. 显式化“计数偿还”语义，精准契约 asyncio.Queue 的底层设计。
    """

    def __init__(
        self,
        batch_size: int,
        schedule_delay: float,
        on_flush_callback: Callable[[List[Any]], Awaitable[None]],
        max_queue_size: int = 10000,
        max_concurrent_flushes: int = 5,
        flush_timeout: float = 10.0,
        shutdown_flush_timeout: float = 5.0,
    ):
        if batch_size <= 0 or schedule_delay < 0:
            raise ValueError("Invalid batch_size or schedule_delay")

        self._batch_size = batch_size
        self._schedule_delay = schedule_delay
        self._on_flush_callback = on_flush_callback
        self._flush_timeout = flush_timeout
        self._shutdown_flush_timeout = shutdown_flush_timeout

        self._async_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._semaphore = asyncio.Semaphore(max_concurrent_flushes)
        self._consume_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._active_flush_tasks: set[asyncio.Task] = set()
        self._state_lock = asyncio.Lock()

    async def start(self) -> None:
        """启动消费者循环（幂等）"""
        async with self._state_lock:
            if self._consume_task and not self._consume_task.done():
                return
            self._shutdown_event.clear()
            self._consume_task = asyncio.create_task(self._core_consume_loop())
            logger.info("WebSocket worker started")

    async def put(self, item: Any) -> None:
        """异步放入队列（支持背压）"""
        if self._shutdown_event.is_set():
            raise RuntimeError("Worker stopped")
        await self._async_queue.put(item)

    def put_nowait(self, item: Any) -> None:
        """非阻塞放入队列"""
        if self._shutdown_event.is_set():
            raise RuntimeError("Worker stopped")
        self._async_queue.put_nowait(item)

    async def _core_consume_loop(self) -> None:
        """核心消费循环：批量聚合 + 并发推送"""
        batch: List[Any] = []
        loop = asyncio.get_running_loop()
        should_exit = False

        try:
            while not self._shutdown_event.is_set() or not self._async_queue.empty():
                if should_exit:
                    break

                # 1. 获取首条数据
                if not batch:
                    try:
                        item = await asyncio.wait_for(
                            self._async_queue.get(), timeout=1.0
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        continue

                    if item is _POISON_PILL:
                        self._async_queue.task_done()
                        break

                    batch.append(item)
                    self._async_queue.task_done()

                # 2. 时间窗口聚合
                last_flush_time = loop.time()
                while len(batch) < self._batch_size:
                    timeout_duration = max(
                        0.0, self._schedule_delay - (loop.time() - last_flush_time)
                    )
                    try:
                        async with asyncio.timeout(timeout_duration):
                            item = await self._async_queue.get()
                            if item is _POISON_PILL:
                                self._async_queue.task_done()
                                should_exit = True
                                break
                            batch.append(item)
                            self._async_queue.task_done()
                    except asyncio.TimeoutError:
                        break

                # 3. 安全凭证分发
                if batch:
                    if self._shutdown_event.is_set() and self._semaphore.locked():
                        break

                    try:
                        await self._semaphore.acquire()
                    except asyncio.CancelledError:
                        raise

                    self._dispatch_flush_with_ticket(list(batch))
                    batch.clear()

        except asyncio.CancelledError:
            logger.info("Consume loop cancelled, draining items")
        finally:
            # 4. 彻底排空队列
            # 💡 核心修正：此处 task_done() 旨在偿还队列内部 unfinished_tasks 计数，防止解释器关闭时报 Pending Task 警告
            while not self._async_queue.empty():
                try:
                    item = self._async_queue.get_nowait()
                    if item is not _POISON_PILL:
                        batch.append(item)
                    self._async_queue.task_done()
                except asyncio.QueueEmpty:
                    break

            # 5. 最终停机冲刷
            if batch:
                shutdown_tasks = [
                    asyncio.create_task(
                        self._safe_shutdown_flush(batch[i : i + self._batch_size])
                    )
                    for i in range(0, len(batch), self._batch_size)
                ]
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)

    def _dispatch_flush_with_ticket(self, packet: List[Any]) -> None:
        """分发推送任务并严格移交凭证"""

        async def wrapped_flush() -> None:
            try:
                async with asyncio.timeout(self._flush_timeout):
                    await self._on_flush_callback(packet)
            except Exception:
                logger.exception("Flush failed")
            finally:
                self._semaphore.release()

        task = asyncio.create_task(wrapped_flush())
        self._active_flush_tasks.add(task)
        task.add_done_callback(self._active_flush_tasks.discard)

    async def _safe_shutdown_flush(self, chunk: List[Any]) -> None:
        """停机兜底冲刷"""
        try:
            async with asyncio.timeout(self._shutdown_flush_timeout):
                await self._on_flush_callback(chunk)
        except Exception:
            logger.exception("Shutdown flush failed")

    async def stop(self, timeout: float = 5.0) -> None:
        """
        优雅停机标准闭环路径（带有防御性强杀加固）
        """
        async with self._state_lock:
            if self._shutdown_event.is_set():
                return

            logger.info("Initiating graceful shutdown...")
            self._shutdown_event.set()

            try:
                self._async_queue.put_nowait(_POISON_PILL)
            except asyncio.QueueFull:
                pass

            # 级联等待：搜集所有需要等待的生命周期目标
            wait_targets = []
            if self._consume_task:
                wait_targets.append(self._consume_task)
            if self._active_flush_tasks:
                wait_targets.extend(self._active_flush_tasks)

            if not wait_targets:
                self._active_flush_tasks.clear()
                logger.info("Worker fully stopped (no active tasks)")
                return

            try:
                async with asyncio.timeout(timeout):
                    await asyncio.gather(*wait_targets, return_exceptions=True)
            except asyncio.TimeoutError:
                logger.warning(
                    "Shutdown timeout reached. Leaving it to the defensive block."
                )
            finally:
                # 🌟 方案一落地：动态计算并强杀真正未完成的“悬空任务”，闭环收尾
                pending = [t for t in wait_targets if not t.done()]
                if pending:
                    logger.warning(
                        f"Force cancelling {len(pending)} pending tasks after timeout"
                    )
                    for t in pending:
                        t.cancel()
                    # 强杀过程同样需要 gather 隐式等待其异常链收尾，确保无 Pending 警告
                    await asyncio.gather(*pending, return_exceptions=True)

                self._active_flush_tasks.clear()
                logger.info("Worker fully stopped")
