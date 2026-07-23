import asyncio
import logging
from typing import Any, List, Optional, Callable, Awaitable, Generator

logger = logging.getLogger(__name__)
_POISON_PILL = object()


class AsyncBatchProcessor:
    """
    【高内聚精简版】

    优化点：
    1. 大幅减少方法数量，逻辑集中在核心循环中。
    2. 关机流程线性化，消除过度的方法跳转。
    3. 保留零拷贝、背压控制和防御性强杀等企业级特性。
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
        self._batch_size = batch_size
        self._schedule_delay = schedule_delay
        self._on_flush_callback = on_flush_callback
        self._flush_timeout = flush_timeout
        self._shutdown_flush_timeout = shutdown_flush_timeout
        self._max_concurrent_flushes = max_concurrent_flushes
        self._max_queue_size = max_queue_size
        self._reset_state()

    def _reset_state(self) -> None:
        """重置内部状态"""
        self._queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._sem = asyncio.Semaphore(self._max_concurrent_flushes)
        self._shutdown_event = asyncio.Event()
        self._consume_task: Optional[asyncio.Task] = None
        self._active_flushes: set[asyncio.Task] = set()

    async def start(self) -> None:
        """启动消费者（幂等）"""
        if self._consume_task and not self._consume_task.done():
            return

        if self._shutdown_event.is_set():
            self._reset_state()

        self._shutdown_event.clear()
        self._consume_task = asyncio.create_task(self._consume_loop())
        logger.info("Batch processor started")

    async def put(self, item: Any) -> None:
        """异步放入队列（背压）"""
        if self._shutdown_event.is_set():
            raise RuntimeError("Worker stopped")
        await self._queue.put(item)

    def put_nowait(self, item: Any) -> None:
        """非阻塞放入队列"""
        if self._shutdown_event.is_set():
            raise RuntimeError("Worker stopped")
        self._queue.put_nowait(item)

    async def _consume_loop(self) -> None:
        """核心消费循环：逻辑高度集中"""
        try:
            while not self._shutdown_event.is_set() or not self._queue.empty():
                batch, should_exit = await self._gather_batch()
                if not batch:
                    if should_exit:
                        break
                    continue
                # 关机状态下不再发起新 flush，直接走收尾逻辑
                if self._shutdown_event.is_set():
                    await self._drain_and_flush(batch)
                    break

                # 获取发送许可（背压）
                try:
                    await self._sem.acquire()
                except asyncio.CancelledError:
                    await self._drain_and_flush(batch)
                    raise

                flush_task = asyncio.create_task(self._on_flush_callback(batch))
                self._active_flushes.add(flush_task)
                flush_task.add_done_callback(self._active_flushes.discard)
                flush_task.add_done_callback(lambda _: self._sem.release())

                if should_exit:
                    break

        except asyncio.CancelledError:
            logger.info("Consumer cancelled, draining remaining items")
        finally:
            # 无论何种退出，确保残留数据被处理
            await self._drain_and_flush()

    async def _gather_batch(self) -> tuple[List[Any], bool]:
        """聚合一个批次的数据"""
        batch: List[Any] = []
        loop = asyncio.get_running_loop()

        # 阻塞等待首个元素
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            return batch, False
        except asyncio.CancelledError:
            raise

        # 检查毒丸
        if item is _POISON_PILL:
            self._queue.task_done()
            return batch, True
        batch.append(item)
        self._queue.task_done()

        # 时间窗口内拉取更多
        start = loop.time()
        while len(batch) < self._batch_size:
            remain = self._schedule_delay - (loop.time() - start)
            if remain <= 0:
                break
            try:
                async with asyncio.timeout(remain):
                    item = await self._queue.get()
                    if item is _POISON_PILL:
                        self._queue.task_done()
                        return batch, True
                    batch.append(item)
                    self._queue.task_done()
            except asyncio.TimeoutError:
                break
        return batch, False

    async def _drain_and_flush(self, initial_batch: Optional[List[Any]] = None) -> None:
        """【核心归并】排空队列并冲刷（零拷贝分片）"""
        batch = list(initial_batch) if initial_batch else []

        # 快速路径：如果已关机且队列空，直接返回
        if self._shutdown_event.is_set() and self._queue.empty() and not batch:
            return

        # 合并队列剩余数据
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item is not _POISON_PILL:
                    batch.append(item)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        if not batch:
            return

        # 零拷贝分片并执行关机冲刷
        chunks = (
            batch[i : i + self._batch_size]
            for i in range(0, len(batch), self._batch_size)
        )
        results = await asyncio.gather(
            *(self._safe_flush(c) for c in chunks),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                logger.exception("Shutdown flush failed")

    async def _safe_flush(self, chunk: List[Any]) -> None:
        """带超时控制的冲刷"""
        try:
            async with asyncio.timeout(self._shutdown_flush_timeout):
                await self._on_flush_callback(chunk)
        except Exception as e:
            raise e

    async def stop(self, timeout: float = 5.0) -> None:
        """优雅停机：状态锁定 -> 唤醒 -> 等待 -> 强杀"""
        if self._shutdown_event.is_set():
            return

        logger.info("Initiating shutdown...")
        self._shutdown_event.set()

        # 唤醒可能阻塞在 get() 上的消费者
        try:
            self._queue.put_nowait(_POISON_PILL)
        except asyncio.QueueFull:
            pass

        # 等待消费者和活跃的 flush 任务
        tasks = {self._consume_task} | self._active_flushes
        pending = {t for t in tasks if t and not t.done()}

        try:
            async with asyncio.timeout(timeout):
                await asyncio.gather(*pending, return_exceptions=True)
        except asyncio.TimeoutError:
            logger.warning("Graceful shutdown timeout, forcing cancellation")
        finally:
            # 防御性强杀，防止 I/O 死锁
            for t in pending:
                if not t.done() and not t.cancelled():
                    t.cancel()

            # 给 cancel 一个极短的宽限期，然后强制返回
            done, pending = await asyncio.wait(pending, timeout=1.0)
            if pending:
                logger.critical(f"{len(pending)} tasks refused to cancel.")

            self._active_flushes.clear()
            logger.info("Worker fully stopped")
