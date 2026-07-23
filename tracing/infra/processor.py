import asyncio
import logging
from typing import Any, List, Optional, Callable, Awaitable, Tuple

logger = logging.getLogger(__name__)
_POISON_PILL = object()


class AsyncBatchProcessor:
    """
    高性能异步批处理器
    - 支持定时/定量双触发
    - 支持最大并发数限制
    - 支持优雅停机（Graceful Shutdown）
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
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_concurrent_flushes <= 0:
            raise ValueError("max_concurrent_flushes must be positive")

        self._batch_size = batch_size
        self._schedule_delay = schedule_delay
        self._on_flush_callback = on_flush_callback
        self._flush_timeout = flush_timeout
        self._shutdown_flush_timeout = shutdown_flush_timeout
        self._max_concurrent_flushes = max_concurrent_flushes
        self._max_queue_size = max_queue_size

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._sem: asyncio.Semaphore = asyncio.Semaphore(self._max_concurrent_flushes)
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._consume_task: Optional[asyncio.Task] = None
        self._active_flushes: set[asyncio.Task] = set()

    async def start(self) -> None:
        """启动消费者循环"""
        if self._consume_task and not self._consume_task.done():
            logger.debug("Batch processor already running.")
            return

        # 如果之前处于 shutdown 状态，重置内部状态
        if self._shutdown_event.is_set():
            self._reset_state()

        self._consume_task = asyncio.create_task(self._consume_loop())
        logger.info(
            "Batch processor started (concurrency=%d)", self._max_concurrent_flushes
        )

    def _reset_state(self) -> None:
        """重置内部状态，用于重启或停止后的清理"""
        self._queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._sem = asyncio.Semaphore(self._max_concurrent_flushes)
        self._shutdown_event.clear()
        self._active_flushes.clear()
        logger.debug("Batch processor state reset.")

    async def put(self, item: Any) -> None:
        """异步放入队列"""
        if self._shutdown_event.is_set():
            raise RuntimeError("Worker is shutting down or stopped")
        await self._queue.put(item)

    def put_nowait(self, item: Any) -> None:
        """非阻塞放入队列"""
        if self._shutdown_event.is_set():
            raise RuntimeError("Worker is shutting down or stopped")
        self._queue.put_nowait(item)

    async def stop(self, timeout: Optional[float] = None) -> None:
        """安全停机"""
        if self._shutdown_event.is_set() and (
            not self._consume_task or self._consume_task.done()
        ):
            return

        logger.info("Initiating graceful shutdown...")
        self._shutdown_event.set()

        # 发送毒丸以唤醒可能在等待 get() 的消费者
        try:
            self._queue.put_nowait(_POISON_PILL)
        except asyncio.QueueFull:
            pass  # 队列满也无所谓，shutdown_event 已经设置

        total_timeout = timeout or (self._flush_timeout + self._shutdown_flush_timeout)

        # 等待消费者结束
        if self._consume_task and not self._consume_task.done():
            try:
                async with asyncio.timeout(total_timeout):
                    await self._consume_task
            except asyncio.TimeoutError:
                logger.warning("Consumer task shutdown timeout, forcing cancellation")
                self._consume_task.cancel()
                try:
                    await self._consume_task
                except asyncio.CancelledError:
                    pass

        # 等待所有活跃的 flush 任务完成
        if self._active_flushes:
            pending = [t for t in self._active_flushes if not t.done()]
            if pending:
                logger.info(f"Waiting for {len(pending)} active flushes to complete...")
                try:
                    async with asyncio.timeout(self._flush_timeout + 1.0):
                        await asyncio.gather(*pending, return_exceptions=True)
                except asyncio.TimeoutError:
                    logger.critical("Active flushes timeout! Force cancelling.")
                    for t in pending:
                        t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

        logger.info("Batch processor stopped completely")

    async def _consume_loop(self) -> None:
        """核心消费循环"""
        last_batch: Optional[List[Any]] = None
        try:
            while True:
                # 1. 获取执行许可
                await self._sem.acquire()

                try:
                    batch, should_exit = await self._gather_batch()
                    last_batch = batch

                    if not batch:
                        self._sem.release()
                        if should_exit:
                            break
                        continue

                except (Exception, asyncio.CancelledError):
                    # 无论发生什么异常或被取消，都要释放信号量
                    self._sem.release()
                    raise

                # 2. 派发刷新任务
                flush_task = asyncio.create_task(
                    self._wrapped_flush(batch, self._flush_timeout)
                )
                self._active_flushes.add(flush_task)

                # 安全的回调移除方式
                if flush_task.done():
                    self._active_flushes.discard(flush_task)
                else:
                    flush_task.add_done_callback(self._active_flushes.discard)

                last_batch = None

                if should_exit:
                    break

        except asyncio.CancelledError:
            logger.info("Consumer loop received cancellation signal")
        except Exception:
            logger.exception("Unexpected error in consume loop")
        finally:
            # 3. 终极兜底：确保残留数据被清洗
            logger.debug("Entering final drain and flush...")
            await asyncio.shield(self._drain_and_flush(last_batch))

    async def _wrapped_flush(self, chunk: List[Any], timeout: float) -> None:
        """包装刷新逻辑，确保信号量释放"""
        try:
            await self._safe_flush(chunk, timeout)
        finally:
            self._sem.release()

    async def _safe_flush(self, chunk: List[Any], timeout: float) -> None:
        """执行刷新回调，包含超时和异常保护"""
        if not chunk:
            return
        try:
            async with asyncio.timeout(timeout):
                await self._on_flush_callback(chunk)
        except asyncio.TimeoutError:
            logger.warning(
                f"Flush callback timed out after {timeout}s for {len(chunk)} items"
            )
        except Exception:
            logger.exception(f"Error during flush callback for {len(chunk)} items")

    async def _gather_batch(self) -> Tuple[List[Any], bool]:
        """
        从队列中收集批次数据
        返回: (batch_data, should_exit_flag)
        """
        batch: List[Any] = []
        loop = asyncio.get_running_loop()

        # 情况 1: 正在关机且队列已空
        if self._shutdown_event.is_set() and self._queue.empty():
            return batch, True

        # 情况 2: 获取第一个元素（非阻塞尝试，然后阻塞等待）
        try:
            item = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            # 如果队列空了且正在关机，退出
            if self._shutdown_event.is_set():
                return batch, True
            # 否则等待下一个元素
            item = await self._queue.get()

        if item is _POISON_PILL:
            self._queue.task_done()
            return batch, True

        batch.append(item)
        self._queue.task_done()

        # 情况 3: 凑单逻辑
        start_time = loop.time()
        while len(batch) < self._batch_size:
            elapsed = loop.time() - start_time
            remain = self._schedule_delay - elapsed
            if remain <= 0:
                break

            try:
                async with asyncio.timeout(remain):
                    item = await self._queue.get()
            except asyncio.TimeoutError:
                break  # 超时，直接返回当前批次

            if item is _POISON_PILL:
                self._queue.task_done()
                return batch, True

            batch.append(item)
            self._queue.task_done()

        return batch, False

    async def _drain_and_flush(self, initial_batch: Optional[List[Any]] = None) -> None:
        """
        关机清洗器：并发受控地刷新剩余数据
        """
        batch = list(initial_batch) if initial_batch else []

        # 1. 清空队列
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

        logger.info(f"Draining {len(batch)} remaining items...")
        chunks = [
            batch[i : i + self._batch_size]
            for i in range(0, len(batch), self._batch_size)
        ]

        # 2. 并发受控地执行最后的刷新
        tasks = []
        for c in chunks:
            # 关键：这里也要等待信号量，防止关机时打爆下游
            await self._sem.acquire()
            t = asyncio.create_task(
                self._wrapped_flush(c, self._shutdown_flush_timeout)
            )
            tasks.append(t)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Drain and flush completed.")
