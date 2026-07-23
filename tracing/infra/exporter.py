# tracing/exporters/batch_exporter.py
import logging
import asyncio
from typing import (
    List,
    Optional,
    Callable,
    Generic,
    TypeVar,
)

from tracing.infra.processor import AsyncBatchProcessor
from tracing.transport.protocol import Transport

logger = logging.getLogger(__name__)

# ==================================================================
# 类型定义 (Type Definitions)
# ==================================================================

EventT = TypeVar("EventT")  # 输入：业务事件（如 StepEvent, MetricPoint）
PayloadT = TypeVar("PayloadT")  # 输出：序列化后的数据（如 dict, bytes）

# 序列化函数签名
Serializer = Callable[[EventT], PayloadT]

# 丢弃事件回调函数签名
# 参数: count (丢弃数量), reason (丢弃原因)
OnDropCallback = Callable[[int, Optional[str]], None]


# ==================================================================
# 核心实现 (Core Implementation)
# ==================================================================


class BatchExporter(Generic[EventT, PayloadT]):
    """
    通用批量导出器（生产级·管道核心版）。

    职责：
    1. 提供异步/同步双模导出接口。
    2. 保证跨线程安全调用（call_soon_threadsafe）。
    3. 管理批量聚合与背压。
    4. 执行序列化回调并调用 Transport。

    非职责（通过回调外置）：
    - 丢弃事件的统计（交还给 Metrics/Monitor 系统）。
    - 丢弃后的降级逻辑（如落盘）。
    """

    def __init__(
        self,
        transport: Transport[PayloadT],
        serializer: Serializer[EventT, PayloadT],
        *,
        batch_size: int = 100,
        schedule_delay: float = 0.5,
        max_queue_size: int = 10000,
        max_concurrent_flushes: int = 5,
        flush_timeout: float = 10.0,
        shutdown_flush_timeout: float = 5.0,
        # --- 关键设计 ---
        # 注入丢弃回调，默认使用静默实现（No-op）
        on_drop: Optional[OnDropCallback] = None,
    ):
        self.transport = transport
        self.serializer = serializer
        self._on_drop = on_drop or (lambda c, r: None)  # 默认空实现，避免 None check

        # 运行时状态
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._is_running: bool = False
        self._is_shutting_down: bool = False

        # 批量处理器
        self._batch_worker = AsyncBatchProcessor(
            batch_size=batch_size,
            schedule_delay=schedule_delay,
            on_flush_callback=self._flush_batch,
            max_queue_size=max_queue_size,
            max_concurrent_flushes=max_concurrent_flushes,
            flush_timeout=flush_timeout,
            shutdown_flush_timeout=shutdown_flush_timeout,
        )
        self._lock = asyncio.Lock()

    # ==================================================================
    # 生命周期管理 (Lifecycle)
    # ==================================================================

    async def start(self) -> None:
        """启动导出器（幂等）"""
        async with self._lock:
            if self._is_running:
                return
            self._loop = asyncio.get_running_loop()
            await self._batch_worker.start()
            self._is_running = True
            logger.info(f"{self.__class__.__name__} started.")

    async def shutdown(self, timeout: float = 10.0) -> None:
        """优雅停机"""
        async with self._lock:
            if not self._is_running or self._is_shutting_down:
                return

            logger.info("[AUDIT] Exporter shutdown initiated.")
            self._is_shutting_down = True

            try:
                await self._batch_worker.stop(timeout=timeout)
            except Exception as e:
                logger.error(f"Error during batch worker stop: {e}", exc_info=True)
            finally:
                self._is_running = False
                self._is_shutting_down = False
                self._loop = None
                logger.info("[AUDIT] Exporter shutdown completed.")

    # ==================================================================
    # 对外接口 (Public APIs)
    # ==================================================================

    async def export(self, event: EventT) -> None:
        """
        异步导出（主事件循环内使用）。
        支持背压，若队列满会阻塞。
        """
        if not self._can_accept():
            self._on_drop(1, "exporter_not_running")
            return

        # 防御性检查：防止跨线程误用导致死锁
        if asyncio.get_running_loop() != self._loop:
            self._on_drop(1, "wrong_thread")
            raise RuntimeError(
                "export() called from wrong thread. Use export_sync() instead."
            )

        try:
            await self._batch_worker.put(event)
        except (RuntimeError, asyncio.QueueFull):
            # QueueFull 理论上不会发生，因为 put 是阻塞的，除非设置了 maxsize 且满了
            self._on_drop(1, "queue_full_or_closed")

    def export_sync(self, event: EventT) -> None:
        """
        同步导出（跨线程安全）。
        适用于 WSGI/ThreadPool 环境。
        """
        target_loop = self._loop
        if not self._can_accept() or target_loop is None:
            self._on_drop(1, "exporter_not_running")
            return

        # 如果在同一个 loop 线程内，直接执行
        if self._is_same_loop():
            self._put_nowait(event)
            return

        # 跨线程投递
        try:
            target_loop.call_soon_threadsafe(self._put_nowait, event)
        except RuntimeError as e:
            # 通常发生在 loop 已关闭的情况下
            self._on_drop(1, "loop_closed")

    # ==================================================================
    # 内部逻辑 (Internal Logic)
    # ==================================================================

    async def _flush_batch(self, batch: List[EventT]) -> None:
        """
        核心回调：由 AsyncBatchProcessor 触发。
        负责序列化并发送数据。
        """
        if not batch:
            return

        try:
            # 1. 序列化：调用外部注入的逻辑
            payloads = [self.serializer(event) for event in batch]
            # 2. 传输：调用 Transport
            await self.transport.send(payloads)
        except asyncio.CancelledError:
            # 必须重新抛出，确保批量处理器能正确响应取消信号
            self._on_drop(len(batch), "task_cancelled")
            raise
        except Exception:
            # 故障隔离：通知外部这批数据丢失了
            self._on_drop(len(batch), "transport_error")
            logger.critical(
                f"Transport send failed. Notifying drop of {len(batch)} events.",
                exc_info=True,
            )

    def _put_nowait(self, event: EventT) -> None:
        """
        内部非阻塞写入（必须运行在 loop 线程内）。
        """
        if not self._can_accept_nolock():
            self._on_drop(1, "shutting_down")
            return
        try:
            self._batch_worker.put_nowait(event)
        except (RuntimeError, asyncio.QueueFull):
            self._on_drop(1, "queue_put_failed")

    def _can_accept(self) -> bool:
        """对外接口使用的状态检查"""
        return self._is_running and not self._is_shutting_down

    def _can_accept_nolock(self) -> bool:
        """内部使用，假设调用方已处理线程安全问题"""
        return self._is_running and not self._is_shutting_down

    def _is_same_loop(self) -> bool:
        """判断当前线程是否为 exporter 所属的事件循环线程"""
        try:
            return asyncio.get_running_loop() == self._loop
        except RuntimeError:
            return False
