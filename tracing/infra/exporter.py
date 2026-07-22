# tracing/infra/exporter.py
from contextlib import asynccontextmanager
from contextvars import ContextVar
import logging
import asyncio
from typing import Any, AsyncIterator, List, Optional

from tracing.core.schema import StepEvent, StepEventType
from tracing.infra.batch_flusher import AsyncBatchProcessor
from tracing.infra.transport import EventTransport

logger = logging.getLogger(__name__)

_active_exporter: ContextVar[Optional["AgentEventExporter"]] = ContextVar(
    "agent_active_exporter", default=None
)


@asynccontextmanager
async def bind_exporter(
    exporter: "AgentEventExporter",
) -> AsyncIterator["AgentEventExporter"]:
    """绑定 exporter 到当前异步上下文"""
    token = _active_exporter.set(exporter)
    try:
        yield exporter
    finally:
        _active_exporter.reset(token)


class AgentEventExporter:
    """
    Agent 事件导出器。
    内部使用 PureAsyncBatchWorker 实现高性能异步批量发送。

    线程 / Task 安全说明：
    - 本类允许多个 Task 并发调用 export / export_nowait
    - 所有可变统计字段（tokens / cost / dropped）均受 _stats_lock 保护
    """

    def __init__(
        self,
        transport: EventTransport,
        batch_size: int = 100,
        schedule_delay: float = 0.5,
        max_queue_size: int = 10000,
        max_concurrent_flushes: int = 5,
        flush_timeout: float = 30.0,
        shutdown_flush_timeout: float = 5.0,
    ):
        self.transport = transport

        self._batch_worker = AsyncBatchProcessor(
            batch_size=batch_size,
            schedule_delay=schedule_delay,
            on_flush_callback=self._export_batch_to_network,
            max_queue_size=max_queue_size,
            max_concurrent_flushes=max_concurrent_flushes,
            flush_timeout=flush_timeout,
            shutdown_flush_timeout=shutdown_flush_timeout,
        )

        self._lock = asyncio.Lock()  # 生命周期锁（start/shutdown）
        self._stats_lock = asyncio.Lock()  # 统计字段锁（高频路径）

        # 统计指标（受 _stats_lock 保护）
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.dropped_events: int = 0

        self._token: Optional[Any] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AgentEventExporter":
        async with self._lock:
            current = _active_exporter.get()
            if current is self:
                raise RuntimeError("Exporter already active in current context")

            await self._batch_worker.start()
            self._token = _active_exporter.set(self)
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------

    async def export(self, event: StepEvent) -> None:
        """
        生产事件（异步背压）。
        队列满时会挂起，防止内存暴涨。
        """
        payload = self._build_payload(event)

        try:
            await self._batch_worker.put(payload)
        except RuntimeError as e:
            logger.debug(f"Exporter stopped, dropping single event: {e}")
            if not self._batch_worker.is_running:
                await self._incr_dropped(1)
            return
        except Exception as e:
            logger.error(f"Failed to enqueue event: {e}", exc_info=True)
            return

        await self._maybe_update_stats(event)

    def export_nowait(self, event: StepEvent) -> None:
        """
        生产事件（非阻塞同步方法）。
        队列满时立即抛出 asyncio.QueueFull，由调用方决定是否重试。
        """
        payload = self._build_payload(event)

        try:
            self._batch_worker.put_nowait(payload)
        except RuntimeError as e:
            logger.debug(f"Exporter stopped, dropping single event: {e}")
            if not self._batch_worker.is_running:
                asyncio.create_task(self._incr_dropped(1))
            return
        except asyncio.QueueFull:
            asyncio.create_task(self._incr_dropped(1))
            logger.warning(
                f"Queue full, dropping event. Total dropped: {self.dropped_events}"
            )
            return
        except Exception as e:
            logger.error(f"Failed to enqueue event: {e}", exc_info=True)
            return

        asyncio.create_task(self._maybe_update_stats(event))

    async def shutdown(self, timeout: float = 10.0) -> None:
        """
        优雅关闭（幂等）。
        """
        async with self._lock:
            if not self._batch_worker.is_running:
                return

            try:
                await self._batch_worker.stop(timeout=timeout)
            finally:
                if self._token:
                    _active_exporter.reset(self._token)
                    self._token = None

                logger.info(
                    "[AUDIT] AgentEventExporter shutdown completed. "
                    f"Metrics summary -> tokens: {self.total_tokens}, "
                    f"cost: {self.total_cost:.4f}, explicitly_dropped: {self.dropped_events}"
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(self, event: StepEvent) -> dict:
        """
        将 StepEvent 转换为 transport 层可用的 payload。
        此处不做 IO，避免引入 await。
        """
        return event.model_dump(mode="json")

    async def _maybe_update_stats(self, event: StepEvent) -> None:
        """
        更新 tokens / cost 统计（受锁保护）。
        """
        if event.event_type != StepEventType.DATA_UPDATE or not event.metadata:
            return

        async with self._stats_lock:
            self.total_tokens += event.metadata.get("tokens", 0)
            self.total_cost += float(event.metadata.get("cost", 0.0))

    async def _incr_dropped(self, count: int = 1) -> None:
        """
        原子递增丢弃事件计数。
        """
        async with self._stats_lock:
            self.dropped_events += count

    async def _export_batch_to_network(self, batch: List[dict]) -> None:
        """
        批量发送网络请求回调。
        网络失败视为整批丢失，不做重试（符合 tracing 的 best-effort 语义）。
        """
        try:
            if hasattr(self.transport, "send_batch") and callable(
                getattr(self.transport, "send_batch")
            ):
                await self.transport.send_batch(batch)
            else:
                for payload in batch:
                    await self.transport.send(payload)
        except Exception:
            await self._incr_dropped(len(batch))
            logger.critical(
                f"Transport batch send failed. Dropped {len(batch)} events. "
                f"Total dropped: {self.dropped_events}",
                exc_info=True,
            )


# ----------------------------------------------------------------------
# Global accessor
# ----------------------------------------------------------------------


def get_global_exporter(silent: bool = False) -> Optional[AgentEventExporter]:
    """
    获取当前异步上下文中激活的全局导出器。
    """
    exporter = _active_exporter.get()
    if exporter is None and not silent:
        raise RuntimeError(
            "No active exporter found. "
            "Did you forget to use 'async with bind_exporter(...)'?"
        )
    return exporter
