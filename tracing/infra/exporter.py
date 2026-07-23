import logging
import asyncio
from typing import List, Optional, Callable, Generic, TypeVar

from tracing.infra.processor import AsyncBatchProcessor
from tracing.infra.transport import Transport

logger = logging.getLogger(__name__)

EventT = TypeVar("EventT")
PayloadT = TypeVar("PayloadT")

Serializer = Callable[[EventT], PayloadT]
OnDropCallback = Callable[[int, Optional[str]], None]


class BatchExporter(Generic[EventT, PayloadT]):
    """优化后的批量导出器（纯异步协程版）"""

    def __init__(
        self,
        transport: "Transport[PayloadT]",
        serializer: Serializer[EventT, PayloadT],
        *,
        batch_size: int = 100,
        schedule_delay: float = 0.5,
        max_queue_size: int = 10000,
        max_concurrent_flushes: int = 5,
        flush_timeout: float = 10.0,
        # 重点：将 shutdown 的超时时间固定在这里，防止调用时混淆
        shutdown_flush_timeout: float = 5.0,
        on_drop: Optional[OnDropCallback] = None,
    ):
        self.transport = transport
        self.serializer = serializer
        self._on_drop = on_drop or (lambda c, r: None)
        self.shutdown_flush_timeout = shutdown_flush_timeout

        # 生命周期状态管理
        self._is_running: bool = False
        self._is_shutting_down: bool = False
        self._lifecycle_lock = asyncio.Lock()

        # 底层异步批量处理器
        self._batch_worker = AsyncBatchProcessor[EventT](
            batch_size=batch_size,
            schedule_delay=schedule_delay,
            on_flush_callback=self._flush_batch,
            max_queue_size=max_queue_size,
            max_concurrent_flushes=max_concurrent_flushes,
            flush_timeout=flush_timeout,
            shutdown_flush_timeout=shutdown_flush_timeout,
        )

    def _serialize_batch_safe(self, batch: List[EventT]) -> List[PayloadT]:
        """
        批量序列化安全包装器（纯协程版）。

        警告：此方法运行于事件循环线程。
        若 serializer 包含重 CPU 逻辑（如大 JSON dumps 或复杂加密），
        应考虑在未来迁移至 asyncio.to_thread 以避免阻塞 IO。
        """
        payloads: List[PayloadT] = []
        drop_count = 0

        for event in batch:
            try:
                payloads.append(self.serializer(event))
            except Exception as e:
                drop_count += 1
                logger.error(
                    "Individual event serialization failed.",
                    exc_info=True,
                )

        if drop_count > 0:
            self._on_drop(drop_count, "serialization_error")

        return payloads

    async def start(self) -> None:
        """启动导出器（幂等，协程安全）"""
        async with self._lifecycle_lock:
            if self._is_running:
                return
            self._is_running = True
            self._is_shutting_down = False
            await self._batch_worker.start()
            logger.info(f"{self.__class__.__name__} started.")

    async def shutdown(self) -> None:
        """
        优雅停机（增强数据保护版）。

        注意：此方法不接受 timeout 参数，统一使用初始化时设定的
        `shutdown_flush_timeout`，以确保配置的一致性。
        """
        async with self._lifecycle_lock:
            if not self._is_running or self._is_shutting_down:
                return

            logger.info("[AUDIT] Exporter shutdown initiated.")
            self._is_shutting_down = True

            try:
                # 使用初始化时设定的超时时间，而非运行时传入
                await self._batch_worker.stop(timeout=self.shutdown_flush_timeout)
            except Exception as e:
                logger.error(f"Error during batch worker stop: {e}", exc_info=True)
            finally:
                # 无论成功与否，都将状态置为 False，防止僵尸进程
                self._is_running = False
                self._is_shutting_down = False
                logger.info("[AUDIT] Exporter shutdown completed.")

    async def export(self, event: EventT) -> None:
        """
        异步导出接口。
        - 状态校验：防止关闭期间的异常流出。
        - 非阻塞异步背压：队列满时挂起当前协程。
        """
        # 快速失败：避免在停机期间接收新数据导致状态混乱
        if not self._is_running or self._is_shutting_down:
            self._on_drop(1, "exporter_not_running")
            return

        try:
            # 压入底层异步队列，队列满时自动 await 挂起产生背压
            await self._batch_worker.put(event)
        except (RuntimeError, asyncio.QueueFull, ConnectionError) as e:
            # 移除了 GeneratorExit，该异常在纯 asyncio 上下文中极少出现
            reason = f"queue_unavailable:{type(e).__name__}"
            self._on_drop(1, reason)
        except Exception as e:
            # 兜底异常处理，防止未捕获异常炸毁事件循环
            self._on_drop(1, "unexpected_export_error")
            logger.error(f"Unexpected error in export: {e}", exc_info=True)

    async def _flush_batch(self, batch: List[EventT]) -> None:
        """底层消费回调：纯异步单线程序列化与发送"""
        if not batch:
            return

        try:
            # 1. 序列化阶段：直接在当前协程流式执行
            payloads = self._serialize_batch_safe(batch)

            # 如果序列化全部失败，直接返回
            if not payloads:
                logger.debug(
                    "All events in batch failed serialization. Dropping batch."
                )
                return

            # 2. 发送阶段：异步网络 IO 发送
            await self.transport.send(payloads)

        except asyncio.CancelledError:
            # 显式记录被取消时的批次大小，用于审计
            self._on_drop(len(batch), "task_cancelled")
            logger.warning(f"Flush task cancelled. {len(batch)} events dropped.")
            raise

        except Exception as e:
            self._on_drop(len(batch), "transport_error")
            logger.critical(
                f"Transport send failed. Notifying drop of {len(batch)} events. "
                f"Error: {e}",
                exc_info=True,
            )
