import asyncio
import logging
import time
from typing import Any, Optional, Dict
from enum import Enum
from collections import deque
import json


class BroadcastActionType(str, Enum):
    """【天网天线】全异步高性能流式广播行为事件类型契约"""

    STEP_ENTER = "STEP_ENTER"
    STEP_EXIT = "STEP_EXIT"
    TOKEN_CONSUME = "TOKEN_CONSUME"
    METADATA_STREAM_UPDATE = "METADATA_STREAM_UPDATE"
    STEP_CRASH = "STEP_CRASH"
    SESSION_TIMEOUT = "SESSION_TIMEOUT"
    SESSION_BUDGET_EXHAUSTED = "SESSION_BUDGET_EXHAUSTED"


class TelemetryEventType(str, Enum):
    """异步高性能流式传输与链路度量事件类型契约"""

    SPAN_STARTED = "SPAN_STARTED"
    SPAN_FINISHED = "SPAN_FINISHED"
    SPAN_CRASHED = "SPAN_CRASHED"
    SPAN_CANCELLED = "SPAN_CANCELLED"
    SPAN_METADATA_UPDATED = "SPAN_METADATA_UPDATED"
    TOKEN_CONSUMED = "TOKEN_CONSUMED"
    SESSION_TIMEOUT = "SESSION_TIMEOUT"
    SESSION_BUDGET_EXHAUSTED = "SESSION_BUDGET_EXHAUSTED"


class TelemetryEventPublisher:
    """🪐 纯异步高吞吐非阻塞广播器：内置背压与优雅排空机制"""

    def __init__(
        self,
        logger: logging.Logger,
        max_queue_size: int = 1000,
        max_retries: int = 3,
        send_timeout: float = 5.0,
    ):
        self.logger = logger
        self.max_queue_size = max_queue_size
        self.max_retries = max_retries
        self.send_timeout = send_timeout

        self._queue: Optional[asyncio.Queue] = None
        self._consume_task: Optional[asyncio.Task] = None
        self._closed: bool = False
        self._initialized: bool = False
        self._dropped_events: int = 0
        self._sent_events: int = 0
        self._recent_errors: deque = deque(maxlen=100)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._consume_task = loop.create_task(self._consume_loop())
        self._initialized = True
        self.logger.info("🪐 [Broadcaster] 纯异步消费引擎初始化成功。")

    async def push(self, **kwargs: Any) -> None:
        if self._closed:
            self._dropped_events += 1
            return
        self._ensure_initialized()
        assert self._queue is not None

        qsize = self._queue.qsize()
        if qsize > self.max_queue_size * 0.8:
            self.logger.warning(
                f"⚠️ [Broadcaster] 队列高水位积压: {qsize}/{self.max_queue_size}"
            )
        if qsize >= self.max_queue_size * 0.95:
            self.logger.error(
                f"🚨 [Broadcaster] 队列接近满载，应用背压！{qsize}/{self.max_queue_size}"
            )

        try:
            await self._queue.put(kwargs)
        except asyncio.CancelledError:
            self._dropped_events += 1
            self.logger.warning("🚫 [Broadcaster] Push cancelled, event dropped")
        except Exception as e:
            self._dropped_events += 1
            self.logger.error(f"❌ [Broadcaster] 投递失败: {e}, packet: {kwargs}")

    async def _consume_loop(self) -> None:
        self.logger.info("🪐 [Broadcaster] 异步消费者 Task 已就位。")
        assert self._queue is not None
        try:
            while True:
                try:
                    packet = await self._queue.get()
                except asyncio.CancelledError:
                    self.logger.info(
                        "🛑 [Broadcaster] 接收到停机信号，开始防丢包排空存量..."
                    )
                    while not self._queue.empty():
                        pkt = self._queue.get_nowait()
                        await self._safe_send(pkt)
                        self._queue.task_done()
                    break

                await self._safe_send(packet)
                self._queue.task_done()
        finally:
            self.logger.info(
                f"🏁 [Broadcaster] 异步消费者 Task 干净退出。"
                f"Sent: {self._sent_events}, Dropped: {self._dropped_events}"
            )

    async def _safe_send(self, packet: dict) -> None:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                await asyncio.wait_for(
                    self._send_to_frontend(packet), timeout=self.send_timeout
                )
                self._sent_events += 1
                return
            except asyncio.TimeoutError as e:
                last_error = e
                self.logger.warning(
                    f"⏰ [Broadcaster] 发送超时 (尝试 {attempt + 1}/{self.max_retries}): "
                    f"{packet.get('span_name', 'UNKNOWN')}"
                )
            except Exception as e:
                last_error = e
                self.logger.error(
                    f"❌ [Broadcaster] 发送失败 (尝试 {attempt + 1}/{self.max_retries}): {e}"
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(0.1 * (2**attempt))

        self._recent_errors.append(
            {"timestamp": time.time(), "packet": packet, "error": str(last_error)}
        )
        self.logger.error(
            f"❌ [Broadcaster] 最终发送失败，丢弃事件: {packet.get('span_name', 'UNKNOWN')}"
        )

    async def _send_to_frontend(self, packet: dict) -> None:
        """对接真实的物理通信库，例如：await websocket.send_json(packet)"""
        self.logger.debug(
            f"📡 [Broadcaster] Sending: {json.dumps(packet, ensure_ascii=False)[:200]}..."
        )
        await asyncio.sleep(0.01)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.logger.info("🔒 [Broadcaster] 启动异步优雅停机流程...")
        if self._consume_task and not self._consume_task.done():
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass
        self.logger.info(
            f"✅ [Broadcaster] 广播管道完全干净关闭。"
            f"Sent: {self._sent_events}, Dropped: {self._dropped_events}"
        )

    def get_stats(self) -> Dict[str, int]:
        """获取广播器统计信息"""
        return {
            "sent_events": self._sent_events,
            "dropped_events": self._dropped_events,
            "queue_size": self._queue.qsize() if self._queue else 0,
            "recent_errors": len(self._recent_errors),
        }
