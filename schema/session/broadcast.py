import time
import asyncio
import logging
from typing import Optional


class DashboardBroadcaster:
    """专职大屏推送的独立广播组件
    对外提供纯同步、0阻塞、线程安全的 push 接口；对内通过常驻协程异步消费。
    """

    def __init__(
        self, session_id: str, logger: logging.Logger, max_queue_size: int = 1000
    ):
        self.session_id = session_id
        self.logger = logger
        self.max_queue_size = max_queue_size

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

        self._queue: asyncio.Queue[Optional[dict]] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._task: Optional[asyncio.Task] = None
        self._running = True

    def _ensure_consumer_started(self):
        """惰性确保消费协程在正确的事件循环中启动，防止同步初始化时报错"""
        if self._task is None and self._running:
            try:
                self._task = asyncio.create_task(self._push_to_frontend_pipeline())
            except RuntimeError:
                self._loop.call_soon(self._start_task_lazy)

    def _start_task_lazy(self):
        if self._task is None and self._running:
            self._task = asyncio.create_task(self._push_to_frontend_pipeline())

    def push(
        self,
        action_type: str,
        node_id: str,
        attempt_idx: int,
        parent_id: Optional[str],
        total_tokens: int,
        session_status: str,
        payload: dict,
    ):
        """0阻塞安全投递快照包至广播队列（纯同步、支持多线程/协程混用）"""
        if not self._running:
            return

        self._ensure_consumer_started()

        event_packet = {
            "session_id": self.session_id,
            "action": action_type,
            "node_id": node_id,
            "attempt_idx": attempt_idx,
            "parent_id": parent_id,
            "timestamp": time.time(),
            "total_tokens": total_tokens,
            "status": session_status,
            "data": payload,
        }

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._safe_put, event_packet)
        else:
            self._safe_put(event_packet)

    def _safe_put(self, packet: Optional[dict]):
        try:
            self._queue.put_nowait(packet)
        except asyncio.QueueFull:
            node_id = packet.get("node_id", "UNKNOWN") if packet else "SYSTEM"
            action = packet.get("action", "UNKNOWN") if packet else "CLOSE"
            self.logger.warning(
                f"🎨 [Dashboard Queue Full] Dropping event for node {node_id}. Action: {action}"
            )

    async def _push_to_frontend_pipeline(self):
        """常驻发射协程：专职喷数据，完全不干涉外部状态"""
        while True:
            packet = None
            try:
                packet = await self._queue.get()
                if packet is None:
                    break

                # 📡 工业标准对接点：此处执行 WebSocket 真正的广播
                # await self.websocket_server.broadcast(packet)
                self.logger.debug(
                    f"📡 [Broadcast] Action: {packet['action']} | Node: {packet['node_id']}"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                node_id = packet.get("node_id", "UNKNOWN") if packet else "UNKNOWN"
                self.logger.error(
                    f"💥 Dashboard broadcast pipe error on node {node_id}: {e}",
                    exc_info=True,
                )
            finally:
                if packet is not None:
                    self._queue.task_done()

    async def close(self):
        """优雅 Drain 冲刷剩余广播包并关闭常驻任务（修复了原生 break 导致的死锁）"""
        if not self._running:
            return
        self._running = False

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._safe_put, None)
        else:
            self._safe_put(None)

        if self._queue.qsize() > 0:
            self.logger.info(
                f"Draining {self._queue.qsize()} remaining dashboard events..."
            )
            try:
                await asyncio.wait_for(self._queue.join(), timeout=3.0)
            except asyncio.TimeoutError:
                self.logger.warning("Dashboard drain timed out, forcing close.")

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                self.logger.debug("Dashboard broadcaster task cancelled successfully.")

        self.logger.info("DashboardBroadcaster closed cleanly.")
