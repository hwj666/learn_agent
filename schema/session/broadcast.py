import time
import asyncio
import logging
from typing import Optional

class DashboardBroadcaster:
    """独立广播管道（对外同步 0 阻塞安全投递，对内常驻协程高效消费）"""
    def __init__(self, session_id: str, logger: logging.Logger, max_queue_size: int = 1000):
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

        self._queue: asyncio.Queue[Optional[dict]] = asyncio.Queue(maxsize=max_queue_size)
        self._task: Optional[asyncio.Task] = None
        self._running = True

    def _ensure_consumer_started(self):
        if self._task is None and self._running:
            try:
                self._task = asyncio.create_task(self._push_to_frontend_pipeline())
            except RuntimeError:
                self._loop.call_soon(self._start_task_lazy)

    def _start_task_lazy(self):
        if self._task is None and self._running:
            self._task = asyncio.create_task(self._push_to_frontend_pipeline())

    def push(self, action_type: str, node_id: str, attempt_idx: int, parent_id: Optional[str], total_tokens: int, session_status: str, payload: dict):
        if not self._running: return
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
            self.logger.warning(f"🎨 [Dashboard Queue Full] Dropping node {node_id} | Action: {action}")

    async def _push_to_frontend_pipeline(self):
        while True:
            packet = None
            try:
                packet = await self._queue.get()
                if packet is None: break
                # 📡 真实生产环境对接点：await self.websocket_server.broadcast(packet)
                self.logger.debug(f"📡 [Broadcast] Action: {packet['action']} | Node: {packet['node_id']}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                node_id = packet.get("node_id", "UNKNOWN") if packet else "UNKNOWN"
                self.logger.error(f"💥 Broadcast pipe error on node {node_id}: {e}", exc_info=True)
            finally:
                if packet is not None:
                    self._queue.task_done()

    async def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._running = False

        # 确保消费者存在
        self._ensure_consumer_started()

        # 投递关闭哨兵
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._safe_put, None)
        elif not self._loop.is_closed():
            self._safe_put(None)
        else:
            self.logger.warning("Event loop closed, skipping drain.")

        # Drain 剩余事件
        if self._queue.qsize() > 0:
            self.logger.info(f"Draining {self._queue.qsize()} dashboard events...")
            try:
                await asyncio.wait_for(self._queue.join(), timeout=3.0)
            except asyncio.TimeoutError:
                self.logger.warning("Dashboard drain timed out.")

        # 取消消费者
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                self.logger.debug("Dashboard broadcaster task cancelled.")

        self.logger.info("DashboardBroadcaster closed cleanly.")