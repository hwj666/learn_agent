import time
import asyncio
import logging


class DashboardBroadcaster:
    """专职大屏推送的独立广播组件（通过组合模式与账本解耦）"""

    def __init__(
        self, session_id: str, logger: logging.Logger, max_queue_size: int = 1000
    ):
        self.session_id = session_id
        self.logger = logger

        # 【修复点 1】必须设置队列上限，防止 OOM
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=max_queue_size)
        self._task = asyncio.create_task(self._push_to_frontend_pipeline())
        self._running = True  # 显式运行状态标志

    def push(
        self,
        action_type: str,
        node_id: str,
        total_tokens: int,
        session_status: str,
        payload: dict,
    ):
        """0阻塞投递快照包至广播队列"""
        if not self._running:
            # 如果广播器已关闭，直接丢弃，防止往死队列里塞数据
            return

        event_packet = {
            "session_id": self.session_id,
            "action": action_type,
            "node_id": node_id,
            "timestamp": time.time(),
            "total_tokens": total_tokens,
            "status": session_status,
            "data": payload,
        }
        try:
            self._queue.put_nowait(event_packet)
        except asyncio.QueueFull:
            # 【优化】降级日志级别，避免刷屏，或者使用 metrics counter
            self.logger.warning(
                f"🎨 [Dashboard Queue Full] Dropping event for node {node_id}. Action: {action_type}"
            )

    async def _push_to_frontend_pipeline(self):
        """常驻发射协程：专职喷数据，完全不干涉外部状态"""
        while self._running:
            try:
                packet = await self._queue.get()
                # 📡 工业标准对接点：此处执行 WebSocket 真正的广播
                # await self.websocket_server.broadcast(packet)
                self.logger.debug(
                    f"📡 [Broadcast] Action: {packet['action']} | Node: {packet['node_id']}"
                )
                self._queue.task_done()
            except asyncio.CancelledError:
                # 收到取消信号，跳出循环
                break
            except Exception as e:
                self.logger.error(
                    f"💥 Dashboard broadcast pipe error: {e}", exc_info=True
                )
                # 即使出错，也要标记 task_done，否则 join() 会永远挂起
                if not self._queue.empty():
                    self._queue.task_done()

    async def close(self):
        """优雅 Drain 冲刷剩余广播包"""
        self._running = False

        # 等待队列清空
        if not self._queue.empty():
            self.logger.info(
                f"Draining {self._queue.qsize()} remaining dashboard events..."
            )
            await self._queue.join()

        # 取消后台任务
        if not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                self.logger.debug("Dashboard broadcaster task cancelled successfully.")

        self.logger.info("DashboardBroadcaster closed.")
