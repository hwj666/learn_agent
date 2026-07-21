# exporter.py
from contextvars import ContextVar
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from infra.event_queue import AsyncEventQueue
from common.schema import StepEvent, StepEventType

logger = logging.getLogger(__name__)

_active_exporter: ContextVar[Optional["AgentEventExporter"]] = ContextVar(
    "agent_active_exporter", default=None
)


class EventTransport(ABC):
    @abstractmethod
    async def send(self, payload: dict) -> None: ...


class WebSocketTransport(EventTransport):
    def __init__(self, ws_manager: Any):
        self.ws_manager = ws_manager

    async def send(self, payload: dict) -> None:
        await self.ws_manager.send_json(payload)


class AgentEventExporter:
    """🌐 Agent 链路事件流式导出器"""

    def __init__(self, transport: EventTransport):
        self.transport = transport
        self._event_queue = AsyncEventQueue(self._export_to_network)
        self._context_token: Any | None = None

    def bind_to_current_context(self) -> None:
        if self._context_token:
            raise RuntimeError("Exporter already bound")
        self._context_token = _active_exporter.set(self)

    def unbind_from_current_context(self) -> None:
        if self._context_token:
            _active_exporter.reset(self._context_token)
            self._context_token = None

    def export(self, event: StepEvent) -> None:
        payload = self._build_payload(event)
        self._event_queue.push(payload)

        # ✅ 预算触发点（只记账，不决策）
        if event.event_type == StepEventType.DATA_UPDATE:
            tokens = event.metadata.get("tokens", 0)
            cost = event.metadata.get("cost", 0.0)

    async def wait_for_drain(self) -> None:
        await self._event_queue.wait_flush()

    async def shutdown(self) -> None:
        await self._event_queue.join_and_close()

    def _build_payload(self, event: StepEvent) -> dict:
        return {
            "event": event.event_type.value,
            "span_id": event.span.span_id,
            "parent_span_id": event.span.parent_span_id,
            "trace_id": event.span.trace_id,
            "session_id": event.span.session_id,
            "name": event.span.span_name,
            "depth": event.span.depth,
            "timestamp": int(event.timestamp.timestamp() * 1000),
            "chunk_text": event.chunk_text,
            "metadata": event.metadata,
            "error_msg": event.error_msg,
        }

    async def _export_to_network(self, payload: dict) -> None:
        try:
            await self.transport.send(payload)
        except Exception:
            logger.exception("Export failed, payload dropped")


def get_global_exporter() -> AgentEventExporter:
    return _active_exporter.get()
