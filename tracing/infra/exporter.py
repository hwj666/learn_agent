# tracing/infra/exporter.py
from contextlib import asynccontextmanager
from contextvars import ContextVar
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from tracing.core.schema import StepEvent, StepEventType
from tracing.infra.transport import EventTransport
from tracing.infra.event_queue import AsyncEventQueue

logger = logging.getLogger(__name__)

_active_exporter: ContextVar[Optional["AgentEventExporter"]] = ContextVar(
    "agent_active_exporter", default=None
)

@asynccontextmanager
async def bind_exporter(exporter: "AgentEventExporter") -> AsyncIterator["AgentEventExporter"]:
    token = _active_exporter.set(exporter)
    try:
        yield exporter
    finally:
        _active_exporter.reset(token)

class AgentEventExporter:
    def __init__(self, transport: EventTransport):
        self.transport = transport
        self._event_queue = AsyncEventQueue(self._export_to_network)
        self._token: Any | None = None

        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.dropped_events: int = 0

    async def __aenter__(self) -> "AgentEventExporter":
        if self._token is not None:
            raise RuntimeError("Exporter already active in current context")
        self._token = _active_exporter.set(self)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self.shutdown()
        finally:
            if self._token:
                _active_exporter.reset(self._token)
                self._token = None

    async def export(self, event: StepEvent) -> None:
        if not self._event_queue.accepting:
            return

        payload = self._build_payload(event)
        await self._event_queue.push(payload)

        if event.event_type == StepEventType.DATA_UPDATE and event.metadata:
            self.total_tokens += event.metadata.get("tokens", 0)
            self.total_cost += float(event.metadata.get("cost", 0.0))

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
            "metadata": event.metadata or {},
            "error_msg": event.error_msg,
        }

    async def _export_to_network(self, payload: dict) -> None:
        try:
            await self.transport.send(payload)
        except Exception:
            self.dropped_events += 1
            logger.warning(
                "Transport send failed (dropped=%d)",
                self.dropped_events,
                exc_info=True,
            )


def get_global_exporter() -> AgentEventExporter:
    exporter = _active_exporter.get()
    if exporter is None:
        raise RuntimeError("get_global_exporter")
    return exporter
