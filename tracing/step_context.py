# context.py
from __future__ import annotations
from contextvars import ContextVar
import uuid
import functools
import logging
from typing import Optional, Any

from common.schema import AgentSpan, StepEvent, StepEventType
from exporter import get_global_exporter

from .exception_handlers import translate_exception_to_event

logger = logging.getLogger(__name__)

_active_span: ContextVar[Optional[AgentSpan]] = ContextVar(
    "agent_active_span", default=None
)

class AgentStepContext:
    """⚙️ 工业级异步上下文管理器"""

    def __init__(
        self,
        span_name: str,
        metadata: Optional[dict] = None,
        parent_span: Optional[AgentSpan] = None,
    ):
        self.span_name = span_name
        self.metadata = metadata or {}
        self.parent_span = parent_span
        self.exporter = get_global_exporter()
        self.span: AgentSpan | None = None
        self._span_token: Any | None = None

    async def __aenter__(self) -> AgentStepContext:
        parent = self.arent_span or _active_span.get()

        span_id = f"sp_{uuid.uuid4().hex[:8]}"
        depth = (parent.depth + 1) if parent else 1

        self.span = AgentSpan(
            span_id=span_id,
            session_id=parent.session_id if parent else "sys_session",
            trace_id=parent.trace_id if parent else f"tr_{uuid.uuid4().hex[:8]}",
            span_name=self.span_name,
            parent_span_id=parent.span_id if parent else None,
            depth=depth,
        )

        self._span_token = _active_span.set(self.span)
        self.exporter.export(StepEvent(StepEventType.ENTER, self.span, self.metadata))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self.span:
            return False
        try:
            if exc_val is not None:
                event = translate_exception_to_event(exc_val, self.span)
                self.exporter.export(event)
            else:
                self.exporter.export(StepEvent(StepEventType.EXIT, self.span))
        except Exception:
            logger.exception("Trace system internal error")
        finally:
            if self._context_token:
                _active_span.reset(self._context_token)
        return False

    def stream_chunk(self, text: str) -> None:
        if not self.span:
            return
        self.exporter.export(
            StepEvent(
                StepEventType.STREAM_CHUNK,
                self.span,
                chunk_text=text,
            )
        )

    def update_metadata(self, **kwargs) -> None:
        if not self.span:
            return
        self.exporter.export(
            StepEvent(
                StepEventType.DATA_UPDATE,
                self.span,
                metadata=kwargs,
            )
        )

# ====== API 层 ======


def trace_step(span_name: str, log_args: bool = False):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            meta = {}
            if log_args:
                meta = {"args": args, "kwargs": kwargs}
            async with AgentStepContext(span_name, metadata=meta):
                return await func(*args, **kwargs)

        return wrapper

    return decorator