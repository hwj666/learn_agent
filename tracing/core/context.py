# tracing/core/context.py
from __future__ import annotations

import uuid
import logging
from contextvars import ContextVar
from typing import Optional, Any

from tracing.core.schema import AgentSpan, StepEvent, StepEventType
from tracing.core.translators import translate_exception_to_event
from tracing.infra.exporter import get_global_exporter

logger = logging.getLogger(__name__)

ROOT_SESSION_ID = "sys_session"
ROOT_TRACE_ID_PREFIX = "tr_"

_active_span: ContextVar[Optional[AgentSpan]] = ContextVar(
    "agent_active_span", default=None
)


class AgentStepContext:
    def __init__(self, span_name: str, metadata: Optional[dict] = None):
        self.span_name = span_name
        self.metadata = metadata or {}
        self.span: AgentSpan | None = None
        self._span_token: Any | None = None
        self._exporter = get_global_exporter()  # ✅ 只取一次

    async def __aenter__(self) -> AgentStepContext:
        parent = _active_span.get()

        span_id = f"sp_{uuid.uuid4().hex[:8]}"
        trace_id = (
            parent.trace_id
            if parent
            else f"{ROOT_TRACE_ID_PREFIX}{uuid.uuid4().hex[:8]}"
        )
        session_id = parent.session_id if parent else ROOT_SESSION_ID

        self.span = AgentSpan(
            span_id=span_id,
            session_id=session_id,
            trace_id=trace_id,
            span_name=self.span_name,
            parent_span_id=parent.span_id if parent else None,
            depth=(parent.depth + 1) if parent else 1,
        )

        self._span_token = _active_span.set(self.span)

        try:
            await self._exporter.export(
                StepEvent(
                    event_type=StepEventType.ENTER,
                    span=self.span,
                    metadata=self.metadata,
                )
            )
        except Exception:
            logger.warning(
                "Failed to export ENTER event for span=%s",
                self.span.span_id,
                exc_info=True,
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self.span:
            return False

        try:
            if exc_val is not None:
                event = translate_exception_to_event(exc_val, self.span)
                await self._exporter.export(event)
            else:
                await self._exporter.export(StepEvent(StepEventType.EXIT, self.span))
        except Exception:
            # ✅ tracing 失败绝不能影响业务异常传播
            logger.warning(
                "Tracing exporter failed during __aexit__ for span=%s",
                self.span.span_id,
                exc_info=True,
            )
        finally:
            if self._span_token:
                _active_span.reset(self._span_token)

        # ✅ 永远不吞异常
        return False


#
# ===== 可选：对外只读 API =====
#
def current_span() -> AgentSpan | None:
    """供业务代码安全读取当前 span（不可修改）"""
    return _active_span.get()
