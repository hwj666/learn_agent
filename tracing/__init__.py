from ..common.schema import AgentSpan, StepEvent, StepEventType
from ..common.exceptions import (
    AgentBaseException,
    BudgetExceededError,
    AgentFuseException,
    TimeoutFuseError,
    CancelledFuseError,
)
from ..core.runtime_context import AgentContextRegistry
from .step_context import AgentStepContext, trace_step
from ..core.budget_guard import BudgetGuard
from .exporter import AgentEventExporter, EventTransport, WebSocketTransport, get_global_exporter
from .exception_handlers import translate_exception_to_event

# 明确声明包的公开 API
# 这是非常重要的工程实践，配合 IDE 自动补全和静态检查工具（如 mypy/pyright）
__all__ = [
    # Version
    # Schema (Data Models)
    "AgentSpan",
    "StepEvent",
    "StepEventType",
    # Exceptions
    "AgentBaseException",
    "BudgetExceededError",
    "AgentFuseException",
    "TimeoutFuseError",
    "CancelledFuseError",
    # Context & Execution
    "AgentStepContext",
    "AgentContextRegistry",
    "get_current_context",
    "trace_step",
    # Budget & Safety
    "BudgetGuard",
    # Export & IO
    "AgentEventExporter",
    "EventTransport",
    "WebSocketTransport",
    "get_global_exporter",
    # Internal Logic (rarely used directly)
    "translate_exception_to_event",
]