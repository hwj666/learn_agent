# exception_handlers.py
import traceback
from typing import Callable, Dict, Type
from common.exceptions import (
    BudgetExceededError,
    TimeoutFuseError,
    CancelledFuseError,
)
from common.schema import AgentSpan, StepEvent, StepEventType

ExceptionHandlerType = Callable[[Exception, AgentSpan], StepEvent]

EXCEPTION_REGISTRY: Dict[Type[Exception], ExceptionHandlerType] = {}


def _handle_timeout(exc: TimeoutFuseError, span: AgentSpan) -> StepEvent:
    return StepEvent(
        event_type=StepEventType.TIMEOUT,
        span=span,
        error_msg=str(exc),
        metadata={"error_code": exc.error_code},
    )


def _handle_cancelled(exc: CancelledFuseError, span: AgentSpan) -> StepEvent:
    return StepEvent(
        event_type=StepEventType.CANCELLED,
        span=span,
        error_msg=str(exc),
        metadata={"error_code": exc.error_code},
    )


def _handle_budget(exc: BudgetExceededError, span: AgentSpan) -> StepEvent:
    return StepEvent(
        event_type=StepEventType.BUDGET_EXCEEDED,
        span=span,
        error_msg=str(exc),
        metadata={
            "error_code": exc.error_code,
            "limit": exc.limit,
            "used": exc.used,
            "unit": exc.unit,
        },
    )


def _handle_generic_crash(exc: Exception, span: AgentSpan) -> StepEvent:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return StepEvent(
        event_type=StepEventType.CRASHED,
        span=span,
        error_msg=str(exc),
        metadata={
            "error_code": "UNHANDLED_CRASH",
            "exc_type": type(exc).__name__,
            "stack_trace": tb,
        },
    )


EXCEPTION_REGISTRY.update(
    {
        TimeoutFuseError: _handle_timeout,
        CancelledFuseError: _handle_cancelled,
        BudgetExceededError: _handle_budget,
    }
)


def translate_exception_to_event(
    exc: Exception,
    span: AgentSpan,
) -> StepEvent:
    exc_type = type(exc)
    for target_class, handler in EXCEPTION_REGISTRY.items():
        if isinstance(exc, target_class):
            return handler(exc, span)
    return _handle_generic_crash(exc, span)
