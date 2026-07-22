# tracing/core/translators.py
import traceback
import logging
from typing import Callable, Dict, Type, Optional
from common.exceptions import (
    CancelledFuseError,
    TimeoutFuseError,
    TokenBudgetExceededError,
)
from tracing.core.schema import AgentSpan, StepEvent, StepEventType

logger = logging.getLogger(__name__)

TranslatorFunction = Callable[[Exception, AgentSpan], StepEvent]
_TRANSLATOR_REGISTRY: Dict[Type[Exception], TranslatorFunction] = {}


def register_translator(*exc_types: Type[Exception]):
    """装饰器：注册异常转换器"""

    def decorator(fn: TranslatorFunction):
        for exc_type in exc_types:
            _TRANSLATOR_REGISTRY[exc_type] = fn
        return fn

    return decorator


@register_translator(TimeoutFuseError)
def _translate_timeout(exc: TimeoutFuseError, span: AgentSpan) -> StepEvent:
    """超时异常转换器"""
    return StepEvent(
        event_type=StepEventType.TIMEOUT,
        span=span,
        error_msg=str(exc),
        metadata={
            "error_code": getattr(exc, "error_code", "TIMEOUT_ERROR"),
            "exc_type": type(exc).__name__,
        },
    )


@register_translator(CancelledFuseError)
def _translate_cancelled(exc: CancelledFuseError, span: AgentSpan) -> StepEvent:
    """取消异常转换器"""
    return StepEvent(
        event_type=StepEventType.CANCELLED,
        span=span,
        error_msg=str(exc),
        metadata={
            "error_code": getattr(exc, "error_code", "CANCELLED_ERROR"),
            "exc_type": type(exc).__name__,
        },
    )


@register_translator(TokenBudgetExceededError)
def _translate_budget(exc: TokenBudgetExceededError, span: AgentSpan) -> StepEvent:
    """预算超限异常转换器"""
    return StepEvent(
        event_type=StepEventType.BUDGET_EXCEEDED,
        span=span,
        error_msg=str(exc),
        metadata={
            "error_code": getattr(exc, "error_code", "BUDGET_EXCEEDED"),
            "limit": getattr(exc, "limit", 0),
            "used": getattr(exc, "used", 0),
            "unit": getattr(exc, "unit", "tokens"),
            "exc_type": type(exc).__name__,
        },
    )


def _translate_generic_crash(exc: Exception, span: AgentSpan) -> StepEvent:
    """通用异常转换器（兜底）"""
    try:
        tb = traceback.format_exc(limit=15)
    except Exception:
        tb = "Failed to extract traceback."

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


def translate_exception_to_event(
    exc: Optional[Exception], span: AgentSpan
) -> StepEvent:
    """
    将异常转换为标准化的StepEvent

    Args:
        exc: 捕获的异常（可能为None）
        span: 关联的AgentSpan

    Returns:
        StepEvent: 标准化的事件对象
    """
    if exc is None:
        exc = RuntimeError(
            "translate_exception_to_event received None instead of an Exception"
        )
        logger.warning("Empty exception captured, creating synthetic RuntimeError")

    # 遍历异常类的MRO查找注册的转换器
    for base_cls in type(exc).__mro__:
        translator = _TRANSLATOR_REGISTRY.get(base_cls)
        if translator:
            try:
                return translator(exc, span)
            except Exception as translator_exc:
                logger.exception(
                    "Translator %s crashed for exception %s: %s. "
                    "Falling back to generic crash handler.",
                    translator.__name__,
                    type(exc).__name__,
                    translator_exc,
                )
                # 转换器崩溃，使用通用转换器
                return _translate_generic_crash(exc, span)

    # 没有找到专用转换器，使用通用转换器
    return _translate_generic_crash(exc, span)
