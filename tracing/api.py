# tracing/api.py
import asyncio
import functools
import warnings
from typing import Callable

from tracing.infra.exporter import (
    AgentEventExporter,
    EventTransport,
    get_global_exporter,
)
from tracing.core.schema import StepEvent, StepEventType
from tracing.core.context import AgentStepContext


class AgentSession:
    """
    顶层会话容器，对应一次 HTTP / WS / Job 生命周期。
    不暴露 exporter 实现细节。
    """

    def __init__(self, transport: EventTransport):
        self.exporter = AgentEventExporter(transport)

    async def __aenter__(self) -> "AgentSession":
        await self.exporter.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exporter.__aexit__(exc_type, exc_val, exc_tb)


def trace_step(span_name: str, log_args: bool = False):
    """
    业务粒度追踪装饰器，兼容 sync / async 函数。
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            meta = {}
            if log_args:
                meta = {"args": args, "kwargs": kwargs}
            async with AgentStepContext(span_name, metadata=meta):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            meta = {}
            if log_args:
                meta = {"args": args, "kwargs": kwargs}

            async def _run():
                async with AgentStepContext(span_name, metadata=meta):
                    return await asyncio.to_thread(func, *args, **kwargs)

            return asyncio.get_event_loop().run_until_complete(_run())

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


async def update_step_metadata(**kwargs) -> None:
    """
    更新当前 span 的元数据（账单、状态等）。
    必须在 AgentStepContext 内调用。
    """
    span = _active_span.get()
    if not span:
        warnings.warn(
            "update_step_metadata called outside of AgentStepContext",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    exporter = get_global_exporter()
    await exporter.export(StepEvent(StepEventType.DATA_UPDATE, span, metadata=kwargs))


async def emit_stream_chunk(text: str) -> None:
    """
    LLM 流式输出专用 API。
    必须在 AgentStepContext 内调用。
    """
    span = _active_span.get()
    if not span:
        warnings.warn(
            "emit_stream_chunk called outside of AgentStepContext",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    exporter = get_global_exporter()
    await exporter.export(StepEvent(StepEventType.STREAM_CHUNK, span, chunk_text=text))
