# utils/trace.py
import contextvars
import uuid

# 全局上下文变量
_trace_id = contextvars.ContextVar("trace_id", default=None)

def set_trace_id(trace_id: str | None = None) -> str:
    """设置 Trace ID，不存在则生成 UUID"""
    if not trace_id:
        trace_id = f"trace_{uuid.uuid4().hex[:16]}"
    _trace_id.set(trace_id)
    return trace_id

def get_trace_id() -> str | None:
    """获取当前 Trace ID"""
    return _trace_id.get()