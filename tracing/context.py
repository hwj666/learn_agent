from __future__ import annotations
import uuid
import traceback
import asyncio
from contextvars import ContextVar
from typing import Optional, Any
from .span import AgentSpan
from .tracker import AgentTracker


_active_span_carrier: ContextVar[Optional[AgentSpan]] = ContextVar(
    "active_span_carrier", default=None
)


class AgentSpanContext:
    """⚙️ 完美异步上下文管理器：实现 OTel 调用栈与资产记账无缝嵌套"""

    MAX_SPAN_DEPTH = 32

    def __init__(
        self,
        tracker: AgentTracker,
        span_name: str,
        metadata: Optional[dict] = None,
        attempt_idx: int = 0,
        seed_span: Optional[AgentSpan] = None,
        kind: str = "INTERNAL",
    ):
        self.tracker = tracker
        self.span_name = span_name
        self.metadata = metadata or {}
        self.seed_span = seed_span
        self.attempt_idx = attempt_idx
        self.kind = kind
        self.span: Optional[AgentSpan] = None
        self._context_token: Optional[Any] = None

    async def __aenter__(self) -> AgentSpanContext:
        # 1. 🟢 优雅推导：优先使用 seed_span，否则从 ContextVar 中动态读取
        parent_span = self.seed_span or _active_span_carrier.get()

        # 2. 深度检查，防止递归爆炸
        current_depth = (parent_span.depth + 1) if parent_span else 1
        if current_depth > self.MAX_SPAN_DEPTH:
            raise RuntimeError(
                f"Span depth limit ({self.MAX_SPAN_DEPTH}) exceeded. "
                f"Possible infinite recursion in '{self.span_name}'"
            )

        # 3. 物理跨度的锁死绑定
        current_span_id = f"sp_{uuid.uuid4().hex}"
        session_id = parent_span.session_id if parent_span else "sys_session"
        trace_id = parent_span.trace_id if parent_span else f"tr_{uuid.uuid4().hex}"
        parent_span_id = parent_span.span_id if parent_span else None

        self.span = AgentSpan(
            span_id=current_span_id,
            session_id=session_id,
            trace_id=trace_id,
            span_name=self.span_name,
            parent_span_id=parent_span_id,
            attempt_idx=self.attempt_idx,
            depth=current_depth,
            kind=self.kind,
        )

        # 4. 将当前最新的 Span 压入协程栈顶
        self._context_token = _active_span_carrier.set(self.span)

        # 5. 异步记录追踪日志入库
        await self.tracker.record_step_enter(span=self.span, metadata=self.metadata)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self.span:
            return False

        try:
            if exc_type is None:
                # 正常退出
                await self.tracker.record_step_exit(span=self.span)
            elif isinstance(exc_val, asyncio.CancelledError):
                # 取消退出
                await self.tracker.record_node_cancelled(span=self.span)
            else:
                # 异常退出：记录完整堆栈
                full_stack_trace = "".join(
                    traceback.format_exception(exc_type, exc_val, exc_tb)
                )
                await self.tracker.record_node_crashed(
                    span=self.span, error_msg=full_stack_trace
                )
        finally:
            # 6. 严格退栈，确保无论 tracker 是否报错，都能把链路控制权完好交还给父级
            if self._context_token:
                _active_span_carrier.reset(self._context_token)

        return False  # 返回 False 以便让异常继续向外抛出，保证业务感应

    @classmethod
    def get_current_span(cls) -> Optional[AgentSpan]:
        """获取当前活跃的 Span（类方法，方便全局访问）"""
        return _active_span_carrier.get()

    @classmethod
    def set_current_span(cls, span: Optional[AgentSpan]) -> Any:
        """设置当前活跃的 Span，返回 token 用于恢复"""
        return _active_span_carrier.set(span)
