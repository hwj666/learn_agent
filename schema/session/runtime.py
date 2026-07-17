import uuid
import contextvars
from typing import Iterable, Tuple, Optional
from contextlib import contextmanager


class RuntimeContext:
    """运行时上下文大管家（工业级零拷贝并发安全版）
    职责：100% 垄断和维护栈的读写权，利用 ContextVar 和不可变 Tuple 提供天然的并发安全隔离。
    """

    _session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
        "ctx_session_id", default="SYSTEM"
    )
    _node_id: contextvars.ContextVar[str] = contextvars.ContextVar(
        "ctx_node_id", default="MAIN"
    )
    _node_stack: contextvars.ContextVar[Tuple[str, ...]] = contextvars.ContextVar(
        "ctx_node_stack", default=()
    )
    _trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
        "ctx_trace_id", default=None
    )

    @classmethod
    def get_trace_id(cls) -> str:
        """获取全链路追踪ID（纯函数，无任何副作用）"""
        tid = cls._trace_id.get()
        if tid is not None:
            return tid
        # 工业级修正：坚决不调用 set()！只返回临时的 RAW ID。
        # 这样即使在 Root Context 触发，也不会污染上下文，guard_session 退出后依然能完美回到 None。
        return f"T-RAW-{uuid.uuid4().hex[:12]}"

    @classmethod
    def has_trace_id(cls) -> bool:
        """辅助方法：判断当前上下文是否已经由 Session 托管注入了合法的 TraceID"""
        return cls._trace_id.get() is not None

    @classmethod
    def get_session_id(cls) -> str:
        return cls._session_id.get()

    @classmethod
    def get_node_id(cls) -> str:
        return cls._node_id.get()

    @classmethod
    def get_stack(cls) -> Tuple[str, ...]:
        return cls._node_stack.get()

    @classmethod
    def set_stack(cls, stack: Iterable[str]) -> contextvars.Token:
        """兜底物理防腐网：强行收敛为不可变元组，杜绝外部脏数据修改写入"""
        return cls._node_stack.set(tuple(stack))

    @classmethod
    @contextmanager
    def guard_session(
        cls, session_id: str, trace_id: Optional[str] = None, flat_mode: bool = False
    ):
        """Session 级安全生命周期守卫"""
        # 所有的正式 TraceID 必须在此处或中间件中被显式、集中地初始化
        actual_trace_id = trace_id or f"T-{uuid.uuid4().hex[:16]}"

        s_token = cls._session_id.set(session_id)
        t_token = cls._trace_id.set(actual_trace_id)
        st_token = cls._node_stack.set(())

        try:
            yield actual_trace_id
        finally:
            # 退出时，_trace_id 必定完美回归到进入前的状态（如果没有被根上下文污染，则回归到 None）
            cls._node_stack.reset(st_token)
            cls._trace_id.reset(t_token)
            cls._session_id.reset(s_token)

    @classmethod
    @contextmanager
    def guard_node(cls, node_id: str, flat_mode: bool = False):
        """Node 级安全生命周期守卫"""
        current_stack = cls._node_stack.get()
        new_stack = (node_id,) if flat_mode else current_stack + (node_id,)

        node_token = cls._node_id.set(node_id)
        stack_token = cls._node_stack.set(new_stack)

        try:
            yield len(new_stack)
        finally:
            cls._node_stack.reset(stack_token)
            cls._node_id.reset(node_token)
