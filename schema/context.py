import uuid
import contextvars
from typing import Iterable, Tuple, Optional
from contextlib import contextmanager


# =====================================================================
# 1. 运行时上下文大管家 (RuntimeContext) - 工业级零拷贝并发安全版
# =====================================================================
class RuntimeContext:
    """
    分布式级运行时上下文大管家（工业防腐、高并发安全版）
    基于元组（Tuple）实现零拷贝、天然并发安全的微观拓扑栈
    """

    _session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
        "ctx_session_id", default="SYSTEM"
    )
    _node_id: contextvars.ContextVar[str] = contextvars.ContextVar(
        "ctx_node_id", default="MAIN"
    )

    # 将 List[str] 改为 Tuple[str, ...]，利用不可变性防并发污染
    _node_stack: contextvars.ContextVar[Tuple[str, ...]] = contextvars.ContextVar(
        "ctx_node_stack", default=()
    )

    _trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
        "ctx_trace_id", default=None
    )

    @classmethod
    def get_trace_id(cls) -> str:
        """获取当前链路的唯一 TraceID。若无则返回悬挂追踪提示，绝不隐式改写上下文"""
        tid = cls._trace_id.get()
        if tid is None:
            # 🌟 修复：只返回，不 set！保证当前上下文依然是 None，让后续的 guard_session 能正常覆盖
            return f"T-RAW-{uuid.uuid4().hex[:12]}"
        return tid

    @classmethod
    def get_session_id(cls) -> str:
        return cls._session_id.get()

    @classmethod
    def get_node_id(cls) -> str:
        return cls._node_id.get()

    @classmethod
    def get_stack(cls) -> Tuple[str, ...]:
        """获取当前拓扑栈（返回不可变元组，调用方可安全读取）"""
        return cls._node_stack.get()

    @classmethod
    def set_stack(cls, stack: Iterable[str]) -> contextvars.Token:
        """
        🌟 统一写入网关：安全覆盖当前协程的拓扑栈镜像
        通过强制 tuple() 转换，建立物理防腐层，100% 杜绝外部错误类型写入引发的数据污染
        """
        # 强行收敛为不可变元组，确保绝对安全
        token = cls._node_stack.set(tuple(stack))
        return token

    @classmethod
    @contextmanager
    def guard_session(
        cls, session_id: str, trace_id: Optional[str] = None, flat_mode: bool = False
    ):
        """
        全局生命周期锚定器（双模高可用版）
        :param flat_mode: 是否开启扁平清洗模式（后台长寿命死循环消费者专用，彻底根除 Token 内存膨胀）
        """
        actual_trace_id = trace_id or f"T-{uuid.uuid4().hex[:16]}"

        if flat_mode:
            cls._session_id.set(session_id)
            cls._trace_id.set(actual_trace_id)
            cls._node_stack.set(())  # 清空，且不产生 st_token
            try:
                yield actual_trace_id
            finally:
                # 🌟 绝不 reset！下一次循环的新事件自会覆盖它。
                # 这样可以确保当前的死循环协程在堆内存中的空间复杂度永远保持在常量级 O(1)
                pass
        else:
            s_token = cls._session_id.set(session_id)
            t_token = cls._trace_id.set(actual_trace_id)
            st_token = cls._node_stack.set(())
            try:
                yield actual_trace_id
            finally:
                cls._node_stack.reset(st_token)
                cls._trace_id.reset(t_token)
                cls._session_id.reset(s_token)

    @classmethod
    @contextmanager
    def guard_node(cls, node_id: str, flat_mode: bool = False):
        """
        微观拓扑生命周期锚定器（双模高可用版）
        :param flat_mode: 是否开启扁平清洗模式（后台长寿命死循环消费者专用，100% 免疫内存泄漏）
        """
        if flat_mode:
            # 🟢 后方管道流派：直接覆盖，不读取旧元组追加，不留任何历史旧值引用
            cls._node_id.set(node_id)
            cls._node_stack.set((node_id,))
            try:
                yield 1
            finally:
                # 🌟 绝不 reset，彻底斩断幽灵令牌残余
                pass
        else:
            # 🟢 前线业务流派：Tuple零拷贝拼接，高精准还原递归与回路调用图谱
            current_stack = cls._node_stack.get()
            new_stack = current_stack + (node_id,)

            node_token = cls._node_id.set(node_id)
            stack_token = cls._node_stack.set(new_stack)
            try:
                yield len(new_stack)
            finally:
                cls._node_id.reset(node_token)
                cls._node_stack.reset(stack_token)
