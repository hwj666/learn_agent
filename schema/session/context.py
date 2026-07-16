import time
import logging
from typing import Any
from schema.context import RuntimeContext
from schema.session.session import AgentSession

logger = logging.getLogger("StandardStepContext")


class StandardStepContext:
    """界面/业务层对接的非阻塞同步上下文管理器"""

    def __init__(
        self, outer: "AgentSession", node_id: str, metadata: Any, attempt_idx: int
    ):
        self.outer = outer
        self.node_id = node_id
        self.metadata = metadata
        self.attempt_idx = attempt_idx
        self._ctx_manager = None  # 【优化点 1】显式初始化，防御性编程

    def __enter__(self):
        logger.debug(f"Entering step: {self.node_id} (Attempt: {self.attempt_idx})")

        # 1. 超出预算实时在前线熔断爆破，0延迟保护钱包
        # 注意：如果此处抛异常，__exit__ 不会被调用，这正是我们想要的
        self.outer.check_budget_pure()

        # 2. 同步构建拓扑并对齐元数据洗涤
        self.outer.record_step_enter(
            node_id=self.node_id, metadata=self.metadata, attempt_idx=self.attempt_idx
        )

        # 3. 激活并绑定当前微观节点的日志追踪管理通道
        self._ctx_manager = RuntimeContext.guard_node(self.node_id)
        self._ctx_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        logger.debug(f"Exiting step: {self.node_id}, Exception: {exc_type}")

        try:
            is_error = exc_type is not None

            if is_error:
                # 【优化点 2】构造更详细的错误信息
                error_msg = (
                    f"{exc_type.__name__}: {str(exc_val)}"
                    if exc_val
                    else exc_type.__name__
                )
                logger.warning(
                    f"Node {self.node_id} crashed. Triggering cascade failure. Error: {error_msg}"
                )
                # 🚀 发生异常，触发严格垂直血缘崩溃传播，级联上游祖先
                self.outer.record_node_crashed(
                    trigger_node_id=self.node_id,
                    trigger_idx=self.attempt_idx,
                    error_msg=error_msg,  # 传递完整的错误字符串
                )
            else:
                # 🚀 正常出站，瞬间闭环结算耗时
                self.outer.record_step_exit(
                    node_id=self.node_id, attempt_idx=self.attempt_idx
                )
        finally:
            # 确保 RuntimeContext 的栈一定被清理干净
            if self._ctx_manager:
                # 这里传递 exc 信息，确保内层 guard 也能正确处理异常状态
                self._ctx_manager.__exit__(exc_type, exc_val, exc_tb)
            else:
                # 防御性日志：如果进入时没有设置 manager，说明在进入阶段就失败了
                logger.error(
                    f"Context manager for {self.node_id} was not initialized. "
                    f"Possible budget exhaustion or early crash during __enter__."
                )

        return False  # 保持错误正常向上冒泡
