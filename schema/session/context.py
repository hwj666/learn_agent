import logging
from typing import Any, Optional
from types import TracebackType
from schema.session.runtime import RuntimeContext

logger = logging.getLogger("StandardStepContext")


class StandardStepContext:
    """同步生命周期上下文管理器
    核心变更：全权垄断生命周期内的压栈（__enter__）与弹栈（__exit__）职责。
    """

    def __init__(self, outer: Any, node_id: str, metadata: Any, attempt_idx: int):
        self.outer = outer
        self.node_id = node_id
        self.metadata = metadata
        self.attempt_idx = attempt_idx
        self._ctx_manager: Optional[Any] = None

    def __enter__(self):
        logger.debug(f"Entering step: {self.node_id} (Attempt: {self.attempt_idx})")

        # 1. 🟢 将微观环境锚定器前提：垄断写栈权，利用 ContextVar Token 机制确保 LIFO 安全
        try:
            self._ctx_manager = RuntimeContext.guard_node(self.node_id)
            self._ctx_manager.__enter__()
        except Exception as e:
            logger.error(
                f"Failed to initialize guard_node context for {self.node_id}: {e}"
            )
            raise

        # 2. 🟢 做账大管家登场：内部第一行 check_budget_pure 实时熔断拦截。
        # 此时读取到的 RuntimeContext.get_stack() 已完美包含当前 node_id，物理拓扑完全闭合。
        try:
            self.outer.record_step_enter(
                node_id=self.node_id,
                metadata=self.metadata,
                attempt_idx=self.attempt_idx,
            )
        except Exception:
            # 如果大管家在入口处熔断或报错，安全弹出刚才压入的栈，防止栈污染，随后向上冒泡
            if self._ctx_manager:
                self._ctx_manager.__exit__(None, None, None)
            raise
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        logger.debug(f"Exiting step: {self.node_id}, Exception: {exc_type}")

        # 3. 🟢 账本核心结算与状态变更（大管家此时沦为“纯只读/只记账”模式，不碰栈）
        try:
            if exc_type is not None:
                raw_err = str(exc_val) if exc_val else exc_type.__name__
                error_msg = f"{exc_type.__name__}: {raw_err[:500]}"  # 防内存放大
                logger.warning(
                    f"Node {self.node_id} (Attempt {self.attempt_idx}) failed: {error_msg}"
                )

                self.outer.record_node_crashed(
                    trigger_node_id=self.node_id,
                    trigger_idx=self.attempt_idx,
                    error_msg=error_msg,
                )
            else:
                self.outer.record_step_exit(
                    node_id=self.node_id, attempt_idx=self.attempt_idx
                )
        except Exception as ledger_err:
            logger.critical(
                f"Ledger accounting crashed during __exit__ for {self.node_id}: {ledger_err}"
            )
        finally:
            # 4. 🟢 遵循 LIFO 原则，最后且必定关闭内层环境，利用原生的 reset(token) 干净地回滚状态栈
            if self._ctx_manager:
                try:
                    self._ctx_manager.__exit__(exc_type, exc_val, exc_tb)
                except Exception as ctx_err:
                    logger.error(
                        f"Error while exiting internal ctx_manager for {self.node_id}: {ctx_err}"
                    )

        return False  # 保持错误正常向上冒泡，由上层重试器裁决
