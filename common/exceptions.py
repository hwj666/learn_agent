# common/exceptions.py
"""
🚨 Agent 系统业务异常定义中心

所有异常均包含稳定 error_code，供 Tracing / Billing / ControlPlane 消费。
"""

class AgentBaseError(Exception):
    """所有 Agent 业务异常的基类"""
    def __init__(self, message: str, error_code: str = "AGENT_INTERNAL_ERROR"):
        super().__init__(message)
        self.error_code = error_code

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.args[0]}"


# ---------------------------
# 执行控制类异常
# ---------------------------

class TimeoutFuseError(AgentBaseError):
    """⏱️ 执行超时（硬熔断）"""
    def __init__(self, message: str, error_code: str = "TIMEOUT_FUSE_ERROR"):
        super().__init__(message, error_code)


class CancelledFuseError(AgentBaseError):
    """🚫 任务被取消（用户 / 上游 / 调度器）"""
    def __init__(self, message: str, error_code: str = "CANCELLED_FUSE_ERROR"):
        super().__init__(message, error_code)


class AgentRecoverableError(AgentBaseError):
    """🔁 可恢复异常（允许 retry / fallback）"""
    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        error_code: str = "RECOVERABLE_ERROR",
    ):
        super().__init__(message, error_code)
        self.retry_after = retry_after


# ---------------------------
# 资源与预算类异常
# ---------------------------

class TokenBudgetExceededError(AgentBaseError):
    """🪙 Token 预算超限（整数计量）"""
    def __init__(self, limit: int, used: int, message: str | None = None):
        message = message or f"Token budget exceeded: {used}/{limit}"
        super().__init__(message, error_code="TOKEN_BUDGET_EXCEEDED")
        self.limit = limit
        self.used = used


class CostBudgetExceededError(AgentBaseError):
    """💰 财务预算超限（浮点计量）"""
    def __init__(self, limit: float, used: float, message: str | None = None):
        message = message or f"Cost budget exceeded: {used:.4f}/{limit:.4f}"
        super().__init__(message, error_code="COST_BUDGET_EXCEEDED")
        self.limit = limit
        self.used = used