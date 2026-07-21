# tracing/exceptions.py
from dataclasses import dataclass

class AgentBaseException(RuntimeError):
    def __init__(self, message: str, error_code: str = "AGENT_ERROR"):
        super().__init__(message)
        self.error_code = error_code

@dataclass
class BudgetExceededError(AgentBaseException):
    limit: float
    used: float
    unit: str = "USD"

    def __post_init__(self):
        msg = f"Budget exceeded: {self.used}{self.unit} / {self.limit}{self.unit}"
        super().__init__(msg, error_code="BUDGET_EXCEEDED")

class AgentFuseException(AgentBaseException):
    """🔥 所有导致 Agent 运行被强制终止的异常基类"""
    pass

class TimeoutFuseError(AgentFuseException):
    def __init__(self, seconds: float):
        super().__init__(
            f"Execution timed out after {seconds}s",
            error_code="TIMEOUT"
        )

class CancelledFuseError(AgentFuseException):
    def __init__(self):
        super().__init__(
            "Execution cancelled by user or system",
            error_code="CANCELLED"
        )