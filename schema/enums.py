from enum import Enum


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class SessionStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    TOKEN_LIMIT = "token_limit"
    PLANNER_FAILED = "planner_failed"
    CANCELLED = "cancelled"
