import time
from enum import Enum
from threading import Lock
from typing import List, Dict, Any, Optional


class SessionStatus(Enum):
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class NodeStatus(Enum):
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PlannerMetadata:
    __slots__ = ("node_type", "description", "planned_tasks", "raw_user_query", "error")

    def __init__(self, description: str = "", raw_user_query: str = "") -> None:
        self.node_type = "Planner"
        self.description = description
        self.planned_tasks: List[str] = []
        self.raw_user_query = raw_user_query
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "node_type": self.node_type,
            "description": self.description,
            "planned_tasks": self.planned_tasks,
            "raw_user_query": self.raw_user_query,
            "error": self.error,
        }


class ReActTurnMetadata:
    __slots__ = (
        "node_type",
        "description",
        "message_count",
        "content",
        "reasoning",
        "has_tool_calls",
        "tool_results",
        "error",
    )

    def __init__(self, description: str = "", message_count: int = 0):
        self.node_type = "ReAct_Micro_Turn"
        self.description = description
        self.message_count = message_count
        self.content = None
        self.reasoning = None
        self.has_tool_calls = False
        self.tool_results: List[Dict[str, Any]] = []
        self.error = None

    def to_dict(self) -> dict:
        return {
            "node_type": self.node_type,
            "description": self.description,
            "message_count": self.message_count,
            "content": self.content,
            "reasoning": self.reasoning,
            "has_tool_calls": self.has_tool_calls,
            "tool_results": self.tool_results,
            "error": self.error,
        }


class SubStepMetadata:
    __slots__ = (
        "node_type",
        "description",
        "tool_name",
        "arguments",
        "output_data",
        "error",
    )

    def __init__(self, tool_name: str = "", arguments: Any = None):
        self.node_type = "SubStep_Action"
        self.description = f"Calling tool [{tool_name}]"
        self.tool_name = tool_name
        self.arguments = arguments
        self.output_data: Dict[str, Any] = {}
        self.error = None

    def to_dict(self) -> dict:
        return {
            "node_type": self.node_type,
            "description": self.description,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "output_data": self.output_data,
            "error": self.error,
        }


class TaskMetadata:
    """通用任务元数据，用于兼容顶层流水线和子任务节点的结构化导出"""

    __slots__ = ("node_type", "description", "input_data", "output_data", "error")

    def __init__(
        self, node_type: str, description: str = "", input_data: Optional[dict] = None
    ) -> None:
        self.node_type = node_type
        self.description = description
        self.input_data = input_data or {}
        self.output_data: Dict[str, Any] = {}
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "node_type": self.node_type,
            "description": self.description,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
        }


class NodeRecord:
    """原子终结、带独立线程锁的高性能物理节点记录"""

    def __init__(self, node_id: str, metadata: Any = None):
        self.node_id = node_id
        self.status = NodeStatus.INITIALIZING
        self.metadata = metadata if metadata is not None else {}
        self.custom_ext: Dict[str, Any] = {}
        self.children: List["NodeRecord"] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self._lock = Lock()

    @property
    def duration_ms(self) -> Optional[float]:
        with self._lock:
            if self.end_time:
                return (self.end_time - self.start_time) * 1000
            return (time.time() - self.start_time) * 1000

    def _atomic_finalize(self, status: NodeStatus, error_msg: Optional[str] = None):
        if self.status in (NodeStatus.COMPLETED, NodeStatus.FAILED):
            return
        self.status = status
        self.end_time = time.time()
        if error_msg:
            if isinstance(self.metadata, dict):
                self.metadata["error"] = error_msg
            elif hasattr(self.metadata, "__slots__") or hasattr(
                self.metadata, "__dict__"
            ):
                object.__setattr__(self.metadata, "error", error_msg)

    def mark_success(self) -> None:
        with self._lock:
            self._atomic_finalize(NodeStatus.COMPLETED)

    def mark_failure(self, error_message: str) -> None:
        with self._lock:
            self._atomic_finalize(NodeStatus.FAILED, error_message)

    def to_dict(self) -> dict:
        with self._lock:
            meta_dict = (
                self.metadata.to_dict()
                if hasattr(self.metadata, "to_dict")
                else str(self.metadata)
            )
            return {
                "node_id": self.node_id,
                "status": self.status.value,
                "duration_ms": self.duration_ms,
                "metadata": meta_dict,
                "custom_ext": self.custom_ext,
                "children": [child.to_dict() for child in self.children],
            }
