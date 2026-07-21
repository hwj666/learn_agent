from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import time


class NodeStatus(str, Enum):
    """节点执行状态枚举"""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CRASHED = "CRASHED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


class SpanKind(str, Enum):
    """Span 类型枚举，对齐 OpenTelemetry 标准"""

    INTERNAL = "INTERNAL"
    CLIENT = "CLIENT"
    SERVER = "SERVER"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


class BaseMetadata(BaseModel):
    """元数据基类，提供统一的字段和行为"""

    node_type: str = Field(..., description="节点类型标识")
    description: str = Field(default="", description="节点描述")
    error: Optional[str] = Field(default=None, description="错误信息")
    timestamp: float = Field(default_factory=time.time, description="时间戳（秒）")
    status: NodeStatus = Field(default=NodeStatus.RUNNING, description="节点执行状态")


class PlannerMetadata(BaseModel):
    """规划阶段元数据契约"""

    node_type: str = "Planner"
    description: str = ""
    planned_tasks: List[str] = Field(default_factory=list)
    raw_user_query: str = ""
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    status: NodeStatus = NodeStatus.RUNNING


class ReActTurnMetadata(BaseModel):
    """ReAct 核心迭代轮次元数据契约"""

    node_type: str = "ReAct_Micro_Turn"
    description: str = ""
    message_count: int = 0
    tool_results: List[Dict[str, str]] = Field(default_factory=list)
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    status: NodeStatus = NodeStatus.RUNNING


class CallLlmMetadata(BaseModel):
    """大语言模型调用元数据契约"""

    node_type: str = "call_llm"
    description: str = ""
    content: Optional[str] = None
    reasoning: Optional[str] = None
    has_tool_calls: bool = False
    message_count: int = 0
    token_usage: Dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    status: NodeStatus = NodeStatus.RUNNING


class ExecuteToolMetadata(BaseModel):
    """工具调用元数据契约"""

    node_type: str = "Execute_Tool"
    description: str = ""
    name: str = ""
    arguments: Any = None
    result: Any = None
    result_truncated: Optional[str] = Field(
        None, description="用于日志/UI展示的截断版本"
    )
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    status: NodeStatus = NodeStatus.RUNNING


class SubStepMetadata(BaseModel):
    """原子工具调用元数据契约"""

    node_type: str = "SubStep_Action"
    description: str = ""
    tool_name: str = ""
    arguments: Any = None
    output_data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    status: NodeStatus = NodeStatus.RUNNING
