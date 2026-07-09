# schema/metadata.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PlannerMetadata(BaseModel):
    """规划阶段元数据契约"""

    node_type: str = "Planner"
    description: str = ""
    planned_tasks: List[str] = Field(default_factory=list)
    raw_user_query: str = ""
    error: Optional[str] = None


class ReActTurnMetadata(BaseModel):
    """ReAct 核心迭代轮次元数据契约"""

    node_type: str = "ReAct_Micro_Turn"
    description: str = ""
    message_count: int = 0
    content: Optional[str] = None
    reasoning: Optional[str] = None
    has_tool_calls: bool = False
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class SubStepMetadata(BaseModel):
    """原子工具调用元数据契约"""

    node_type: str = "SubStep_Action"
    description: str = ""
    tool_name: str = ""
    arguments: Any = None
    output_data: Dict[str, Any] = Field(default_factory=list)
    error: Optional[str] = None
