# schema.py
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timezone


class StepEventType(Enum):
    ENTER = "enter"
    EXIT = "exit"
    CRASHED = "crashed"
    STREAM_CHUNK = "chunk"
    DATA_UPDATE = "update"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class AgentSpan:
    """🌐 锁死的物理跨度节点：记录绝对的拓扑关系、会话与嵌套深度"""

    span_id: str
    session_id: str
    trace_id: str
    span_name: str
    parent_span_id: Optional[str] = None
    depth: int = 1


@dataclass(frozen=True)
class StepEvent:
    """✉️ 跨解耦边界发送的纯净事件数据载体"""

    event_type: StepEventType
    span: AgentSpan
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_text: str = ""
    error_msg: str = ""

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        init=False,
        repr=False,
    )
