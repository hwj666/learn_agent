from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .enums import NodeStatus


@dataclass
class NodeRecord:
    node_id: str
    node_type: str
    status: NodeStatus = NodeStatus.PENDING
    description: str = ""

    # 输入输出
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)

    # 工具调用记录
    tool_calls: List[Any] = field(default_factory=list)
    tool_results: List[Any] = field(default_factory=list)

    # 层级关系
    children: List["NodeRecord"] = field(default_factory=list)

    # 错误信息
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 时间追踪
    start_time: float = field(default_factory=lambda: __import__("time").time())
    end_time: Optional[float] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def mark_success(self):
        self.status = NodeStatus.SUCCESS
        self.end_time = __import__("time").time()

    def mark_failure(self, error: str):
        self.status = NodeStatus.FAILURE
        self.error = error
        self.end_time = __import__("time").time()
