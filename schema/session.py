import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

from schema.enums import SessionStatus
from schema.message import LLMMessage
from schema.node import NodeRecord

@dataclass(frozen=True)
class ReadonlySessionView:
    """
    权限隔离视图：Worker 只能看到这些，不能碰 SessionContext 本身
    frozen=True 保证运行时不可变
    """
    session_id: str
    max_token_budget: int
    total_tokens: int
    fingerprints: frozenset
    consensus_pool: Dict[str, Any]
    is_expired: bool
    remaining_time: float

    def has_fingerprint(self, fp: str) -> bool:
        return fp in self.fingerprints

@dataclass
class SessionContext:
    """全局会话上下文 - 上帝视角"""

    session_id: str
    user_query: str

    # ===== 生命周期管理 =====
    status: SessionStatus = SessionStatus.INITIALIZING
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    timeout_limit_seconds: float = 60.0
    max_token_budget: int = 500_000

    # ===== 业务数据 =====
    consensus_pool: Dict[str, Any] = field(default_factory=dict)
    global_history: List[LLMMessage] = field(default_factory=list)

    # ===== 追踪树 =====
    root_nodes: List[NodeRecord] = field(default_factory=list)

    # ===== 全局防重 =====
    fingerprints: Set[str] = field(default_factory=set)

    # ===== 指标 =====
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    logger: logging.Logger = field(default=None, repr=False)

    def __post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger(f"Session[{self.session_id}]")

    # ==================== 属性访问器 ====================

    @property
    def is_expired(self) -> bool:
        """检查会话是否超时"""
        return (time.time() - self.start_time) > self.timeout_limit_seconds

    @property
    def remaining_time(self) -> float:
        """剩余可用时间"""
        elapsed = time.time() - self.start_time
        return max(0.0, self.timeout_limit_seconds - elapsed)

    @property
    def token_usage_percent(self) -> float:
        """Token 使用百分比"""
        return (
            (self.total_tokens / self.max_token_budget) * 100
            if self.max_token_budget > 0
            else 0
        )

    # ==================== 状态管理 ====================

    def finalize(self, status: SessionStatus):
        """结束会话"""
        self.status = status
        self.end_time = time.time()
        self.logger.info(f"Session finalized with status: {status.value}")

    # ==================== 数据操作 ====================

    def add_global_message(self, msg: LLMMessage):
        """添加全局消息"""
        self.global_history.append(msg)

    def add_root_node(self, node: NodeRecord):
        """添加根节点"""
        self.root_nodes.append(node)

    def add_fingerprint(self, fp: str):
        """添加全局指纹"""
        self.fingerprints.add(fp)

    def has_fingerprint(self, fp: str) -> bool:
        """检查全局指纹"""
        return fp in self.fingerprints

    def add_token_cost(self, tokens: int):
        """记录 Token 消耗"""
        self.total_tokens += tokens
        if self.total_tokens >= self.max_token_budget:
            self.logger.warning(
                f"Token budget exceeded: {self.total_tokens}/{self.max_token_budget}"
            )

    # ==================== 序列化 ====================

    def to_trace(self, verbose: bool = True) -> Dict[str, Any]:
        """生成完整追踪数据"""

        def _serialize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
            if not tool_calls:
                return []
            serialized = []
            for tc in tool_calls:
                if hasattr(tc, "name"):
                    name = tc.name
                    args = tc.arguments if hasattr(tc, "arguments") else {}
                elif isinstance(tc, dict):
                    name = tc.get("name", "unknown")
                    args = tc.get("arguments", {})
                else:
                    name = "unknown"
                    args = {}
                serialized.append({"name": name, "arguments": args})
            return serialized

        def _serialize_node(node: NodeRecord) -> Dict[str, Any]:
            return {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "status": node.status.value,
                "description": node.description,
                "duration_ms": node.duration_ms,
                "input": node.input_data if verbose else {"truncated": True},
                "output": node.output_data if verbose else {"truncated": True},
                "tool_calls": _serialize_tool_calls(node.tool_calls),
                "tool_results": node.tool_results if verbose else "[truncated]",
                "error": node.error,
                "metadata": node.metadata,
                "children": [_serialize_node(child) for child in node.children],
            }

        return {
            "session_id": self.session_id,
            "user_query": self.user_query,
            "session_metrics": {
                "status": self.status.value,
                "total_elapsed_seconds": round(
                    (self.end_time or time.time()) - self.start_time, 2
                ),
                "total_tokens_consumed": self.total_tokens,
                "token_usage_percent": round(self.token_usage_percent, 2),
                "token_budget_limit": self.max_token_budget,
                "time_limit_seconds": self.timeout_limit_seconds,
                "actual_end_time": self.end_time,
            },
            "consensus_pool": self.consensus_pool,
            "global_history": [msg.to_dict() for msg in self.global_history],
            "topology_tree": [_serialize_node(root) for root in self.root_nodes],
            "global_fingerprints": list(self.fingerprints),
            "metadata": self.metadata,
        }

    def save_trace(self, path: str, verbose: bool = True):
        """保存追踪数据到文件"""
        try:
            trace_data = self.to_trace(verbose=verbose)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Trace saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save trace: {e}", exc_info=True)

    # ==================== 只读视图 ====================

    def get_readonly_view(self):
        """获取只读视图（用于传递给 ExecutionContext）"""
        return ReadonlySessionView(
            session_id=self.session_id,
            max_token_budget=self.max_token_budget,
            total_tokens=self.total_tokens,
            fingerprints=frozenset(self.fingerprints),
            consensus_pool=self.consensus_pool.copy(),
            is_expired=self.is_expired,
            remaining_time=self.remaining_time,
        )
