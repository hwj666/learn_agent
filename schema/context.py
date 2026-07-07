import time
from dataclasses import dataclass, field
from typing import List, Optional, Set

from schema.node import NodeRecord
from schema.session import ReadonlySessionView, SessionContext


@dataclass
class ExecutionContext:
    """单次执行上下文 - 工人视角"""

    execution_id: str
    parent_node: NodeRecord
    session_view: ReadonlySessionView  # 只读视图
    session: SessionContext  # 真实 Session（仅用于上报）

    # ===== 局部限制 =====
    deadline: float
    local_token_budget: int = 25_000

    # ===== 局部状态 =====
    prompt_tokens: int = 0
    completion_tokens: int = 0
    local_fingerprints: Set[str] = field(default_factory=set)
    active_node_stack: List[NodeRecord] = field(default_factory=list)

    # ===== 属性访问器 =====
    @property
    def remaining_time(self) -> float:
        """当前 Execution 的剩余时间（基于 deadline）"""
        return max(0.0, self.deadline - time.time())

    @property
    def is_expired(self) -> bool:
        """检查是否超时"""
        return time.time() > self.deadline

    @property
    def local_tokens_used(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def remaining_local_tokens(self) -> int:
        return max(0, self.local_token_budget - self.local_tokens_used)

    # ===== 操作方法 =====

    def check_expiration(self):
        """三重熔断检查"""
        # 1. 局部时间检查
        if time.time() > self.deadline:
            raise TimeoutError(f"Local deadline exceeded: {self.deadline}")

        # 2. 局部 Token 检查
        if self.local_tokens_used >= self.local_token_budget:
            raise RuntimeError(
                f"Local token budget exceeded: "
                f"{self.local_tokens_used}/{self.local_token_budget}"
            )

        # 3. 全局检查（通过只读视图）
        if self.session_view.is_expired:
            raise TimeoutError("Global session expired")

        if self.session_view.total_tokens >= self.session_view.max_token_budget:
            raise RuntimeError("Global token budget exceeded")

    def add_token_cost(self, prompt: int, completion: int):
        """记录 Token 消耗（双向上报）"""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        # 上报给全局 Session
        self.session.add_token_cost(prompt + completion)

    def add_fingerprint(self, fp: str):
        """添加指纹（双写）"""
        self.local_fingerprints.add(fp)
        self.session.add_fingerprint(fp)

    def has_fingerprint(self, fp: str) -> bool:
        """检查指纹（双查）"""
        return (fp in self.local_fingerprints) or self.session_view.has_fingerprint(fp)

    def push_node(self, node: NodeRecord):
        """压入执行栈"""
        self.active_node_stack.append(node)

    def pop_node(self) -> Optional[NodeRecord]:
        """弹出执行栈"""
        if self.active_node_stack:
            return self.active_node_stack.pop()
        return None

    @property
    def current_node(self) -> Optional[NodeRecord]:
        """获取当前节点"""
        return self.active_node_stack[-1] if self.active_node_stack else None
