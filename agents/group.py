"""
多 Agent 管理器
管理多个 Agent 的生命周期、消息传递和任务分配
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger("MultiAgentGroup")


class AgentRole(Enum):
    """Agent 角色"""

    SUPERVISOR = "supervisor"  # 监督者，负责分配任务
    PLANNER = "planner"  # 规划者，负责制定计划
    WORKER = "worker"  # 执行者，负责具体任务
    CRITIC = "critic"  # 批评者，负责评估和反思


@dataclass
class AgentMember:
    """Agent 成员"""

    name: str
    agent: Any  # Agent 实例
    role: AgentRole = AgentRole.WORKER
    description: str = ""
    capabilities: List[str] = field(default_factory=list)  # 擅长的能力
    is_busy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "description": self.description,
            "capabilities": self.capabilities,
            "is_busy": self.is_busy,
        }


class Message:
    """Agent 间消息"""

    def __init__(
        self,
        sender: str,
        receiver: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = None  # 可扩展：添加时间戳


class MultiAgentGroup:
    """
    多 Agent 组管理器

    功能：
    - 添加/移除 Agent 成员
    - 消息传递
    - 任务分配
    - 状态管理
    """

    def __init__(
        self,
        group_id: str,
        name: str = "",
        shared_context: Optional[Dict[str, Any]] = None,
    ):
        self.group_id = group_id
        self.name = name or f"Group-{group_id}"
        self.members: Dict[str, AgentMember] = {}
        self.messages: List[Message] = []
        self.shared_context = shared_context or {}
        self.logger = logger

    def add_member(self, member: AgentMember) -> None:
        """添加 Agent 成员"""
        if member.name in self.members:
            raise ValueError(f"Agent '{member.name}' 已存在")
        self.members[member.name] = member
        self.logger.info(f"[{self.name}] 添加成员: {member.name} ({member.role.value})")

    def remove_member(self, name: str) -> None:
        """移除 Agent 成员"""
        if name not in self.members:
            raise ValueError(f"Agent '{name}' 不存在")
        del self.members[name]
        self.logger.info(f"[{self.name}] 移除成员: {name}")

    def get_member(self, name: str) -> Optional[AgentMember]:
        """获取 Agent 成员"""
        return self.members.get(name)

    def get_members_by_role(self, role: AgentRole) -> List[AgentMember]:
        """获取特定角色的所有成员"""
        return [m for m in self.members.values() if m.role == role]

    def get_available_worker(self) -> Optional[AgentMember]:
        """获取空闲的 Worker"""
        workers = self.get_members_by_role(AgentRole.WORKER)
        available = [w for w in workers if not w.is_busy]
        return available[0] if available else None

    def select_best_worker(
        self, required_capabilities: List[str]
    ) -> Optional[AgentMember]:
        """选择最适合的 Worker（基于能力匹配）"""
        workers = self.get_members_by_role(AgentRole.WORKER)
        available = [w for w in workers if not w.is_busy]

        if not available:
            return None

        # 简单匹配：优先选择能力完全匹配的
        for worker in available:
            if all(cap in worker.capabilities for cap in required_capabilities):
                return worker

        # 如果没有完全匹配，返回第一个可用的
        return available[0]

    def set_busy(self, name: str, busy: bool = True) -> None:
        """设置 Agent 忙碌状态"""
        if name in self.members:
            self.members[name].is_busy = busy

    def send_message(
        self,
        sender: str,
        receiver: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """发送消息"""
        msg = Message(
            sender=sender, receiver=receiver, content=content, metadata=metadata
        )
        self.messages.append(msg)
        self.logger.debug(f"[{self.name}] 消息: {sender} -> {receiver}")
        return msg

    def broadcast(
        self, sender: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Message]:
        """广播消息给所有成员"""
        msgs = []
        for name in self.members:
            if name != sender:  # 不发给自己
                msg = self.send_message(sender, name, content, metadata)
                msgs.append(msg)
        return msgs

    def get_messages_for(self, receiver: str) -> List[Message]:
        """获取发给特定 Agent 的消息"""
        return [m for m in self.messages if m.receiver == receiver]

    def clear_messages(self) -> None:
        """清空消息历史"""
        self.messages.clear()

    def update_shared_context(self, key: str, value: Any) -> None:
        """更新共享上下文"""
        self.shared_context[key] = value
        self.logger.debug(f"[{self.name}] 更新共享上下文: {key}")

    def get_shared_context(self, key: str, default: Any = None) -> Any:
        """获取共享上下文"""
        return self.shared_context.get(key, default)

    def get_status(self) -> Dict[str, Any]:
        """获取组状态"""
        return {
            "group_id": self.group_id,
            "name": self.name,
            "member_count": len(self.members),
            "members": [m.to_dict() for m in self.members.values()],
            "message_count": len(self.messages),
            "shared_context_keys": list(self.shared_context.keys()),
        }
