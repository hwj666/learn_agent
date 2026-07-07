"""
agents 模块
多 Agent 系统：Agent 组、监督者、层级编排、计划执行等

架构说明：
- group.py: MultiAgentGroup 管理多个 Agent
- supervisor.py: SupervisorAgent 协调子 Agent
- hierarchical.py: HierarchicalOrchestrator 任务树编排
- plan_policy.py: PlanPolicy 计划执行策略
"""

from agents.group import (
    MultiAgentGroup,
    AgentMember,
    AgentRole,
    Message,
)
from agents.supervisor import SupervisorAgent
from agents.hierarchical import (
    HierarchicalOrchestrator,
    OrchestrationMode,
    TaskNode,
)
from agents.plan_policy import PlanPolicy

__all__ = [
    # Group
    "MultiAgentGroup",
    "AgentMember",
    "AgentRole",
    "Message",
    # Supervisor
    "SupervisorAgent",
    # Hierarchical
    "HierarchicalOrchestrator",
    "OrchestrationMode",
    "TaskNode",
    # Policy
    "PlanPolicy",
]
