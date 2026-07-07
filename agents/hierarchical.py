"""
层级编排器
支持多层级、多 Agent 的复杂任务编排

注意：核心编排逻辑（Orchestrator）在 core/orchestrator.py
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum

from schema.context import StepStatus

logger = logging.getLogger("HierarchicalOrchestrator")


class OrchestrationMode(Enum):
    """编排模式"""

    SEQUENTIAL = "sequential"  # 顺序执行
    PARALLEL = "parallel"  # 并行执行
    SUPERVISED = "supervised"  # 监督者模式
    HIERARCHICAL = "hierarchical"  # 层级编排


class TaskNode:
    """任务节点"""

    def __init__(
        self,
        task_id: str,
        description: str,
        agent_name: Optional[str] = None,
        sub_tasks: Optional[List["TaskNode"]] = None,
    ):
        self.task_id = task_id
        self.description = description
        self.agent_name = agent_name
        self.sub_tasks = sub_tasks or []
        self.status = StepStatus.PENDING
        self.result: Optional[str] = None
        self.error: Optional[str] = None

    def is_leaf(self) -> bool:
        """是否是叶子节点（可执行）"""
        return len(self.sub_tasks) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "sub_tasks": [t.to_dict() for t in self.sub_tasks],
        }


class HierarchicalOrchestrator:
    """
    层级编排器（用于 Multi-Agent 系统）

    支持：
    - 任务树构建和执行
    - 多种编排模式（顺序/并行）
    - 任务依赖管理
    - 执行状态追踪

    注意：单 Agent 场景使用 core.Orchestrator
    """

    def __init__(
        self,
        mode: OrchestrationMode = OrchestrationMode.SEQUENTIAL,
        max_depth: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        self.mode = mode
        self.max_depth = max_depth
        self.logger = logger or logging.getLogger("HierarchicalOrchestrator")
        self.task_tree: Optional[TaskNode] = None
        self.execution_log: List[Dict[str, Any]] = []

    def build_task_tree(
        self,
        root_description: str,
        sub_tasks: List[Dict[str, Any]],
    ) -> TaskNode:
        """
        构建任务树

        sub_tasks 格式：
        [
            {
                "id": "task-1",
                "description": "任务描述",
                "agent_name": "worker-1",
                "sub_tasks": [...]  # 嵌套子任务
            }
        ]
        """
        self.logger.info(f"[Hierarchical] 构建任务树: {root_description}")

        def build_node(task_dict: Dict[str, Any]) -> TaskNode:
            return TaskNode(
                task_id=task_dict["id"],
                description=task_dict["description"],
                agent_name=task_dict.get("agent_name"),
                sub_tasks=[build_node(st) for st in task_dict.get("sub_tasks", [])],
            )

        self.task_tree = TaskNode(
            task_id="root",
            description=root_description,
            sub_tasks=[build_node(t) for t in sub_tasks],
        )

        return self.task_tree

    async def execute_leaf_task(
        self,
        node: TaskNode,
        agent_registry: Dict[str, Any],
    ) -> str:
        """执行叶子任务"""
        if not node.is_leaf():
            raise ValueError(f"任务 {node.task_id} 不是叶子节点")

        if not node.agent_name:
            return f"任务完成: {node.description}"

        agent = agent_registry.get(node.agent_name)
        if not agent:
            raise ValueError(f"Agent '{node.agent_name}' 不存在")

        self.logger.info(f"[Hierarchical] 执行任务 {node.task_id} by {node.agent_name}")

        if hasattr(agent, "run"):
            result = await agent.run(node.description)
            return result

        return f"Agent {node.agent_name} 不支持 run 方法"

    async def execute_node(
        self,
        node: TaskNode,
        agent_registry: Dict[str, Any],
        depth: int = 0,
    ) -> str:
        """递归执行任务节点"""
        if depth > self.max_depth:
            raise RecursionError(f"超出最大深度 {self.max_depth}")

        indent = "  " * depth
        self.logger.info(f"{indent}[执行] {node.task_id}: {node.description}")

        node.status = StepStatus.RUNNING
        self._log_execution(node.task_id, "start", depth)

        try:
            if node.is_leaf():
                result = await self.execute_leaf_task(node, agent_registry)
                node.result = result
                node.status = StepStatus.SUCCESS
                self._log_execution(node.task_id, "success", depth, result)
                return result

            else:
                if self.mode == OrchestrationMode.SEQUENTIAL:
                    results = []
                    for sub_task in node.sub_tasks:
                        result = await self.execute_node(
                            sub_task, agent_registry, depth + 1
                        )
                        results.append(result)
                    node.result = "\n".join(results)

                elif self.mode == OrchestrationMode.PARALLEL:
                    import asyncio

                    coros = [
                        self.execute_node(sub_task, agent_registry, depth + 1)
                        for sub_task in node.sub_tasks
                    ]
                    results = await asyncio.gather(*coros, return_exceptions=True)
                    node.result = "\n".join(
                        str(r) if not isinstance(r, Exception) else f"Error: {r}"
                        for r in results
                    )

                node.status = StepStatus.SUCCESS
                self._log_execution(node.task_id, "success", depth, node.result)
                return node.result

        except Exception as e:
            node.status = StepStatus.FAILURE
            node.error = str(e)
            self.logger.error(f"{indent}[错误] {node.task_id}: {e}")
            self._log_execution(node.task_id, "failure", depth, str(e))
            raise

    def _log_execution(
        self,
        task_id: str,
        status: str,
        depth: int,
        result: Optional[str] = None,
    ) -> None:
        """记录执行日志"""
        self.execution_log.append(
            {
                "task_id": task_id,
                "status": status,
                "depth": depth,
                "result": result,
            }
        )

    async def run(
        self,
        query: str,
        agent_registry: Dict[str, Any],
        task_structure: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        运行编排器

        参数：
        - query: 用户查询
        - agent_registry: Agent 注册表 {name: agent}
        - task_structure: 任务结构（可选）
        """
        self.logger.info(f"[Hierarchical] 开始运行，模式: {self.mode.value}")
        self.execution_log.clear()

        if task_structure:
            self.build_task_tree(query, task_structure)
        else:
            self.task_tree = TaskNode(
                task_id="root",
                description=query,
                agent_name=None,
            )

        if self.task_tree:
            try:
                result = await self.execute_node(self.task_tree, agent_registry)
                return result
            except Exception as e:
                return f"执行失败: {e}"

        return "没有可执行的任务"

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "mode": self.mode.value,
            "execution_count": len(self.execution_log),
            "log": self.execution_log,
            "task_tree": self.task_tree.to_dict() if self.task_tree else None,
        }

    @staticmethod
    def create_parallel(max_depth: int = 5) -> "HierarchicalOrchestrator":
        """创建并行编排器"""
        return HierarchicalOrchestrator(
            mode=OrchestrationMode.PARALLEL,
            max_depth=max_depth,
        )

    @staticmethod
    def create_sequential(max_depth: int = 5) -> "HierarchicalOrchestrator":
        """创建顺序编排器"""
        return HierarchicalOrchestrator(
            mode=OrchestrationMode.SEQUENTIAL,
            max_depth=max_depth,
        )
