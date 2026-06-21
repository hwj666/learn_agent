"""
计划模块
用于生成、管理和执行任务计划
"""
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from core.message import LLMMessage


class PlanStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanTask:
    id: str
    description: str
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Plan:
    """任务计划"""
    id: str
    query: str
    tasks: List[PlanTask] = field(default_factory=list)
    status: PlanStatus = PlanStatus.CREATED
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def completed_tasks(self) -> List[PlanTask]:
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    @property
    def pending_tasks(self) -> List[PlanTask]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_next_tasks(self) -> List[PlanTask]:
        """获取可以立即执行的任务（依赖已完成）"""
        completed_ids = {t.id for t in self.completed_tasks}
        return [
            t for t in self.pending_tasks
            if all(dep in completed_ids for dep in t.dependencies)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status.value,
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "tool_name": t.tool_name,
                    "arguments": t.arguments,
                    "status": t.status.value,
                    "dependencies": t.dependencies,
                    "result": t.result,
                    "error": t.error,
                }
                for t in self.tasks
            ],
            "metadata": self.metadata,
        }


class PlanGenerator(ABC):
    """计划生成器接口"""

    def __init__(self, client=None):
        self.client = client
        self.logger = logging.getLogger("PlanGenerator")

    @abstractmethod
    async def generate(self, query: str, history: List[LLMMessage]) -> Plan:
        """生成计划"""
        pass


class SimplePlanGenerator(PlanGenerator):
    """简单计划生成器"""

    PLAN_PROMPT = """你是一个任务规划专家，请将以下查询分解为一系列步骤：

查询：{query}

请以 JSON 格式输出计划：
{{
    "tasks": [
        {{
            "id": "task-1",
            "description": "第一步做什么",
            "tool_name": "可选：调用的工具名",
            "arguments": "可选：工具参数",
            "dependencies": ["task-0"]
        }}
    ]
}}

注意：
- 每个任务必须有唯一的 id
- dependencies 是前置任务的 id 列表
- 最后一步必须调用 task_completed 工具
"""

    async def generate(self, query: str, history: List[LLMMessage]) -> Plan:
        messages = [
            LLMMessage.system("你是一个任务规划专家"),
            LLMMessage.user(self.PLAN_PROMPT.format(query=query)),
        ]

        resp = await self.client.chat(messages=messages)

        try:
            data = json.loads(resp.content or "{}")
            tasks = []

            for i, task_data in enumerate(data.get("tasks", [])):
                tasks.append(PlanTask(
                    id=task_data.get("id", f"task-{i+1}"),
                    description=task_data.get("description", ""),
                    tool_name=task_data.get("tool_name"),
                    arguments=task_data.get("arguments"),
                    dependencies=task_data.get("dependencies", []),
                ))

            return Plan(
                id=f"plan-{id(self)}",
                query=query,
                tasks=tasks,
                status=PlanStatus.CREATED,
            )

        except Exception as e:
            self.logger.error(f"计划生成失败: {e}")
            return Plan(
                id=f"plan-{id(self)}",
                query=query,
                tasks=[],
                status=PlanStatus.FAILED,
            )


class PlanExecutor:
    """计划执行器"""

    def __init__(self, executor, client=None):
        self.executor = executor
        self.client = client
        self.logger = logging.getLogger("PlanExecutor")

    async def execute(self, plan: Plan, ctx: Dict[str, Any]) -> Plan:
        """执行计划"""
        plan.status = PlanStatus.RUNNING

        while plan.pending_tasks:
            next_tasks = plan.get_next_tasks()

            if not next_tasks:
                self.logger.warning("无法继续执行：存在未解决的依赖")
                plan.status = PlanStatus.FAILED
                break

            for task in next_tasks:
                task.status = TaskStatus.RUNNING
                self.logger.info(f"执行任务: {task.description}")

                try:
                    if task.tool_name:
                        tool_results = await self.executor.execute(
                            tool_calls=[{
                                "name": task.tool_name,
                                "arguments": json.dumps(task.arguments or {}),
                            }],
                            ctx=ctx,
                        )
                        task.result = "\n".join(
                            msg.content or "" for msg in tool_results
                        )
                    else:
                        task.result = f"任务完成: {task.description}"

                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    self.logger.error(f"任务失败: {task.description} - {e}")

        if not plan.pending_tasks:
            plan.status = PlanStatus.COMPLETED

        return plan
