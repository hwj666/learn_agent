"""
监督者 Agent
负责协调多个子 Agent，分配任务，汇总结果
"""

import logging
import json
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from schema.message import LLMMessage

if TYPE_CHECKING:
    from agents.group import MultiAgentGroup, AgentMember


class SupervisorAgent:
    """
    监督者 Agent

    职责：
    - 理解用户任务
    - 将任务分解并分配给合适的子 Agent
    - 收集子 Agent 的结果
    - 判断任务是否完成
    """

    SYSTEM_PROMPT = """你是一个智能任务监督者。

职责：
1. 分析用户请求，将其分解为子任务
2. 选择最合适的 Agent 执行每个子任务
3. 收集结果并判断是否完成

输出格式（JSON）：
{
    "task_type": "single|parallel|sequential",
    "subtasks": [
        {
            "description": "子任务描述",
            "agent_name": "选择的 Agent 名称",
            "input": "传给 Agent 的输入",
            "wait_result": true/false  // 是否等待结果
        }
    ],
    "final_answer": "如果任务完成，给出最终答案"
}

规则：
- 只输出 JSON，不要其他内容
- 选择 Agent 时考虑其能力和当前状态
- 复杂任务优先分解为并行子任务
"""

    def __init__(
        self,
        group: "MultiAgentGroup",
        client,
        max_iterations: int = 10,
        logger: Optional[logging.Logger] = None,
    ):
        self.group = group
        self.client = client
        self.max_iterations = max_iterations
        self.logger = logger or logging.getLogger("SupervisorAgent")
        self.iteration = 0

    def _build_members_info(self) -> str:
        """构建成员信息供 LLM 参考"""
        members = []
        for name, member in self.group.members.items():
            status = "忙碌" if member.is_busy else "空闲"
            caps = ", ".join(member.capabilities) if member.capabilities else "通用"
            members.append(f"- {name}: {member.role.value}, {status}, 能力: {caps}")
        return "\n".join(members)

    async def decide(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        决定如何处理任务

        返回：
        {
            "task_type": "single|parallel|sequential",
            "subtasks": [...],
            "final_answer": None or str
        }
        """
        self.iteration += 1
        self.logger.info(f"[迭代 {self.iteration}] 分析任务: {query[:50]}...")

        members_info = self._build_members_info()

        prompt = f"""用户任务：{query}

可用 Agent：
{members_info}

{self.SYSTEM_PROMPT}
"""

        messages = [
            LLMMessage.system("你是一个智能任务监督者"),
            LLMMessage.user(prompt),
        ]

        try:
            response = await self.client.chat(messages=messages)
            result = json.loads(response.content or "{}")
            self.logger.info(
                f"决策结果: {json.dumps(result, ensure_ascii=False)[:200]}..."
            )
            return result
        except Exception as e:
            self.logger.error(f"决策失败: {e}")
            return {
                "task_type": "single",
                "subtasks": [],
                "error": str(e),
            }

    async def execute_subtask(
        self,
        agent_name: str,
        task_input: str,
        wait_result: bool = True,
    ) -> Dict[str, Any]:
        """执行子任务"""
        member = self.group.get_member(agent_name)
        if not member:
            return {"success": False, "error": f"Agent '{agent_name}' 不存在"}

        self.group.set_busy(agent_name, True)
        try:
            self.logger.info(f"分配任务给 {agent_name}: {task_input[:50]}...")

            if hasattr(member.agent, "run"):
                result = await member.agent.run(task_input)
                return {"success": True, "result": result, "agent": agent_name}
            else:
                return {"success": False, "error": "Agent 不支持 run 方法"}

        except Exception as e:
            self.logger.error(f"Agent {agent_name} 执行失败: {e}")
            return {"success": False, "error": str(e), "agent": agent_name}
        finally:
            self.group.set_busy(agent_name, False)

    async def execute_plan(
        self,
        plan: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """
        执行计划

        支持三种任务类型：
        - single: 单 Agent 执行
        - parallel: 并行执行多个子任务
        - sequential: 顺序执行多个子任务
        """
        task_type = plan.get("task_type", "single")
        subtasks = plan.get("subtasks", [])
        final_answer = plan.get("final_answer")

        # 如果已经有最终答案，直接返回
        if final_answer:
            return final_answer

        results = []

        if task_type == "single" and subtasks:
            # 单任务
            task = subtasks[0]
            result = await self.execute_subtask(
                agent_name=task["agent_name"],
                task_input=task["input"],
                wait_result=task.get("wait_result", True),
            )
            results.append(result)

        elif task_type == "parallel" and subtasks:
            # 并行执行
            import asyncio

            coros = [
                self.execute_subtask(
                    agent_name=task["agent_name"],
                    task_input=task["input"],
                    wait_result=task.get("wait_result", True),
                )
                for task in subtasks
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

        elif task_type == "sequential" and subtasks:
            # 顺序执行
            for task in subtasks:
                result = await self.execute_subtask(
                    agent_name=task["agent_name"],
                    task_input=task["input"],
                    wait_result=task.get("wait_result", True),
                )
                results.append(result)
                # 将结果添加到上下文供后续任务使用
                if result.get("success"):
                    context[f"result_{task['agent_name']}"] = result.get("result")

        # 汇总结果
        return self._summarize_results(results, context)

    def _summarize_results(
        self,
        results: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> str:
        """汇总子任务结果"""
        success_count = sum(
            1 for r in results if isinstance(r, dict) and r.get("success")
        )
        total_count = len(results)

        summaries = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                summaries.append(f"任务 {i + 1}: 失败 ({result})")
            elif isinstance(result, dict):
                if result.get("success"):
                    summaries.append(f"任务 {i + 1}: 成功")
                else:
                    summaries.append(f"任务 {i + 1}: 失败 ({result.get('error')})")

        summary = f"完成 {success_count}/{total_count} 个子任务\n"
        summary += "\n".join(summaries)
        return summary

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        运行监督者

        迭代循环：
        1. 分析任务，决定如何分配
        2. 执行子任务
        3. 检查是否完成，否则继续
        """
        context = context or {}
        self.iteration = 0

        self.logger.info(f"[Supervisor] 开始处理: {query[:50]}...")

        while self.iteration < self.max_iterations:
            # 1. 决策
            plan = await self.decide(query, context)

            # 2. 执行
            result = await self.execute_plan(plan, context)

            # 3. 检查是否完成
            if plan.get("final_answer"):
                self.logger.info("[Supervisor] 任务完成")
                return plan["final_answer"]

            # 更新上下文，准备下一轮
            context["last_result"] = result
            context["iteration"] = self.iteration

            # 如果所有子任务都成功，可能可以结束
            if plan.get("task_type") in ["single", "parallel", "sequential"]:
                subtasks = plan.get("subtasks", [])
                if not subtasks:
                    break

        return f"任务未在 {self.max_iterations} 次迭代内完成"
