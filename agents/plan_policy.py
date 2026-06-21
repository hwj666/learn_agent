"""
计划执行策略
实现 Plan-and-Execute 模式：先生成计划，再逐步执行

注意：核心类型（Plan, PlanTask, PlanGenerator 等）定义在 core/plan.py
"""
import json
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from core.message import LLMMessage
from core.policy import ExecutionPolicy
from core.plan import Plan, PlanTask, PlanStatus, TaskStatus, SimplePlanGenerator

if TYPE_CHECKING:
    from tools.execute import ToolExecutor


class PlanPolicy(ExecutionPolicy):
    """
    计划执行策略 (Plan-and-Execute)

    两个阶段：
    1. Planning: 使用 LLM 生成任务计划（使用 core.plan 中的类型）
    2. Execution: 按计划逐步执行任务

    优点：
    - 计划可以在执行前被审查/修改
    - 更好的长期规划能力
    - 更容易调试和干预
    """

    EXECUTOR_SYSTEM_PROMPT = """你是一个任务执行专家。

你已经有一个计划，请按计划逐步执行。

当前计划：
{plan}

执行规则：
1. 每次只执行一个步骤
2. 执行完成后检查结果是否符合预期
3. 如果步骤失败，尝试调整参数重试
4. 所有步骤完成后，调用 task_completed
"""

    FINISH_TOOL_NAME = "task_completed"

    def __init__(
        self,
        executor: "ToolExecutor",
        ctx: Dict[str, Any],
        max_history_turns: int = 5,
        client=None,
        max_plan_steps: int = 10,
    ):
        super().__init__(executor, ctx, max_history_turns)
        self.logger = logging.getLogger("PlanPolicy")
        self.client = client
        self.max_plan_steps = max_plan_steps
        self.current_plan: Optional[Plan] = None
        self.current_step_index: int = 0

    def get_system_prompt(self) -> str:
        if not self.current_plan:
            return self.EXECUTOR_SYSTEM_PROMPT.format(plan="无计划")
        return self.EXECUTOR_SYSTEM_PROMPT.format(
            plan=json.dumps(self.current_plan.to_dict(), ensure_ascii=False, indent=2)
        )

    def _format_plan_for_context(self, plan: Plan) -> str:
        """格式化计划为上下文字符串"""
        lines = [f"目标: {plan.query}", ""]
        lines.append("执行计划：")
        for i, task in enumerate(plan.tasks, 1):
            tool_info = f" → {task.tool_name}({task.arguments})" if task.tool_name else ""
            lines.append(f"  {i}. {task.description}{tool_info}")
        return "\n".join(lines)

    async def decide(self, query: str, history: List[LLMMessage]) -> Any:
        """
        决策逻辑：
        1. 如果没有计划，先生成计划
        2. 如果有计划，基于当前步骤构建执行消息
        """
        if self.current_plan is None:
            # 阶段 1: 生成计划
            generator = SimplePlanGenerator(client=self.client)
            self.current_plan = await generator.generate(query, history)
            self.current_step_index = 0
            self.logger.info(f"[PlanPolicy] 生成计划，包含 {len(self.current_plan.tasks)} 个步骤")

        # 阶段 2: 执行当前步骤
        if self.current_step_index >= len(self.current_plan.tasks):
            self.logger.info("[PlanPolicy] 所有步骤已完成")
            return None

        current_task = self.current_plan.tasks[self.current_step_index]
        plan_context = self._format_plan_for_context(self.current_plan)

        step_instruction = f"""当前执行计划：
{plan_context}

当前步骤 ({self.current_step_index + 1}/{len(self.current_plan.tasks)})：
{current_task.description}

请执行当前步骤。"""

        messages = [
            LLMMessage.system(self.get_system_prompt()),
            LLMMessage.user(step_instruction),
            *history,
        ]

        return await self.client.chat(messages=messages, tools=self.executor.tools)

    async def execute(self, decision: Any, ctx: Any) -> List[LLMMessage]:
        """执行决策"""
        if not decision:
            return []

        if not decision.tool_calls:
            return []

        tool_messages = await self.executor.execute(
            tool_calls=decision.tool_calls, ctx=self.ctx
        )

        if decision.tool_calls:
            called_tool = decision.tool_calls[0].name
            self.logger.info(f"[PlanPolicy] 已执行步骤 {self.current_step_index + 1}: {called_tool}")

            if called_tool == self.FINISH_TOOL_NAME:
                self.current_step_index = len(self.current_plan.tasks)
            else:
                self.current_step_index += 1

        out = "\n".join(msg.content or "" for msg in tool_messages)
        self.logger.debug(f"[PlanPolicy] 步骤结果: {out[:200]}...")

        return tool_messages

    def should_stop(self, decision: Any, execution_result: List[LLMMessage]) -> bool:
        """判断是否应该停止"""
        if not decision:
            return self.current_plan is not None and \
                   self.current_step_index >= len(self.current_plan.tasks)

        if not decision.tool_calls:
            return False

        return self.FINISH_TOOL_NAME in [tc.name for tc in decision.tool_calls]

    def get_finish_result(self, decision: Any) -> str:
        """获取完成结果"""
        if not decision or not decision.tool_calls:
            return "任务完成"

        for tc in decision.tool_calls:
            if tc.name == self.FINISH_TOOL_NAME:
                try:
                    args = json.loads(tc.arguments)
                    summary = args.get("summary", "")
                    result = args.get("result", "")
                    return f"[PLAN COMPLETED]\n\n总结：{summary}\n\n结果：{result or '无额外输出'}"
                except Exception:
                    return f"[PLAN COMPLETED]\n\n{tc.arguments}"

        return "任务完成"

    def reset(self) -> None:
        """重置计划状态"""
        self.current_plan = None
        self.current_step_index = 0

    def get_current_progress(self) -> Dict[str, Any]:
        """获取当前进度"""
        if not self.current_plan:
            return {"has_plan": False}

        tasks = self.current_plan.tasks
        return {
            "has_plan": True,
            "plan_id": self.current_plan.id,
            "total_steps": len(tasks),
            "current_step": self.current_step_index + 1,
            "current_step_description": tasks[self.current_step_index].description
                if tasks and self.current_step_index < len(tasks) else None,
        }
