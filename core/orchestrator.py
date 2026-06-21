"""
编排器
负责执行循环的核心逻辑：决策、执行、记录
"""
import logging
from typing import Optional, Any

from core.message import LLMMessage
from core.policy import ExecutionPolicy
from core.context import ExecutionContext, StepRecord, StepStatus


class Orchestrator:
    def __init__(
        self,
        policy: ExecutionPolicy,
        max_steps: int = 10,
        logger: Optional[logging.Logger] = None,
    ):
        self.policy = policy
        self.max_steps = max_steps
        self.logger = logger or logging.getLogger("Orchestrator")

    async def run(self, query: str, ctx: ExecutionContext) -> str:
        ctx.user_query = query
        step = 0

        while step < self.max_steps:
            step += 1
            self.logger.info(f"🔁 Step {step}/{self.max_steps}")

            step_record = StepRecord(step=step, status=StepStatus.RUNNING)
            ctx.add_step(step_record)

            try:
                recent_history = ctx.get_recent_turns(self.policy.max_history_turns)
                decision = await self.policy.decide(query, recent_history)

                step_record.decision = decision
                step_record.tool_calls = getattr(decision, "tool_calls", []) or []

                if decision.tool_calls:
                    tool_results = await self.policy.execute(decision, ctx)
                    step_record.tool_results = tool_results
                    ctx.history.extend(tool_results)

                if hasattr(decision, "content") or hasattr(decision, "tool_calls"):
                    ctx.add_message(LLMMessage.assistant(
                        content=getattr(decision, "content", None),
                        tool_calls=getattr(decision, "tool_calls", None),
                    ))

                tool_results_for_check = step_record.tool_results
                if self.policy.should_stop(decision, tool_results_for_check):
                    step_record.status = StepStatus.SUCCESS
                    if hasattr(self.policy, "get_finish_result"):
                        return self.policy.get_finish_result(decision)
                    return self._format_result(decision, tool_results_for_check)

                fp = self.policy.get_fingerprint(decision)
                if fp:
                    if ctx.has_fingerprint(fp):
                        self.logger.warning("🚨 死循环检测触发")
                        ctx.add_message(LLMMessage.user(
                            "框架拦截：你正在重复执行相同操作，请立刻更换工具或调整参数。"
                        ))
                        step_record.status = StepStatus.SKIPPED
                        continue
                    ctx.add_fingerprint(fp)

                if tool_results_for_check:
                    out = "\n".join(msg.content or "" for msg in tool_results_for_check)
                    ctx.add_message(LLMMessage.user(out))

                step_record.status = StepStatus.SUCCESS

            except Exception as e:
                self.logger.exception("执行异常")
                step_record.status = StepStatus.FAILURE
                step_record.error = str(e)
                ctx.add_message(LLMMessage.user(f"⚠️ 系统异常: {e}"))

        if not ctx.tool_called:
            return "模型从未调用工具"

        return f"任务未在规定步数内完成"

    def _format_result(self, decision: Any, tool_results: list) -> str:
        if tool_results:
            return "\n".join(msg.content or "" for msg in tool_results)

        content = getattr(decision, "content", "")
        if content:
            return content

        return "任务完成"
