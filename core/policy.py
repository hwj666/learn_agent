"""
策略接口与实现
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from core.message import LLMMessage

if TYPE_CHECKING:
    from tools.execute import ToolExecutor


class ExecutionPolicy(ABC):
    def __init__(
        self,
        executor: "ToolExecutor",
        ctx: Dict[str, Any],
        max_history_turns: int = 5,
    ):
        self.executor = executor
        self.ctx = ctx
        self.max_history_turns = max_history_turns

    @abstractmethod
    async def decide(self, query: str, history: List[LLMMessage]) -> Any:
        pass

    @abstractmethod
    async def execute(self, decision: Any, ctx: Any) -> List[LLMMessage]:
        pass

    @abstractmethod
    def should_stop(self, decision: Any, execution_result: List[LLMMessage]) -> bool:
        pass

    def get_system_prompt(self) -> str:
        return ""

    def build_messages(self, query: str, history: List[LLMMessage]) -> List[LLMMessage]:
        return [
            LLMMessage.system(self.get_system_prompt()),
            LLMMessage.user(query),
            *history,
        ]

    def get_fingerprint(self, decision: Any) -> Optional[str]:
        if hasattr(decision, "tool_calls") and decision.tool_calls:
            sorted_calls = sorted(
                decision.tool_calls, key=lambda x: (x.name, str(x.arguments))
            )
            return "||".join(f"{c.name}:{c.arguments}" for c in sorted_calls)
        return None


class ReactPolicy(ExecutionPolicy):
    SYSTEM_PROMPT = """你是一个严格执行 ReAct 的智能体。
每一步只能做一件事：
1. 调用业务工具解决问题
2. 或调用 task_completed 工具结束任务

规则：
- 在解决问题前，必须先调用业务工具，禁止直接调用 task_completed
- 工具失败必须修正参数重试
- 禁止解释、禁止废话
- 只有确认问题已完全解决时，才能调用 task_completed
"""

    FINISH_TOOL_NAME = "task_completed"

    def __init__(
        self,
        executor: "ToolExecutor",
        ctx: Dict[str, Any],
        max_history_turns: int = 5,
        client=None,
    ):
        super().__init__(executor, ctx, max_history_turns)
        self.logger = logging.getLogger("ReactPolicy")
        self.client = client

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    async def decide(self, query: str, history: List[LLMMessage]) -> Any:
        messages = self.build_messages(query, history)
        return await self.client.chat(messages=messages, tools=self.executor.tools)

    async def execute(self, decision: Any, ctx: Any) -> List[LLMMessage]:
        if not decision.tool_calls:
            return []

        tool_messages = await self.executor.execute(
            tool_calls=decision.tool_calls, ctx=self.ctx
        )

        out = "\n".join(msg.content or "" for msg in tool_messages)
        tools_str = ", ".join(tc.name for tc in decision.tool_calls)
        short_out = out if len(out) < 600 else f"{out[:300]}...\n[截断]\n...{out[-300:]}"
        self.logger.info(f"[{tools_str}]: {short_out}")

        return tool_messages

    def should_stop(self, decision: Any, execution_result: List[LLMMessage]) -> bool:
        if not decision.tool_calls:
            return False
        return self.FINISH_TOOL_NAME in [tc.name for tc in decision.tool_calls]

    def get_finish_result(self, decision: Any) -> str:
        if not decision.tool_calls:
            return "任务完成"

        for tc in decision.tool_calls:
            if tc.name == self.FINISH_TOOL_NAME:
                try:
                    args = json.loads(tc.arguments)
                    summary = args.get("summary", "")
                    result = args.get("result", "")
                    return f"✅ 任务已完成\n\n总结：{summary}\n\n结果：{result or '无额外输出'}"
                except Exception:
                    return f"✅ 任务已完成\n\n{tc.arguments}"

        return "任务完成"
