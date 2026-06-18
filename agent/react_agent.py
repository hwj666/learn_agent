import logging
from agent.base_agent import BaseAgent
from core.message import LLMMessage

logger = logging.getLogger("ReactAgent")


class ReactAgent(BaseAgent):
    async def run(self, user_query: str) -> str:
        await self._add_message_async(LLMMessage.user(user_query))

        step = 0
        while step < self.max_steps:
            step += 1
            logger.info(
                f"==================== STEP {step}/{self.max_steps} 开始 ===================="
            )

            try:
                llm_response = await self.client.chat(
                    messages=self.memory, tools=self.executor.tools
                )
                print("\033[0m")
            except Exception as e:
                return f"错误：调用模型时发生异常: {str(e)}"

            assistant_msg = LLMMessage.assistant(
                content=llm_response.content or "",
                tool_calls=llm_response.tool_calls or None,
            )
            await self._add_message_async(assistant_msg)

            if not llm_response.tool_calls:
                return llm_response.content or "未生成有效回答"

            tool_responses = await self.executor.execute(
                tool_calls=llm_response.tool_calls, ctx=self.ctx, timeout=30.0
            )
            for response in tool_responses:
                await self._add_message_async(response)

        return "错误：超过最大迭代步数，未能生成有效解答。"
