import os
from typing import Callable, Awaitable
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionToolUnionParam
from core.message import LLMMessage, LLMResponse, ToolCall
from core.base_client import BaseLLMClient
from core.config import ModelConfig

_ = load_dotenv()

class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig):
        # 关键修复：BaseLLMClient 接收 config 而非 provider
        super().__init__(config.provider)
        
        self.client = AsyncOpenAI(
            api_key=config.provider.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.provider.base_url or os.getenv("OPENAI_BASE_URL")
        )
        
        self.model = config.model
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.temperature = config.temperature

    async def chat(
        self, 
        messages: list[LLMMessage], 
        tools: list[ChatCompletionToolUnionParam] | None = None,
        on_chunk: Callable[[str, str, str], Awaitable[None]] | None = None
    ) -> LLMResponse:
        """
        流式输出 + 思考链 + 工具调用 三合一
        on_chunk: (delta_reasoning, delta_content, status)
        """
        extra_body = {}
        if "qwen" in self.model.lower() or "deepseek" in self.model.lower():
            extra_body = {
                "enable_thinking": True,
                "top_k": self.top_k,
                "min_p": 0
            }

        chat_messages = [msg.to_dict() for msg in messages]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            tools=tools,
            stream=True,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            extra_body=extra_body
        )

        content = ""
        reasoning = ""
        tool_calls_buffer = {}

        async for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # --------------------------
            # 1. 处理思考内容（兼容多家）
            # --------------------------
            r = getattr(delta, "reasoning_content", None)
            if r:
                reasoning += r
                if on_chunk:
                    await on_chunk(r, "", "🤔 思考中...")
                continue

            # --------------------------
            # 2. 处理正常回复内容
            # --------------------------
            if delta.content:
                content += delta.content
                if on_chunk:
                    await on_chunk("", delta.content, "正在生成回复...") # 触发回调
                continue

            # --------------------------
            # 3. 处理工具调用流式分片
            # --------------------------
            if delta.tool_calls:
                self._merge_tool_calls(delta.tool_calls, tool_calls_buffer)
                if on_chunk:
                    await on_chunk("", "", "⚙️ 构思工具调用...")

        # 转换为 ToolCall 列表
        tool_calls = []
        for idx, tool in sorted(tool_calls_buffer.items()):
            tool_calls.append(
                ToolCall(
                    name=tool["name"],
                    id=tool["id"],
                    arguments=tool["arguments"]
                )
            )

        return LLMResponse(
            content=content.strip(),
            reasoning_content=reasoning.strip(),
            tool_calls=tool_calls or None
        )

    def _merge_tool_calls(self, delta_tool_calls, tool_calls_buffer):
        """合并流式分片的 tool_calls"""
        for tool_call_delta in delta_tool_calls:
            idx = tool_call_delta.index
            if idx not in tool_calls_buffer:
                tool_calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}

            if tool_call_delta.id:
                tool_calls_buffer[idx]["id"] = tool_call_delta.id

            func = tool_call_delta.function
            if func:
                if func.name:
                    tool_calls_buffer[idx]["name"] = func.name
                if func.arguments:
                    tool_calls_buffer[idx]["arguments"] += func.arguments