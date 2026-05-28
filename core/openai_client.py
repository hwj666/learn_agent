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
        super().__init__(config.provider)
        self.client = AsyncOpenAI(
            api_key=config.provider.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.provider.base_url or os.getenv("OPENAI_BASE_URL")
        )
        self.model = config.model
        self.top_p = config.top_p
        self.top_k = config.top_k if config.top_k is not None else 50
        self.temperature = config.temperature

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[ChatCompletionToolUnionParam] | None = None,
        on_chunk: Callable[[str, str, str], Awaitable[None]] | None = None
    ) -> LLMResponse:
        """
        三合一能力：文本流式、思考链流式、工具调用流式合并
        """
        extra_body = {}
        temperature = self.temperature

        # 自动识别推理/思考模型（可根据自己的模型名扩展）
        model_lower = self.model.lower()
        is_reasoning_model = any(
            keyword in model_lower 
            for keyword in ["r1", "thinking", "qwen", "deepseek"]
        )

        if is_reasoning_model:
            extra_body["enable_thinking"] = True
            if self.top_k is not None:
                extra_body["top_k"] = self.top_k
            extra_body["min_p"] = 0
            # 推理模型官方要求固定 temperature=1.0
            temperature = 1.0

        chat_messages = [msg.to_dict() for msg in messages]

        # 发起流式请求
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            tools=tools or None,
            stream=True,
            temperature=temperature,
            top_p=self.top_p,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            extra_body=extra_body if extra_body else None,
        )

        content = ""
        reasoning = ""
        tool_calls_buffer = {}

        async for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 1. 思考链
            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning += reasoning_delta
                if on_chunk:
                    await on_chunk(reasoning_delta, "", "thinking")
                continue

            # 2. 正常文本
            if delta.content:
                content += delta.content
                if on_chunk:
                    await on_chunk("", delta.content, "responding")
                continue

            # 3. 工具调用（真正流式输出，不卡顿）
            if delta.tool_calls:
                valid_tool_calls = [tc for tc in delta.tool_calls if tc is not None]
                if valid_tool_calls:
                    # 实时推送工具参数片段
                    if on_chunk:
                        for tc in valid_tool_calls:
                            func = getattr(tc, "function", None)
                            arg_delta = func.arguments if (func and func.arguments) else ""
                            if arg_delta:
                                await on_chunk("", arg_delta, "tool_calling")
                    # 合并工具调用
                    self._merge_tool_calls(valid_tool_calls, tool_calls_buffer)
        # 工具调用排序 & 格式化
        tool_calls = [
            ToolCall(
                id=t["id"],
                name=t["name"],
                arguments=t["arguments"]
            )
            for _, t in sorted(tool_calls_buffer.items())
        ]

        return LLMResponse(
            content=content.strip() or None,
            reasoning_content=reasoning.strip() or None,
            tool_calls=tool_calls or None
        )

    def _merge_tool_calls(self, delta_tool_calls, tool_calls_buffer: dict):
        """流式工具调用分片合并（OpenAI 标准 SSE 协议）"""
        for tool_call_delta in delta_tool_calls:
            idx = getattr(tool_call_delta, "index", None)
            if idx is None:
                continue

            if idx not in tool_calls_buffer:
                tool_calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}

            # 合并 ID
            if tool_call_delta.id:
                tool_calls_buffer[idx]["id"] = tool_call_delta.id

            # 合并函数信息
            func = getattr(tool_call_delta, "function", None)
            if not func:
                continue

            if func.name:
                tool_calls_buffer[idx]["name"] = func.name
            if func.arguments:
                tool_calls_buffer[idx]["arguments"] += func.arguments