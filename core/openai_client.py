import os
import json
import logging
import time
import asyncio
from typing import AsyncGenerator, List, Optional, Tuple
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError
from core.message import LLMMessage, LLMResponse, ToolCall
from core.base_client import BaseLLMClient
from core.config import ModelConfig
from handlers.base import BaseStreamHandler
from tools.extract import extract_implicit_tool_calls
from handlers.print_handler import PrintStreamHandler
from handlers.rich_handler import RichStreamHandler

_ = load_dotenv()

trace_logger = logging.getLogger("trace")
trace_logger.setLevel(logging.DEBUG)


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, trace_enabled: bool = False):
        super().__init__(config.provider)
        self.client = AsyncOpenAI(
            api_key=config.provider.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.provider.base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.model = config.model
        self.top_p = config.top_p
        self.top_k = config.top_k if config.top_k is not None else 50
        self.temperature = config.temperature
        self.trace_enabled = trace_enabled

    def _log_trace(self, phase: str, data: any):
        """记录 trace 日志"""
        if self.trace_enabled:
            trace_logger.debug(f"[{phase}] {json.dumps(data, ensure_ascii=False, indent=2)}")

    def _merge_tool_calls(self, cleaned_tool_deltas: list, buffer: dict):
        """安全合并并发工具流，确保 index 绝对对齐"""
        for delta in cleaned_tool_deltas:
            idx = delta["index"]
            if idx not in buffer:
                buffer[idx] = {"id": "", "name": "", "arguments": ""}

            # 仅在非空时追加/覆盖，防止被后续的空值覆盖
            if delta["id"]:
                buffer[idx]["id"] = delta["id"]
            if delta["name"]:
                buffer[idx]["name"] = delta["name"]
            if delta["arguments"]:
                buffer[idx]["arguments"] += delta["arguments"]  # 字符串追加

    async def _create_chat_completion_stream(
        self, messages: List[LLMMessage], tools: Optional[list]
    ):
        """封装底层的请求发起，便于配置 tenacity 实施指数退避重试"""
        extra_body = {}
        temperature = self.temperature
        model_lower = self.model.lower()

        # 兼容国内 Qwen/DeepSeek/R1 等模型的思考链与推理参数限制
        is_reasoning_model = any(
            k in model_lower for k in ["r1", "thinking", "qwen", "deepseek"]
        )
        if is_reasoning_model:
            extra_body["enable_thinking"] = False
            if self.top_k is not None:
                extra_body["top_k"] = self.top_k
            extra_body["min_p"] = 0

        return await self.client.chat.completions.create(
            model=self.model,
            messages=[msg.to_dict() for msg in messages],
            tools=tools or None,
            stream=True,
            temperature=temperature,
            top_p=self.top_p,
            presence_penalty=1.1,
            frequency_penalty=1.1,
            extra_body=extra_body if extra_body else None,
        )

    async def _stream_chat_inner(
        self, messages: List[LLMMessage], tools: Optional[list] = None
    ) -> AsyncGenerator[Tuple[str, str, list, str], None]:
        response = await self._create_chat_completion_stream(messages, tools)
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            reasoning_delta = getattr(delta, "reasoning_content", None) or getattr(
                delta, "reasoning", None
            )
            if reasoning_delta:
                yield reasoning_delta, "", [], "thinking"
                continue

            if delta.content:
                yield "", delta.content, [], "responding"
                continue

            if getattr(delta, "tool_calls", None):
                valid_tool_calls = [tc for tc in delta.tool_calls if tc is not None]
                cleaned_tool_deltas = []
                if valid_tool_calls:
                    for tc in valid_tool_calls:
                        idx = getattr(tc, "index", 0) or 0
                        func = getattr(tc, "function", None)
                        args = getattr(func, "arguments", "") if func else ""

                        cleaned_tool_deltas.append(
                            {
                                "index": idx,
                                "id": getattr(tc, "id", None),
                                "name": getattr(func, "name", None),
                                "arguments": args,
                            }
                        )
                yield "", "", cleaned_tool_deltas, "tool_calling"

    async def _fallback_non_streaming(
        self, messages: List[LLMMessage], tools: Optional[list] = None
    ) -> AsyncGenerator[Tuple[str, str, list, str], None]:
        """非流式降级调用：当流式调用失败时使用"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[msg.to_dict() for msg in messages],
            tools=tools or None,
            stream=False,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        if not response.choices:
            return

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = getattr(choice.message, "tool_calls", [])

        if content:
            yield "", content, [], "responding"

        if tool_calls:
            cleaned_tool_deltas = []
            for tc in tool_calls:
                idx = getattr(tc, "index", 0) or 0
                func = getattr(tc, "function", None)
                args = getattr(func, "arguments", "") if func else ""
                cleaned_tool_deltas.append(
                    {
                        "index": idx,
                        "id": getattr(tc, "id", None),
                        "name": getattr(func, "name", None),
                        "arguments": args,
                    }
                )
            yield "", "", cleaned_tool_deltas, "tool_calling"

    async def stream_chat(
        self, messages: List[LLMMessage], tools: Optional[list] = None
    ) -> AsyncGenerator[Tuple[str, str, list, str], None]:
        """【底层数据源】流式生成器：清洗数据，透传碎片段与工具增量字典"""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                async for item in self._stream_chat_inner(messages, tools):
                    yield item
                return
            except APIError as e:
                if "peg-native" in str(e).lower() and attempt < max_retries - 1:
                    self._log_trace("RETRY", {
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "error": str(e),
                        "action": "retrying stream_chat"
                    })
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                elif "peg-native" in str(e).lower():
                    self._log_trace("FALLBACK", {
                        "error": str(e),
                        "action": "falling back to non-streaming"
                    })
                    async for item in self._fallback_non_streaming(messages, tools):
                        yield item
                    return
                else:
                    raise e
            except Exception as e:
                self._log_trace("ERROR", {"error": str(e)})
                raise e

    async def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[list] = None,
        handler: BaseStreamHandler = PrintStreamHandler(),
    ) -> LLMResponse:
        """【高层业务接口】消灭异形判断，确保工具并发调用绝对不丢包、不错位"""
        content = ""
        reasoning = ""
        tool_calls_buffer = {}

        # Trace: 记录发送给模型的请求
        request_data = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in messages],
            "tools": [t["function"]["name"] for t in tools] if tools else None,
        }
        self._log_trace("REQUEST", request_data)

        async with handler as print_handler:
            # 注意：第三个参数变为了 cleaned_tool_deltas 列表
            async for think, text, cleaned_tool_deltas, chunk_type in self.stream_chat(
                messages, tools
            ):
                reasoning += think
                content += text

                if cleaned_tool_deltas:
                    self._merge_tool_calls(cleaned_tool_deltas, tool_calls_buffer)

                if print_handler:
                    await print_handler(think, text, cleaned_tool_deltas, chunk_type)
                else:
                    if think:
                        print(think, end="", flush=True)
                    if text:
                        print(text, end="", flush=True)
                    if cleaned_tool_deltas:
                        # 兜底打印：仅打印当前帧出来的参数碎片
                        for d in cleaned_tool_deltas:
                            print(d["arguments"], end="", flush=True)

        final_tool_calls = [
            ToolCall(id=t["id"], name=t["name"], arguments=t["arguments"])
            for _, t in sorted(tool_calls_buffer.items())
        ]

        # 2. 🛡️ 漏斗防御：如果官方通道未触发，直接调用外部独立的工具函数进行抢救
        if len(final_tool_calls) == 0 and content:
            tool_calls = extract_implicit_tool_calls(content, tools)
            final_tool_calls = [
                ToolCall(id=t["id"], name=t["name"], arguments=t["arguments"])
                for t in tool_calls
            ]

        # Trace: 记录模型的响应
        response_data = {
            "content": content[:2000] + "..." if len(content) > 2000 else content,
            "reasoning_content": reasoning[:2000] + "..." if len(reasoning) > 2000 else reasoning,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in final_tool_calls
            ] if final_tool_calls else None,
        }
        self._log_trace("RESPONSE", response_data)

        return LLMResponse(
            content=content,
            reasoning_content=reasoning,
            tool_calls=final_tool_calls if final_tool_calls else None,
        )
