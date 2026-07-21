import os
from typing import AsyncGenerator, List, Optional, Tuple, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
from common.message import LLMMessage, LLMResponse, ToolCall
from common.config import ModelConfig
from core.base_client import BaseLLMClient
from handlers.base import BaseStreamHandler
from handlers.print_handler import PrintStreamHandler
from tools import extract_implicit_tool_calls

_ = load_dotenv()


class OpenAIClient(BaseLLMClient):
    """
    工业级大模型客户端（极简高性能版）
    内置 DeepSeek R1 / Qwen 思考链与参数解耦兼容、并发工具流状态对齐及强安全异步上下文渲染机制
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config.provider)
        self.client = AsyncOpenAI(
            api_key=config.provider.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.provider.base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.model = config.model
        self.top_p = config.top_p
        self.top_k = config.top_k if config.top_k is not None else 50
        self.temperature = config.temperature

    def _merge_tool_calls(self, cleaned_tool_deltas: list, buffer: dict):
        """安全合并并发工具流碎片，确保多路并发的工具索引 index 绝对对齐"""
        for delta in cleaned_tool_deltas:
            idx = delta["index"]
            if idx not in buffer:
                buffer[idx] = {"id": "", "name": "", "arguments": ""}

            # 仅在非空时追加或覆盖，防止被后续帧的空值冲掉
            if delta["id"]:
                buffer[idx]["id"] = delta["id"]
            if delta["name"]:
                buffer[idx]["name"] = delta["name"]
            if delta["arguments"]:
                buffer[idx]["arguments"] += delta["arguments"]  # 参数字符串增量追加

    async def _create_chat_completion_stream(
        self, messages: List[LLMMessage], tools: Optional[list]
    ):
        """封装底层的流式 Completion 请求发起"""
        extra_body = {}
        temperature = self.temperature
        model_lower = self.model.lower()

        # 兼容国内 Qwen/DeepSeek/R1 等模型的思考链与推理参数限制
        is_reasoning_model = any(
            k in model_lower for k in ["r1", "thinking", "qwen", "deepseek"]
        )

        actual_tools = tools or None

        if is_reasoning_model:
            # 针对国内各大服务商平台，在关闭思维链或调整采样限制时的最佳防御实践
            extra_body["enable_thinking"] = False
            if self.top_k is not None:
                extra_body["top_k"] = self.top_k
            extra_body["min_p"] = 0

            # 防御性规避：部分推理模型（如原生 DeepSeek R1 官方版）在 API 层面不支持传 tools 显式 Function Calling
            if "r1" in model_lower:
                actual_tools = None
        return await self.client.chat.completions.create(
            model=self.model,
            messages=[msg.to_dict() for msg in messages],
            tools=actual_tools,
            stream=True,
            temperature=temperature,
            top_p=self.top_p,
            presence_penalty=1.1,
            frequency_penalty=1.1,
            extra_body=extra_body if extra_body else None,
        )

    async def stream_chat(
        self, messages: List[LLMMessage], tools: Optional[list] = None
    ) -> AsyncGenerator[Tuple[str, str, list, str], None]:
        """【底层数据源生成器】清洗数据，透传碎片段、工具增量字典以及强状态 stage 标签"""
        response = await self._create_chat_completion_stream(messages, tools)
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 兼容不同供应商平台对思考链（Reasoning Content）的混杂字段返回命名
            reasoning_delta = getattr(delta, "reasoning_content", None) or getattr(
                delta, "reasoning", None
            )
            if reasoning_delta:
                yield reasoning_delta, "", [], "thinking"
                continue

            # 文本响应流分流
            if delta.content:
                yield "", delta.content, [], "responding"
                continue

            # 工具调用流规范清洗与组装
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

    async def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[list] = None,
        handler: BaseStreamHandler = None,
    ) -> LLMResponse:
        """【高层统一业务接口】消灭各种异形条件判断，实现函数式单入口派发与异常强安全隔离"""
        # 如果未指定处理器，则提供一个不进行任何操作的空壳 Handler 维持核心代码流畅度
        if handler is None:
            handler = PrintStreamHandler()

        content = ""
        reasoning = ""
        tool_calls_buffer = {}
        # 🎯 异步上下文管理器切入：网络异常、Task 被取消或崩溃都能 100% 捕获并重置 UI/终端 状态
        async with handler:
            async for r_delta, c_delta, t_deltas, stage in self.stream_chat(
                messages, tools
            ):
                # 1. 内存静态数据累加合并（解耦 UI 渲染层）
                if stage == "thinking":
                    reasoning += r_delta
                elif stage == "responding":
                    content += c_delta
                elif stage == "tool_calling":
                    self._merge_tool_calls(t_deltas, tool_calls_buffer)

                # 2. 极致清爽的函数式单入口派发，将渲染状态机的职责彻底转交给 Handler 内部
                await handler(r_delta, c_delta, t_deltas, stage)

        # 3. 将对齐的合并 buffer 转换为强类型的 ToolCall 元组
        final_tool_calls = []
        if tool_calls_buffer:
            # 严格按照 index 升序排列还原数组，彻底解决乱序小碎包引起的丢包错位问题
            for idx in sorted(tool_calls_buffer.keys()):
                tc_info = tool_calls_buffer[idx]
                final_tool_calls.append(
                    ToolCall(
                        id=tc_info["id"],
                        name=tc_info["name"],
                        arguments=tc_info["arguments"],
                    )
                )

        # 4. 隐式提取兜底机制：当显式工具未触发但模型输出了内容，尝试从中提取 Markdown / JSON 的工具指令
        if not final_tool_calls and content:
            implicit_calls = extract_implicit_tool_calls(content)
            if implicit_calls:
                final_tool_calls.extend(implicit_calls)

        # 5. 组装并返回标准高层响应契约对象
        return LLMResponse(
            content=content,
            reasoning_content=reasoning if reasoning else None,
            tool_calls=tuple(final_tool_calls) if final_tool_calls else None,
        )
