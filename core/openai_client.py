import os
from typing import AsyncGenerator, List, Optional, Tuple
from dotenv import load_dotenv
from openai import AsyncOpenAI
from core.message import LLMMessage, LLMResponse, ToolCall
from core.base_client import BaseLLMClient
from core.config import ModelConfig
from utils.base_stream_handler import BaseStreamHandler
from utils.extract_tool import extract_implicit_tool_calls
from utils.print_stream_handler import PrintStreamHandler
from utils.rich_stream_handler import RichStreamHandler

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
                buffer[idx]["arguments"] += delta["arguments"] # 字符串追加

    async def _create_chat_completion_stream(self, messages: List[LLMMessage], tools: Optional[list]):
        """封装底层的请求发起，便于配置 tenacity 实施指数退避重试"""
        extra_body = {}
        temperature = self.temperature
        model_lower = self.model.lower()
        
        # 兼容国内 Qwen/DeepSeek/R1 等模型的思考链与推理参数限制
        is_reasoning_model = any(k in model_lower for k in ["r1", "thinking", "qwen", "deepseek"])
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

    async def stream_chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[list] = None
    ) -> AsyncGenerator[Tuple[str, str, list, str], None]:
        """【底层数据源】流式生成器：清洗数据，透传碎片段与工具增量字典"""
        try:
            response = await self._create_chat_completion_stream(messages, tools)
        except Exception as e:
            print(f"Stream connection failed: {str(e)}")
            raise e

        async for chunk in response:
            if not chunk.choices: 
                continue
            delta = chunk.choices[0].delta

            # 1. 思考链兼容性获取
            reasoning_delta = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if reasoning_delta:
                yield reasoning_delta, "", [], "thinking"
                continue

            # 2. 正常文本
            if delta.content:
                yield "", delta.content, [], "responding"
                continue

            # 3. 工具调用安全清洗 (精准提取并发片段)
            if getattr(delta, "tool_calls", None):
                valid_tool_calls = [tc for tc in delta.tool_calls if tc is not None]
                cleaned_tool_deltas = []
                # 🎯 如果有有效数据，正常解析
                if valid_tool_calls:
                    for tc in valid_tool_calls:
                        idx = getattr(tc, "index", 0) or 0
                        func = getattr(tc, "function", None)
                        args = getattr(func, "arguments", "") if func else ""
                        
                        cleaned_tool_deltas.append({
                            "index": idx,
                            "id": getattr(tc, "id", None),
                            "name": getattr(func, "name", None),
                            "arguments": args
                        })
                yield "", "", cleaned_tool_deltas, "tool_calling"


    async def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[list] = None,
        handler: BaseStreamHandler = RichStreamHandler()
    ) -> LLMResponse:
        """【高层业务接口】消灭异形判断，确保工具并发调用绝对不丢包、不错位"""
        content = ""
        reasoning = ""
        tool_calls_buffer = {}

        async with handler as print_handler:
            # 注意：第三个参数变为了 cleaned_tool_deltas 列表
            async for think, text, cleaned_tool_deltas, chunk_type in self.stream_chat(messages, tools):
                reasoning += think
                content += text
                
                if cleaned_tool_deltas:
                    self._merge_tool_calls(cleaned_tool_deltas, tool_calls_buffer)

                if print_handler:
                    await print_handler(think, text, cleaned_tool_deltas, chunk_type)
                else:
                    if think: print(think, end="", flush=True)
                    if text: print(text, end="", flush=True)
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

        return LLMResponse(
            content=content,
            reasoning_content=reasoning,
            tool_calls=final_tool_calls if final_tool_calls else None
        )