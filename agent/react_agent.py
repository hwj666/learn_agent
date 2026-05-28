import json
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional

from core.config import AgentConfig
from core.message import LLMMessage, LLMMessageBuilder
from core.openai_client import OpenAIClient
from tools.execute import ToolExecutor 
import aiofiles 
from utils.stream_print_handler import StreamPrintHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReactAgent")

class ReactAgent:
    def __init__(self, config: AgentConfig, session_id: str):
        self.session_id = session_id
        self.max_steps = config.max_steps
        self.max_memory_len = getattr(config, "max_memory_len", 20) 
        
        work_dir = "./work"
        os.makedirs(work_dir, exist_ok=True)
        self.history_file = os.path.join(work_dir, f"history_{session_id}.jsonl")

        self.client = OpenAIClient(config.model_config)
        self.executor = ToolExecutor(allowed_toolsets=config.tool_set)
        
        self.system_prompt = "你是一个具备自主思考和行动能力的 ReAct Agent，工作目录为./work，后续要创建的文件都在这个目录下"
        self.memory: List[LLMMessage] = []
        self._file_lock = asyncio.Lock()
        
        self.ctx = {
            "todo_store": {}, 
            "session_id": session_id, 
            "agent_id": 1, 
            "sandbox_read_dirs": ["./"],
            "sandbox_write_dirs": ["./work"]
        }
        self.timeline_events: List[Dict[str, Any]] = []

    async def initialize(self):
        if not self.memory:
            await self._add_message_async(LLMMessage(role="system", content=self.system_prompt))

    def _trim_memory(self):
        if len(self.memory) <= self.max_memory_len:
            return

        system_msg = self.memory[0]
        non_system_memory = self.memory[1:]
        target_keep = self.max_memory_len - 1

        safe_cut_idx = None
        for i in reversed(range(len(non_system_memory))):
            if non_system_memory[i].role == "user":
                safe_cut_idx = i
                break

        if safe_cut_idx is not None:
            trimmed = non_system_memory[safe_cut_idx:]
        else:
            trimmed = non_system_memory[-target_keep:] if len(non_system_memory) > target_keep else non_system_memory

        self.memory = [system_msg] + trimmed

    async def _add_message_async(self, msg: LLMMessage):
        self.memory.append(msg)
        self._trim_memory()
        
        async with self._file_lock:
            try:
                msg_dict = msg.to_dict()
                async with aiofiles.open(self.history_file, "a", encoding="utf-8") as f:
                    await f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"消息持久化失败: {e}")

    async def run(self, user_query: str) -> str:
        await self.initialize()
        await self._add_message_async(LLMMessageBuilder.user(user_query))
        
        step = 0
        while step < self.max_steps:
            step += 1
            logger.info(f"==================== STEP {step}/{self.max_steps} 开始 ====================")
            
            sph = StreamPrintHandler()
            try:
                available_tools = self.executor.get_schemas() if step < self.max_steps else None
                
                llm_response = await self.client.chat(
                    messages=self.memory,
                    tools=available_tools,
                    on_chunk=sph.on_chunk
                )
                print("\033[0m")  # 仅关闭颜色，不输出多余空行
            except Exception as e:
                logger.error(f"LLM 调用异常: {str(e)}")
                print(f"\033[31m[❌ 错误]: 大模型调用失败\033[0m")
                return f"错误：调用模型时发生异常: {str(e)}"

            assistant_msg = LLMMessageBuilder.assistant(
                content=llm_response.content or "",
                reasoning=getattr(llm_response, "reasoning_content", ""),
                tool_calls=llm_response.tool_calls or None
            )
            await self._add_message_async(assistant_msg)

            # 最终回答 → logger
            if not llm_response.tool_calls:
                answer = llm_response.content or "未生成有效回答"
                logger.info(f"[最终解答]: {answer}")
                logger.info("======================== ✨ 任务完成 ========================")
                return answer

            # 工具调用 → logger
            logger.info("[工具调用请求]")
            for tc in llm_response.tool_calls:
                logger.info(f"  调用工具: {tc.name} | 参数: {tc.arguments}")
            
            tool_responses = await self.executor.execute(
                tool_calls=llm_response.tool_calls, 
                ctx=self.ctx,
                timeout=30.0
            )
            logger.info("所有并行工具执行完毕")

            for response in tool_responses:
                await self._add_message_async(response)

        error_msg = "超过最大迭代步数限制。"
        logger.error(error_msg)
        return "错误：超过最大迭代步数，未能生成有效解答。"