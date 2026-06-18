import json
import os
import logging
import asyncio
from typing import List

from core.config import AgentConfig
from core.message import LLMMessage
from core.openai_client import OpenAIClient
from tools.execute import ToolExecutor 
from tools.storage import MemoryStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class BaseAgent:
    def __init__(self, config: AgentConfig, session_id: str):
        self.session_id = session_id
        self.max_steps = config.max_steps
        self.max_memory_len = getattr(config, "max_memory_len", 5) 
        
        self.work_dir = "/home/mint/learn_agent/work"
        os.makedirs(self.work_dir, exist_ok=True)
        self.history_file = os.path.join(self.work_dir, f"history_{session_id}.jsonl")

        self.client = OpenAIClient(config.model_config)
        storage = MemoryStorage()
        self.executor = ToolExecutor(storage, allowed_toolsets=config.tool_set)

        # 1. 同步加载并注入 System Prompt
        with open("pt.md", encoding='utf-8') as f:
            raw_prompt = f.read()
        self.system_prompt = raw_prompt.format(work_dir=self.work_dir)
        
        # 2. 直接在内存中初始化 System 消息
        system_msg = LLMMessage(role="system", content=self.system_prompt)
        self.memory: List[LLMMessage] = [system_msg]
        
        # 3. 同步将第一条 System 消息持久化到本地文件
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(system_msg.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logging.getLogger(self.__class__.__name__).warning(f"初始 System 消息持久化失败: {e}")

        self._file_lock = asyncio.Lock()
        
        self.ctx = {
            "todo_store": {}, 
            "session_id": session_id, 
            "agent_id": 1, 
            "sandbox_read_dirs": ["./"],
            "sandbox_write_dirs": ["./work"]
        }

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
        """后续的用户和工具交互依然走异步追加"""
        self.memory.append(msg)
        self._trim_memory()
        
        async with self._file_lock:
            try:
                msg_dict = msg.to_dict()
                # 引入 aiofiles 保持后续运行期 I/O 异步非阻塞
                import aiofiles 
                async with aiofiles.open(self.history_file, "a", encoding="utf-8") as f:
                    await f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")
            except Exception as e:
                logging.getLogger(self.__class__.__name__).warning(f"消息持久化失败: {e}")

    async def run(self, user_query: str) -> str:
        raise NotImplementedError("子类必须实现 run 方法")