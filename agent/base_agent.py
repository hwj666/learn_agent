import json
import os
import logging
import asyncio
from typing import List

from schema.config import AgentConfig
from schema.message import LLMMessage
from core.openai_client import OpenAIClient
from tools.execute import ToolExecutor
from tools.storage import MemoryStorage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class BaseAgent:
    def __init__(self, config: AgentConfig):
        self.max_steps = config.max_steps
        self.max_memory_len = getattr(config, "max_memory_len", 5)

        self.client = OpenAIClient(config.model_config)
        self.executor = ToolExecutor(allowed_toolsets=config.tool_set)

    async def run(self, user_query: str) -> str:
        raise NotImplementedError("子类必须实现 run 方法")
