from abc import ABC, abstractmethod
from typing import Optional, Callable, Awaitable
from schema.message import LLMMessage, LLMResponse
from schema.config import ProviderConfig

# 流式回调函数类型定义（和你的 Agent 完全匹配）
ChunkCallback = Callable[[str, str, str], Awaitable[None]]


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类（兼容 ReAct Agent + 流式输出）"""

    def __init__(self, config: ProviderConfig | None):
        # 配置对齐上层，防止空值崩溃
        self.config = config
        self.api_key: str = config.api_key if config else ""
        self.base_url: Optional[str] = config.base_url if config else None

        # 内部消息历史（可选）
        self.messages = []

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        tools: Optional[list] = None,
        reuse_history: bool = True,
        on_chunk: Optional[ChunkCallback] = None,  # 关键：流式回调
    ) -> LLMResponse:
        """
        异步聊天接口（必须实现）
        :param messages: 对话消息列表
        :param tools: 工具 schema 列表
        :param reuse_history: 是否复用历史
        :param on_chunk: 流式输出回调 (reasoning, content, status)
        """
        pass

    def add_message(self, message: LLMMessage):
        """添加消息到历史"""
        self.messages.append(message)
