from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ToolCall:
    """解析后的工具调用对象"""
    name: str
    arguments: str  # 必须是 JSON 字符串
    id: Optional[str] = None

    def __str__(self) -> str:
        return f"ToolCall(name={self.name}, arguments={self.arguments}, id={self.id})"


@dataclass
class LLMMessage:
    """LLM 标准消息体（兼容 OpenAI / 通义 / 深度思考）"""
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None  # 思维链/思考过程
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """转为 LLM API 标准字典（完全兼容 OpenAI 格式）"""
        # 1. 工具返回结果
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content or ""
            }

        # 2. 助手调用工具
        if self.role == "assistant" and self.tool_calls:
            return {
                "role": "assistant",
                "content": self.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments
                        }
                    } for call in self.tool_calls
                ]
            }

        # 3. 普通消息（system / user / 纯文本 assistant）
        result = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.reasoning_content is not None:
            result["reasoning_content"] = self.reasoning_content
        return result


@dataclass
class LLMResponse:
    """LLM 响应统一格式"""
    content: str
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class LLMMessageBuilder:
    """
    消息构造器（业务层专用，避免手写 role 出错）
    你的 ReactAgent 正在用这个，完全匹配！
    """
    @staticmethod
    def system(content: str) -> LLMMessage:
        return LLMMessage(role="system", content=content)

    @staticmethod
    def user(content: str) -> LLMMessage:
        return LLMMessage(role="user", content=content)

    @staticmethod
    def assistant(
        content: Optional[str] = None,
        reasoning: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None
    ) -> LLMMessage:
        return LLMMessage(
            role="assistant",
            content=content,
            reasoning_content=reasoning,
            tool_calls=tool_calls
        )

    @staticmethod
    def tool(tool_call_id: str, content: str) -> LLMMessage:
        return LLMMessage(
            role="tool",
            tool_call_id=tool_call_id,
            content=content
        )