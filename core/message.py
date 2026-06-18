from dataclasses import dataclass, asdict
from typing import Any, Optional, Tuple
import sys


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # 必须是 JSON 字符串


from pydantic import BaseModel
from typing import Optional


class ToolResult(BaseModel):
    success: bool
    content: str
    error: Optional[str] = None


@dataclass(slots=True)
class LLMMessage:
    """LLM 标准消息体（终极内存优化 + 流式安全版）"""

    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[Tuple[ToolCall, ...]] = None
    tool_call_id: Optional[str] = None

    def __post_init__(self):
        # 角色字符串全局共享，海量消息内存暴跌
        self.role = sys.intern(str(self.role))

    """业务层消息构造器（完全兼容旧代码）"""

    @classmethod
    def system(cls, content: str):
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str):
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: Optional[str] = None,
        reasoning: Optional[str] = None,
        tool_calls: Optional[list | Tuple[ToolCall, ...]] = None,
    ):
        t_calls = tuple(tool_calls) if tool_calls is not None else None

        return cls(
            role="assistant",
            content=content,
            reasoning_content=reasoning,
            tool_calls=t_calls,
        )

    @classmethod
    def tool(cls, id: str, content: str):
        return cls(role="tool", tool_call_id=id, content=content)

    def to_dict(self) -> dict[str, Any]:
        """转为 LLM API 标准字典（严格对齐 OpenAI）"""

        # 1. 工具返回消息
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content or "",
            }

        # 2. 助手调用工具
        if self.role == "assistant" and self.tool_calls:
            res = {
                "role": "assistant",
                "content": self.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in self.tool_calls
                ],
            }
            if self.reasoning_content is not None:
                res["reasoning_content"] = self.reasoning_content
            return res

        # 3. 普通消息
        result = {"role": self.role}

        if self.role == "assistant":
            result["content"] = self.content
        elif self.content is not None:
            result["content"] = self.content

        if self.reasoning_content is not None:
            result["reasoning_content"] = self.reasoning_content

        return result


@dataclass(slots=True)
class LLMResponse:
    """LLM 响应统一格式"""

    content: str
    reasoning_content: Optional[str] = None
    tool_calls: Optional[Tuple[ToolCall, ...]] = None
