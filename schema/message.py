import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Sequence

# ---------- 常量驻留（减少重复字符串对象） ----------
_ROLE_SYSTEM = sys.intern("system")
_ROLE_USER = sys.intern("user")
_ROLE_ASSISTANT = sys.intern("assistant")
_ROLE_TOOL = sys.intern("tool")


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # 标准 JSON 字符串


@dataclass(slots=True)
class ToolResult:
    success: bool
    content: str
    error: Optional[str] = None


@dataclass(slots=True)
class LLMResponse:
    content: str
    reasoning_content: Optional[str] = None
    tool_calls: Optional[Tuple[ToolCall, ...]] = None
    usage: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens") or self.usage.get("input_tokens") or 0

    @property
    def completion_tokens(self) -> int:
        return (
            self.usage.get("completion_tokens") or self.usage.get("output_tokens") or 0
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.usage.get("total_tokens")
            or (self.prompt_tokens + self.completion_tokens)
            or 0
        )


class BaseLLMMessage:
    __slots__ = ("role", "content", "reasoning_content")

    def __init__(
        self,
        role: str,
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
    ):
        self.role = sys.intern(str(role))
        self.content = content
        self.reasoning_content = reasoning_content

    def to_dict(self) -> dict[str, Any]:
        res = {"role": self.role}
        if self.content is not None:
            res["content"] = self.content
        if self.reasoning_content is not None:
            res["reasoning_content"] = self.reasoning_content
        return res


class SystemMessage(BaseLLMMessage):
    __slots__ = ()

    def __init__(self, content: str):
        super().__init__(_ROLE_SYSTEM, content=content)


class UserMessage(BaseLLMMessage):
    __slots__ = ()

    def __init__(self, content: str):
        super().__init__(_ROLE_USER, content=content)


class ToolResultMessage(BaseLLMMessage):
    __slots__ = ("tool_call_id",)

    def __init__(self, tool_call_id: str, content: str):
        super().__init__(_ROLE_TOOL, content=content)
        self.tool_call_id = sys.intern(str(tool_call_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": _ROLE_TOOL,
            "tool_call_id": self.tool_call_id,
            "content": self.content or "",
        }


class AssistantMessage(BaseLLMMessage):
    __slots__ = ("tool_calls",)

    def __init__(
        self,
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        tool_calls: Optional[Sequence[ToolCall]] = None,
    ):
        super().__init__(
            _ROLE_ASSISTANT, content=content, reasoning_content=reasoning_content
        )
        self.tool_calls = tuple(tool_calls) if tool_calls else None

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {"role": _ROLE_ASSISTANT, "content": self.content}
        if self.tool_calls:
            res["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        if self.reasoning_content is not None:
            res["reasoning_content"] = self.reasoning_content
        return res


class LLMMessage:
    @staticmethod
    def system(content: str) -> SystemMessage:
        return SystemMessage(content)

    @staticmethod
    def user(content: str) -> UserMessage:
        return UserMessage(content)

    @staticmethod
    def tool(id: str, content: str) -> ToolResultMessage:
        return ToolResultMessage(tool_call_id=id, content=content)

    @staticmethod
    def assistant(
        content: Optional[str] = None,
        reasoning: Optional[str] = None,
        tool_calls: Optional[Sequence[ToolCall]] = None,
    ) -> AssistantMessage:
        return AssistantMessage(
            content=content, reasoning_content=reasoning, tool_calls=tool_calls
        )
