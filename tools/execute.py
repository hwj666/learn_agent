import asyncio
import inspect
import json
import re
from typing import Any, Dict, List, Set

from pydantic import ValidationError

from schema.message import LLMMessage, ToolCall, ToolResult
from tools.registry import ToolRegistry


# =====================================================================
# 5. 极致并发、完全无状态的工具执行核心器
# =====================================================================
class ToolExecutor:
    def __init__(
        self,
        allowed_toolsets: Set[str] | None = None,
        max_concurrency: int = 16,
    ) -> None:
        self.allowed_toolsets = allowed_toolsets or set()
        self._json_pattern = re.compile(r"(\{.*\})", re.DOTALL)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tools_schema = self._get_schemas()

    def _get_schemas(self) -> List[Dict[str, Any]]:
        tool_classes = ToolRegistry.get_tools_by_set(self.allowed_toolsets)
        return [cls.to_schema() for _, cls in sorted(tool_classes.items())]

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return self._tools_schema

    def _parse_arguments(self, arguments: Any) -> Dict[str, Any]:
        if not arguments:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            raise ValueError("Arguments must be str or dict")

        cleaned_args = arguments.strip()
        if cleaned_args.startswith("```"):
            cleaned_args = re.sub(
                r"^```(?:json)?\n?|\n?```$", "", cleaned_args, flags=re.IGNORECASE
            ).strip()

        match = self._json_pattern.search(cleaned_args)
        if match:
            cleaned_args = match.group(1)

        return json.loads(cleaned_args)

    async def _execute_core_with_timeout(
        self, tool_instance: Any, ctx: Dict[str, Any], args: Any, timeout: float
    ) -> Any:
        # 精准切分：异步函数直接走 wait_for，同步函数通过 to_thread 扔给线程池，绝不卡死主线程事件循环
        if inspect.iscoroutinefunction(tool_instance.execute):
            return await asyncio.wait_for(
                tool_instance.execute(ctx, args), timeout=timeout
            )
        else:
            return await asyncio.wait_for(
                asyncio.to_thread(tool_instance.execute, ctx, args), timeout=timeout
            )

    async def execute(
        self, tool_calls: List[ToolCall], ctx: Dict[str, Any], timeout: float = 30.0
    ) -> List[LLMMessage]:
        if not tool_calls:
            return []

        # 🌟 降维打击：无状态下直接全量并行下发，共享基础上下文，但互不污染
        tasks = [
            self._execute_single(tool_call, ctx, timeout) for tool_call in tool_calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_messages = []
        for i, res in enumerate(results):
            tool_call = tool_calls[i]
            if isinstance(res, Exception):
                tool_result = ToolResult(
                    success=False, content=f"调度异常: {str(res)}", error=str(res)
                )
            elif isinstance(res, ToolResult):
                tool_result = res
            else:
                tool_result = ToolResult(
                    success=False, content="错误：非标的响应协议。"
                )
            final_messages.append(
                LLMMessage.tool(
                    tool_call.id,
                    content=json.dumps(
                        tool_result.model_dump(exclude_none=True), ensure_ascii=False
                    ),
                )
            )
        return final_messages

    async def _execute_single(
        self, tool_call: ToolCall, ctx: Dict[str, Any], timeout: float
    ) -> ToolResult:
        tool_cls = ToolRegistry.get_tool(tool_call.name)
        if not tool_cls or tool_cls.toolset not in self.allowed_toolsets:
            return ToolResult(
                success=False, content=f"❌ 权限拒绝：未获准使用工具 `{tool_call.name}`"
            )

        try:
            arguments_dict = self._parse_arguments(tool_call.arguments)
        except Exception:
            return ToolResult(
                success=False, content="参数 JSON 解析失败", error="JSONDecodeError"
            )

        # 建立不可变的影子副本，注入隔离的 Trace 凭证，不破坏外层上下文
        shadow_ctx = ctx.copy()
        shadow_ctx["trace_id"] = "trace_id_12345678"

        async with self._semaphore:
            try:
                tool_instance = tool_cls()
                validated_args = tool_instance.args_schema.model_validate(
                    arguments_dict
                )

                # 🌟 纯净的无锁化多路复用执行
                res_content = await self._execute_core_with_timeout(
                    tool_instance, shadow_ctx, validated_args, timeout
                )
                return ToolResult(success=True, content=str(res_content))

            except ValidationError as ve:
                return ToolResult(
                    success=False,
                    content=f"契约校验失败: {str(ve)}",
                    error="ValidationError",
                )
            except asyncio.TimeoutError:
                return ToolResult(
                    success=False, content="网络执行超时", error="TimeoutError"
                )
            except Exception as e:
                return ToolResult(
                    success=False, content=f"内部崩溃: {str(e)}", error=type(e).__name__
                )
