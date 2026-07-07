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

        tasks = [
            self._execute_single(tool_call, ctx, timeout) for tool_call in tool_calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_messages = []
        for i, res in enumerate(results):
            tool_call = tool_calls[i]

            # 1. 统一封装为 ToolResult 容器
            if isinstance(res, Exception):
                # 捕获 asyncio.gather 级别或线程池本身抛出的极罕见异常
                tool_result = ToolResult(
                    success=False,
                    content="系统异常，请稍后再试。",
                    error=f"Task Interrupted: {str(res)}",
                )
            elif isinstance(res, ToolResult):
                tool_result = res
            else:
                tool_result = ToolResult(
                    success=False,
                    content="系统调度错误。",
                    error="Non-standard protocol response",
                )

            # 2. 💰 释放 ToolResult 的工程价值：统一监控与埋点
            if not tool_result.success:
                # 在后端使用真正的错误日志（error 字段），报警系统（如 Sentry）会在这里拦截
                print(
                    f"[ERROR][Trace:{ctx.get('trace_id')}] 工具 `{tool_call.name}` 执行失败. 详情: {tool_result.error}"
                )
            else:
                # 可以做成功率统计或耗时统计
                pass

            # 3. 🛡️ 完美解耦：只把安全的、纯净的 content 交付给 Agent
            final_messages.append(
                LLMMessage.tool(
                    tool_call.id,
                    content=tool_result.content,  # 👈 大模型只看这个，不再需要 json.dumps
                )
            )
        return final_messages

    async def _execute_single(
        self, tool_call: ToolCall, ctx: Dict[str, Any], timeout: float
    ) -> ToolResult:
        print(tool_call.name)
        tool_cls = ToolRegistry.get_tool(tool_call.name)
        if not tool_cls or tool_cls.toolset not in self.allowed_toolsets:
            # 权限拒绝属于工程错误，但需要清晰告诉 Agent 别再试了
            return ToolResult(
                success=False,
                content=f"❌ 权限拒绝：当前策略未获准使用工具 `{tool_call.name}`。",
                error="PermissionDenied",
            )

        try:
            arguments_dict = self._parse_arguments(tool_call.arguments)
        except Exception as e:
            # 参数解析失败，告诉 Agent 它生成的 JSON 格式不对，让它修正后重试
            return ToolResult(
                success=False,
                content="参数解析失败，请检查你生成的工具入参 JSON 格式是否正确。",
                error=f"JSONDecodeError: {str(e)}",
            )

        shadow_ctx = ctx.copy()

        async with self._semaphore:
            try:
                tool_instance = tool_cls()
                validated_args = tool_instance.args_schema.model_validate(
                    arguments_dict
                )

                res_content = await self._execute_core_with_timeout(
                    tool_instance, shadow_ctx, validated_args, timeout
                )

                # 成功场景：真正在这里展现 ToolResult 结构体的规范价值
                return ToolResult(success=True, content=str(res_content))

            except ValidationError as ve:
                # 字段校验失败（如少传了必填参数），content 提示 Agent 缺了什么，以便 Agent 自行修复
                return ToolResult(
                    success=False,
                    content=f"工具调用契约校验失败，可能缺少必填字段或类型错误: {str(ve.errors())}",
                    error="ValidationError",
                )
            except asyncio.TimeoutError:
                # 超时错误
                return ToolResult(
                    success=False,
                    content="该工具响应超时，请稍后重试或尝试调用其他工具。",
                    error="TimeoutError",
                )
            except Exception as e:
                # ⚠️ 关键安全保护：防止内部崩溃日志（如 SQL 报错、IP 暴露）污染大模型上下文
                return ToolResult(
                    success=False,
                    content="工具内部执行遇到未预期错误，暂时无法获取结果。",  # 面向 Agent：安全无害
                    error=f"InternalCrash: {type(e).__name__} - {str(e)}",  # 面向后端：精准排查
                )
