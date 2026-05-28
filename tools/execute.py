import json
import re
import asyncio
import logging
import copy
from typing import Any, Dict, List, Set
from pydantic import ValidationError
from tools.registry import ToolRegistry
from core.message import LLMMessage, LLMMessageBuilder, ToolCall
from utils.trace import set_trace_id, get_trace_id 

logger = logging.getLogger(__name__)

class ToolExecutor:
    def __init__(self, allowed_toolsets: Set[str] | None = None) -> None:
        self.allowed_toolsets = allowed_toolsets or set()
        self._json_pattern = re.compile(r"(\{.*\})", re.DOTALL)

    def get_schemas(self) -> List[Dict[str, Any]]:
        tool_classes = ToolRegistry.get_tools_by_set(self.allowed_toolsets)
        return [cls.to_schema() for _, cls in sorted(tool_classes.items())]

    def _parse_arguments(self, arguments: Any) -> Dict[str, Any]:
        if not arguments:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            raise json.JSONDecodeError("Arguments must be str or dict", str(arguments), 0)

        cleaned_args = arguments.strip()
        if cleaned_args.startswith("```"):
            cleaned_args = re.sub(r"^```(?:json)?\n?|\n?```$", "", cleaned_args, flags=re.IGNORECASE).strip()

        match = self._json_pattern.search(cleaned_args)
        if match:
            cleaned_args = match.group(1)

        return json.loads(cleaned_args)

    async def _execute_single(self, tool_call: ToolCall, ctx: Dict[str, Any], timeout: float) -> LLMMessage:
        """
        内部私有方法：由于 contextvars 会自动随 asyncio 任务复制，
        这里无需手动设置 trace_id，直接调用 logger 即可自动带上正确的 ID。
        """
        tool_cls = ToolRegistry.get_tool(tool_call.name)
        if not tool_cls or tool_cls.toolset not in self.allowed_toolsets:
            logger.warning("🚫 权限拒绝：Agent 尝试调用未授权工具 [%s]", tool_call.name)
            return LLMMessageBuilder.tool(
                tool_call.id,
                f"❌ 权限拒绝：未在当前 Agent 授权集中找到工具 `{tool_call.name}`"
            )

        try:
            arguments_dict = self._parse_arguments(tool_call.arguments)
            tool_instance = tool_cls()
            validated_args = tool_instance.args_schema.model_validate(arguments_dict)

            # 在上下文影子中注入当前的 trace_id，方便下游工具链（如 Http 客户端）透传
            shadow_ctx = copy.deepcopy(ctx) 
            shadow_ctx["trace_id"] = get_trace_id()

            logger.info("🚀 开始执行工具 [%s] 参数: %s", tool_call.name, arguments_dict)
            
            result = await asyncio.wait_for(
                tool_instance.execute(shadow_ctx, validated_args), 
                timeout=timeout
            )
            
            logger.info("✅ 工具 [%s] 执行成功", tool_call.name)
            return LLMMessageBuilder.tool(tool_call.id, str(result))

        except json.JSONDecodeError:
            logger.error("❌ 参数解析失败，输入: %s", tool_call.arguments)
            return LLMMessageBuilder.tool(tool_call.id, f"❌ 参数解析失败...")
        except ValidationError as e:
            logger.error("❌ 参数校验失败: %s", str(e))
            return LLMMessageBuilder.tool(tool_call.id, f"❌ 参数校验失败...")
        except asyncio.TimeoutError:
            logger.warning("⏳ 工具执行超时 [%s]", tool_call.name)
            return LLMMessageBuilder.tool(tool_call.id, f"❌ 工具执行超时...")
        except Exception as e:
            logger.exception("💥 工具内部执行异常 [%s]", tool_call.name)
            return LLMMessageBuilder.tool(tool_call.id, f"❌ 工具执行内部异常...")

    async def execute(self, tool_calls: List[ToolCall], ctx: Dict[str, Any], timeout: float = 30.0) -> List[LLMMessage]:
        if not tool_calls:
            return []

        if not ctx or "agent_id" not in ctx:
            return [
                LLMMessageBuilder.tool(tc.id, "❌ 安全拒绝：运行时上下文缺失")
                for tc in tool_calls
            ]

        # 核心：在入口处初始化 Trace ID（优先使用上游传入的 trace_id）
        upstream_trace_id = ctx.get("trace_id")
        current_trace_id = set_trace_id(upstream_trace_id)

        logger.info("📥 收到批量工具调用请求，包含 %d 个任务", len(tool_calls))

        # asyncio.create_task 会自动复制当前的 contextvars 上下文
        # 这确保了并行的子协程 _execute_single 共享同一个 trace_id
        tasks = [asyncio.create_task(self._execute_single(tc, ctx, timeout)) for tc in tool_calls]
        results = await asyncio.gather(*tasks)
        
        logger.info("📤 批量工具调用执行完毕")
        return list(results)