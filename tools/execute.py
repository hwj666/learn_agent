import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from pydantic import ValidationError
from schema.message import ToolCall, ToolResult
from .base import BaseTool
from .extract import parse_llm_json_arguments

logger = logging.getLogger("ToolExecutor")


class ToolExecutor:
    """【精简高可用版】去除非必要防御，保留核心高并发调度与契约校验的纯净发动机"""

    def __init__(
        self,
        max_concurrency: int = 16,
        thread_pool_size: Optional[int] = None,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._thread_pool = ThreadPoolExecutor(
            max_workers=thread_pool_size or max_concurrency,
            thread_name_prefix="ToolExecutor",
        )
        self._is_shutdown = False
        self._shutdown_event = asyncio.Event()
        self._active_tasks: List[asyncio.Task] = []
        self._lock = threading.RLock()

    async def execute(
        self,
        tool_call: ToolCall,
        tool_instance: BaseTool,
        ctx: Dict[str, Any],
        timeout: float,
    ) -> ToolResult:
        """单管道隔离执行：参数解析 -> Schema校验 -> 异步/同步路由 -> 统一输出"""
        if self._is_shutdown:
            return ToolResult(
                success=False,
                content="Executor is shutting down.",
                error="Shutdown",
                structured_content={"error": "shutdown"},
            )

        # 权限检查
        allowed_toolsets = ctx.get("allowed_toolsets", set())
        if not tool_instance.is_allowed(allowed_toolsets):
            return ToolResult(
                success=False,
                content="Permission denied: tool not in allowed toolsets",
                error="PermissionDenied",
                structured_content={"error": "permission_denied"},
            )

        # 1. 契约校验与参数解析
        parse_result = self._parse_and_validate_args(tool_call, tool_instance)
        if not parse_result.success:
            return parse_result

        validated_args = parse_result.structured_content["validated_args"]

        # 2. 核心并发调度
        async with self._semaphore:
            if self._is_shutdown:
                return ToolResult(
                    success=False,
                    content="Executor shutdown during wait",
                    error="Shutdown",
                    structured_content={"error": "shutdown"},
                )

            task = asyncio.current_task()
            if task:
                with self._lock:
                    self._active_tasks.append(task)

            try:
                result = await self._execute_with_timeout(
                    tool_instance, ctx, validated_args, timeout
                )
                return self._format_result(result)
            except asyncio.TimeoutError:
                return ToolResult(
                    success=False,
                    content=f"工具执行超时({timeout}秒)",
                    error="TimeoutError",
                    structured_content={"error": "timeout", "timeout_seconds": timeout},
                )
            except Exception as e:
                logger.exception(f"Tool {tool_call.name} failed")
                return ToolResult(
                    success=False,
                    content=f"工具执行内部错误: {type(e).__name__}: {str(e)}",
                    error="RuntimeError",
                    structured_content={
                        "error": "runtime_error",
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                    },
                )
            finally:
                if task:
                    with self._lock:
                        if task in self._active_tasks:
                            self._active_tasks.remove(task)

    def _parse_and_validate_args(
        self, tool_call: ToolCall, tool_instance: BaseTool
    ) -> ToolResult:
        """解析并验证参数"""
        try:
            args_dict = parse_llm_json_arguments(tool_call.arguments)
            validated_args = tool_instance.args_schema.model_validate(args_dict)
            return ToolResult(
                success=True,
                content="Arguments validated",
                structured_content={"validated_args": validated_args},
            )
        except ValidationError as ve:
            errors = [
                f"Field '{e['loc'][-1]}': {e['msg']}" if e["loc"] else e["msg"]
                for e in ve.errors()
            ]
            error_details = ve.errors()
            return ToolResult(
                success=False,
                content=f"参数校验失败: {'; '.join(errors)}",
                error="ValidationError",
                structured_content={
                    "error": "validation_error",
                    "details": error_details,
                },
            )
        except ValueError as e:
            return ToolResult(
                success=False,
                content=f"JSON解析错误: {str(e)}",
                error="JSONDecodeError",
                structured_content={"error": "json_decode_error", "message": str(e)},
            )

    async def _execute_with_timeout(
        self,
        tool_instance: BaseTool,
        ctx: Dict[str, Any],
        validated_args: Any,
        timeout: float,
    ) -> Any:
        """带超时的执行包装"""
        if asyncio.iscoroutinefunction(tool_instance.execute):
            return await asyncio.wait_for(
                tool_instance.execute(ctx, validated_args), timeout=timeout
            )
        else:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self._thread_pool, tool_instance.execute, ctx, validated_args
                ),
                timeout=timeout,
            )

    def _format_result(self, result: Any) -> ToolResult:
        """格式化执行结果"""
        if isinstance(result, ToolResult):
            return result

        structured_content = (
            result if isinstance(result, (dict, list)) else {"result": result}
        )
        content_str = (
            json.dumps(result, ensure_ascii=False)
            if isinstance(result, (dict, list))
            else str(result)
        )

        return ToolResult(
            success=True, content=content_str, structured_content=structured_content
        )

    async def shutdown(self, timeout: float = 5.0) -> None:
        """优雅停机：取消所有活动任务并关闭线程池"""
        self._is_shutdown = True
        self._shutdown_event.set()

        # 取消所有活动任务
        with self._lock:
            active_tasks = self._active_tasks.copy()

        if active_tasks:
            logger.info(
                f"Cancelling {len(active_tasks)} active tool execution tasks..."
            )
            for task in active_tasks:
                if not task.done():
                    task.cancel()

            # 等待任务取消完成
            done, pending = await asyncio.wait(
                active_tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED
            )

            if pending:
                logger.warning(f"{len(pending)} tasks did not cancel within {timeout}s")

        # 关闭线程池
        self._thread_pool.shutdown(wait=False)
        logger.info("ToolExecutor shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """获取执行器统计信息"""
        with self._lock:
            active_tasks = len(self._active_tasks)

        return {
            "active_tasks": active_tasks,
            "max_concurrency": self._semaphore._value
            if hasattr(self._semaphore, "_value")
            else None,
            "thread_pool_size": self._thread_pool._max_workers,
            "is_shutdown": self._is_shutdown,
        }
