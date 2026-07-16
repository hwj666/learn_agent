import asyncio
import copy
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple, Union

from pydantic import ValidationError
from schema.message import LLMMessage, ToolCall, ToolResult
from tools.base import BaseTool

logger = logging.getLogger("ToolExecutor")


class ToolExecutor:
    """【业界标准：纯粹的执行发动机 - 终极安全无疵版】

    具备动态线程熔断、深层上下文隔离、防 ReDoS 参数解析以及全链路异常降级的生产级工具执行器。
    """

    def __init__(self, max_concurrency: int = 16) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_workers = max_concurrency * 4

        self._thread_pool = ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix="tool_sync_worker"
        )
        self._is_shutdown = False
        self._lock = threading.Lock()  # 真正用于保护和切换 shutdown 状态的原子锁

    def _parse_arguments(
        self, arguments: Union[str, Dict[str, Any], None]
    ) -> Dict[str, Any]:
        """流式兼容的参数解析器 (完全移除正则，根除 ReDoS 并提升性能)"""
        if not arguments:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            raise ValueError("Arguments must be str or dict")

        cleaned_args = arguments.strip()

        if cleaned_args.startswith("```"):
            if cleaned_args.startswith("```json"):
                cleaned_args = cleaned_args[7:]
            else:
                cleaned_args = cleaned_args[3:]
            if cleaned_args.endswith("```"):
                cleaned_args = cleaned_args[:-3]
            cleaned_args = cleaned_args.strip()

        try:
            return json.loads(cleaned_args)
        except json.JSONDecodeError:
            start = cleaned_args.find("{")
            end = cleaned_args.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned_args[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise

    async def _execute_core_with_timeout(
        self,
        tool_instance: BaseTool,
        ctx: Dict[str, Any],
        args: Any,
        timeout: float,
    ) -> Any:
        """底层核心执行器：精准区分同步/异步，实施运行时双轨调度与线程池过载熔断"""
        # 线程安全地检查关闭状态
        with self._lock:
            if self._is_shutdown:
                raise RuntimeWarning("Executor is shutting down, request rejected.")

        async with self._semaphore:
            # 1. 异步工具执行轨道
            if asyncio.iscoroutinefunction(tool_instance.execute):
                return await asyncio.wait_for(
                    tool_instance.execute(ctx, args), timeout=timeout
                )

            # 2. 同步工具执行轨道
            # 动态检测同步线程池水位，防止发生超时逃逸的孤儿线程堆积撑爆系统
            current_active_sync_tasks = self._thread_pool._work_queue.qsize()
            if current_active_sync_tasks >= self._max_workers * 2:
                logger.critical(
                    f"[EngineBlock] 同步工具线程池排队队列过长 ({current_active_sync_tasks})，"
                    f"触发过载保护。强制熔断当前工具 `{tool_instance.__class__.__name__}` 的执行。"
                )
                raise RuntimeWarning(
                    "System sync thread pool is overloaded. Request throttled."
                )

            loop = asyncio.get_running_loop()
            try:
                future = loop.run_in_executor(
                    self._thread_pool, tool_instance.execute, ctx, args
                )
            except RuntimeError as re:
                if "executor is shutdown" in str(re).lower():
                    raise RuntimeWarning("Executor was shut down during scheduling.")
                raise

            try:
                return await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                active_threads = threading.active_count()
                logger.critical(
                    f"[EngineWarning] 同步工具 `{tool_instance.__class__.__name__}` 触发协程级超时！"
                    f"该同步线程无法被外部强行中止，可能已变为孤儿挂起线程。当前进程总线程数: {active_threads}。"
                    f"请务必检查该工具内部是否缺失网络 timeout 参数设置！"
                )
                raise

    async def execute(
        self,
        resolved_calls: List[Tuple[ToolCall, BaseTool]],
        ctx: Dict[str, Any],
        timeout: float = 30.0,
    ) -> List[LLMMessage]:
        """批量并发执行工具入口（对外核心 API）"""
        if not resolved_calls:
            return []

        with self._lock:
            if self._is_shutdown:
                logger.error("拒绝执行：ToolExecutor 已经处于关闭状态。")
                return [
                    LLMMessage.tool(
                        call.id, content="系统正在维护，暂时无法处理工具调用。"
                    )
                    for call, _ in resolved_calls
                ]

        # 【修复点 1】为了防止并发任务之间由于共享 ctx 导致竞态冲突，
        # 在传递给每个独立协程任务前，就在最外层为每个 Task 单独生成独立的 ctx 深拷贝快照
        tasks = []
        for tool_call, tool_instance in resolved_calls:
            try:
                task_ctx = copy.deepcopy(ctx) if isinstance(ctx, dict) else {}
            except Exception:
                logger.warning("Context deepcopy failed, falling back to shallow copy.")
                task_ctx = ctx.copy() if isinstance(ctx, dict) else {}

            tasks.append(
                self._execute_single(tool_call, tool_instance, task_ctx, timeout)
            )

        # 协程级全面并发隔离
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_messages = []

        for i, res in enumerate(results):
            tool_call, _ = resolved_calls[i]

            if isinstance(res, Exception):
                logger.critical(f"发动机管线破裂崩溃: {str(res)}", exc_info=res)
                tool_result = ToolResult(
                    success=False,
                    content="系统遇到了未预期的执行底层故障。",
                    error=f"UnhandledEngineException: {type(res).__name__}",
                )
            elif isinstance(res, ToolResult):
                tool_result = res
            else:
                # 【优化点 4】如果返回值是复杂结构（dict/list），用 json.dumps 保持标准 JSON 字符串
                if isinstance(res, (dict, list)):
                    content_str = json.dumps(res, ensure_ascii=False)
                else:
                    content_str = str(res)
                tool_result = ToolResult(success=True, content=content_str)

            if not tool_result.success:
                # 获取当次请求特有的 trace_id
                trace_id = (
                    ctx.get("trace_id", "N/A") if isinstance(ctx, dict) else "N/A"
                )
                logger.error(
                    f"[Trace:{trace_id}] 工具 `{tool_call.name}`(CallID:{tool_call.id}) 执行失败. "
                    f"错误信息: {tool_result.error}"
                )

            final_messages.append(
                LLMMessage.tool(tool_call.id, content=tool_result.content)
            )

        return final_messages

    async def _execute_single(
        self,
        tool_call: ToolCall,
        tool_instance: BaseTool,
        ctx: Dict[str, Any],
        timeout: float,
    ) -> ToolResult:
        """单个工具的单向隔离执行管道"""
        trace_id = ctx.get("trace_id", "N/A")

        # 1. 参数初步解析
        try:
            arguments_dict = self._parse_arguments(tool_call.arguments)
        except Exception as e:
            return ToolResult(
                success=False,
                content="参数解析失败，请检查你生成的工具入参 JSON 格式是否正确。",
                error=f"JSONDecodeError: {str(e)}",
            )

        # 2. 触发强类型 Pydantic 契约校验
        try:
            validated_args = tool_instance.args_schema.model_validate(arguments_dict)
        except ValidationError as ve:
            logger.warning(
                f"[Trace:{trace_id}] 工具 `{tool_call.name}` 参数校验未通过. 详情: {ve.errors()}"
            )
            readable_errors = [
                f"Field '{e['loc'][-1]}': {e['msg']}" for e in ve.errors()
            ]
            return ToolResult(
                success=False,
                content=f"工具调用契约校验失败。具体错误: {'; '.join(readable_errors)}。请修正后重新调用。",
                error=f"ValidationError: {str(ve.errors())}",
            )

        # 3. 触发带超时防御的核心执行
        try:
            res_content = await self._execute_core_with_timeout(
                tool_instance, ctx, validated_args, timeout
            )

            if isinstance(res_content, ToolResult):
                return res_content
            return res_content

        except asyncio.TimeoutError:
            logger.warning(
                f"[Trace:{trace_id}] 工具 `{tool_call.name}` 执行超时 (限额 {timeout} 秒)."
            )
            return ToolResult(
                success=False,
                content=f"工具执行超时，未能及时返回结果（时限：{timeout}秒）。",
                error="AsyncioTimeoutError",
            )
        except RuntimeWarning as rw:
            return ToolResult(
                success=False,
                content="由于系统负载过高或正在维护，该工具调用被拒绝。",
                error=f"RuntimeWarning: {str(rw)}",
            )

    def shutdown(self, wait: bool = True) -> None:
        """【修复点 2】显式提供线程安全的关闭接口，杜绝进程挂起与线程泄露"""
        with self._lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True
        logger.info("ToolExecutor 正在关闭，开始释放底层同步线程池...")
        self._thread_pool.shutdown(wait=wait)
        logger.info("ToolExecutor 线程池已成功释放。")
