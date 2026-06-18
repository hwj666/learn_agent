import asyncio
import inspect
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from contextlib import asynccontextmanager

from pydantic import ValidationError
from core.message import LLMMessage, ToolCall, ToolResult
from tools.base import EmptyState
from tools.registry import ToolRegistry
from tools.storage import BaseStorage
from utils.trace import get_trace_id

logger = logging.getLogger(__name__)

class ToolExecutor:
    def __init__(
        self, 
        storage: BaseStorage,
        allowed_toolsets: Set[str] | None = None, 
        max_concurrency: int = 16
    ) -> None:
        self.storage = storage
        self.allowed_toolsets = allowed_toolsets or set()
        self._json_pattern = re.compile(r"(\{.*\})", re.DOTALL)
        
        # 内存锁字典
        self._session_locks: Dict[str, Tuple[asyncio.Lock, int]] = {}
        self._lock_creation_mutex = asyncio.Lock()
        
        # 依赖强约束注册表，动态计算 Schema
        self._tools_schema = self._get_schemas()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def _get_schemas(self) -> List[Dict[str, Any]]:
        """基于被授权的 toolset 过滤并生成标准的 OpenAPI Schema 列表"""
        tool_classes = ToolRegistry.get_tools_by_set(self.allowed_toolsets)
        return [cls.to_schema() for _, cls in sorted(tool_classes.items())]
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """安全获取指定工具名对应的最新 Schema"""
        tool = ToolRegistry.get_tool(name)
        if tool:
            # 去掉包裹的外层方括号，直接返回字典
            return tool.to_schema()
        return None


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
            cleaned_args = re.sub(r"^```(?:json)?\n?|\n?```$", "", cleaned_args, flags=re.IGNORECASE).strip()

        match = self._json_pattern.search(cleaned_args)
        if match:
            cleaned_args = match.group(1)

        return json.loads(cleaned_args)

    @asynccontextmanager
    async def _use_tool_lock(self, session_id: str, agent_id: str, tool_name: str):
        """
        工业级原子上下文锁管理器（完美替代原先松散的 acquire/release 分配）
        确保即使在极端的 asyncio.gather 任务撤销/崩溃场景下，锁的引用计数和释放也绝对精准。
        """
        lock_key = f"{session_id}:{agent_id}:{tool_name}"
        
        # 1. 极短的临界区：安全递增引用计数并获取锁实例
        async with self._lock_creation_mutex:
            if lock_key not in self._session_locks:
                self._session_locks[lock_key] = (asyncio.Lock(), 0)
            lock, ref_count = self._session_locks[lock_key]
            self._session_locks[lock_key] = (lock, ref_count + 1)
        
        try:
            # 2. 移交控制权：让业务代码在安全的锁保护区内执行
            async with lock:
                yield
        finally:
            # 3. 收尾临界区：安全递减引用计数并及时擦除无用锁
            async with self._lock_creation_mutex:
                if lock_key in self._session_locks:
                    lock, ref_count = self._session_locks[lock_key]
                    if ref_count <= 1:
                        self._session_locks.pop(lock_key, None)
                    else:
                        self._session_locks[lock_key] = (lock, ref_count - 1)

    async def _execute_core_with_timeout(self, tool_instance: Any, ctx: Dict[str, Any], args: Any, timeout: float) -> Any:
        """多路复用：精准识别异步/同步函数，防外部 I/O 阻塞主线程"""
        if inspect.iscoroutinefunction(tool_instance.execute):
            return await asyncio.wait_for(tool_instance.execute(ctx, args), timeout=timeout)
        else:
            return await asyncio.wait_for(
                asyncio.to_thread(tool_instance.execute, ctx, args), 
                timeout=timeout
            )

    async def execute(self, tool_calls: List[ToolCall], ctx: Dict[str, Any], timeout: float = 1000) -> List[LLMMessage]:
        if not tool_calls:
            return []

        # 并行分发所有工具调用，完全共享且隔离的 ctx
        tasks = [self._execute_single(tool_call, ctx, timeout) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_messages = []
        for i, res in enumerate(results):
            tool_call = tool_calls[i]

            # 1. 顶层致命系统崩溃捕获
            if isinstance(res, Exception):
                logger.error(f"💥 工具调度层严重崩溃 [{tool_call.name}]: {res}", exc_info=True)
                tool_result = ToolResult(
                    success=False,
                    content=f"系统调度层异常: {str(res)}",
                    exit_code=-1,
                    error=str(res)
                )
            # 2. 标准 ToolResult 响应
            elif isinstance(res, ToolResult):
                tool_result = res
        final_messages = []
        for i, res in enumerate(results):
            tool_call = tool_calls[i]

            # 1. 顶层致命系统崩溃捕获（如：超时、未捕获的代码 Bug）
            if isinstance(res, Exception):
                logger.error(f"💥 工具调度层严重崩溃 [{tool_call.name}]: {res}", exc_info=True)
                tool_result = ToolResult(
                    success=False,
                    content=f"系统调度层异常: {str(res)}",
                    exit_code=-1,
                    error=str(res)
                )
            
            # 2. 核心现代通路：强制所有工具执行必须生成标准的 ToolResult
            elif isinstance(res, ToolResult):
                tool_result = res
            
            # 3. 严格拒绝任何非标返回（防止脏数据污染大模型）
            else:
                logger.error(f"❌ 工具 [{tool_call.name}] 返回了不合规的数据类型: {type(res)}")
                tool_result = ToolResult(
                    success=False,
                    content="错误：工具未返回标准的 ToolResult 协议对象。",
                    exit_code=-98
                )

            status = "✅ 成功" if tool_result.success else "❌ 失败"
            logger.info(f"📊 [工具报表] 名称: {tool_call.name} | 状态: {status} | 状态码: {tool_result.exit_code}")

            final_messages.append(
                LLMMessage.tool(
                    tool_call.id,
                    content=json.dumps(tool_result.model_dump(), ensure_ascii=False)
                )
            )
        return final_messages

    async def _execute_single(self, tool_call: ToolCall, ctx: Dict[str, Any], timeout: float) -> ToolResult:
        # 从新版全局注册表中精准提取类模板
        tool_cls = ToolRegistry.get_tool(tool_call.name)
        if not tool_cls or tool_cls.toolset not in self.allowed_toolsets:
            return ToolResult(success=False, content=f"❌ 权限拒绝：当前 Agent 未获准使用工具 `{tool_call.name}`", exit_code=-99)

        # 2. 统一参数解析，避免重复解析
        try:
            arguments_dict = self._parse_arguments(tool_call.arguments)
            clean_log_args = json.dumps(arguments_dict, ensure_ascii=False)
        except Exception:
            arguments_dict = {}
            clean_log_args = str(tool_call.arguments)

        # 美化中文日志输出
        logger.info(f"🛠️ [工具调用流] [{tool_call.name}] -> {clean_log_args}")

        session_id = ctx.get("session_id", "default")
        agent_id = ctx.get("agent_id", "main")
        state_key = f"agent_state:{session_id}:{agent_id}:{tool_call.name}"
        has_state = getattr(tool_cls, "state_schema", None) is not EmptyState

        # 3. 建立上下文影子副本，注入 Trace 凭证
        shadow_ctx = {**ctx, "trace_id": get_trace_id()}

        async with self._semaphore:
            try:
                # 参数硬核契约校验
                tool_instance = tool_cls()
                validated_args = tool_instance.args_schema.model_validate(arguments_dict)

                # 建立上下文影子副本，注入不可变的 Trace 凭证
                shadow_ctx = ctx.copy()
                shadow_ctx["trace_id"] = get_trace_id()

                if has_state:
                    # 使用封装好的安全异步锁上下文
                    async with self._use_tool_lock(session_id, agent_id, tool_call.name):
                        logger.info(f"🔒 [串行加锁区] 正在加载状态数据: {tool_call.name}")
                        raw_state = await self.storage.get_state(state_key)
                        
                        # 在锁安全的保护伞内实例化并注入专属影子状态
                        shadow_ctx["tool_state"] = tool_cls.state_schema(**raw_state) if raw_state else tool_cls.state_schema()
                        
                        logger.info(f"🚀 [有状态通道] 启动串行调用 -> {tool_call.name}")
                        result = await self._execute_core_with_timeout(tool_instance, shadow_ctx, validated_args, timeout)
                        
                        # 执行成功立即刷回底层存储（如 Redis/Memory），随后安全释放锁
                        await self.storage.set_state(state_key, shadow_ctx["tool_state"].model_dump())
                else:
                    # 无状态高并发快车道，零锁延迟，最大化拉满吞吐率
                    logger.info(f"🚀 [无状态通道] 启动高并发调用 -> {tool_call.name}")
                    result = await self._execute_core_with_timeout(tool_instance, shadow_ctx, validated_args, timeout)
                # 7. 结构化结果序列化（防止 str() 导致大模型无法识别）
                if hasattr(result, "model_dump_json"):
                    content_str = result.model_dump_json()
                elif isinstance(result, (dict, list)):
                    content_str = json.dumps(result, ensure_ascii=False)
                else:
                    content_str = str(result)
                return ToolResult(success=True, content=str(content_str), exit_code=0)

            except ValidationError as e:
                return ToolResult(success=False, content=f"错误：参数不匹配大模型契约 → {str(e)}", exit_code=-400)
            except asyncio.TimeoutError:
                return ToolResult(success=False, content=f"错误：该工具执行已超时（阈值 {timeout}s）", exit_code=-255)
            except Exception as e:
                logger.exception(f"❌ 工具 {tool_call.name} 运行期间抛出业务异常")
                return ToolResult(success=False, content=f"错误：{str(e)}", exit_code=-500)