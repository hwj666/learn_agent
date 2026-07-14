# schema/session_types.py
import copy
import time
from enum import Enum, unique
from typing import Any
from dataclasses import dataclass, field
import logging
import asyncio
from typing import Dict, Any, Optional, List, Set, Callable
from copy import deepcopy

from schema.context import RuntimeContext


@unique
class SessionEventType(Enum):
    TOKEN_CONSUME = "session.metrics.token_consume"
    METADATA_UPDATE = "session.state.metadata_update"
    STEP_ENTER = "session.lifecycle.step_enter"
    STEP_EXIT = "session.lifecycle.step_exit"
    NODE_CRASHED = "session.lifecycle.crashed"


@dataclass
class SessionEvent:
    event_type: SessionEventType
    node_id: str
    depth: int
    attempt_idx: int
    data: Any = None
    ctx_session_id: str = "SYSTEM"
    ctx_node_id: str = "MAIN"
    timestamp: float = field(default_factory=time.time)


class StandardStepContext:
    """业界通用的非阻塞扁平同步上下文管理器"""

    def __init__(
        self, outer: "AsyncAgentSession", node_id: str, metadata: Any, attempt_idx: int
    ):
        self.outer = outer
        self.node_id = node_id
        self.metadata = metadata
        self.attempt_idx = attempt_idx
        self.depth = 0
        self._ctx_manager = None

    def __enter__(self):
        """同步管道切入拦截器：高内聚状态安全校验与拓扑图谱精准锁定"""
        # 1. 物理账本核心风控校验（超出 Token 预算在前线直接熔断爆破）
        self.outer.check_budget_pure()

        # 🟢 2. 先执行自有入栈记账逻辑：
        # 此时大管家里的栈还是干净的旧状态，record_step_enter 能够 100% 精准抓到真正的 parent_id！
        # 并且在 record 内部完成对 RuntimeContext 栈镜像的追加更新。
        self.depth = self.outer.record_step_enter(
            node_id=self.node_id, metadata=self.metadata, attempt_idx=self.attempt_idx
        )

        # 🟢 3. 最后一步：正式激活并绑定当前微观节点的日志追踪管理通道
        # 锁死当前协程的 current_node_id 标签，完美对接底层的 SessionTraceFilter
        self._ctx_manager = RuntimeContext.guard_node(self.node_id)
        self._ctx_manager.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """同步管道出口拦截器：实现资产 100% 结构化包装与大管家双向安全解构"""
        try:
            is_error = exc_type is not None
            # 🟢 修正：格式化纯文本错误堆栈，用于后续大快照及日志的高保真落盘
            error_trace = f"{exc_type.__name__}: {exc_val}" if is_error else None

            # 🟢 核心修复：100% 对齐后台处理器需要的解构资产包，
            # 彻底阻断后台因执行 event.data.get("is_error") 引发的 AttributeError 瘫痪！
            payload_package = {"is_error": is_error, "error_msg": error_trace}

            # 🚀 0阻塞秒回投递，前线不处理耗时逻辑
            self.outer.record_step_exit(
                node_id=self.node_id,
                attempt_idx=self.attempt_idx,
                # 统一打包向下透传
                is_error=is_error,
                error_msg=payload_package,
            )
        finally:
            # 🟢 核心加固：无论中间的事件入队运行是否遭遇惨烈崩溃，
            # 退出当前 with 管道的瞬间，必须无条件、强行通过大管家安全解构回收 Token！
            # 100% 免疫调用栈漂移与日志标签悬挂污染
            if hasattr(self, "_ctx_manager") and self._ctx_manager:
                self._ctx_manager.__exit__(exc_type, exc_val, exc_tb)

        return False  # 保持错误正常向上冒泡，不恶意吞掉业务异常


class AsyncAgentSession:
    """事件驱动型 Agent 会话控制器（Handler 路由加固版）"""

    def __init__(
        self,
        session_id: str,
        max_token_budget: int = 100000,
        timeout_limit: float = 15.0,
        global_deadline: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.session_id = session_id
        self.max_token_budget = max_token_budget
        self.timeout_limit = timeout_limit
        self.logger = logger or logging.getLogger("AsyncAgentSession")

        now = time.time()
        base_deadline = now + timeout_limit
        self.local_deadline = (
            min(base_deadline, global_deadline) if global_deadline else base_deadline
        )

        self.status = "RUNNING"
        self.total_tokens = 0
        self.global_fingerprints: Set[str] = set()

        self.nodes: Dict[str, dict] = {}

        # 🎯 核心优化：解耦事件类型与底层处理器函数的映射关系
        self._event_handlers: Dict[SessionEventType, Callable[[SessionEvent], None]] = {
            SessionEventType.STEP_ENTER: self._handle_step_enter,
            SessionEventType.STEP_EXIT: self._handle_step_exit,
            SessionEventType.TOKEN_CONSUME: self._handle_token_consume,
            SessionEventType.METADATA_UPDATE: self._handle_metadata_update,
            SessionEventType.NODE_CRASHED: self._handle_node_crashed,
        }

        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue()
        self._consumer_task = asyncio.create_task(self._consume_events_pipeline())

    def step(
        self, node_id: str, metadata: Any = None, attempt_idx=0
    ) -> StandardStepContext:
        return StandardStepContext(self, node_id, metadata, attempt_idx)

    def _post_event(
        self,
        event_type: SessionEventType,
        node_id: str,
        depth: int,
        attempt_idx: int,
        data: Any,
    ) -> None:
        """
        【工业级 0阻塞自闭环核心投递网关 - 完美时空固化版】
        """
        # 🌟 核心对齐：ctx_node_id 优先信任上层穿透进来的确切 node_id，防止异步上下文切换瞬间发生时空错位
        actual_ctx_node_id = node_id or RuntimeContext.get_node_id()

        evt = SessionEvent(
            event_type=event_type,
            node_id=node_id,
            depth=depth,
            attempt_idx=attempt_idx,
            data=data,
            ctx_session_id=self.session_id,
            ctx_node_id=actual_ctx_node_id,  # 🌟 使用固化后的确定节点 ID
        )

        try:
            self._event_queue.put_nowait(evt)
        except asyncio.QueueFull:
            # 这里的 self.logger 会触发 MainThreadCaptureFilter，在入队前固化日志层面的 TraceID，非常完美！
            self.logger.error(
                f"[Queue Full] Drop event {event_type} for node {node_id} to protect memory."
            )

    def update_metadata_stream(
        self, node_id: str, metadata: Any, attempt_idx: int = 0
    ) -> None:
        """
        🌟 工业级高保真状态流更新网关
        职责：在事件脱离前线的瞬间，执行原子深拷贝，100% 物理消除流式并发竞态下的键克隆畸变。
        """
        depth = len(RuntimeContext.get_stack())
        try:
            payload = copy.deepcopy(metadata)
        except Exception:
            # 防御极端非标准对象，如果无法深拷贝，采用安全的字典浅拷贝防线降级兜底
            payload = dict(metadata) if isinstance(metadata, dict) else metadata

        # 3. 稳稳投递到 0 阻塞自闭环核心网关
        self._post_event(
            event_type=SessionEventType.METADATA_UPDATE,
            node_id=node_id,
            depth=depth,
            attempt_idx=attempt_idx,
            data=payload,  # 🌟 投递完全物理独立的纯净资产包
        )

    def record_step_enter(
        self, node_id: str, metadata: Any, attempt_idx: int = 0
    ) -> int:
        """无条件保护入栈，完美还原回路与递归调用图谱（Tuple 零拷贝、纯净对齐版）"""
        # 1. 🌟 安全读取当前栈（此时大管家返回的是不可变元组 Tuple[str, ...]）
        current_tuple_stack = RuntimeContext.get_stack()

        # 2. 100% 精准、零干扰地抓到真正的 parent_id（元组同样支持索引 [-1]）
        parent_id = current_tuple_stack[-1] if current_tuple_stack else None
        new_tuple_stack = current_tuple_stack + (node_id,)
        depth = len(new_tuple_stack)

        # 4. 🌟 将这个全新的、天然并发安全的元组资产同步塞回底层 ContextVar
        RuntimeContext.set_stack(new_tuple_stack)

        payload_package = {
            "user_metadata": metadata,  # 投递原始指针，不阻塞前线
            "parent_id": parent_id,
        }

        # 5. 稳稳投递至 0 阻塞自闭环核心网关
        self._post_event(
            event_type=SessionEventType.STEP_ENTER,
            node_id=node_id,
            depth=depth,
            attempt_idx=attempt_idx,
            data=payload_package,
        )
        return depth

    def record_step_exit(
        self,
        node_id: str,
        attempt_idx: int = 0,
        is_error: bool = False,
        error_msg: Optional[
            dict
        ] = None,  # 💡 保持与 StandardStepContext 传入的包装字典一致
    ) -> None:
        """
        安全执行物理出栈，并根据错误状态分流投递核心事件总线
        🌟 终极进化版：全 Tuple 原生驱动，极致零拷贝，免疫异常瘫痪
        """
        # 1. 安全读取当前只读元组栈
        current_tuple_stack = RuntimeContext.get_stack()

        # 2. 🌟 核心修复：将元组转化为局部列表，完美承载你优秀的递归回路裁剪算法
        # 此时的 stack 是纯粹的局部可变副本，随用随弃，100% 隔离高并发污染
        stack = list(current_tuple_stack)

        # 3. 完美保留你原有的全自动链路裁剪容错逻辑
        if stack and stack[-1] == node_id:
            stack.pop()
        else:
            if node_id in stack:
                # 倒序检索，精准剥离错位悬挂的微观拓扑节点
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i] == node_id:
                        stack.pop(i)
                        break

        # 4. 🌟 优雅桥接：调用刚才定义的 set_stack 写入网关，强行收敛写回底层
        RuntimeContext.set_stack(stack)

        # 5. 还原计算退出时的绝对时空深度（depth 恢复至进站前的深度 + 1，保持记账一致）
        depth = len(stack) + 1

        # 精准判定事件类型
        etype = (
            SessionEventType.NODE_CRASHED if is_error else SessionEventType.STEP_EXIT
        )

        # 6. 稳稳投递至 0 阻塞自闭环核心网关
        self._post_event(
            event_type=etype,
            node_id=node_id,
            depth=depth,
            attempt_idx=attempt_idx,
            data=error_msg if is_error else None,
        )

    def _handle_token_consume(self, event: SessionEvent) -> None:
        node_id = event.node_id
        idx = event.attempt_idx
        tokens = int(event.data)

        self.total_tokens += tokens

        self.logger.info(
            "💰 [Accounting] Token burned: +%d | Session Cumulative Total: %d",
            tokens,
            self.total_tokens,
        )

        if node_id in self.nodes:
            attempts = self.nodes[node_id]["attempts"]

            # 🛡️ 安全加固：延续你优秀的防越界防崩溃设计，如果事件乱序触发，自动补全占位
            while len(attempts) <= idx:
                attempts.append(
                    {
                        "attempt_idx": len(attempts),
                        "status": "UNKNOWN",
                        "tokens_consumed": 0,
                        "metadata": {},
                    }
                )

            if attempts[idx]:
                current_consumed = attempts[idx].get("tokens_consumed", 0)
                attempts[idx]["tokens_consumed"] = current_consumed + tokens

    def check_and_record_fingerprint(self, fp: str) -> bool:
        if fp in self.global_fingerprints:
            self.status = "FAILED"
            self.logger.log(
                f"⚠️ [LOOP_BLOCKED] Fingerprint [{fp[:8]}] matched.",
                level=logging.WARNING,
            )
            return True
        self.global_fingerprints.add(fp)
        return False

    def check_budget_pure(self) -> None:
        if self.status in ("FAILED", "TIMEOUT"):
            raise RuntimeError("Execution abandoned: Global session already terminated")
        if time.time() > self.local_deadline:
            self.status = "TIMEOUT"
            raise TimeoutError("Local time budget exceeded")
        if self.total_tokens >= self.max_token_budget:
            self.status = "FAILED"
            raise RuntimeError("Token budget exhausted")

    async def _consume_events_pipeline(self) -> None:
        while True:
            got_item = False
            try:
                event = await self._event_queue.get()
                got_item = True

                handler = self._event_handlers.get(event.event_type)
                if handler:
                    # 🌟 统一开启 flat_mode=True 模式！
                    # 大管家在底层会无感切换为单层物理覆盖清洗，消灭所有的 reset 回滚开销与 Token 堆积
                    with (
                        RuntimeContext.guard_session(
                            event.ctx_session_id, flat_mode=True
                        ),
                        RuntimeContext.guard_node(event.ctx_node_id, flat_mode=True),
                    ):
                        handler(event)  # 安全执行回调处理

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "💥 Event pipeline internal anomaly: %s", e, exc_info=True
                )
            finally:
                if got_item:
                    self._event_queue.task_done()

    def _handle_step_enter(self, event: SessionEvent) -> None:
        node_id = event.node_id
        idx = event.attempt_idx

        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "node_id": node_id,
                "parent_id": event.data.get("parent_id"),
                "status": "RUNNING",
                "attempts": [],
            }
        else:
            if not self.nodes[node_id].get("parent_id") and event.data.get("parent_id"):
                self.nodes[node_id]["parent_id"] = event.data.get("parent_id")

        while len(self.nodes[node_id]["attempts"]) <= idx:
            self.nodes[node_id]["attempts"].append({})

        self.nodes[node_id]["attempts"][idx] = {
            "attempt_idx": idx,
            "status": "RUNNING",
            "start_time": event.timestamp,
            "duration_ms": 0.0,
            "tokens_consumed": 0,
            "metadata": event.data.get("user_metadata"),
            "error": None,
        }
        self.nodes[node_id]["status"] = "RUNNING"

    def _handle_step_exit(self, event: SessionEvent) -> None:
        """后台专职处理器：负责在隔离空间内完成高精度耗时回填与状态安全关闭（100%免疫空指针/乱序）"""
        node_id = event.node_id
        idx = event.attempt_idx

        if node_id not in self.nodes:
            return

        attempts = self.nodes[node_id]["attempts"]

        while len(attempts) <= idx:
            attempts.append(
                {
                    "attempt_idx": len(attempts),
                    "status": "UNKNOWN",
                    "tokens_consumed": 0,
                    "metadata": {},
                }
            )

        att = attempts[idx]

        # 🟢 2. 延续你最优秀的防覆盖核心防线：只有当前槽位是 RUNNING 或 UNKNOWN 乱序时才允许回填
        if att.get("status") in ("RUNNING", "UNKNOWN"):
            # 🟢 3. 防御性安全解构网关：完美兼容 dict 投递以及单指针 None 投递
            raw_data = event.data
            is_error = False
            error_msg = None

            if isinstance(raw_data, dict):
                is_error = raw_data.get("is_error", False)
                error_msg = raw_data.get("error_msg")
            elif event.event_type == SessionEventType.NODE_CRASHED:
                # 兼容直接透传错误堆栈字符串的分流机制
                is_error = True
                error_msg = str(raw_data) if raw_data else "Unknown execution error"

            # 🟢 4. 你的高精度时钟对齐闭环（秒转毫秒，确保 start_time 缺失时有安全兜底）
            start_time = att.get("start_time", event.timestamp)
            duration_ms = (event.timestamp - start_time) * 1000.0

            # 🚀 在 trace_node 护航下，后台清洗完毕瞬间打印一条带有精准 [SID:xxx] [NODE:xxx] 标签的审计日志
            if is_error:
                self.logger.error(
                    "🔥 [Alert] Node [%s] (Attempt #%d) failed in %.2f ms.",
                    node_id,
                    idx,
                    duration_ms,
                )
                final_status = "FAILED"
            else:
                self.logger.info(
                    "✅ [Topology] Node [%s] (Attempt #%d) completed in %.2f ms.",
                    node_id,
                    idx,
                    duration_ms,
                )
                final_status = "COMPLETED"

            # 5. 结构化原子固化回填
            att["status"] = final_status
            att["duration_ms"] = round(duration_ms, 2)
            att["error"] = error_msg
            if "end_time" not in att or att["end_time"] == 0:
                att["end_time"] = event.timestamp

            self.nodes[node_id]["status"] = final_status

    def _handle_token_consume(self, event: SessionEvent) -> None:
        # 1. 累加全局总账
        self.total_tokens += event.data

        # 2. 累加微观明细
        node_id = event.node_id
        idx = event.attempt_idx

        if node_id in self.nodes:
            attempts = self.nodes[node_id]["attempts"]
            # 🛡️ 安全加固：防止因乱序导致 attempt 数组未扩容到位
            while len(attempts) <= idx:
                attempts.append(
                    {
                        "attempt_idx": len(attempts),
                        "status": "UNKNOWN",
                        "tokens_consumed": 0,
                    }
                )

            if attempts[idx]:
                attempts[idx]["tokens_consumed"] = (
                    attempts[idx].get("tokens_consumed", 0) + event.data
                )

    def _handle_metadata_update(self, event: SessionEvent) -> None:
        """后台专职处理器：负责在隔离空间内完成深拷贝与纯净快照合并"""
        node_id = event.node_id
        idx = event.attempt_idx
        raw_meta = event.data

        if node_id not in self.nodes:
            return
        try:
            if hasattr(raw_meta, "__dataclass_fields__"):
                from dataclasses import asdict

                # 转换为纯 dict 并深度拷贝，彻底切断前线和后方的内存指针联系
                cleaned_data = deepcopy(asdict(raw_meta))
            elif hasattr(raw_meta, "__dict__"):
                cleaned_data = deepcopy(raw_meta.__dict__)
            else:
                cleaned_data = deepcopy(raw_meta)
        except Exception as clean_err:
            cleaned_data = {"error_parsing_meta": str(clean_err), "raw": str(raw_meta)}

        self.logger.info(
            "📊 [Snapshot] Metadata streamed and sanitized into node attempts slot [%d].",
            idx,
        )

        attempts = self.nodes[node_id]["attempts"]

        # 🛡️ 安全加固：延续你的防越界防崩溃设计
        while len(attempts) <= idx:
            attempts.append(
                {
                    "attempt_idx": len(attempts),
                    "status": "UNKNOWN",
                    "tokens_consumed": 0,
                    "metadata": {},  # 🟢 预留初始化字典，确保增量 update 时必为 dict
                }
            )

        if attempts[idx]:
            # 🟢 此时 cleaned_data 已经是100%纯净独立的非强类型数据，安全进行增量更新或覆盖
            if isinstance(attempts[idx].get("metadata"), dict) and isinstance(
                cleaned_data, dict
            ):
                attempts[idx]["metadata"].update(cleaned_data)
            else:
                attempts[idx]["metadata"] = cleaned_data

    def _handle_node_crashed(self, event: SessionEvent) -> None:
        """
        【生产级防卡死异常冒泡处理器 - 严格垂直血缘完全体】
        🌟 优化：彻底肃清横向兄弟污染、杜绝看门狗级联耗时通货膨胀。
        """
        trigger_node_id = event.node_id
        trigger_idx = event.attempt_idx

        # 1. 安全数据提取网关
        raw_data = event.data
        if isinstance(raw_data, dict):
            error_msg = (
                raw_data.get("error_msg") or raw_data.get("error") or "Unknown crashed"
            )
        else:
            error_msg = str(raw_data) if raw_data else "Unknown crashed"

        # 🛡️ 核心防线：彻底切断图回路中的死循环可能
        visited_nodes: Set[str] = set()

        # 从直系触发源开始单向垂直向上爬
        current_id = trigger_node_id

        while current_id is not None and current_id not in visited_nodes:
            node = self.nodes.get(current_id)
            if not node:
                break

            visited_nodes.add(current_id)
            node["status"] = "FAILED"

            # 2. 🌟 区分【直系触发源】与【垂直祖先】的报错文案，绝对不污染不相关的兄弟
            if current_id == trigger_node_id:
                self.logger.error(
                    "🔥 [Alert] Node [%s] (Attempt #%d) direct crashed: %s",
                    current_id,
                    trigger_idx,
                    error_msg,
                )
                node["error"] = f"Direct Error: {error_msg}"
                # 触发源节点精准使用前线传过来的 idx
                target_idx = trigger_idx
            else:
                self.logger.warning(
                    "⚠️ [Cascade] Tail-Whip! Cascading error to parent node [%s] from child [%s]",
                    current_id,
                    trigger_node_id,
                )
                node["error"] = (
                    f"Cascaded Error from Child [{trigger_node_id}]: {error_msg}"
                )

                # 🌟 核心解锁点一：如果是垂直上游的祖先节点，不能无脑用儿子的 target_idx！
                # 我们应当去检索祖先自己当前哪个槽位是处于运行中（RUNNING）或者是最新的，精准对其精准动刀
                attempts = node.get("attempts", [])
                target_idx = len(attempts) - 1 if attempts else 0
                if target_idx < 0:
                    target_idx = 0

            # 3. 稳健槽位对齐
            attempts = node.get("attempts", [])
            while len(attempts) <= target_idx:
                attempts.append(
                    {
                        "attempt_idx": len(attempts),
                        "status": "UNKNOWN",
                        "tokens_consumed": 0,
                        "metadata": {},
                    }
                )

            att = attempts[target_idx]

            # 🌟 核心解锁点二：时空防污染网关
            # 如果这个槽位已经是 FAILED 或 COMPLETED（说明是早就结束的大哥哥，比如 Turn_1）
            # 那么【绝对禁止】改写它的耗时和报错！直接跳过它，维护历史账本的高保真度！
            if att.get("status") in ("FAILED", "COMPLETED"):
                # 如果是大哥哥节点，我们只负责顺着它的 parent_id 继续往上爬，绝对不对它内部的数据泼脏水
                current_id = node.get("parent_id")
                continue

            # 只有处于活动状态的槽位才允许回填
            att["status"] = "FAILED"
            att["error"] = node["error"]

            # 4. 🌟 彻底消灭 35秒大锅饭时间通胀
            start_time = att.get("start_time")
            if start_time is None or start_time == 0:
                # 乱序导致的缺失，直接记为 0 毫秒的原子闪击事件，绝不吃大锅饭
                duration_ms = 0.0
                att["start_time"] = event.timestamp
            else:
                # 正常的运行中上游节点，高精度客观结算它被逼崩溃时的真实耗时
                duration_ms = (event.timestamp - start_time) * 1000.0

            att["duration_ms"] = round(duration_ms, 2)
            if "end_time" not in att or att["end_time"] == 0:
                att["end_time"] = event.timestamp

            # 5. 🌟 绝对纯净的垂直单向向上追溯，绝不横向横穿
            current_id = node.get("parent_id")

    async def close(
        self, exc_type: Optional[type] = None, exc_val: Optional[Any] = None
    ) -> None:
        """
        【标准异步关闭接口 - 终极对齐版】
        1. 若传入异常，触发看门狗强杀逻辑，近乎零延迟回写所有正在运行的悬挂节点为 FAILED。
        2. 等待队列冲刷完毕后，彻底安全地撤销常驻消费任务。
        """
        if self.status not in ("FINISHED", "CLOSED"):
            # 🎯 自动化三：如果是因为外部异常/超时强杀，立即批量修正内存中所有悬挂节点的状态
            if exc_type is not None:
                self.status = (
                    "TIMEOUT"
                    if issubclass(exc_type, (TimeoutError, asyncio.TimeoutError))
                    else "FAILED"
                )
                error_msg = str(exc_val) or exc_type.__name__

                # 遍历内存账本，强杀所有还在 RUNNING 的节点
                for node_id, node in self.nodes.items():
                    if node.get("status") == "RUNNING":
                        node["status"] = "FAILED"
                        node["error"] = f"Watchdog Aborted: {error_msg}"

                        # 连带修正其微观重试槽位的状态
                        for att in node.get("attempts", []):
                            if att and att.get("status") == "RUNNING":
                                att["status"] = "FAILED"
                                att["error"] = f"Watchdog Aborted: {error_msg}"
            else:
                if self.status == "RUNNING":
                    self.status = "FINISHED"

            # 优雅清空、冲刷队列中的残留追踪数据
            await self._event_queue.join()

            # 强力终止后台常驻 Task
            if not self._consumer_task.done():
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    pass

            # 如果之前不是超时或失败，则正常标记为关闭
            if self.status not in ("TIMEOUT", "FAILED"):
                self.status = "CLOSED"

    def to_snapshot(self) -> Dict[str, Any]:
        """
        【工业级读写分离纯净大快照】
        特性：万能类型擦除、指针彻底解耦、100% 免疫 JSON 序列化崩溃。
        """
        from copy import deepcopy
        from dataclasses import asdict, is_dataclass

        # 1. 提取宏观元数据骨架
        snapshot = {
            "session_id": self.session_id,
            "status": self.status,
            "total_tokens": self.total_tokens,
            "max_token_budget": self.max_token_budget,
            "timeout_limit": self.timeout_limit,
            # 采用深拷贝，防止外部拿到快照后恶意修改，破坏内部状态树的客观真实性
            "nodes": deepcopy(self.nodes),
        }

        # 🟢 2. 核心防御天网：递归清洗器
        def sanitize_node(obj: Any) -> Any:
            """递归擦除所有强类型 Dataclass 痕迹，转换为纯净的基础 JSON 数据类型"""
            if is_dataclass(obj):
                # 如果漏网了 Dataclass 对象，瞬间剥离外壳，转为纯净 dict
                return sanitize_node(asdict(obj))
            elif isinstance(obj, dict):
                # 递归清洗字典的每一个 Value
                return {k: sanitize_node(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                # 递归清洗列表的每一个元素
                return [sanitize_node(item) for item in obj]
            elif hasattr(obj, "__dict__") and not isinstance(
                obj, (str, int, float, bool, type(None))
            ):
                # 兼容普通第三方类对象的反射降级
                return sanitize_node(obj.__dict__)
            else:
                # 基础原生类型直接放行
                return obj

        # 🟢 3. 瞬间洗涤整棵拓扑树，完成全线彻底的读写分离
        cleaned_snapshot = sanitize_node(snapshot)

        self.logger.info(
            "📊 [Audit] Pure structural OTel-grade snapshot generated successfully."
        )
        return cleaned_snapshot
