import asyncio
import time
import logging
from typing import Any, Dict, Optional, Set
from dataclasses import asdict, is_dataclass
from copy import deepcopy

from pydantic import BaseModel
from .broadcast import TelemetryEventPublisher, BroadcastActionType
from .span import AgentSpan, AgentSpanRecord, SessionStatus


class AgentTracker:
    """👑 工业级全异步大管家：内置非阻塞协程锁与资产熔断防御"""

    def __init__(
        self,
        max_token_budget: int = 100000,
        timeout_limit: float = 15.0,
        global_deadline: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.max_token_budget = max_token_budget
        self.timeout_limit = timeout_limit
        self.logger = logger or logging.getLogger("AgentTracker")

        now = time.time()
        base_deadline = now + timeout_limit
        self.local_deadline = (
            min(base_deadline, global_deadline) if global_deadline else base_deadline
        )

        self.total_tokens = 0
        self.status = SessionStatus.RUNNING
        self.global_fingerprints: Set[str] = set()
        self.nodes: Dict[str, AgentSpanRecord] = {}

        self.broadcaster = TelemetryEventPublisher(logger=self.logger)
        self._ledger_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

    async def _trigger_broadcast(
        self, action_type: BroadcastActionType, span: AgentSpan, payload: dict
    ) -> None:
        """触发广播，确保在锁外执行 I/O"""
        async with self._ledger_lock:
            snapshot = self.nodes.get(span.span_id)
            parent_span_id = (
                snapshot.parent_span_id if snapshot else span.parent_span_id
            )

        await self.broadcaster.push(
            span_id=span.span_id,
            action_type=action_type,
            session_id=span.session_id,
            trace_id=span.trace_id,
            span_name=span.span_name,
            attempt_idx=span.attempt_idx,
            parent_span_id=parent_span_id,
            total_tokens=self.total_tokens,
            session_status=self.status.value,
            payload=payload,
        )

    async def check_budget_pure(self, span: AgentSpan) -> None:
        """纯计算预算复核，必须在锁内调用"""
        if self.status != SessionStatus.RUNNING:
            raise RuntimeError(
                f"Execution abandoned: Session terminated ({self.status})"
            )

        if time.time() > self.local_deadline:
            self.status = SessionStatus.TIMEOUT
            await self._trigger_broadcast(
                BroadcastActionType.SESSION_TIMEOUT,
                span,
                {"message": "Session expired!", "deadline": self.local_deadline},
            )
            raise TimeoutError("Local time budget exceeded")

    async def record_step_enter(self, span: AgentSpan, metadata: Any = None) -> None:
        """记录步骤进入，包含预算检查"""
        async with self._ledger_lock:
            await self.check_budget_pure(span)
            cleaned_meta = self._clean_metadata(metadata)
            snapshot = AgentSpanRecord.from_span(span, metadata=cleaned_meta)
            self.nodes[span.span_id] = snapshot
            payload = {"depth": snapshot.depth, "metadata": cleaned_meta}

        await self._trigger_broadcast(BroadcastActionType.STEP_ENTER, span, payload)

    async def record_step_exit(self, span: AgentSpan) -> None:
        """记录步骤正常退出"""
        payload = None
        async with self._ledger_lock:
            if span.span_id not in self.nodes:
                self.logger.error(f"❌ [Ledger] 账本严重失联节点: {span.span_name}")
                return

            snapshot = self.nodes[span.span_id]
            if snapshot.status != SessionStatus.RUNNING.value:
                return

            snapshot.mark_completed()
            payload = {
                "status": snapshot.status,
                "duration_ms": snapshot.duration_ms,
                "tokens_consumed": snapshot.tokens_consumed,
            }

        if payload:
            await self._trigger_broadcast(BroadcastActionType.STEP_EXIT, span, payload)

    async def record_node_crashed(self, span: AgentSpan, error_msg: str) -> None:
        """记录节点崩溃"""
        now = time.time()
        payload = None

        async with self._ledger_lock:
            if self.status != SessionStatus.RUNNING:
                return

            if span.span_id not in self.nodes:
                self.nodes[span.span_id] = AgentSpanRecord.from_span(
                    span, start_time=now, metadata={"raw": "Emergency recovered record"}
                )

            snapshot = self.nodes[span.span_id]
            if snapshot.status != SessionStatus.RUNNING.value:
                return

            snapshot.mark_crashed(error_msg)
            payload = {
                "status": snapshot.status,
                "duration_ms": snapshot.duration_ms,
                "error": snapshot.error,
            }

        if payload:
            await self._trigger_broadcast(BroadcastActionType.STEP_CRASH, span, payload)

    async def record_node_cancelled(self, span: AgentSpan) -> None:
        """记录节点被取消"""
        payload = None
        async with self._ledger_lock:
            if span.span_id not in self.nodes:
                return

            snapshot = self.nodes[span.span_id]
            if snapshot.status != SessionStatus.RUNNING.value:
                return

            snapshot.mark_cancelled()
            payload = {
                "status": snapshot.status,
                "duration_ms": snapshot.duration_ms,
            }

        if payload:
            await self._trigger_broadcast(BroadcastActionType.STEP_CRASH, span, payload)

    async def record_token_consume(self, span: AgentSpan, tokens: int) -> None:
        """记录 Token 消耗，包含预算检查"""
        tokens = int(tokens)
        payload = {}

        async with self._ledger_lock:
            # 1. 先检查预算（TOCTOU 防护）
            if self.total_tokens + tokens > self.max_token_budget:
                self.status = SessionStatus.FAILED
                await self._trigger_broadcast(
                    BroadcastActionType.SESSION_BUDGET_EXHAUSTED,
                    span,
                    {
                        "message": "Token budget exhausted!",
                        "requested": tokens,
                        "current_total": self.total_tokens,
                        "limit": self.max_token_budget,
                    },
                )
                raise RuntimeError("Token budget exhausted")

            # 2. 再更新账本
            self.total_tokens += tokens
            if span.span_id in self.nodes:
                snapshot = self.nodes[span.span_id]
                snapshot.accumulate_tokens(tokens)
                payload = {
                    "incremental_tokens": tokens,
                    "node_total": snapshot.tokens_consumed,
                }

        self.logger.info(
            "💰 [Accounting] Token burned: +%d | Total: %d", tokens, self.total_tokens
        )
        await self._trigger_broadcast(BroadcastActionType.TOKEN_CONSUME, span, payload)

    async def update_metadata_stream(self, span: AgentSpan, metadata: Any) -> None:
        """更新元数据流，支持深度合并"""
        cleaned_meta = self._clean_metadata(metadata)
        if not cleaned_meta:
            return

        async with self._ledger_lock:
            if span.span_id not in self.nodes:
                return
            snapshot = self.nodes[span.span_id]

            if isinstance(snapshot.metadata, dict) and isinstance(cleaned_meta, dict):
                self._deep_merge_dict(snapshot.metadata, cleaned_meta)
            else:
                snapshot.metadata = cleaned_meta

        await self._trigger_broadcast(
            BroadcastActionType.METADATA_STREAM_UPDATE,
            span,
            {"metadata_delta": cleaned_meta},
        )

    def _deep_merge_dict(
        self, target: dict, source: dict, max_depth: int = 10, current_depth: int = 0
    ) -> None:
        """深度合并字典，防止递归过深"""
        if current_depth >= max_depth:
            target.update(source)
            return

        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge_dict(target[key], value, max_depth, current_depth + 1)
            else:
                target[key] = deepcopy(value)

    async def close(
        self,
        exc_type: Optional[type[BaseException]] = None,
        exc_val: Optional[BaseException | str] = None,
    ) -> None:
        """优雅关闭 Tracker"""
        async with self._ledger_lock:
            if self.status != SessionStatus.RUNNING:
                # 已经处于终态，不再更改
                pass
            elif exc_type:
                self.status = SessionStatus.CRASHED
                self.logger.error(
                    f"❌ [Session Close] 会话因异常不幸终结! 原因: {exc_val}"
                )
            else:
                self.status = SessionStatus.COMPLETED
                self.logger.info("🏁 [Session Close] 会话生命周期正常圆满终结。")

        self._shutdown_event.set()

        if hasattr(self, "broadcaster") and self.broadcaster:
            try:
                await self.broadcaster.close()
                self.logger.info("🔒 [Session Close] 大管家账本完全优雅停机成功。")
            except Exception as e:
                self.logger.error(
                    f"❌ [Session Close] 广播管道关闭失败: {e}", exc_info=True
                )

    def _clean_metadata(self, metadata: Any) -> dict:
        """清洗元数据，确保可序列化"""
        if metadata is None:
            return {}

        if isinstance(metadata, BaseModel):
            cleaned = metadata.model_dump()
        elif is_dataclass(metadata):
            cleaned = asdict(metadata)
        elif hasattr(metadata, "__dict__"):
            cleaned = metadata.__dict__.copy()
        elif isinstance(metadata, dict):
            cleaned = metadata.copy()
        else:
            cleaned = {"raw": str(metadata)}

        try:
            return deepcopy(cleaned)
        except Exception:
            return {k: str(v) for k, v in cleaned.items()}

    async def __aenter__(self) -> "AgentTracker":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self.close(exc_type=exc_type, exc_val=exc_val)
        return False
