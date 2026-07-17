import time
import logging
import asyncio
from typing import Any, Dict, Optional, Set, Tuple
from dataclasses import asdict, is_dataclass
from copy import deepcopy

from schema.session.runtime import RuntimeContext
from schema.session.broadcast import DashboardBroadcaster
from schema.session.context import StandardStepContext


class AgentSession:
    """扁平流式状态账本大管家：后端随发随丢，100% 只读上下文，专注于账本裁决与熔断"""

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
        self.logger = logger or logging.getLogger("AgentSession")

        now = time.time()
        base_deadline = now + timeout_limit
        self.local_deadline = (
            min(base_deadline, global_deadline) if global_deadline else base_deadline
        )

        self.status = "RUNNING"
        self.total_tokens = 0
        self.global_fingerprints: Set[str] = set()
        self.nodes: Dict[Tuple[str, int], dict] = {}

        self.broadcaster = DashboardBroadcaster(
            session_id=self.session_id, logger=self.logger
        )

    def step(
        self, node_id: str, metadata: Any = None, attempt_idx: int = 0
    ) -> StandardStepContext:
        return StandardStepContext(self, node_id, metadata, attempt_idx)

    def _trigger_broadcast(
        self, action_type: str, node_id: str, attempt_idx: int, payload: dict
    ):
        parent_id = self.nodes.get((node_id, attempt_idx), {}).get("parent_id")
        self.broadcaster.push(
            action_type=action_type,
            node_id=node_id,
            attempt_idx=attempt_idx,
            parent_id=parent_id,
            total_tokens=self.total_tokens,
            session_status=self.status,
            payload=payload,
        )

    def check_budget_pure(self) -> None:
        """🔒 前线熔断爆破：严格状态锁定，0延迟保护钱包并通知大屏"""
        if self.status in ("FAILED", "TIMEOUT"):
            raise RuntimeError(
                f"Execution abandoned: Global session already terminated ({self.status})"
            )

        if time.time() > self.local_deadline:
            self.status = "TIMEOUT"
            self._trigger_broadcast(
                action_type="SESSION_TIMEOUT",
                node_id="SYSTEM",
                attempt_idx=0,
                payload={
                    "message": "Session expired local deadline!",
                    "deadline": self.local_deadline,
                },
            )
            raise TimeoutError("Local time budget exceeded")

        if self.total_tokens >= self.max_token_budget:
            self.status = "FAILED"
            self._trigger_broadcast(
                action_type="SESSION_BUDGET_EXHAUSTED",
                node_id="SYSTEM",
                attempt_idx=0,
                payload={
                    "message": "Token budget exhausted!",
                    "max_budget": self.max_token_budget,
                    "consumed": self.total_tokens,
                },
            )
            raise RuntimeError("Token budget exhausted")

    # =====================================================================
    # 核心做账逻辑：随发随丢的高效流式状态同步（纯只读上下文）
    # =====================================================================

    def record_step_enter(
        self, node_id: str, metadata: Any, attempt_idx: int = 0
    ) -> int:
        self.check_budget_pure()

        # 🟢 采纳建议：大管家只读不写！准确拿到由 StandardStepContext 刚刚压入的只读栈快照
        current_stack = RuntimeContext.get_stack()
        # 因为当前 node_id 已经在栈顶，它的 parent_id 就是它的上一个元素
        parent_id = current_stack[-2] if len(current_stack) >= 2 else None
        depth = len(current_stack)

        if metadata is None:
            cleaned_meta = {}
        elif is_dataclass(metadata):
            cleaned_meta = asdict(metadata)
        else:
            cleaned_meta = getattr(metadata, "__dict__", metadata)

        if not isinstance(cleaned_meta, dict):
            cleaned_meta = {"raw": str(cleaned_meta)}

        payload = {
            "node_id": node_id,
            "parent_id": parent_id,
            "attempt_idx": attempt_idx,
            "status": "RUNNING",
            "start_time": time.time(),
            "end_time": 0.0,
            "duration_ms": 0.0,
            "tokens_consumed": 0,
            "metadata": deepcopy(cleaned_meta),
            "error": None,
        }

        self.nodes[(node_id, attempt_idx)] = payload
        self._trigger_broadcast("STEP_ENTER", node_id, attempt_idx, payload)
        return depth

    def record_token_consume(
        self, node_id: str, tokens: int, attempt_idx: int = 0
    ) -> None:
        tokens = int(tokens)
        self.total_tokens += tokens
        self.logger.info(
            "💰 [Accounting] Token burned: +%d | Total: %d", tokens, self.total_tokens
        )

        self.check_budget_pure()

        cache_key = (node_id, attempt_idx)
        if cache_key in self.nodes:
            self.nodes[cache_key]["tokens_consumed"] += tokens
            self._trigger_broadcast(
                "TOKEN_CONSUME",
                node_id,
                attempt_idx,
                {
                    "attempt_idx": attempt_idx,
                    "incremental_tokens": tokens,
                    "node_total": self.nodes[cache_key]["tokens_consumed"],
                },
            )

    def update_metadata_stream(
        self, node_id: str, metadata: Any, attempt_idx: int = 0
    ) -> None:
        cache_key = (node_id, attempt_idx)
        if cache_key not in self.nodes:
            return

        if is_dataclass(metadata):
            cleaned_meta = asdict(metadata)
        else:
            cleaned_meta = getattr(metadata, "__dict__", metadata)

        target_meta = self.nodes[cache_key]["metadata"]
        if isinstance(target_meta, dict) and isinstance(cleaned_meta, dict):
            target_meta.update(deepcopy(cleaned_meta))
            self._trigger_broadcast(
                "METADATA_UPDATE",
                node_id,
                attempt_idx,
                {"attempt_idx": attempt_idx, "metadata": target_meta},
            )

    def record_step_exit(self, node_id: str, attempt_idx: int = 0) -> None:
        """🟢 采纳建议：干净简化的出栈结算，绝不插手 RuntimeContext 修改"""
        cache_key = (node_id, attempt_idx)
        if cache_key in self.nodes:
            att = self.nodes[cache_key]
            if att.get("status") == "RUNNING":
                now = time.time()
                duration_ms = (now - att["start_time"]) * 1000.0
                att.update(
                    {
                        "status": "COMPLETED",
                        "end_time": now,
                        "duration_ms": round(duration_ms, 2),
                    }
                )
                self._trigger_broadcast("STEP_EXIT", node_id, attempt_idx, att)

    def record_node_crashed(
        self, trigger_node_id: str, trigger_idx: int, error_msg: str
    ) -> None:
        """🟢 采纳建议：大管家不碰栈，只专注于修改目标重试周期的故障快照"""
        now = time.time()
        cache_key = (trigger_node_id, trigger_idx)

        if cache_key in self.nodes:
            att = self.nodes[cache_key]
            if att.get("status") == "RUNNING":
                duration_ms = (now - att["start_time"]) * 1000.0
                att.update(
                    {
                        "status": "FAILED",
                        "end_time": now,
                        "duration_ms": round(duration_ms, 2),
                        "error": error_msg,
                    }
                )
                self._trigger_broadcast(
                    "NODE_CRASHED", trigger_node_id, trigger_idx, att
                )

    # =====================================================================
    # 会话终结网关：纯同步、零阻塞的后台分发模式
    # =====================================================================
    async def close(self) -> None:
        """
        优雅关闭当前会话的大管家账本
        核心任务：全面终止并安全冲刷内部组合的异步广播大管道，100% 确保数据发送完整
        """
        self.logger.info(
            "⏳ [Session Control] Closing AgentSession %s. Final status: %s | Total Tokens: %d",
            self.session_id,
            self.status,
            self.total_tokens,
        )

        # 1. 锁定状态，防止关闭过程中还有新的异步刺入修改
        if self.status == "RUNNING":
            self.status = "COMPLETED"

        # 2. 严格等待广播通道将积压的“大结局”数据通过网络冲刷干净
        try:
            await self.broadcaster.close()
        except Exception as e:
            self.logger.error(
                f"💥 Error occurred while closing broadcaster for session {self.session_id}: {e}",
                exc_info=True,
            )

        self.logger.info(
            "✅ [Session Control] AgentSession %s closed cleanly.", self.session_id
        )
