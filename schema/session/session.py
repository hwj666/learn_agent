import time
import logging
from typing import Any, Dict, Optional, Set
from dataclasses import asdict, is_dataclass
from copy import deepcopy

from schema.context import RuntimeContext
from schema.session.broadcast import DashboardBroadcaster
from schema.session.context import StandardStepContext


class AgentSession:
    """纯粹的确定性同步状态账本大管家（组合模式版）"""

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
        self.nodes: Dict[str, dict] = {}

        # 🟢 通过【组合】注入广播组件，把 session 实例和 logger 传过去完成绑定
        self.broadcaster = DashboardBroadcaster(
            session_id=self.session_id, logger=self.logger
        )

    def step(
        self, node_id: str, metadata: Any = None, attempt_idx=0
    ) -> StandardStepContext:
        return StandardStepContext(self, node_id, metadata, attempt_idx)

    def _trigger_broadcast(self, action_type: str, node_id: str, payload: dict):
        """内部辅助函数：代理调用组合对象的 push 能力"""
        self.broadcaster.push(
            action_type=action_type,
            node_id=node_id,
            total_tokens=self.total_tokens,
            session_status=self.status,
            payload=payload,
        )

    # =====================================================================
    # 核心业务逻辑：修改完同步真理源后，通过代理函数将最新的局部快照推给组合的广播器
    # =====================================================================

    def record_step_enter(
        self, node_id: str, metadata: Any, attempt_idx: int = 0
    ) -> int:
        current_stack = RuntimeContext.get_stack()
        parent_id = current_stack[-1] if current_stack else None
        RuntimeContext.set_stack(current_stack + (node_id,))
        depth = len(current_stack) + 1

        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "node_id": node_id,
                "parent_id": parent_id,
                "status": "RUNNING",
                "attempts": [],
            }

        attempts = self.nodes[node_id]["attempts"]
        while len(attempts) <= attempt_idx:
            attempts.append({})

        cleaned_meta = (
            asdict(metadata)
            if is_dataclass(metadata)
            else getattr(metadata, "__dict__", metadata)
        )
        if not isinstance(cleaned_meta, dict):
            cleaned_meta = {"raw": str(cleaned_meta)}

        attempts[attempt_idx] = {
            "attempt_idx": attempt_idx,
            "status": "RUNNING",
            "start_time": time.time(),
            "end_time": 0.0,
            "duration_ms": 0.0,
            "tokens_consumed": 0,
            "metadata": deepcopy(cleaned_meta),
            "error": None,
        }
        self.nodes[node_id]["status"] = "RUNNING"

        # 🟢 代理通知组合的广播器
        self._trigger_broadcast("STEP_ENTER", node_id, attempts[attempt_idx])
        return depth

    def record_token_consume(
        self, node_id: str, tokens: int, attempt_idx: int = 0
    ) -> None:
        tokens = int(tokens)
        self.total_tokens += tokens
        self.logger.info(
            "💰 [Accounting] Token burned: +%d | Session Cumulative Total: %d",
            tokens,
            self.total_tokens,
        )

        if node_id in self.nodes:
            attempts = self.nodes[node_id]["attempts"]
            if attempt_idx < len(attempts) and attempts[attempt_idx]:
                attempts[attempt_idx]["tokens_consumed"] = (
                    attempts[attempt_idx].get("tokens_consumed", 0) + tokens
                )

                # 🟢 代理通知组合的广播器
                self._trigger_broadcast(
                    "TOKEN_CONSUME",
                    node_id,
                    {
                        "incremental_tokens": tokens,
                        "node_total": attempts[attempt_idx]["tokens_consumed"],
                    },
                )

    def update_metadata_stream(
        self, node_id: str, metadata: Any, attempt_idx: int = 0
    ) -> None:
        if node_id not in self.nodes:
            return
        attempts = self.nodes[node_id]["attempts"]
        if attempt_idx >= len(attempts) or not attempts[attempt_idx]:
            return

        cleaned_meta = (
            asdict(metadata)
            if is_dataclass(metadata)
            else getattr(metadata, "__dict__", metadata)
        )
        target_meta = attempts[attempt_idx]["metadata"]
        if isinstance(target_meta, dict) and isinstance(cleaned_meta, dict):
            target_meta.update(deepcopy(cleaned_meta))

            # 🟢 代理通知组合的广播器
            self._trigger_broadcast("METADATA_UPDATE", node_id, target_meta)

    def record_step_exit(self, node_id: str, attempt_idx: int = 0) -> None:
        stack = list(RuntimeContext.get_stack())
        if stack and stack[-1] == node_id:
            stack.pop()
        else:
            if node_id in stack:
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i] == node_id:
                        stack.pop(i)
                        break
        RuntimeContext.set_stack(tuple(stack))

        if node_id in self.nodes:
            attempts = self.nodes[node_id]["attempts"]
            if attempt_idx < len(attempts) and attempts[attempt_idx]:
                att = attempts[attempt_idx]
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
                    self.nodes[node_id]["status"] = "COMPLETED"

                    # 🟢 代理通知组合的广播器
                    self._trigger_broadcast("STEP_EXIT", node_id, att)

    def record_node_crashed(
        self, trigger_node_id: str, trigger_idx: int, error_msg: str
    ) -> None:
        now = time.time()
        visited_nodes: Set[str] = set()
        current_id = trigger_node_id

        while current_id is not None and current_id not in visited_nodes:
            node = self.nodes.get(current_id)
            if not node:
                break
            visited_nodes.add(current_id)
            node["status"] = "FAILED"

            if current_id == trigger_node_id:
                node["error"] = f"Direct Error: {error_msg}"
                target_idx = trigger_idx
            else:
                node["error"] = (
                    f"Cascaded Error from Child [{trigger_node_id}]: {error_msg}"
                )
                attempts = node.get("attempts", [])
                target_idx = max(0, len(attempts) - 1)

            attempts = node.get("attempts", [])
            if target_idx < len(attempts) and attempts[target_idx]:
                att = attempts[target_idx]
                if att.get("status") in ("FAILED", "COMPLETED"):
                    current_id = node.get("parent_id")
                    continue

                duration_ms = (now - att["start_time"]) * 1000.0
                att.update(
                    {
                        "status": "FAILED",
                        "error": node["error"],
                        "end_time": now,
                        "duration_ms": round(duration_ms, 2),
                    }
                )

                # 🟢 代理通知组合的广播器
                self._trigger_broadcast("NODE_CRASHED", current_id, att)

            current_id = node.get("parent_id")

    def check_budget_pure(self) -> None:
        if self.status in ("FAILED", "TIMEOUT"):
            raise RuntimeError("Execution abandoned: Global session already terminated")
        if time.time() > self.local_deadline:
            self.status = "TIMEOUT"
            raise TimeoutError("Local time budget exceeded")
        if self.total_tokens >= self.max_token_budget:
            self.status = "FAILED"
            raise RuntimeError("Token budget exhausted")

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "total_tokens": self.total_tokens,
            "max_token_budget": self.max_token_budget,
            "timeout_limit": self.timeout_limit,
            "nodes": deepcopy(self.nodes),
        }

    async def close(
        self, exc_type: Optional[type] = None, exc_val: Optional[Any] = None
    ) -> None:
        if self.status not in ("FINISHED", "CLOSED", "TIMEOUT", "FAILED"):
            if exc_type is not None:
                self.status = (
                    "TIMEOUT"
                    if issubclass(exc_type, (TimeoutError, asyncio.TimeoutError))
                    else "FAILED"
                )
                error_msg = str(exc_val) or exc_type.__name__
                for node_id, node in self.nodes.items():
                    if node.get("status") == "RUNNING":
                        self.record_node_crashed(
                            node_id,
                            max(0, len(node.get("attempts", [])) - 1),
                            f"Watchdog Aborted: {error_msg}",
                        )
            else:
                self.status = "CLOSED"

        # 🟢 优雅关闭直接调用组合对象的 close() 即可，干净清爽！
        await self.broadcaster.close()
