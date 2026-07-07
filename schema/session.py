import logging
import time
import copy
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Dict, Any, Optional, Set, Tuple

from .node import SessionStatus, NodeStatus, NodeRecord
from .message import BaseLLMMessage


@dataclass(frozen=True)
class ReadonlySessionView:
    """务实的不可变只读快照：在创建时直接转为纯基础数据副本。"""

    session_id: str
    max_token_budget: int
    total_tokens: int
    global_deadline: float
    is_expired: bool
    status: SessionStatus
    consensus_pool: Dict[str, Any]  # 直接提供独立副本，下游改动不影响源数据
    fingerprints: Set[str]

    def has_fingerprint(self, fp: str) -> bool:
        return fp in self.fingerprints


@dataclass
class SessionContext:
    session_id: str
    user_query: str
    status: SessionStatus = SessionStatus.INITIALIZING
    timeout_limit_seconds: float = 60.0
    global_deadline: float = field(init=False)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    max_token_budget: int = 500_000

    consensus_pool: Dict[str, Any] = field(default_factory=dict)
    global_history: List[BaseLLMMessage] = field(default_factory=list)
    root_nodes: List[NodeRecord] = field(default_factory=list)
    fingerprints: Set[str] = field(default_factory=set)
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    logger: logging.Logger = field(default=None, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger(f"Session[{self.session_id}]")
        self.global_deadline = self.start_time + self.timeout_limit_seconds

    @property
    def is_expired(self) -> bool:
        return time.time() > self.global_deadline

    def _is_terminal(self) -> bool:
        return self.status in (
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.TIMEOUT,
        )

    # ---------- 写入 API (线程安全) ----------

    def add_root_node(self, node: NodeRecord) -> None:
        with self._lock:
            self.root_nodes.append(node)

    def add_fingerprint(self, fp: str) -> None:
        with self._lock:
            self.fingerprints.add(fp)

    def update_consensus(self, key: str, value: Any) -> None:
        with self._lock:
            # 写入时进行深拷贝，确保 Session 内部数据的独立性
            self.consensus_pool[key] = copy.deepcopy(value)

    def add_token_cost(self, tokens: int) -> None:
        log_msg = None
        with self._lock:
            if self._is_terminal():
                return
            self.total_tokens += tokens
            if self.total_tokens >= self.max_token_budget:
                self.status = SessionStatus.FAILED
                self.end_time = time.time()
                self.metadata["reason"] = "budget exhausted"
                log_msg = f"Session FAILED: budget exhausted. Duration: {self.end_time - self.start_time:.3f}s"

        # 绝不在锁内部执行日志 I/O
        if log_msg:
            self.logger.warning(log_msg)

    def finalize(self, status: SessionStatus, reason: str = "") -> None:
        log_msg = None
        with self._lock:
            if self._is_terminal():
                return
            self.status = status
            self.end_time = time.time()
            self.metadata["reason"] = reason or f"session {status.name.lower()}"
            log_msg = f"Session transitioned to {status.name}, reason={self.metadata['reason']}, duration={self.end_time - self.start_time:.3f}s"

        if log_msg:
            self.logger.info(log_msg)

    # ---------- 读取/快照 API (线程安全) ----------

    def get_termination_reason(self) -> Tuple[SessionStatus, str]:
        with self._lock:
            return self.status, str(self.metadata.get("reason", ""))

    def get_readonly_view(self) -> ReadonlySessionView:
        with self._lock:
            # 仅在生成 View 的瞬间做一次深拷贝，后续下游任意读写 `.consensus_pool` 都是 O(1) 且安全的
            return ReadonlySessionView(
                session_id=self.session_id,
                max_token_budget=self.max_token_budget,
                total_tokens=self.total_tokens,
                global_deadline=self.global_deadline,
                is_expired=self.is_expired,
                status=self.status,
                consensus_pool=copy.deepcopy(self.consensus_pool),
                fingerprints=self.fingerprints.copy(),
            )
