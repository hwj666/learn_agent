import time
import logging
from threading import RLock
from typing import Dict, Any, Optional, List, Set, TypedDict
from copy import deepcopy
from collections import deque
from schema.enums import SessionStatus, NodeStatus


class NodeDict(TypedDict):
    """通过 TypedDict 规范原生的 dict 字段，开发期享受完美的 Key 智能补全"""

    node_id: str
    parent_id: Optional[str]
    status: NodeStatus
    start_time: float
    duration_ms: float
    metadata: Any
    error: Optional[str]


class AgentStepContext:
    """顶级扁平化步骤生命周期上下文管理器。"""

    __slots__ = ("outer", "node_id", "metadata", "start_time", "_landed_normally")

    def __init__(self, outer: "AgentSession", node_id: str, metadata: Any):
        self.outer = outer  # 💡 字符串前向引用提示，物理上无需导入 AgentSession
        self.node_id = node_id
        self.metadata = metadata
        self.start_time = time.time()
        self._landed_normally = False

    def __enter__(self):
        self.outer.check_budget()
        with self.outer._lock:
            parent_id = self.outer._node_stack[-1] if self.outer._node_stack else None

            # 🔄 升级版终极多态判定：完美兼容 Pydantic、标准 Dict 和普通 Slots 类
            if hasattr(self.metadata, "model_dump"):
                meta_snapshot = self.metadata.model_dump()  # 🚀 Pydantic v2 标准导出
            elif hasattr(self.metadata, "to_dict"):
                meta_snapshot = self.metadata.to_dict()  # 兼容传统 Slots 类
            elif isinstance(self.metadata, dict):
                meta_snapshot = deepcopy(self.metadata)
            else:
                meta_snapshot = str(self.metadata)

            self.outer.nodes[self.node_id] = {
                "node_id": self.node_id,
                "parent_id": parent_id,
                "status": NodeStatus.RUNNING,
                "start_time": self.start_time,
                "duration_ms": 0.0,
                "metadata": meta_snapshot,
                "error": None,
            }
            self.outer._node_stack.append(self.node_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        with self.outer._lock:
            if self.outer._node_stack and self.outer._node_stack[-1] == self.node_id:
                self.outer._node_stack.pop()
            node = self.outer.nodes.get(self.node_id)
            if not node:
                return False

            node["duration_ms"] = (end_time - self.start_time) * 1000

            # 🔄 退出生命周期时，同步触发 Pydantic 快照回刷
            if hasattr(self.metadata, "model_dump"):
                node["metadata"] = self.metadata.model_dump()
            elif hasattr(self.metadata, "to_dict"):
                node["metadata"] = self.metadata.to_dict()
            elif isinstance(self.metadata, dict):
                node["metadata"] = deepcopy(self.metadata)

            if exc_type is not None:
                err_msg = f"{exc_type.__name__}: {exc_val}"
                node["status"] = NodeStatus.FAILED
                node["error"] = err_msg

                # 🚀 兼容 Pydantic 和 Slots 的异常回写
                if hasattr(self.metadata, "error"):
                    try:
                        setattr(self.metadata, "error", err_msg)
                        if hasattr(self.metadata, "model_dump"):
                            node["metadata"] = self.metadata.model_dump()
                    except Exception:
                        pass
        return False

    def __del__(self):
        if not self._landed_normally:
            try:
                with self.outer._lock:
                    node = self.outer.nodes.get(self.node_id)
                    if node and node["status"] == NodeStatus.RUNNING:
                        node["status"] = NodeStatus.FAILED
                        node["error"] = "[Forced Abort]"
                        self.outer.log_trace(
                            f"🔶 [SPAN_ABORT] Node {self.node_id} aborted implicitly",
                            level=logging.WARNING,
                        )
            except Exception:
                pass


class AgentSession:
    """工业级全功能单类会话控制器 (无 Metadata 物理耦合版)"""

    def __init__(
        self,
        session_id: str,
        max_token_budget: int = 100000,
        timeout_limit: float = 15.0,
        global_deadline: Optional[float] = None,
        max_trace_logs: int = 5000,
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

        self.status = SessionStatus.RUNNING
        self.total_tokens = 0
        self.global_fingerprints: Set[str] = set()

        self.nodes: Dict[str, NodeDict] = {}
        self.trace_logs: deque = deque(maxlen=max_trace_logs)

        self._node_stack: List[str] = []
        self._lock = RLock()

        self.log_trace(
            f"🚀 AgentSession [{self.session_id}] initialized. Local quota: {self.timeout_limit}s.",
            level=logging.INFO,
        )

    def log_trace(self, message: str, level: int = logging.INFO) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        level_name = logging.getLevelName(level)
        with self._lock:
            depth = len(self._node_stack)
            indent = "  " * depth
            current_span = self._node_stack[-1] if self._node_stack else "Root"
            trace_line = f"[{timestamp}] [{level_name}] [{self.session_id}] [{current_span}]{indent} {message}"
            self.trace_logs.append(trace_line)
        if self.logger:
            self.logger.log(level, f"[{current_span}]{indent} {message}")

    def check_budget(self) -> None:
        with self._lock:
            if self.status in (SessionStatus.FAILED, SessionStatus.TIMEOUT):
                raise RuntimeError(
                    "Execution abandoned: Global session already terminated"
                )
            if time.time() > self.local_deadline:
                self.status = SessionStatus.TIMEOUT
                self.log_trace(
                    f"🚨 Local execution quota [{self.timeout_limit}s] exceeded",
                    level=logging.ERROR,
                )
                raise TimeoutError("Local time budget exceeded")
            if self.total_tokens >= self.max_token_budget:
                self.status = SessionStatus.FAILED
                self.log_trace("🚨 Global token budget exhausted", level=logging.ERROR)
                raise RuntimeError("Token budget exhausted")

    def consume_tokens(self, tokens: int) -> None:
        with self._lock:
            self.total_tokens += tokens
            self.log_trace(
                f"💸 [METRICS] Step Cost: {tokens} | Accumulate={self.total_tokens}",
                level=logging.INFO,
            )

    def check_and_record_fingerprint(self, fp: str) -> bool:
        with self._lock:
            if fp in self.global_fingerprints:
                self.status = SessionStatus.FAILED
                self.log_trace(
                    f"⚠️ [LOOP_BLOCKED] Fingerprint [{fp[:8]}] matched. Loop defense active.",
                    level=logging.WARNING,
                )
                return True
            self.global_fingerprints.add(fp)
            self.log_trace(
                f"💾 [FINGERPRINT] Recorded new signature [{fp[:8]}]",
                level=logging.DEBUG,
            )
            return False

    def step(self, node_id: str, metadata: Any = None) -> AgentStepContext:
        return AgentStepContext(self, node_id, metadata)

    def close(self, exc_type=None, exc_val=None) -> None:
        master_reason = (
            f"{exc_type.__name__}: {exc_val}" if exc_type else "Context Extinguished"
        )
        with self._lock:
            while self._node_stack:
                stale_node_id = self._node_stack.pop()
                node = self.nodes.get(stale_node_id)
                if node and node["status"] == NodeStatus.RUNNING:
                    node["status"] = NodeStatus.FAILED
                    node["error"] = f"[Cascaded Abort] Master: {master_reason}"
                    self.log_trace(
                        f"🧹 [CASCADED_ABORT] Cleaned hanging span: {stale_node_id}",
                        level=logging.WARNING,
                    )
            if exc_type:
                if self.status != SessionStatus.TIMEOUT:
                    self.status = SessionStatus.FAILED
                self.log_trace(
                    f"🛑 会话因外部异常强行关闭: {master_reason}", level=logging.ERROR
                )
            else:
                if self.status == SessionStatus.RUNNING:
                    self.status = SessionStatus.COMPLETED
                self.log_trace(
                    f"🏁 会话安全结束。最终状态: {self.status}", level=logging.INFO
                )

    def to_snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            nodes_copy = deepcopy(self.nodes)
            for node_id, node in nodes_copy.items():
                if node["status"] == NodeStatus.RUNNING:
                    node["duration_ms"] = (now - node["start_time"]) * 1000
            return {
                "session_id": self.session_id,
                "status": self.status.value,
                "total_tokens": self.total_tokens,
                "nodes": nodes_copy,
                "trace_logs": list(self.trace_logs),
            }
