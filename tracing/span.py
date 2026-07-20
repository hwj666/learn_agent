from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum
import time
import uuid


class SessionStatus(str, Enum):
    """大管家会话宏观生命周期状态机"""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CRASHED = "CRASHED"


class SpanKind(str, Enum):
    """Span 类型枚举，对齐 OpenTelemetry 标准"""

    INTERNAL = "INTERNAL"
    CLIENT = "CLIENT"
    SERVER = "SERVER"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


@dataclass(frozen=True)
class AgentSpan:
    """🪐 物理生命周期通行证：纯净且符合行业 OTel 标准"""

    span_id: str = field(default_factory=lambda: f"sp_{uuid.uuid4().hex}")
    session_id: str = ""
    trace_id: str = ""
    span_name: str = ""
    attempt_idx: int = 0
    parent_span_id: Optional[str] = None
    depth: int = 1
    kind: SpanKind = SpanKind.INTERNAL
    baggage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSpanRecord:
    """💰 内存中央账本快照：自动固化标量数据"""

    span_id: str
    session_id: str
    trace_id: str
    span_name: str
    attempt_idx: int
    parent_span_id: Optional[str]
    depth: int
    kind: SpanKind = SpanKind.INTERNAL
    status: str = SessionStatus.RUNNING.value
    start_monotonic: float = field(default_factory=time.monotonic)
    start_wall: float = field(default_factory=time.time)
    end_monotonic: float = 0.0
    end_wall: float = 0.0
    duration_ms: float = 0.0
    tokens_consumed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    baggage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def mark_completed(self) -> None:
        self.status = SessionStatus.COMPLETED.value
        self.end_monotonic = time.monotonic()
        self.end_wall = time.time()
        self.duration_ms = round(
            (self.end_monotonic - self.start_monotonic) * 1000.0, 2
        )

    def mark_crashed(self, error_msg: str) -> None:
        self.status = SessionStatus.CRASHED.value
        self.end_monotonic = time.monotonic()
        self.end_wall = time.time()
        self.duration_ms = round(
            (self.end_monotonic - self.start_monotonic) * 1000.0, 2
        )
        self.error = error_msg

    def mark_timeout(self) -> None:
        self.status = SessionStatus.TIMEOUT.value
        self.end_monotonic = time.monotonic()
        self.end_wall = time.time()
        self.duration_ms = round(
            (self.end_monotonic - self.start_monotonic) * 1000.0, 2
        )

    def mark_cancelled(self) -> None:
        self.status = SessionStatus.CRASHED.value
        self.end_monotonic = time.monotonic()
        self.end_wall = time.time()
        self.duration_ms = round(
            (self.end_monotonic - self.start_monotonic) * 1000.0, 2
        )

    def accumulate_tokens(self, tokens: int) -> None:
        self.tokens_consumed += int(tokens)

    @classmethod
    def from_span(
        cls,
        span: AgentSpan,
        metadata: Dict[str, Any] = None,
        start_time: Optional[float] = None,
    ) -> "AgentSpanRecord":
        init_kwargs = {
            "span_id": span.span_id,
            "session_id": span.session_id,
            "trace_id": span.trace_id,
            "span_name": span.span_name,
            "attempt_idx": span.attempt_idx,
            "parent_span_id": span.parent_span_id,
            "depth": span.depth,
            "kind": span.kind,
            "baggage": span.baggage.copy(),
            "metadata": metadata or {},
        }
        if start_time is not None:
            init_kwargs["start_wall"] = start_time
        return cls(**init_kwargs)
