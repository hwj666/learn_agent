"""
执行上下文抽象
用于 trace / replay / debug
"""
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from core.message import LLMMessage


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


@dataclass
class StepRecord:
    step: int
    status: StepStatus
    decision: Optional[Any] = None
    tool_calls: List[Any] = field(default_factory=list)
    tool_results: List[Any] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    session_id: str
    user_query: str = ""
    history: List[LLMMessage] = field(default_factory=list)
    step_records: List[StepRecord] = field(default_factory=list)
    fingerprints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    _logger: logging.Logger = field(default=None, repr=False)

    def __post_init__(self):
        if self._logger is None:
            self._logger = logging.getLogger(f"Context[{self.session_id}]")

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def add_message(self, msg: LLMMessage):
        self.history.append(msg)

    def add_step(self, step_record: StepRecord):
        self.step_records.append(step_record)

    def add_fingerprint(self, fp: str):
        if fp not in self.fingerprints:
            self.fingerprints.append(fp)

    def has_fingerprint(self, fp: str) -> bool:
        return fp in self.fingerprints

    def get_recent_turns(self, max_turns: int) -> List[LLMMessage]:
        if len(self.history) <= max_turns * 3:
            return self.history

        turn_count = 0
        start_idx = len(self.history)

        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i].role == "user":
                turn_count += 1
                if turn_count >= max_turns:
                    start_idx = i
                    break

        return self.history[start_idx:]

    def to_trace(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_query": self.user_query,
            "total_steps": len(self.step_records),
            "history": [msg.to_dict() for msg in self.history],
            "steps": [
                {
                    "step": r.step,
                    "status": r.status.value,
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in r.tool_calls
                    ],
                    "error": r.error,
                    "metadata": r.metadata,
                }
                for r in self.step_records
            ],
            "fingerprints": self.fingerprints,
        }

    def save_trace(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_trace(), f, ensure_ascii=False, indent=2)

    @property
    def tool_called(self) -> bool:
        return any(
            msg.role == "assistant" and msg.tool_calls
            for msg in self.history
        )
