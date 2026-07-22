# tracing/infra/transport.py
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable
import logging

logger = logging.getLogger(__name__)


class TracingTransportError(Exception):
    """Transport 层统一异常，便于 exporter 区分处理。"""

    pass


@runtime_checkable
class SupportsSendJson(Protocol):
    async def send_json(self, data: dict[str, Any]) -> None: ...


class EventTransport(ABC):
    """
    事件传输抽象层。
    不关心序列化、队列、重试策略。
    """

    @abstractmethod
    async def send(self, payload: dict[str, Any]) -> None:
        """
        发送单个事件 payload。
        实现方应抛出 TracingTransportError 或其子类。
        """
        raise NotImplementedError

    @property
    def healthy(self) -> bool:
        """可选：标识当前 transport 是否可接受新事件。"""
        return True


class WebSocketTransport(EventTransport):
    """
    基于 WebSocket 的事件传输。
    假设 ws_manager 由外部生命周期管理（连接 / 重连 / 关闭）。
    """

    def __init__(self, ws_manager: SupportsSendJson) -> None:
        self.ws_manager = ws_manager

    @property
    def healthy(self) -> bool:
        # 如果 ws_manager 有类似属性，可以委托给它
        return True

    async def send(self, payload: dict[str, Any]) -> None:
        try:
            await self.ws_manager.send_json(payload)
        except Exception as exc:
            logger.warning(
                "WebSocketTransport send failed",
                exc_info=True,
            )
            raise TracingTransportError(f"WebSocket send failed: {exc}") from exc


class InMemoryTransport(EventTransport):
    """
    用于测试 / 调试。
    不真正发送，仅追加到列表。
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)


class LoggingTransport(EventTransport):
    """
    本地调试用：打印到日志。
    """

    def __init__(self, level: int = logging.DEBUG) -> None:
        self.level = level

    async def send(self, payload: dict[str, Any]) -> None:
        logger.log(self.level, "TracingEvent: %r", payload)


class NoopTransport(EventTransport):
    """
    完全静默，用于禁用 tracing 的场景。
    """

    async def send(self, payload: dict[str, Any]) -> None:
        return


class ConsoleJsonTransport(EventTransport):
    """自定义传输通道：将接收到的链路数据打印到控制台"""

    async def send(self, payload: dict) -> None:
        event = payload["event"]
        name = payload["name"]
        # 只显示 Span ID 后 4 位，减少视觉噪音
        span_id = payload["span_id"][-4:]
        depth = "  " * (payload["depth"] - 1)

        data = ""
        if payload.get("chunk_text"):
            data = f" -> Chunk: '{payload['chunk_text']}'"
        elif payload.get("metadata"):
            data = f" -> Meta: {payload['metadata']}"
        elif payload.get("error_msg"):
            data = f" -> ❌ Error: {payload['error_msg']}"

        print(f"[Network Output] {depth}📍 [{event}] {name} ({span_id}){data}")
