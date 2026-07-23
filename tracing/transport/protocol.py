# tracing/infra/transport/protocol.py
from typing import Protocol, List, TypeVar, runtime_checkable

PayloadT = TypeVar("PayloadT")


@runtime_checkable
class Transport(Protocol[PayloadT]):
    """
    通用传输层协议（接口）。

    所有具体传输实现（HTTP、Kafka、UDP、File）都必须实现此接口。
    它定义了 BatchExporter 与底层 IO 之间的契约。
    """

    async def send(self, batch: List[PayloadT]) -> None:
        """
        发送一批载荷。

        Args:
            batch: 由 BatchExporter 序列化好的数据列表。

        Raises:
            任何异常都会被 BatchExporter 捕获并触发 on_drop 回调。
            实现者无需在内部吞掉异常，但应保证异常信息足够详细。
        """
        ...
