# tracing/infra/transport/protocol.py
from typing import Protocol, List, TypeVar

PayloadT = TypeVar("PayloadT")


class Transport(Protocol[PayloadT]):
    async def send(self, batch: List[PayloadT]) -> None: ...
