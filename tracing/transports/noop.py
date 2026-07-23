# tracing/infra/transport/noop.py
from typing import List

from tracing.infra.transport import Transport


class NoopTransport(Transport[object]):
    """空实现，用于测试或禁用导出功能"""

    async def send(self, batch: List[object]) -> None:
        # 什么都不做，也不抛异常
        print(batch)
