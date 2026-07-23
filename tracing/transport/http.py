# tracing/infra/transport/http.py
import asyncio
import json
import logging
from typing import List, Optional
from aiohttp import ClientSession, TCPConnector, ClientError

from tracing.transport.protocol import Transport


logger = logging.getLogger(__name__)


class HttpTransport(Transport[dict]):
    """
    HTTP 批量传输器。
    适用于 JSON 格式的 Trace / Log / Metric 数据。
    """

    def __init__(
        self,
        endpoint: str,
        *,
        headers: Optional[dict] = None,
        timeout: float = 10.0,
        max_connections: int = 100,
        compress: bool = False,
    ):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
        self.compress = compress

        # 共享 Session（非常重要，避免每次建连）
        self._session: Optional[ClientSession] = None
        self._connector = TCPConnector(
            limit=max_connections,
            ttl_dns_cache=300,
            keepalive_timeout=60,
        )
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> ClientSession:
        """懒加载 Session（线程安全）"""
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    self._session = ClientSession(
                        connector=self._connector,
                        timeout=self._make_timeout(),
                        headers=self.headers,
                    )
        return self._session

    def _make_timeout(self):
        from aiohttp import ClientTimeout
        return ClientTimeout(
            total=self.timeout,
            connect=5.0,
            sock_read=self.timeout,
        )

    async def send(self, batch: List[dict]) -> None:
        """
        发送 JSON 批量数据。
        异常会直接抛出，由 BatchExporter 统一处理。
        """
        if not batch:
            return

        session = await self._ensure_session()

        # HTTP 本身支持批量，不需要拆包
        payload = json.dumps(batch).encode("utf-8")

        try:
            async with session.post(
                self.endpoint,
                data=payload,
                compress="gzip" if self.compress else None,
            ) as resp:
                if resp.status >= 400:
                    # 4xx/5xx 抛异常，触发 on_drop
                    text = await resp.text()
                    raise RuntimeError(
                        f"HTTP {resp.status}: {text[:200]}"
                    )
        except ClientError:
            # aiohttp 的网络异常直接向上抛
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            # 兜底，防止未知异常逃逸
            logger.exception("Unexpected HTTP transport error")
            raise

    async def close(self) -> None:
        """显式关闭 Session（可选，供应用退出时调用）"""
        if self._session and not self._session.closed:
            await self._session.close()
            await self._connector.close()