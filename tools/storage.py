import json
import abc
from typing import Any, Dict, Optional

class BaseStorage(abc.ABC):
    """插件状态存储抽象基类，用于解耦内存与分布式存储"""
    @abc.abstractmethod
    async def get_state(self, state_key: str) -> Optional[Dict[str, Any]]:
        """异步获取状态，返回字典或 None"""
        pass

    @abc.abstractmethod
    async def set_state(self, state_key: str, state_data: Dict[str, Any]) -> None:
        """异步保存状态"""
        pass

class MemoryStorage(BaseStorage):
    """纯内存状态存储（适合单机本地极速开发与测试）"""
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    async def get_state(self, state_key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(state_key)

    async def set_state(self, state_key: str, state_data: Dict[str, Any]) -> None:
        self._data[state_key] = state_data

class RedisStorage(BaseStorage):
    """Redis 异步状态存储（适合单机生产环境：防崩溃、持久化、支持状态过期）"""
    def __init__(self, redis_client: Any, ttl: int = 86400) -> None:
        """
        :param redis_client: 已初始化好的 redis.asyncio.Redis 客户端实例
        :param ttl: 状态在 Redis 中的生存时间（秒），默认 1 天
        """
        self.redis = redis_client
        self.ttl = ttl

    async def get_state(self, state_key: str) -> Optional[Dict[str, Any]]:
        raw_data = await self.redis.get(state_key)
        if raw_data:
            return json.loads(raw_data)
        return None

    async def set_state(self, state_key: str, state_data: Dict[str, Any]) -> None:
        await self.redis.set(state_key, json.dumps(state_data), ex=self.ttl)