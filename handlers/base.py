from abc import ABC, abstractmethod
from types import TracebackType
from typing import Optional, Type, Self


class BaseStreamHandler(ABC):
    """
    工业标准：LLM 流式数据处理器抽象基类。
    规范了异步上下文管理器 (async with) 和统一的回调触发接口。
    """

    async def __aenter__(self) -> Self:
        """
        进入异步上下文的默认实现。
        子类如果需要初始化连接（如开启画布、打开 WebSocket），请重写 `open` 方法。
        """
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """
        退出异步上下文的默认实现。
        子类如果需要收尾（如关闭连接、强制刷盘），请重写 `close` 方法。
        """
        await self.close(exc_type, exc_val)
        return False  # 默认不拦截异常，允许异常向上传播

    async def open(self) -> None:
        """
        异步初始化生命周期钩子（可选实现）。
        子类可以重写此方法以处理进入 `async with` 时的准备工作。
        """
        pass

    async def close(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException]
    ) -> None:
        """
        异步清理生命周期钩子（可选实现）。
        子类可以重写此方法以处理离开 `async with` 时的收尾工作。
        """
        pass

    @abstractmethod
    async def __call__(
        self, think: str, text: str, tool_args: str, chunk_type: str
    ) -> None:
        """
        抽象方法：子类必须实现。
        高层业务的回调渲染接口，接收底层流透传的碎片段。
        """
        pass


# 确保引入了你之前定义的基类
class NullStreamHandler(BaseStreamHandler):
    """
    空对象模式（Null Object Pattern）流处理器。

    继承自 BaseStreamHandler，实现所有必须的契约接口，但不做任何实际处理。
    专用于后台静默运行、自动化跑批、单元测试，或作为 handler=None 时的安全兜底。
    """

    def __init__(self) -> None:
        super().__init__()

    async def open(self) -> None:
        """静默进入上下文，不初始化任何画布或连接"""
        pass

    async def close(
        self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException]
    ) -> None:
        """静默退出上下文，不进行任何刷盘或关闭操作"""
        pass

    async def __call__(
        self, think: str, text: str, tool_args: str, chunk_type: str
    ) -> None:
        """
        核心回调接口的空实现。
        接收底层流透传的碎片段并直接丢弃，不消耗任何 CPU 资源，不产生任何控制台输出。
        """
        pass
