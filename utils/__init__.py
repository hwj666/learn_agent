"""
utils 模块
工具函数：装饰器、模式类、流处理器等
"""
from utils.singleton import singleton
from utils.base_stream_handler import BaseStreamHandler, NullStreamHandler

__all__ = [
    "singleton",
    "BaseStreamHandler",
    "NullStreamHandler",
]
