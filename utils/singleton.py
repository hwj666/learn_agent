"""
单例模式装饰器
线程安全的单例实现，支持装饰器和元类两种方式
"""
import threading
from functools import wraps
from typing import Any, Callable, Type


def singleton(cls: Type[Any]) -> Type[Any]:
    """
    线程安全的单例装饰器

    使用方式:
    @singleton
    class MySingleton:
        pass
    """
    _instance: Any = None
    _lock = threading.Lock()

    @wraps(cls)
    def get_instance(*args: Any, **kwargs: Any) -> Any:
        if _instance is None:
            with _lock:
                if _instance is None:
                    _instance = cls(*args, **kwargs)
        return _instance

    return get_instance


class SingletonMeta(type):
    """
    单例元类

    使用方式:
    class MySingleton(metaclass=SingletonMeta):
        pass
    """
    _instances: dict = {}
    _lock = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]
