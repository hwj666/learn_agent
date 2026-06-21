"""
utils 模块
工具函数：装饰器、模式类等

注意：流处理器已移至 handlers/ 模块
注意：工具提取已移至 tools/extract.py 模块
"""
from utils.singleton import singleton

# 兼容旧导入路径
from handlers import BaseStreamHandler, NullStreamHandler, PrintStreamHandler, RichStreamHandler
from handlers.base import BaseStreamHandler, NullStreamHandler
from handlers.print_handler import PrintStreamHandler
from handlers.rich_handler import RichStreamHandler

# 兼容旧导入路径 - tools.extract
import tools.extract as extract_module
from tools.extract import extract_implicit_tool_calls

__all__ = [
    "singleton",
    "BaseStreamHandler",
    "NullStreamHandler",
    "PrintStreamHandler",
    "RichStreamHandler",
    "extract_implicit_tool_calls",
]
