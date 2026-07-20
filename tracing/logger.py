"""
schema/logger.py
🚀 工业级异步高可用日志系统（完全体）

核心特性：
1. 异步 QueueHandler + QueueListener，零阻塞落盘
2. PID 路由防多进程冲突
3. 基于 AgentSpanContext 的微观执行上下文绑定
4. 优雅停机 + 队列强制排空
5. 防御型 Formatter，绝不 KeyError
"""

from __future__ import annotations

import os
import sys
import atexit
import logging
import queue
import threading
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from typing import Tuple, Optional, Any

from .context import AgentSpanContext


# =====================================================================
# 1. 日志支撑组件 (Formatter & Adapter)
# =====================================================================
class DynamicContextFormatter(logging.Formatter):
    """
    工业级防御型格式化器：
    确保即使第三方库（如 httpx）的日志没有 Context 字段，
    也能安全格式化，绝不抛出 KeyError。
    """

    def format(self, record: logging.LogRecord) -> str:
        # 提供安全的全局兜底值
        if not hasattr(record, "session_id"):
            record.session_id = "SYSTEM"
        if not hasattr(record, "step_tag"):
            record.step_tag = "MAIN"
        if not hasattr(record, "trace_id"):
            record.trace_id = "T-RAW-UNKNOWN"

        # 确保所有字段都是字符串，防止 % 格式化失败
        record.session_id = str(record.session_id)
        record.step_tag = str(record.step_tag)
        record.trace_id = str(record.trace_id)

        return super().format(record)


class AgentContextLoggerAdapter(logging.LoggerAdapter):
    """
    【核心】显式日志适配器

    在业务层调用 log.info() 时，自动从 AgentSpanContext 中
    提取当前活跃的 Span ID，并牢固绑定到每一条 LogRecord。

    这解决了 asyncio 下上下文漂移的根本问题。
    """

    def __init__(self, logger: logging.Logger, ctx: Any):
        """
        Args:
            logger: 底层 logger 实例
            ctx: AgentContext 实例（业务身份）
        """
        super().__init__(logger, extra={})
        self.ctx = ctx

    def process(self, msg: Any, kwargs: Any) -> Tuple[Any, Any]:
        # 1. 从 ContextVar 获取当前活跃的 Span（关键修复点）
        current_span = None
        if AgentSpanContext is not None:
            current_span = AgentSpanContext.get_current_span()

        # 2. 构建日志上下文
        extra = {
            "session_id": getattr(self.ctx, "session_id", "SYSTEM"),
            "trace_id": getattr(self.ctx, "trace_id", "T-RAW-UNKNOWN"),
            "step_tag": current_span.span_id if current_span else "MAIN",
        }

        # 3. 合并用户可能自带的其他 extra 字段
        if "extra" in kwargs:
            kwargs["extra"].update(extra)
        else:
            kwargs["extra"] = extra

        return msg, kwargs


# =====================================================================
# 2. 异步生产日志工厂 (完全体)
# =====================================================================
def create_async_production_logger(
    logger_name: str = "AgentEngine",
    log_dir: str = "logs",
    log_file_name: str = "session_audit.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    max_queue_size: int = 100000,
    log_level: int = logging.INFO,
) -> Tuple[logging.Logger, Optional[QueueListener], queue.Queue]:
    """
    创建异步高可用日志系统。

    Returns:
        Tuple[logger, listener, log_queue]
        - logger: 配置好的 logger 实例
        - listener: QueueListener 实例（需保存引用）
        - log_queue: 日志队列（用于优雅停机）
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # PID 路由：防止多进程写同一文件
    pid = os.getpid()
    name_parts = os.path.splitext(log_file_name)
    actual_file_name = f"{name_parts[0]}_{pid}{name_parts[1]}"
    log_path = os.path.join(log_dir, actual_file_name)

    # 获取或创建 logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False  # 防止日志冒泡到 root logger

    # 避免重复添加 handler
    if logger.handlers:
        # 返回现有的 logger 和 None（表示没有新建 listener）
        return logger, None, None

    # 使用防御型格式化器
    formatter = DynamicContextFormatter(
        fmt="%(asctime)s [%(levelname)s] [SID:%(session_id)s] [NODE:%(step_tag)s] [TID:%(trace_id)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件 Handler（RotatingFileHandler）
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # 异步落盘缓冲区队列
    log_queue = queue.Queue(maxsize=max_queue_size)
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    # 启动后台解耦落盘专用监听线程
    listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
    listener.start()

    # =================================================================
    # 3. 优雅停机机制（关键）
    # =================================================================
    def _safe_exit_hook(
        origin_pid: int,
        listen_obj: QueueListener,
        log_queue_obj: queue.Queue,
        file_handler_obj: RotatingFileHandler,
    ):
        """安全的退出钩子，确保日志不丢失"""
        if os.getpid() != origin_pid:
            return

        try:
            # 1. 停止接受新日志
            listen_obj.stop()

            # 2. 强制排空队列（关键修复点）
            drained_count = 0
            while not log_queue_obj.empty():
                try:
                    record = log_queue_obj.get_nowait()
                    file_handler_obj.emit(record)
                    drained_count += 1
                except Exception:
                    break

            # 3. 确保文件句柄关闭
            file_handler_obj.close()

            if drained_count > 0:
                print(
                    f"[Logger] Drained {drained_count} pending log records",
                    file=sys.stderr,
                )

        except Exception as e:
            print(f"[Logger] Error during shutdown: {e}", file=sys.stderr)

    # 注册退出钩子
    atexit.register(
        _safe_exit_hook,
        pid,
        listener,
        log_queue,
        file_handler,
    )

    # 可选：注册信号处理（用于容器环境）
    def _signal_handler(signum, frame):
        print(
            f"[Logger] Received signal {signum}, initiating shutdown...",
            file=sys.stderr,
        )
        _safe_exit_hook(pid, listener, log_queue, file_handler)
        sys.exit(0)

    # 仅在主线程中注册信号处理器
    if threading.current_thread() is threading.main_thread():
        try:
            import signal

            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)
        except (ImportError, ValueError):
            pass  # Windows 或特殊环境可能不支持

    return logger, listener, log_queue


# =====================================================================
# 4. 便捷工具函数
# =====================================================================
def get_agent_logger(
    ctx: Any, logger_name: str = "AgentEngine", **kwargs
) -> AgentContextLoggerAdapter:
    """
    获取绑定了 AgentContext 的日志适配器。

    这是业务代码中应该使用的主要接口。

    Example:
        logger = get_agent_logger(agent_context)
        logger.info("Starting task execution")
    """
    base_logger, _, _ = create_async_production_logger(
        logger_name=logger_name, **kwargs
    )
    return AgentContextLoggerAdapter(base_logger, ctx)
