import os
import sys
import queue
import atexit
import logging
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from typing import Tuple, Optional
from schema.session.runtime import RuntimeContext


# =====================================================================
# 2. 日志支撑组件 (Formatter & Filter)
# =====================================================================
class DynamicContextFormatter(logging.Formatter):
    """
    工业级防御型格式化器：确保即使没有被 Filter 拦截的第三方日志，
    在格式化时也能拿到安全的兜底值，绝对不报 KeyError。
    """

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "session_id"):
            record.session_id = "SYSTEM"
        if not hasattr(record, "node_id"):
            record.node_id = "MAIN"
        if not hasattr(record, "trace_id"):
            record.trace_id = "T-RAW-UNKNOWN"
        return super().format(record)


class MainThreadCaptureFilter(logging.Filter):
    """
    物理现场打标机：挂载在 Logger 上，运行在【主业务线程/协程】中。
    在日志入队前的一瞬间，强制把大管家的安全资产剥离出来，死死固化在 record 身上！
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = RuntimeContext.get_session_id()
        record.node_id = RuntimeContext.get_node_id()
        record.trace_id = RuntimeContext.get_trace_id()
        return True


# =====================================================================
# 3. 异步生产日志工厂 (纯净后台落盘完全体)
# =====================================================================
def create_async_production_logger(
    logger_name: str = "AgentEngine",
    log_dir: str = "logs",
    log_file_name: str = "session_audit.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    max_queue_size: int = 100000,
) -> Tuple[logging.Logger, Optional[QueueListener]]:

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    pid = os.getpid()
    name_parts = os.path.splitext(log_file_name)
    # 🌟 核心修正：修复原工厂函数中元组错乱拼接引发的路径挂掉 Bug
    actual_file_name = f"{name_parts[0]}_{pid}{name_parts[1]}"
    log_path = os.path.join(log_dir, actual_file_name)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 锁闭冒泡，防止异步双写

    if logger.handlers:
        return logger, None

    # 🟢 在主 Logger 上挂载物理现场打标机！
    logger.addFilter(MainThreadCaptureFilter())

    formatter = DynamicContextFormatter(
        fmt="%(asctime)s [%(levelname)s] [SID:%(session_id)s] [NODE:%(node_id)s] [TID:%(trace_id)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # 🌟 核心修正：彻底拔掉 console_handler！让终端保持绝对的工业级静音
    log_queue = queue.Queue(maxsize=max_queue_size)
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    # 🌟 监听器只绑定 file_handler，让审计日志只在后台异步解耦落盘
    listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
    listener.start()

    def safe_exit_hook(origin_pid: int, listen_obj: QueueListener):
        if os.getpid() == origin_pid:
            try:
                listen_obj.stop()
            except Exception:
                pass

    atexit.register(safe_exit_hook, pid, listener)
    return logger, listener
