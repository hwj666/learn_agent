"""
Agent 类 - Policy + 配置
"""
import logging
from typing import Optional, Type, TYPE_CHECKING

from core.config import AgentConfig
from core.openai_client import OpenAIClient
from core.context import ExecutionContext
from core.policy import ExecutionPolicy
from core.orchestrator import Orchestrator

from tools.storage import MemoryStorage

if TYPE_CHECKING:
    from tools.execute import ToolExecutor


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        policy_class: Type[ExecutionPolicy],
        session_id: str,
        max_steps: int = 10,
        max_history_turns: int = 5,
        logger: Optional[logging.Logger] = None,
        trace_enabled: bool = False,
    ):
        self.logger = logger or logging.getLogger(f"Agent[{session_id}]")
        self.session_id = session_id
        self.config = config

        self.client = OpenAIClient(config.model_config, trace_enabled=trace_enabled)
        self.storage = MemoryStorage()
        from tools.execute import ToolExecutor
        self.executor = ToolExecutor(self.storage, allowed_toolsets=config.tool_set)

        self.ctx = {
            "session_id": session_id,
            "agent_id": 1,
            "sandbox_read_dirs": ["./"],
            "sandbox_write_dirs": ["./work"],
        }

        self.policy = policy_class(
            executor=self.executor,
            ctx=self.ctx,
            max_history_turns=max_history_turns,
            client=self.client,
        )

        self.orchestrator = Orchestrator(
            policy=self.policy,
            max_steps=max_steps,
            logger=self.logger,
        )

    async def run(self, user_query: str) -> str:
        self.logger.info(f"🚀 开始执行任务: {user_query[:50]}...")

        context = ExecutionContext(session_id=self.session_id)
        result = await self.orchestrator.run(user_query, context)

        self.logger.info(f"✅ 任务完成: {result[:100]}...")
        return result


def create_react_agent(config: AgentConfig, session_id: str, **kwargs) -> Agent:
    from core.policy import ReactPolicy

    return Agent(
        config=config,
        policy_class=ReactPolicy,
        session_id=session_id,
        **kwargs
    )


def setup_trace_logging(log_file: str = "trace.log", level: int = logging.DEBUG):
    """配置 trace 日志输出到文件"""
    trace_logger = logging.getLogger("trace")
    trace_logger.setLevel(level)

    # 清除已有的 handlers
    trace_logger.handlers.clear()

    # 添加文件 handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    trace_logger.addHandler(file_handler)

    return trace_logger
