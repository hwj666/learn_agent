"""
main.py
🚀 Agent Runtime 总入口
负责：初始化基础设施 → 编排 Session → 优雅启停 → 全链路可观测
"""

import asyncio
import logging
import sys
import uuid

from schema.context import AgentContext
from schema.metadata import PlannerMetadata, SubStepMetadata
from core.openai_client import OpenAIClient
from tools import ToolExecutor
from tools import discover_and_load_tools
from schema.config import AppConfig
from agent.react_agent import ReActExecution
from tracing import AgentTracker
from tracing import AgentSpanContext

from tracing import get_agent_logger

# =====================================================================
# 1. 基础设施初始化（全局单例）
# =====================================================================

# 扫描并注册所有工具（必须在 Executor 之前）
discover_and_load_tools()

# 加载应用配置
config = AppConfig.from_yaml("config.yaml")
agent_config = config.get_agent("simple_agent")

# 初始化核心引擎
client = OpenAIClient(agent_config.model_config)
executor = ToolExecutor()
agent = ReActExecution(client, executor)


# =====================================================================
# 2. 核心任务编排管线
# =====================================================================
async def _execute_task(
    tracker: AgentTracker,
    context: AgentContext,
    worker: ReActExecution,
) -> None:
    """
    微观执行管线：
    1. 创建任务级 Span
    2. 运行 ReAct Agent
    3. 记录最终结果
    """
    t_idx = 0
    t_desc = "run test.py file"
    span_name = "execute_test_file"

    task_meta = SubStepMetadata(
        description=f"Executing subtask {t_idx}: {t_desc}",
        tool_name="subtask_runner",
    )

    async with AgentSpanContext(
        tracker,
        span_name=span_name,
        metadata=task_meta,
        kind="INTERNAL",
    ) as step:
        # 记录输入参数
        task_meta.arguments = {"task_desc": t_desc}

        # 执行核心 ReAct 循环
        task_report = await worker.run(
            current_task_desc=t_desc,
            tracker=tracker,
            context=context,
        )

        # 记录输出结果
        task_meta.output_data = {"final_report": task_report}
        task_meta.status = "COMPLETED"

        # 最后一次刷新元数据
        await tracker.update_metadata_stream(step.span, metadata=task_meta)

        # 打印最终结果
        print("\n" + "=" * 50)
        print("✅ 任务执行完成")
        print(f"📊 最终报告: {task_report[:200]}...")
        print("=" * 50 + "\n")


async def main() -> None:
    """主异步入口"""
    user_query = "帮我运行一下./work/test.py文件，如果报错则进行修复后再次执行"

    # 闭环 1：激活异步高可用日志系统

    session_id = f"session_{uuid.uuid4().hex[:8]}"
    trace_id = f"tr_{uuid.uuid4().hex[:8]}"
    init_context = AgentContext(
        session_id=session_id,
        tenant_id="tenant_enterprise_group_01",
        user_id="user_staff_045",
        trace_id=trace_id,
        allowed_toolsets={"system_utils", "code_utils"},
        payload={
            "user_query": user_query,
            "working_dir": "./work",
        },
    )
    logger = get_agent_logger(init_context)

    # 创建 Session 级 Tracker（大管家）
    async with AgentTracker(
        max_token_budget=100000,
        timeout_limit=40.0,
        logger=logger,
    ) as tracker:
        # 构建初始上下文（不可变数据，支持 fork）

        logger.info(
            f"🚀 启动 Agent Session | session_id={session_id} | trace_id={trace_id}"
        )

        try:
            # 核心微观任务管线（35秒硬超时熔断）
            await asyncio.wait_for(
                _execute_task(tracker, init_context, agent),
                timeout=35.0,
            )

        except asyncio.TimeoutError as t_err:
            # 超时熔断：异常会由 AgentTracker.__aexit__ 自动处理
            logger.error(
                "⏰ Pipeline timeout: global limit exceeded",
                exc_info=True,
            )
            raise TimeoutError(
                "Pipeline stalled and exploded due to global timeout limit."
            ) from t_err

        except Exception as e:
            # 未捕获的业务异常
            logger.exception("💥 Unhandled exception in main task pipeline")
            raise


if __name__ == "__main__":
    # =================================================================
    # 全局日志抑制（防止第三方库污染控制台）
    # =================================================================
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # =================================================================
    # 启动异步事件循环
    # =================================================================

    asyncio.run(main())
