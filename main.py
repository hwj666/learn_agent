import asyncio
import json
import time
from core.openai_client import OpenAIClient
from schema.context import ExecutionContext
from schema.enums import NodeStatus, SessionStatus
from schema.message import LLMMessage
from schema.node import NodeRecord, TaskMetadata
from schema.session import SessionContext
from tools.execute import ToolExecutor
from tools.loader import discover_and_load_tools
from schema.config import AppConfig
from agent.react_agent import ReActExecution

# =====================================================================
# 4. 初始化全局应用桩与调度（完全融合对齐您的原生逻辑）
# =====================================================================

discover_and_load_tools()

config = AppConfig.from_yaml("config.yaml")
agent_config = config.get_agent("simple_agent")
client = OpenAIClient(agent_config.model_config)
executor = ToolExecutor(allowed_toolsets=agent_config.tool_set)
agent = ReActExecution(client, executor)


async def _execute_task(session: SessionContext, worker: ReActExecution):
    """执行单个子任务"""
    t_idx = 0
    t_desc = "check file"

    # 创建任务节点，使用 TaskMetadata 统一格式
    task_meta = TaskMetadata(
        node_type="SubTask_Execution",
        description=f"Executing task {t_idx}: {t_desc}",
        input_data={"consensus_snapshot": session.consensus_pool.copy()},
    )
    task_node = NodeRecord(node_id=f"Task_{t_idx}", metadata=task_meta)
    task_node.status = NodeStatus.RUNNING
    session.add_root_node(task_node)

    try:
        # 计算局部截止时间
        session_remaining = session.remaining_time
        local_deadline = time.time() + min(30.0, session_remaining)

        # 创建执行上下文
        local_ctx = ExecutionContext(
            execution_id=f"Task_{t_idx}_Exec_0",
            parent_node=task_node,
            session_view=session.get_readonly_view(),
            session=session,
            deadline=local_deadline,
            local_token_budget=25000,
        )

        # 执行任务
        task_report = await worker.run(
            current_task_desc=t_desc,
            context_data=session.consensus_pool,
            ctx=local_ctx,
        )

        # 记录结果并原子终结任务节点
        task_meta.output_data = {"final_report": task_report}
        task_node.mark_success()

        # 更新共识池
        session.consensus_pool[f"fact_of_task_{t_idx}"] = task_report

        # 记录指标
        session.metadata[f"task_{t_idx}_metrics"] = {
            "prompt_tokens": local_ctx.prompt_tokens,
            "completion_tokens": local_ctx.completion_tokens,
            "turns": len(task_node.children),
            "fingerprints": len(local_ctx.local_fingerprints),
        }

        session.add_global_message(
            LLMMessage.system(f"Task {t_idx} completed: {task_report[:100]}...")
        )

        print(f"Task {t_idx} completed successfully")

    except Exception as e:
        task_node.mark_failure(str(e))
        raise


async def main():
    user_query = "帮我运行一下./work/test.py文件，如果报错则进行修复后再次执行"
    session = SessionContext(
        session_id="session_1001", user_query=user_query, status=SessionStatus.RUNNING
    )
    session.logger.info(f"Starting workflow for query: {user_query[:50]}...")

    # 创建 Planner 根节点
    planner_meta = TaskMetadata(
        node_type="Planner", description="Analyzing user request and planning tasks"
    )
    planner_node = NodeRecord(node_id="Planner_Root", metadata=planner_meta)
    planner_node.status = NodeStatus.RUNNING
    session.add_root_node(planner_node)

    # 步骤1: 任务拆解
    await asyncio.sleep(0.05)
    planner_meta.output_data = {
        "planned_tasks": "检查一下当前的路径并修复 test.py 脚本中的错误"
    }
    planner_node.mark_success()

    session.global_history.append(LLMMessage.assistant(f"Planned 2 subtasks"))

    # 步骤2: 触发子任务原子链条
    await _execute_task(session, agent)

    # 步骤3: 汇总关闭会话并盘点资产
    session.close()

    # 打印最终追溯拓扑树快照
    print(
        "\n"
        + "=" * 20
        + " 全链路嵌套树状审计日志大快照 (由高效线程安全单体节点驱动) "
        + "=" * 20
    )
    print(json.dumps(session.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
