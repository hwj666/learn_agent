import time
import logging
from typing import List, Dict, Any

from schema.session import SessionContext
from schema.execution import ExecutionContext, ReadonlySessionView
from schema.node import NodeRecord
from schema.enums import NodeStatus
from .react_executor import ReActExecution


class MacroPlanner:
    """宏观规划器 - 上帝视角"""

    def __init__(self, planner_client, react_executor: ReActExecution):
        self.client = planner_client  # 用于规划的 LLM 客户端
        self.worker = react_executor
        self.logger = logging.getLogger("MacroPlanner")

    async def run_workflow(self, user_query: str, session: SessionContext):
        """执行宏观工作流"""
        session.user_query = user_query
        session.status = SessionStatus.RUNNING
        session.logger.info(f"Starting workflow for query: {user_query[:50]}...")

        # 创建 Planner 根节点
        planner_node = NodeRecord(
            node_id="Planner_Root",
            node_type="Planner",
            status=NodeStatus.RUNNING,
            description="Analyzing user request and planning tasks",
        )
        session.add_root_node(planner_node)

        try:
            # 步骤1: 任务拆解
            sub_tasks = await self._plan_tasks(user_query, session)
            planner_node.output_data = {"planned_tasks": sub_tasks}
            planner_node.mark_success()

            session.add_global_message(
                LLMMessage.assistant(f"Planned {len(sub_tasks)} subtasks")
            )

            # 步骤2: 顺序执行子任务
            for task in sub_tasks:
                await self._execute_task(task, session)

            # 步骤3: 完成会话
            session.finalize(SessionStatus.COMPLETED)
            session.logger.info("Workflow completed successfully")

        except TimeoutError as e:
            session.finalize(SessionStatus.TIMEOUT)
            planner_node.mark_failure(str(e))
            session.logger.error(f"Workflow timeout: {e}")
            raise
        except Exception as e:
            session.finalize(SessionStatus.PLANNER_FAILED)
            planner_node.mark_failure(str(e))
            session.logger.error(f"Workflow failed: {e}", exc_info=True)
            raise

    async def _plan_tasks(
        self, user_query: str, session: SessionContext
    ) -> List[Dict[str, Any]]:
        """规划子任务（模拟实现）"""
        # 实际应调用 LLM 进行规划
        # 这里返回模拟任务
        return [
            {
                "idx": "1",
                "desc": "Query sales data for top 3 products in East China region, May 2026",
            },
            {
                "idx": "2",
                "desc": "Analyze logistics delays and customer feedback for these products",
            },
        ]

    async def _execute_task(self, task: Dict[str, Any], session: SessionContext):
        """执行单个子任务"""
        t_idx = task["idx"]
        t_desc = task["desc"]

        # 创建任务节点
        task_node = NodeRecord(
            node_id=f"Task_{t_idx}",
            node_type="SubTask_Execution",
            status=NodeStatus.RUNNING,
            description=f"Executing task {t_idx}: {t_desc}",
            input_data={"consensus_snapshot": session.consensus_pool.copy()},
        )
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

            self.logger.info(f"Starting task {t_idx} with deadline {local_deadline}")

            # 执行任务
            task_report = await self.worker.run(
                current_task_desc=t_desc,
                context_data=session.consensus_pool,
                ctx=local_ctx,
            )

            # 记录结果
            task_node.output_data = {"final_report": task_report}
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

            self.logger.info(f"Task {t_idx} completed successfully")

        except Exception as e:
            task_node.mark_failure(str(e))
            self.logger.error(f"Task {t_idx} failed: {e}")
            raise
