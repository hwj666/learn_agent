import json
import re
import logging
from typing import Any, Dict, List, Tuple

from agent.base_agent import BaseAgent
from core.config import AgentConfig
from core.message import LLMMessage

logger = logging.getLogger(__name__)


class DynamicPlanExecuteAgent(BaseAgent):
    def __init__(self, config: AgentConfig, session_id: str):
        super().__init__(config, session_id)

        # ====================== 阶段 A：提示词优化 ======================
        self.plan_system = """
        你是一个具备高度自适应能力的【任务规划与动态调试专家】。你负责主导整个任务的生命周期。

        【核心运行权责】
        1. 维护动态待办清单：你手头有一个 `manager_todo` 工具。在开始复杂任务前，或在后续遇到报错、计划受阻时，你必须优先调用它来更新你的清单。
        ▪ 如果遇到错误/失败：绝不能盲目推进！你必须立刻调用 `manager_todo` 追加“调试/修复/参数修改”步骤，原地重新尝试。

        2. 严格工具调用：你必须且只能通过【调用工具 (Tool Calls)】来维护清单。
        3. 推进业务阶段：如果当前清单已是最新的，不需要修改，你必须输出【纯文本决策】，清晰描述你下一步想要执行的具体业务动作。严禁直接在回复中写类似 "python test.py" 的低级代码命令，用人类语言描述你的意图。
        4. 判定最终结束：只有当你对照最新的待办清单，确认【原始任务的所有子任务已彻底、完全、成功搞定】，你才允许输出纯文本：[FINISH]
        """.strip()

        self.plan_prompt = """
        ### 当前执行上下文 ###
        【原始用户任务】: {user_query}
        【当前动态待办清单 (来自你之前记录的状态)】: 
        {todo_list_summary}

        【最近行动历史】: 
        {history_str}
        【上一步工具真实返回】: {last_result}

        ### 任务指令 ###
        请审阅上述状态，做出当前步骤的决策：
        ▪ 如果判定所有拆解的任务已全部圆满成功完成，请直接回复纯文本：[FINISH]

        ▪ 如果需要创建、修改、打勾待办清单，请直接调用 `manager_todo` 工具。

        ▪ 如果不需要修改清单，请输出【一段纯文本】，清晰宣告你下一步要执行的业务动作（例如：“我现在需要运行测试脚本来验证接口返回值”）。

        """.strip()

        # ====================== 阶段 B：提示词优化 (补齐参数提取源) ======================
        self.exec_system = """
        你是一个极其严谨、不带任何感情色彩的【低级工具调用翻译器】。
        你的唯一任务：将用户的【当前工具动作】精准翻译成具体的【工具调用 (Tool Call)】。

        【硬性死命令】
        1. 绝对不要解释！绝对不要回复任何纯文字！
        2. 必须且只能输出指定的 Tool Call 格式。
        3. 转换动作所需的具体参数（如代码内容、文件路径、接口ID等），你必须且只能从【上一步工具真实返回】或【全局终极任务】中提取，严禁凭空捏造和盲目猜测参数！
        """.strip()

        self.exec_prompt = """
        【全局终极任务】: {user_query}
        【上一步工具真实返回（提取参数、代码、路径的唯一核心依据）】: 
        {last_result}

        【当前需要转化的工具动作】: {decision}

        请立刻根据上面的动作，从授权的工具集中选择最匹配的工具进行调用。仔细阅读“上一步工具真实返回”以保证参数完全正确。不要解释，直接调用工具！
        """.strip()
        self.ctx["_executor"] = self.executor
        self.ctx["_client"] = self.client

    async def run(self, user_query: str) -> str:
        """引擎主控制流入口"""
        # 核心状态初始化
        todo_list_summary = "【初始状态】尚未调用 todo 工具。请你在第一步先调用 action='add' 规划并拆解你的任务清单！"
        last_result = "系统刚刚启动，请开始进行任务拆解并给出第一步的具体工具行动。"

        has_called_tool = False
        has_success = False
        execution_history: List[str] = []
        continuous_todo_count = 0

        step = 0
        while step < self.max_steps:
            step += 1
            logger.info(
                f"==================== plan 步骤 {step}/{self.max_steps} ===================="
            )

            # ------------------ 阶段 A：滚动规划与决策 ------------------
            decision, resp = await self._stage_a_plan(
                user_query, todo_list_summary, execution_history, last_result
            )

            # 强防御性结束拦截
            is_empty_tool_calls = resp.tool_calls is None or len(resp.tool_calls) == 0
            if "[FINISH]" in decision and is_empty_tool_calls:
                exit_msg, should_continue = self._handle_finish_intercept(
                    has_called_tool, has_success
                )
                if should_continue:
                    last_result = exit_msg
                    continue
                return exit_msg

            # TodoTool 拦截分流引擎
            if resp.tool_calls and len(resp.tool_calls) > 0:
                has_called_tool = True
                continuous_todo_count += 1

                # 熔断保护：防止模型原地无限维护清单
                if continuous_todo_count > 3:
                    logger.warning(
                        "🚨 触发架构熔断：模型陷入连续维护 TODO 清单死循环，强行拦截。"
                    )
                    last_result = "系统警告：检测到你连续多次在原地更新待办清单，请立刻停止调用 manager_todo！请直接输出纯文本宣告下一步的具体业务行动。"
                    continuous_todo_count = 0
                    continue

                # 执行清单维护并切断通路，直接进入下一轮循环
                todo_list_summary, last_result = await self._execute_todo_tool(
                    resp.tool_calls, step, execution_history
                )
                continue

            # 成功执行非 Todo 决策，重置熔断计数器
            continuous_todo_count = 0

            # 强防御性纯文本拦截
            if not decision:
                logger.warning("❌ 拦截：模型决策内容为空且未触发工具")
                last_result = "错误：检测到你没有在文本中给出明确的下一步行动指令，也未调用 todo 工具。请给出具体的行动指示。"
                execution_history.append("模型无有效输出，已被系统拦截。")
                continue

            execution_history.append(f"步骤 {step} 决策: {decision}")
            logger.info(
                f"==================== exec 步骤 {step}/{self.max_steps} ===================="
            )
            # ------------------ 阶段 B：业务参数提取 (正则兜底) ------------------
            final_tool_calls = await self._stage_b_extract_parameters(
                user_query, decision, last_result
            )

            # 提取失败拦截
            if len(final_tool_calls) == 0:
                last_result = f"错误：工具引擎未能将动作 [{decision}] 转化为有效的业务 Tool Call。请在下一轮规划中给出更清晰的指令。"
                execution_history.append(
                    f"步骤 {step} 异常: 动作无法转化为业务 Tool Call。"
                )
                continue

            # ------------------ 阶段 C：执行业务工具并获取返回 ------------------
            has_success, output_text = await self._stage_c_execute_business_tools(
                final_tool_calls
            )
            has_called_tool = True
            last_result = output_text

        return "超过最大步骤限制，任务未能完成。"

    # =========================================================================
    # 私有辅助函数：各阶段业务原子化抽离
    # =========================================================================

    async def _stage_a_plan(
        self, user_query: str, todo_summary: str, history: List[str], last_res: str
    ) -> Tuple[str, Any]:
        """阶段 A：获取规划层 LLM 决策结果"""
        history_str = "\n".join(history[-4:]) if history else "无"
        plan_prompt = self.plan_prompt.format(
            user_query=user_query,
            todo_list_summary=todo_summary,
            history_str=history_str,
            last_result=last_res,
        )
        planner_msgs = [
            LLMMessage.system(self.plan_system),
            LLMMessage.user(plan_prompt),
        ]
        resp = await self.client.chat(
            messages=planner_msgs, tools=[self.executor.get_tool("manager_todo")]
        )
        decision = resp.content.strip() if resp.content else ""
        return decision, resp

    def _handle_finish_intercept(
        self, has_called_tool: bool, has_success: bool
    ) -> Tuple[str, bool]:
        """校验 [FINISH] 状态，防止模型未做工作或带着错误摆烂退出"""
        if not has_called_tool:
            return (
                "错误：检测到你试图在未执行任何工具前结束任务。请先调用工具拆解或执行任务。",
                True,
            )
        if not has_success:
            return (
                "错误：任务未成功完成（历史步骤存在未解决的错误），请给出修复/调试动作继续执行。",
                True,
            )
        logger.info(f"✅ 任务顺利完成退出")
        return "任务已成功通过 TODO 清单动态拆解并执行完毕。", False

    async def _execute_todo_tool(
        self, tool_calls: List[Any], step: int, history: List[str]
    ) -> Tuple[str, str]:
        """执行本地的 TodoTool 任务管理工具"""
        logger.info(f"📋 检测到 TodoTool 调用，启动框架层本地执行拦截。")
        try:
            results = await self.executor.execute(
                tool_calls=tool_calls, ctx=self.ctx, timeout=30
            )
            tool_output = "\n".join(m.content for m in results)
            logger.info(f"[Todo工具调用结果] {tool_output}")

            # 尝试格式化最新的待办统计简报
            todo_summary = self._parse_todo_summary(tool_output)
            history.append(f"步骤 {step} 维护了待办清单")
            return todo_summary, tool_output
        except Exception as e:
            err_msg = f"错误：调用 TODO 工具失败，原因: {str(e)}"
            history.append(f"步骤 {step} 维护待办清单失败")
            return "【异常】清单更新失败", err_msg

    def _parse_todo_summary(self, tool_output: str) -> str:
        """解析 Todo 工具的返回数据，生成摘要字符串"""
        try:
            res_data = json.loads(tool_output)
            if res_data.get("success"):
                todos = res_data.get("todos", [])
                lines = [
                    f"- [Index: {t['index']}] {t['title']} (状态: {t['status']})"
                    for t in todos
                ]
                summary = res_data.get("summary", {})
                lines.append(
                    f"\n📊 统计简报: 总计 {summary['total']} 项 | 进行中 {summary['in_progress']} | 已完成 {summary['completed']}"
                )
                return "\n".join(lines)
        except Exception:
            pass
        return tool_output  # 降级退化处理

    async def _stage_b_extract_parameters(
        self, user_query: str, decision: str, last_result: str
    ) -> List[Dict[str, Any]]:
        """阶段 B：利用参数提取层 LLM 生成工具参数，并配合正则进行隐式 JSON 纠错"""
        exec_prompt = self.exec_prompt.format(
            user_query=user_query, decision=decision, last_result=last_result
        )
        exec_msgs = [LLMMessage.system(self.exec_system), LLMMessage.user(exec_prompt)]
        llm_result = await self.client.chat(
            messages=exec_msgs, tools=self.executor.tools
        )

        final_tool_calls = llm_result.tool_calls or []
        return final_tool_calls

    async def _stage_c_execute_business_tools(
        self, tool_calls: List[Any]
    ) -> Tuple[bool, str]:
        """阶段 C：下发真正的业务工具并做深层结果审计"""
        logger.info(f"[工具调用中... 触发了 {len(tool_calls)} 个业务工具调用]")
        try:
            results = await self.executor.execute(tool_calls=tool_calls, ctx=self.ctx)
            output_text = "\n".join(m.content for m in results)

            # 深层业务成功审计，规避“运行未崩溃、业务实则报错”的盲区
            if any(
                keyword in output_text.lower() for keyword in ["错误", "fail", "error"]
            ):
                logger.warning(f"⚠️ 业务工具运行未崩溃，但内容中包含业务级报错标识。")
                return False, output_text

            return True, output_text
        except Exception as e:
            logger.error(f"❌ 业务工具内部发生系统级崩溃或超时")
            return False, f"错误：业务工具内部发生崩溃或超时: {str(e)}"
