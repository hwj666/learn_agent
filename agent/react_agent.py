from dataclasses import asdict
import json
import hashlib
import logging
from typing import List
from core.openai_client import OpenAIClient

from schema.context import ExecutionContext
from schema.message import LLMMessage, LLMResponse, ToolCall
from schema.node import NodeRecord
from schema.enums import NodeStatus
from tools.execute import ToolExecutor


class ReActExecution:
    """微观 ReAct 物理探针"""

    def __init__(
        self,
        openai_client: OpenAIClient,
        tool_executor: ToolExecutor,
        max_turns: int = 6,
    ):
        self.client = openai_client
        self.tool_executor = tool_executor
        self.max_turns = max_turns
        self.logger = logging.getLogger("ReActExecutor")
        self.model = "gpt-4o-mini"

    async def _call_llm(
        self, messages: List[LLMMessage], ctx: ExecutionContext
    ) -> LLMResponse:
        try:
            # ✅ 透传消息与工具 Schema
            llm_response: LLMResponse = await self.client.chat(
                messages=messages, tools=self.tool_executor.tools
            )

            if llm_response.usage:
                ctx.add_token_cost(
                    llm_response.usage.get("prompt_tokens", 0),
                    llm_response.usage.get("completion_tokens", 0),
                )

            self.logger.debug(
                f"LLM returned | "
                f"content_len={len(llm_response.content or '')} | "
                f"tool_calls={len(llm_response.tool_calls or [])}"
            )

            return llm_response

        except Exception as e:
            self.logger.error("LLM chat call failed", exc_info=True)
            raise RuntimeError(f"LLM 调用失败: {e}")

    async def run(
        self,
        current_task_desc: str,
        context_data: dict,
        ctx: ExecutionContext,
    ) -> str:
        """执行 ReAct 循环"""

        system_prompt = (
            "你是一个勤奋的体力工作者,用注册的工具完成当前任务。\n"
            "当你收集到足够的信息时，在回复中提供详细的最终答案。\n"
            "信息够了之后不要再使用工具。始终用用户提问的语言回答。 "
        )

        messages = [
            LLMMessage.system(system_prompt),
            LLMMessage.user(
                f"Context: {json.dumps(context_data, ensure_ascii=False)}\n"
                f"Current Task: {current_task_desc}"
            ),
        ]

        current_turn_node = None

        for turn in range(1, self.max_turns + 1):
            ctx.check_expiration()

            current_turn_node = NodeRecord(
                node_id=f"{ctx.execution_id}_Turn_{turn}",
                node_type="ReAct_Micro_Turn",
                status=NodeStatus.RUNNING,
                description=f"Turn {turn}: Thought-Action-Observation",
                input_data={
                    "message_count": len(messages),
                    "task_description": current_task_desc,
                },
            )
            ctx.parent_node.children.append(current_turn_node)
            ctx.push_node(current_turn_node)

            try:
                llm_response: LLMResponse = await self._call_llm(messages, ctx)

                current_turn_node.output_data = {
                    "content": llm_response.content,
                    "reasoning": llm_response.reasoning_content,
                    "has_tool_calls": bool(llm_response.tool_calls),
                }

                messages.append(
                    LLMMessage.assistant(
                        llm_response.content,
                        llm_response.reasoning_content,
                        llm_response.tool_calls,
                    )
                )

                if not llm_response.tool_calls:
                    if llm_response.content and llm_response.content.strip():
                        current_turn_node.mark_success()
                        ctx.pop_node()
                        self.logger.info(f"Task completed at turn {turn}")
                        return llm_response.content.strip()
                    else:
                        messages.append(
                            LLMMessage.user(
                                "Please provide your final answer in the response text. "
                                "If you need to use tools, include tool calls. "
                                "Otherwise, provide a complete answer now."
                            )
                        )
                        current_turn_node.status = NodeStatus.SUCCESS
                        ctx.pop_node()
                        continue

                current_turn_node.tool_calls = llm_response.tool_calls
                fp = self._get_fingerprint(llm_response.tool_calls)

                # 🛡️ 体验优化：发现死循环时不直接挂断，喂给大模型一条警告，逼它自愈
                if ctx.has_fingerprint(fp):
                    self.logger.warning(
                        f"Loop detected! Fingerprint: {fp[:8]}. Warning LLM."
                    )
                    messages.append(
                        LLMMessage.user(
                            "【系统警告】检测到你正在用完全相同的参数重复调用工具，系统已拦截该行为。\n"
                            "请仔细检查你是否陷入了逻辑死循环。如果是工具结果不符合预期，请更换参数、换用其他工具，或直接基于现有信息给出最终回答。"
                        )
                    )
                    current_turn_node.status = NodeStatus.SUCCESS
                    ctx.pop_node()
                    continue

                ctx.add_fingerprint(fp)

                # 执行工具
                tool_results = await self._execute_tools(llm_response.tool_calls, ctx)

                # 🛠️ 修复点 1：使用模型统一的序列化方法 model_dump() 代替非标的 to_dict()
                current_turn_node.tool_results = [r.to_dict() for r in tool_results]
                current_turn_node.mark_success()
                ctx.pop_node()

                messages.extend(tool_results)
                ctx.check_expiration()

            except Exception as e:
                if current_turn_node:
                    current_turn_node.mark_failure(str(e))
                ctx.pop_node()
                self.logger.error(f"Turn {turn} failed: {e}", exc_info=True)
                raise

        error_msg = f"Max turns ({self.max_turns}) exceeded without completion"
        if current_turn_node:
            current_turn_node.mark_failure(error_msg)
        raise RuntimeError(error_msg)

    def _get_fingerprint(self, tool_calls: List[ToolCall]) -> str:
        features = []
        for tc in tool_calls:
            args = tc.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw_string": args}
            elif not isinstance(args, dict):
                args = {}

            features.append(
                f"{tc.name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            )

        return hashlib.md5("||".join(features).encode()).hexdigest()

    async def _execute_tools(
        self, tool_calls: List[ToolCall], ctx: ExecutionContext
    ) -> List[LLMMessage]:
        """执行工具调用"""
        # 🛠️ 修复点 2：防御性时间限幅，防止计算出 0 或负数导致异步 wait_for 崩溃
        safe_remaining = max(1.0, ctx.remaining_time - 0.5)
        tool_timeout = min(10.0, safe_remaining)

        executor_ctx = {
            "session_id": ctx.session.session_id,
            "execution_id": ctx.execution_id,
            "agent_id": f"worker_{ctx.execution_id}",
            "timeout_limit_seconds": tool_timeout,
            "remaining_session_time": ctx.session_view.remaining_time,
        }
        results = await self.tool_executor.execute(tool_calls, ctx=executor_ctx)
        return results
