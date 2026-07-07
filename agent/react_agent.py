import json
import hashlib
import logging
from typing import List
from openai import AsyncOpenAI  # 或其他 LLM 客户端

from schema.context import ExecutionContext
from schema.message import LLMMessage, LLMResponse, ToolCall
from schema.node import NodeRecord
from schema.enums import NodeStatus


class ReActExecution:
    """微观 ReAct 物理探针"""

    def __init__(self, openai_client: AsyncOpenAI, tool_executor, max_turns: int = 6):
        self.client = openai_client  # 现在明确要求 AsyncOpenAI 类型
        self.tool_executor = tool_executor
        self.max_turns = max_turns
        self.logger = logging.getLogger("ReActExecutor")
        self.model = "gpt-4o-mini"  # 或从配置读取

    async def _call_llm(
        self, messages: List[LLMMessage], ctx: ExecutionContext
    ) -> LLMResponse:
        """
        严格使用你提供的 client.chat 接口：
        - messages: List[LLMMessage]
        - tools: List[dict]
        - tool_choice: "auto"
        """
        try:
            # ✅ 不做任何格式转换，直接透传
            llm_response: LLMResponse = await self.client.chat(
                messages=messages, tools=self.tool_executor.tools
            )

            # ✅ 假设 LLMResponse 已经由 client 正确构造
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
            "你是一个勤奋的体力工作者,用注册的工具完成当前任务"
            "当你收集到足够的信息时，在回复中提供详细的最终答案。"
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
            # 前置检查
            ctx.check_expiration()

            # 创建 Turn 节点
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
                # 调用 LLM（现在使用真实的调用）
                llm_response: LLMResponse = await self._call_llm(messages, ctx)

                # 记录输出
                current_turn_node.output_data = {
                    "content": llm_response.content,
                    "reasoning": llm_response.reasoning_content,
                    "has_tool_calls": bool(llm_response.tool_calls),
                }

                # 添加助手消息到历史
                messages.append(
                    LLMMessage.assistant(
                        llm_response.content,
                        llm_response.reasoning_content,
                        llm_response.tool_calls,
                    )
                )

                # 检查是否完工（无工具调用 = 完成）
                if not llm_response.tool_calls:
                    if llm_response.content and llm_response.content.strip():
                        current_turn_node.mark_success()
                        ctx.pop_node()
                        self.logger.info(f"Task completed at turn {turn}")
                        return llm_response.content.strip()
                    else:
                        # 空应答，提示重试
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

                # 反死循环检查
                current_turn_node.tool_calls = llm_response.tool_calls
                fp = self._get_fingerprint(llm_response.tool_calls)

                if ctx.has_fingerprint(fp):
                    raise RuntimeError(f"Loop detected! Fingerprint: {fp[:8]}")

                ctx.add_fingerprint(fp)

                # 执行工具
                tool_results = await self._execute_tools(llm_response.tool_calls, ctx)

                current_turn_node.tool_results = [r.to_dict() for r in tool_results]
                current_turn_node.mark_success()
                ctx.pop_node()

                # 添加工具结果到消息历史
                messages.extend(tool_results)

                # 工具执行后再次检查超时（防止长时间工具调用）
                ctx.check_expiration()

            except Exception as e:
                if current_turn_node:
                    current_turn_node.mark_failure(str(e))
                ctx.pop_node()
                self.logger.error(f"Turn {turn} failed: {e}", exc_info=True)
                raise

        # 达到最大轮次
        error_msg = f"Max turns ({self.max_turns}) exceeded without completion"
        if current_turn_node:
            current_turn_node.mark_failure(error_msg)
        raise RuntimeError(error_msg)

    def _get_fingerprint(self, tool_calls: List[ToolCall]) -> str:
        """生成工具调用指纹"""
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
        results = []

        # 计算工具执行的超时时间
        tool_timeout = min(10.0, ctx.remaining_time - 0.5)  # 留出 0.5 秒缓冲

        for tc in tool_calls:
            try:
                self.logger.debug(
                    f"Executing tool: {tc.name} with args: {tc.arguments}"
                )

                # 准备工具执行上下文
                executor_ctx = {
                    "session_id": ctx.session.session_id,
                    "execution_id": ctx.execution_id,
                    "agent_id": f"worker_{ctx.execution_id}",
                    "timeout_limit_seconds": tool_timeout,
                    "remaining_session_time": ctx.session_view.remaining_time,
                }

                # 执行工具
                result_content = await self.tool_executor.execute(
                    tool_name=tc.name, arguments=tc.arguments, ctx=executor_ctx
                )

                # 创建工具结果消息
                result_msg = LLMMessage.tool(
                    content=str(result_content)
                    if result_content is not None
                    else "No result",
                    tool_call_id=tc.id,
                )
                results.append(result_msg)

            except Exception as e:
                self.logger.error(
                    f"Tool execution failed for {tc.name}: {e}", exc_info=True
                )
                # 即使工具失败，也要返回错误消息，让 LLM 知道
                error_msg = LLMMessage.tool(
                    content=f"Error executing tool {tc.name}: {str(e)}",
                    tool_call_id=tc.id,
                )
                results.append(error_msg)

        return results
