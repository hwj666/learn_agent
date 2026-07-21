import asyncio
import json
import logging
import re
from typing import List, Any, Optional
from core.agent_context import AgentContext
from common.metadata import ReActTurnMetadata, CallLlmMetadata, ExecuteToolMetadata
from tracing import AgentSpanContext
from tracing import AgentTracker
from common.message import LLMMessage, ToolCall, ToolResult
from core.openai_client import OpenAIClient
from tools import ToolExecutor

logger = logging.getLogger("ReActExecution")


class ReActExecution:
    """微观 ReAct 物理探针核心引擎（全异步·零死锁·极简契约完美对齐版）"""

    FINAL_ANSWER_PATTERN = re.compile(r"^\s*【最终答案】：", re.MULTILINE)

    def __init__(
        self,
        openai_client: OpenAIClient,
        tool_executor: ToolExecutor,
        max_turns: int = 60,
    ):
        self.client = openai_client
        self.tool_executor = tool_executor
        self.max_turns = max_turns

    async def run(
        self, current_task_desc: str, tracker: AgentTracker, context: AgentContext
    ) -> str:
        system_prompt = (
            "你是一个勤奋的体力工作者，用注册的工具完成当前任务。\n"
            "当你收集到足够的信息时，在回复中提供最终答案，但必须且只能在结果开头包含标记'【最终答案】：'。\n"
            "始终用用户提问的语言回答。"
        )

        messages = [
            LLMMessage.system(system_prompt),
            LLMMessage.user(f"Context Facts: \nCurrent Task: {current_task_desc}"),
        ]

        for turn in range(1, self.max_turns + 1):
            span_name = f"react_turn_{turn}"

            turn_meta = ReActTurnMetadata(
                description=f"Turn {turn}: Thought-Action-Observation",
                message_count=len(messages),
            )

            async with AgentSpanContext(
                tracker, span_name=span_name, metadata=turn_meta, kind="INTERNAL"
            ) as ctx:
                # 1. 异步请求大模型进行思考
                llm_response = await self._call_llm(messages, tracker, ctx.span)

                # 2. 最终答案拦截判定
                if llm_response.content and self.FINAL_ANSWER_PATTERN.search(
                    llm_response.content
                ):
                    logger.info("[ReAct] Found final answer. Exiting loop.")
                    return llm_response.content

                # 3. 无工具调用且无最终答案，防止死锁
                if not llm_response.tool_calls:
                    logger.warning(
                        "[ReAct] No tool calls and no final answer. Retrying turn..."
                    )
                    continue

                # 4. 并发执行物理工具调用
                tool_results = await self._execute_tools(
                    llm_response.tool_calls, tracker, context
                )

                # 5. 补充工具返回结果元数据
                turn_meta.tool_results = [
                    {
                        "role": r.role.value,
                        "content": r.content[:60] if r.content else "",
                        "success": r.success,
                    }
                    for r in tool_results
                ]

                # 6. 强制刷新第二次工具元数据流
                await tracker.update_metadata_stream(span=ctx.span, metadata=turn_meta)
                messages.extend(
                    [
                        LLMMessage.assistant(
                            llm_response.content,
                            llm_response.reasoning_content,
                            llm_response.tool_calls,
                        ),
                        *tool_results,
                    ]
                )

                # 7. 纯计算预算复核
                await tracker.check_budget_pure(span=ctx.span)

        raise RuntimeError(f"Max turns limit reached ({self.max_turns}).")

    async def _call_llm(
        self, messages: List[LLMMessage], tracker: AgentTracker, span: Any
    ) -> Any:
        span_name = "call_llm"
        metadata = CallLlmMetadata(description="LLM Thought Reflection")

        async with AgentSpanContext(
            tracker, span_name=span_name, metadata=metadata, kind="CLIENT"
        ) as ctx:
            llm_response = await self.client.chat(messages, tools=None)

            if hasattr(llm_response, "usage") and llm_response.usage:
                tokens = llm_response.usage.get(
                    "prompt_tokens", 0
                ) + llm_response.usage.get("completion_tokens", 0)
                await tracker.record_token_consume(span=ctx.span, tokens=tokens)
                metadata.token_usage = llm_response.usage

            metadata.content = llm_response.content
            metadata.reasoning = llm_response.reasoning_content
            metadata.has_tool_calls = bool(llm_response.tool_calls)

            await tracker.update_metadata_stream(span=ctx.span, metadata=metadata)

            messages.append(
                LLMMessage.assistant(
                    llm_response.content,
                    llm_response.reasoning_content,
                    llm_response.tool_calls,
                )
            )
        return llm_response

    async def _execute_tool(
        self,
        tracker: AgentTracker,
        tool_call: ToolCall,
        context: AgentContext,
        timeout: int = 100,
    ) -> ToolResult:
        """单管道工具执行核心"""
        span_name = f"tool_call:{tool_call.name}"
        metadata = ExecuteToolMetadata(
            description=f"Executing tool {tool_call.name}",
            name=tool_call.name,
            arguments=tool_call.arguments,
        )

        async with AgentSpanContext(
            tracker, span_name=span_name, metadata=metadata, kind="CLIENT"
        ) as ctx:
            tool_instance = self.tool_executor.get_tool(tool_call.name)

            res = await self.tool_executor.execute(
                tool_call=tool_call,
                tool_instance=tool_instance,
                ctx=context.model_dump(),
                timeout=timeout,
            )

            metadata.result = res.content
            metadata.result_truncated = res.content[:200] if res.content else ""
            metadata.status = "COMPLETED" if res.success else "FAILED"

            await tracker.update_metadata_stream(span=ctx.span, metadata=metadata)
            return res

    async def _execute_tools(
        self, tool_calls: List[ToolCall], tracker: AgentTracker, context: AgentContext
    ) -> List[ToolResult]:
        now = time.time()
        session_remaining_budget = min(
            10.0, max(1.0, tracker.local_deadline - now - 0.5)
        )

        logger.info(
            "[Tool_Dispatcher] 🔌 [DISPATCH] Invoking tool executor. Global remaining budget: %.2fs",
            session_remaining_budget,
        )

        span_name = "Batch Tool Dispatcher"
        batch_metadata = ExecuteToolMetadata(
            description="Concurrent tool execution manager"
        )

        async with AgentSpanContext(
            tracker=tracker,
            span_name=span_name,
            metadata=batch_metadata,
            kind="INTERNAL",
        ) as ctx:
            tasks = [
                self._execute_tool(
                    tracker=tracker,
                    tool_call=call,
                    context=context,
                    timeout=int(session_remaining_budget),
                )
                for call in tool_calls
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            final_messages: List[ToolResult] = []
            for i, res in enumerate(results):
                tool_call = tool_calls[i]

                if isinstance(res, Exception):
                    logger.error(
                        "[Tool_Dispatcher] Tool %s crashed: %s",
                        tool_call.name,
                        str(res),
                    )
                    final_messages.append(
                        ToolResult(
                            success=False,
                            content=f"Error: {str(res)}",
                            error=str(res),
                            structured_content={"error": str(res)},
                        )
                    )
                else:
                    final_messages.append(res)

            await tracker.update_metadata_stream(span=ctx.span, metadata=batch_metadata)
        return final_messages
