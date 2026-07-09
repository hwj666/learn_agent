# engine/react.py
import json
import time
import hashlib
import logging
from typing import List, Any
from schema.message import LLMMessage

# 🚀 顶层显式汇聚下游的控制层与数据契约层
from schema.session import AgentSession
from schema.metadata import ReActTurnMetadata


class ReActExecution:
    """微观 ReAct 物理探针（防空转硬管制版）"""

    def __init__(self, openai_client: Any, tool_executor: Any, max_turns: int = 6):
        self.client = openai_client
        self.tool_executor = tool_executor
        self.max_turns = max_turns

    async def _call_llm(self, messages: List[Any], session: AgentSession) -> Any:
        session.log_trace(
            f"🤖 Interrogating LLM with {len(messages)} historical records..."
        )
        llm_response = await self.client.chat(
            messages=messages, tools=self.tool_executor.tools
        )

        if llm_response.usage:
            p_tok = llm_response.usage.get("prompt_tokens", 0)
            c_tok = llm_response.usage.get("completion_tokens", 0)
            session.consume_tokens(tokens=p_tok + c_tok)
        return llm_response

    async def run(
        self,
        current_task_desc: str,
        context_data: dict,
        session: AgentSession,
        execution_id: str,
    ) -> str:
        system_prompt = (
            "你是一个勤奋的体力工作者,用注册的工具完成当前任务。\n"
            "当你收集到足够的信息时，在回复中提供最终答案，但必须且只能在结果开头包含标记‘【最终答案】：’。\n"
            "始终用用户提问的语言回答。"
        )

        messages = [
            LLMMessage.system(system_prompt),
            LLMMessage.user(
                f"Context Facts: {json.dumps(context_data, ensure_ascii=False)}\nCurrent Task: {current_task_desc}"
            ),
        ]

        for turn in range(1, self.max_turns + 1):
            session.check_budget()

            # 实例化从 schema.metadata 导入的强类型契约
            turn_meta = ReActTurnMetadata(
                description=f"Turn {turn}: Thought-Action-Observation",
                message_count=len(messages),
            )

            with session.step(
                node_id=f"{execution_id}_Turn_{turn}", metadata=turn_meta
            ):
                llm_response = await self._call_llm(messages, session)
                session.log_trace(
                    f"💭 Thought Content: {llm_response.reasoning_content or 'None'}"
                )

                turn_meta.content = llm_response.content
                turn_meta.reasoning = llm_response.reasoning_content
                turn_meta.has_tool_calls = bool(llm_response.tool_calls)

                messages.append(
                    LLMMessage.assistant(
                        llm_response.content,
                        llm_response.reasoning_content,
                        llm_response.tool_calls,
                    )
                )

                # 空转消杀判定防御线
                if not llm_response.tool_calls:
                    reply_text = llm_response.content or ""
                    if "最终答案" in reply_text or turn == self.max_turns:
                        session.log_trace(
                            f"🎯 Verified final answer signature obtained at turn {turn}."
                        )
                        return reply_text.strip()
                    else:
                        session.log_trace(
                            "⚠️ [VACANT_ALERT] LLM is idling. Issuing forced guidance.",
                            level=logging.WARNING,
                        )
                        messages.append(
                            LLMMessage.user(
                                "【系统策略引导】检测到你当前既没有调用工具，也没给交付成果。请立刻开始工作。"
                            )
                        )
                        continue

                # 特征指纹注册去重
                fp = self._get_fingerprint(llm_response.tool_calls)
                if session.check_and_record_fingerprint(fp):
                    messages.append(
                        LLMMessage.user(
                            "【系统警告】检测到你正用完全相同的参数重复调用工具，已被强制拦截。"
                        )
                    )
                    continue

                # 执行物理工具调用
                tool_results = await self._execute_tools(
                    llm_response.tool_calls, session, execution_id
                )
                turn_meta.tool_results = [
                    {"role": r.role, "content": r.content[:60]} for r in tool_results
                ]
                messages.extend(tool_results)

                session.check_budget()

        raise RuntimeError(f"Max turns limit reached.")

    def _get_fingerprint(self, tool_calls: List[Any]) -> str:
        features = [
            f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False)}"
            for tc in tool_calls
        ]
        return hashlib.md5("||".join(features).encode()).hexdigest()

    async def _execute_tools(
        self, tool_calls: List[Any], session: AgentSession, execution_id: str
    ) -> List[Any]:
        now = time.time()
        tool_timeout = min(10.0, max(1.0, session.local_deadline - now - 0.5))
        session.log_trace(
            f"🔌 [DISPATCH] Invoking tool executor. Timeout barrier: {tool_timeout:.2f}s"
        )

        results = await self.tool_executor.execute(
            tool_calls,
            ctx={"session_id": session.session_id, "execution_id": execution_id},
        )
        return results
