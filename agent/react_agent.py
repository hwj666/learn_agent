import json
import time
import hashlib
from typing import List, Any, Dict
from schema.message import LLMMessage
from schema.metadata import ReActTurnMetadata


class ReActExecution:
    """微观 ReAct 物理探针核心引擎（极致去冗余·完全体·最新修正版）"""

    def __init__(self, openai_client: Any, tool_executor: Any, max_turns: int = 60):
        self.client = openai_client
        self.tool_executor = tool_executor
        self.max_turns = max_turns

    async def run(
        self,
        current_task_desc: str,
        context_data: dict,
        session: AgentSession,
        execution_id: str,
        attempt_idx: int = 0,  # 🎯 核心补齐：无缝接收外层拓扑重试循环的 micro-slot 槽位序号
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
            # 1. 看门狗纯计算原子判定，稳死拦截超时
            session.check_budget_pure()

            # 2. 实例化强类型元数据契约（用于承载大模型反思所需的结构化记忆）
            turn_meta = ReActTurnMetadata(
                description=f"Turn {turn}: Thought-Action-Observation",
                message_count=len(messages),
            )
            node_id = f"{execution_id}_Turn_{turn}"

            with session.step(
                node_id=node_id, metadata=turn_meta, attempt_idx=attempt_idx
            ):
                # 3. 大模型物理网络 IO 探测
                llm_response = await self._call_llm(messages, session, attempt_idx)

                session.logger.info(
                    "💭 Thought Content: %s", llm_response.reasoning_content or "None"
                )

                # 4. 动态填充记忆资产
                turn_meta.content = llm_response.content
                turn_meta.reasoning = llm_response.reasoning_content
                turn_meta.has_tool_calls = bool(llm_response.tool_calls)

                # 🛡️ 架构对齐 5: 大模型的思考记忆被完美、按轮次按重试落入明细账本中
                session.update_metadata_stream(
                    node_id=node_id, metadata=turn_meta, attempt_idx=attempt_idx
                )

                messages.append(
                    LLMMessage.assistant(
                        llm_response.content,
                        llm_response.reasoning_content,
                        llm_response.tool_calls,
                    )
                )

                # 5. 空转消杀判定防御线
                if not llm_response.tool_calls:
                    reply_text = llm_response.content or ""
                    if "最终答案" in reply_text or turn == self.max_turns:
                        # 🟢 5. 延迟占位插值修复
                        session.logger.info(
                            "🎯 Verified final answer signature obtained at turn %d.",
                            turn,
                        )
                        return reply_text.strip()
                    else:
                        # 🟢 6. 延迟占位插值修复
                        session.logger.warning(
                            "⚠️ [VACANT_ALERT] LLM is idling. Issuing forced guidance."
                        )
                        messages.append(
                            LLMMessage.user(
                                "【系统策略引导】检测到你当前既没有调用工具，也没给交付成果。请立刻开始工作。"
                            )
                        )
                        continue

                # 6. 特征指纹注册去重（纯同步原子拦截判定，绝不给卡死留下任何弹药）
                fp = self._get_fingerprint(llm_response.tool_calls)
                if session.check_and_record_fingerprint(fp):
                    session.logger.warning(
                        "🚨 [REPETITION_BLOCKED] Repetitive call signature found: %s",
                        fp,
                    )
                    messages.append(
                        LLMMessage.user(
                            "【系统警告】检测到你正用完全相同的参数重复调用工具，已被强制拦截。"
                        )
                    )
                    continue

                # 7. 执行物理工具调用（依然涉及外部网络 IO，保持 await）
                tool_results = await self._execute_tools(
                    llm_response.tool_calls, session, execution_id
                )

                # 8. 补充工具返回结果元数据，并再次精准流式推送进当前重试位的结构化记忆中
                turn_meta.tool_results = [
                    {"role": r.role, "content": r.content[:60] if r.content else ""}
                    for r in tool_results
                ]
                session.update_metadata_stream(
                    node_id=node_id, metadata=turn_meta, attempt_idx=attempt_idx
                )

                messages.extend(tool_results)

                # 9. 步骤结束前做最后的纯计算预算复核
                session.check_budget_pure()

        raise RuntimeError(f"Max turns limit reached ({self.max_turns}).")

    def _get_fingerprint(self, tool_calls: List[Any]) -> str:
        """
        确定性特征指纹注册
        🔒 保证：利用 sort_keys 与 ensure_ascii=False 消除任何序列化扰动，产出绝对唯一的哈希
        """
        features = [
            f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False)}"
            for tc in tool_calls
        ]
        return hashlib.md5("||".join(features).encode()).hexdigest()

    async def _call_llm(
        self, messages: List[Any], session: AgentSession, attempt_idx: int
    ) -> Any:
        """执行异步非阻塞 LLM 网络 IO 交互，并安全划扣 Token 预算"""
        llm_response = await self.client.chat(
            messages=messages, tools=self.tool_executor.tools
        )
        if llm_response.usage:
            p_tok = llm_response.usage.get("prompt_tokens", 0)
            c_tok = llm_response.usage.get("completion_tokens", 0)
            # 🛡️ 显式透传当前大圈的 attempt_idx，彻底免除漂移
            session.consume_tokens_stream(tokens=p_tok + c_tok, attempt_idx=attempt_idx)
        return llm_response

    async def _execute_tools(
        self, tool_calls: List[Any], session: AgentSession, execution_id: str
    ) -> List[Any]:
        now = time.time()
        tool_timeout = min(10.0, max(1.0, session.local_deadline - now - 0.5))

        session.logger.info(
            "[Tool_Dispatcher] 🔌 [DISPATCH] Invoking tool executor. Timeout barrier: %.2fs",
            tool_timeout,
        )

        results = await self.tool_executor.execute(
            tool_calls,
            ctx={
                "session_id": session.session_id,
                "execution_id": execution_id,
                "timeout": tool_timeout,
            },
        )

        for r in results:
            raw_content = r.content if r.content else ""
            truncated_content = (
                raw_content[:300] + f" ... [Truncated, Total {len(raw_content)} chars]"
                if len(raw_content) > 300
                else raw_content
            )

            session.logger.info(
                "🔍 [Tool_Observation] Name: %s | Call_ID: %s\n--- OBSERVATION CONTENT ---\n%s\n---------------------------",
                getattr(r, "name", "UNKNOWN"),
                getattr(r, "tool_call_id", "UNKNOWN"),
                truncated_content,
            )

        return results
