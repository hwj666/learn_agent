import asyncio
from core.openai_client import OpenAIClient
from core.message import LLMMessage, LLMMessageBuilder
from core.config import ModelConfig
from tools.execute import ToolExecutor

# 引入 Rich 相关组件
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.console import Console

# 初始化控制台（避免多线程冲突）
console = Console()

class ReactAgent:
    def __init__(self, model_config: ModelConfig, max_steps=5, allowed_toolsets=None):
        self.client = OpenAIClient(model_config)
        self.max_steps = max_steps
        self.system_prompt = "你是一个具备自主思考和行动能力的 ReAct Agent"
        self.memory: list[LLMMessage] = [LLMMessage(role="system", content=self.system_prompt)]
        self.executor = ToolExecutor(allowed_toolsets=allowed_toolsets)

    def _create_layout(self, step: int, status: str, reasoning: str, content: str, logs: list) -> Layout:
        """构建高级双栏全屏 UI 布局"""
        layout = Layout()
        layout.split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )

        # 左侧：状态与思维链看板
        status_text = Text.assemble(
            ("当前进度: ", "bold white"), (f"{step}/{self.max_steps}\n", "bold cyan"),
            ("当前状态: ", "bold white"), (f"{status}\n\n", "bold yellow"),
            ("🤔 模型思考思维链 (Reasoning):\n", "bold magenta"),
            (reasoning if reasoning else "等待大模型思考...", "white" if reasoning else "dim white")
        )
        layout["left"].update(Panel(status_text, title="📊 Agent 核心状态", border_style="cyan"))

        # 右侧：拆分为“模型原生回复”与“工具执行日志”上下两部分
        layout["right"].split_column(
            Layout(name="reply", ratio=1),
            Layout(name="logs", ratio=1)
        )

        # 右上：大模型正文输出
        layout["right"]["reply"].update(Panel(
            content if content else "等待正文输出...",
            title="💬 模型原生回复 (Content)",
            border_style="blue"
        ))

        # 右下：运行日志
        log_content = "\n".join(logs[-10:])  # 限制最新 10 行
        layout["right"]["logs"].update(Panel(log_content, title="📜 系统工具日志", border_style="green"))

        return layout

    async def run(self, user_query: str) -> str:
        self.memory.append(LLMMessageBuilder.user(user_query))
        step = 0
        logs = [f"[bold blue]🧑💻 用户输入:[/bold blue] {user_query}"]

        # 实时渲染的核心状态变量
        current_reasoning = ""
        current_content = ""
        current_status = "初始化画布"

        # 开启全屏锁定模式
        with Live(
            self._create_layout(step, current_status, current_reasoning, current_content, logs),
            refresh_per_second=8,
            screen=False,
            console=console
        ) as live:
            while step < self.max_steps:
                step += 1
                current_status = f"正在请求模型 (Step {step})"
                # 每一轮开始，清空上一轮的单次流式输出缓存
                current_reasoning = ""
                current_content = ""
                live.update(self._create_layout(step, current_status, current_reasoning, current_content, logs))
                await asyncio.sleep(0.01)  # 让出事件循环，避免UI卡死

                # 流式更新函数（定义在循环内，绑定当前 step）
                async def on_chunk_received(delta_reasoning: str, delta_content: str, status_text: str):
                    nonlocal current_reasoning, current_content, current_status
                    current_status = f"Step {step} - {status_text}"

                    # 增量拼接
                    if delta_reasoning:
                        current_reasoning += delta_reasoning
                    if delta_content:
                        current_content += delta_content

                    # 刷新UI
                    live.update(self._create_layout(step, current_status, current_reasoning, current_content, logs))

                # 1. 传入回调，请求大模型
                llm_response = await self.client.chat(
                    messages=self.memory,
                    tools=self.executor.get_schemas(),
                    on_chunk=on_chunk_received
                )

                # 最终状态同步
                current_reasoning = llm_response.reasoning_content or ""
                current_content = llm_response.content or ""
                live.update(self._create_layout(step, current_status, current_reasoning, current_content, logs))

                # 助手消息入记忆
                assistant_msg = LLMMessageBuilder.assistant(
                    content=llm_response.content,
                    reasoning=llm_response.reasoning_content,
                    tool_calls=llm_response.tool_calls if llm_response.tool_calls else None
                )
                self.memory.append(assistant_msg)

                # 情况 A：模型给出最终回答
                if not llm_response.tool_calls:
                    current_status = "🎉 任务完成"
                    logs.append(f"[bold green]✅ 最终回答已就绪[/bold green]")
                    live.update(self._create_layout(step, current_status, current_reasoning, current_content, logs))
                    await asyncio.sleep(1)
                    return llm_response.content

                # 情况 C：执行工具
                current_status = f"Step {step} - 正在处理工具调用"
                for tool_call in llm_response.tool_calls:
                    logs.append(f"[bold yellow]🛠️ 触发工具:[/bold yellow] {tool_call.name}")
                    logs.append(f"[dim white]  参数: {tool_call.arguments}[/dim white]")
                    live.update(self._create_layout(step, current_status, current_reasoning, current_content, logs))
                    await asyncio.sleep(0.01)

                    # 执行工具
                    tool_response = await self.executor.execute(tool_call, ctx={"agent_id": "react1"})
                    self.memory.append(tool_response)

                    # 日志截断
                    truncated_res = tool_response.content[:80] + "..." if len(tool_response.content) > 80 else tool_response.content
                    logs.append(f"[bold magenta]📥 工具返回:[/bold magenta] {truncated_res}")
                    live.update(self._create_layout(step, current_status, current_reasoning, current_content, logs))
                    await asyncio.sleep(0.01)

            # 情况 B：步数耗尽
            current_status = "❌ 步数超限拦截"
            logs.append("[bold red]❌ 错误: 达到最大迭代步数，中断后续工具执行。[/bold red]")
            live.update(self._create_layout(step, current_status, current_reasoning, current_content, logs))
            await asyncio.sleep(1)
            return "错误：超过最大迭代步数，未能生成有效解答。"