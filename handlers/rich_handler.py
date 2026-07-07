import time
import json
import re
from collections import deque
from typing import Optional, Type, Dict, Deque, Any, List
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from handlers.base import BaseStreamHandler


class RichStreamHandler(BaseStreamHandler):
    def __init__(
        self,
        console: Optional[Console] = None,
        refresh_interval: float = 0.05,
        max_buffer_size: int = 50000,
        theme: str = "monokai",
        panel_styles: Optional[Dict] = None,
    ):
        super().__init__()
        self.console = console or Console()
        self.live: Optional[Live] = None
        self.refresh_interval = max(0.016, min(refresh_interval, 0.1))
        self.last_refresh_time = 0.0
        self.max_buffer_size = max_buffer_size
        self.theme = theme

        self.panel_styles = panel_styles or {
            "think": {
                "active": {
                    "border_style": "yellow",
                    "title": "[THINK] 思考链 (Reasoning)",
                },
                "done": {"border_style": "dim", "title": "[THINK] 思考链 (已结束)"},
            },
            "tool": {
                "active": {
                    "border_style": "magenta",
                    "title": "[TOOL] 工具调用参数 (Tool Calls)",
                },
                "done": {"border_style": "dim", "title": "[TOOL] 工具调用 (已就绪)"},
            },
            "respond": {
                "active": {
                    "border_style": "green",
                    "title": "[RESPOND] 模型回复 (Assistant Response)",
                },
                "done": {"border_style": "dim", "title": "[RESPOND] 模型回复 (完成)"},
            },
        }

        self.stats = {
            "total_updates": 0,
            "total_renders": 0,
            "max_render_time": 0.0,
            "slow_renders": 0,
        }

        self.reset()

    def reset(self) -> None:
        """彻底清空所有缓冲区，防止多轮对话数据污染"""
        self.thinking_buffer: Deque[str] = deque(maxlen=self.max_buffer_size)
        self.responding_buffer: Deque[str] = deque(maxlen=self.max_buffer_size)
        # 工具调用参数使用列表存储原始片段
        self.tool_calling_fragments: List[str] = []
        self.current_stage = "idle"
        self.is_complete = False

    async def open(self) -> None:
        self.reset()
        self.last_refresh_time = time.time()
        self.is_complete = False

        self.live = Live(
            self._build_renderable(),
            console=self.console,
            auto_refresh=False,
            transient=False,
            vertical_overflow="visible",
        )
        self.live.start()

    async def close(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException]
    ) -> None:
        if self.live:
            self.is_complete = True
            try:
                self.live.update(self._build_renderable(), refresh=True)
            except Exception as e:
                self.console.print(f"[red]X 最终渲染失败: {e}[/red]")
            finally:
                self.live.stop()
                self.live = None

                if self.stats["total_updates"] > 0:
                    avg_render_time = (
                        self.stats["total_renders"] / self.stats["total_updates"]
                    )
                    self.console.print(
                        f"\n[dim]渲染统计: "
                        f"总更新 {self.stats['total_updates']} 次, "
                        f"平均渲染 {avg_render_time * 1000:.1f}ms, "
                        f"最慢 {self.stats['max_render_time'] * 1000:.1f}ms[/dim]"
                    )
                self.console.print()

    def _ensure_string(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        elif isinstance(value, (list, dict)):
            try:
                return json.dumps(value, ensure_ascii=False, indent=2)
            except:
                return str(value)
        elif value is None:
            return ""
        else:
            return str(value)

    def _extract_tool_calls(self) -> str:
        """
        提取所有工具调用片段中的 arguments 并拼接成一个完整的 JSON 字符串。

        关键逻辑：
        1. 每个片段都是 [{"index":0, "id":..., "name":..., "arguments":"..."}]
        2. 我们只需要提取每个片段中的 arguments 字符串部分
        3. 将所有 arguments 字符串拼接起来，形成完整的 JSON
        """
        if not self.tool_calling_fragments:
            return ""

        # 收集所有 arguments 字符串片段
        arguments_parts = []

        for fragment in self.tool_calling_fragments:
            fragment = fragment.strip()
            if not fragment:
                continue

            try:
                # 解析片段为 JSON 数组
                parsed = json.loads(fragment)
                if isinstance(parsed, list) and len(parsed) > 0:
                    obj = parsed[0]
                    if isinstance(obj, dict) and "arguments" in obj:
                        arg_str = obj["arguments"]
                        if arg_str:
                            arguments_parts.append(arg_str)
            except json.JSONDecodeError:
                # 如果解析失败，尝试直接提取 arguments 字段
                # 使用正则表达式匹配 "arguments": "..." 模式
                match = re.search(r'"arguments"\s*:\s*"([^"]*)"', fragment)
                if match:
                    arguments_parts.append(match.group(1))
                else:
                    # 最后手段：直接添加片段
                    arguments_parts.append(fragment)

        # 拼接所有 arguments 片段
        full_arguments = "".join(arguments_parts)

        # 尝试构建完整的工具调用对象
        try:
            # 尝试解析完整的 arguments JSON
            parsed_args = json.loads(full_arguments)
            # 构建完整的工具调用对象
            tool_call = {
                "index": 0,
                "id": "merged",
                "name": "manager_todo",
                "arguments": parsed_args,
            }
            return json.dumps([tool_call], ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            # 如果 arguments 还不是完整的 JSON，返回原始拼接
            return json.dumps(
                [
                    {
                        "index": 0,
                        "id": "merged",
                        "name": "manager_todo",
                        "arguments": full_arguments,
                    }
                ],
                ensure_ascii=False,
                indent=2,
            )

    def _build_renderable(self) -> Group:
        parts = []

        try:
            thinking_text = "".join(self.thinking_buffer).strip()
            responding_text = "".join(self.responding_buffer).strip()

            # 提取并合并工具调用参数
            tool_text = self._extract_tool_calls().strip()

            # 1. 思考链卡片
            if thinking_text:
                is_active = (self.current_stage == "think") and not self.is_complete
                style_key = "active" if is_active else "done"
                style = self.panel_styles["think"][style_key]
                parts.append(
                    Panel(
                        thinking_text,
                        title=style["title"],
                        border_style=style["border_style"],
                        title_align="left",
                        padding=(0, 1),
                    )
                )

            # 2. 工具调用卡片（显示合并后的单一工具调用）
            if tool_text:
                is_active = (self.current_stage == "tool") and not self.is_complete
                style_key = "active" if is_active else "done"
                style = self.panel_styles["tool"][style_key]

                # 尝试美化 JSON 显示
                try:
                    parsed = json.loads(tool_text)
                    formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
                    tool_renderable = Syntax(
                        formatted, "json", theme=self.theme, background_color="default"
                    )
                except (json.JSONDecodeError, ValueError):
                    tool_renderable = Text(tool_text)

                parts.append(
                    Panel(
                        tool_renderable,
                        title=style["title"],
                        border_style=style["border_style"],
                        title_align="left",
                        padding=(0, 1),
                    )
                )

            # 3. 模型回复卡片
            if responding_text:
                is_active = (self.current_stage == "respond") and not self.is_complete
                style_key = "active" if is_active else "done"
                style = self.panel_styles["respond"][style_key]
                respond_renderable = Markdown(responding_text)
                parts.append(
                    Panel(
                        respond_renderable,
                        title=style["title"],
                        border_style=style["border_style"],
                        title_align="left",
                        padding=(1, 2),
                    )
                )

            # 4. 初始化占位
            if not thinking_text and not tool_text and not responding_text:
                placeholder = Text("[SYSTEM] 正在连接模型并初始化会话...", style="dim")
                parts.append(Panel(placeholder, border_style="blue"))

            # 5. 状态栏
            if self.stats["total_updates"] > 0 and any(
                [thinking_text, responding_text, tool_text]
            ):
                char_count = len(thinking_text) + len(responding_text) + len(tool_text)
                status_text = Text(
                    f"已接收 {char_count} 字符 | 更新 {self.stats['total_updates']} 次",
                    style="dim",
                )
                parts.append(status_text)

        except Exception as e:
            parts.append(
                Panel(
                    f"[red]渲染错误: {str(e)}[/red]",
                    border_style="red",
                    title="X 渲染异常",
                )
            )
            self.console.print(f"[red]X _build_renderable 错误: {e}[/red]")

        return Group(*parts)

    async def __call__(
        self,
        think: str = "",
        respond: str = "",
        tool_args: Any = "",
        chunk_type: str = "",
    ) -> None:
        try:
            has_changed = False

            # 状态机切换
            if chunk_type:
                ct = chunk_type.lower()
                if "think" in ct or "reason" in ct:
                    self.current_stage = "think"
                elif "tool" in ct:
                    self.current_stage = "tool"
                elif "respond" in ct or "content" in ct:
                    self.current_stage = "respond"

            # 思考链
            if think:
                think_str = self._ensure_string(think)
                if think_str:
                    if self.current_stage == "idle":
                        self.current_stage = "think"
                    self.thinking_buffer.append(think_str)
                    has_changed = True

            # 工具调用：收集片段
            if tool_args:
                tool_str = self._ensure_string(tool_args)
                if tool_str:
                    if self.current_stage != "tool":
                        self.current_stage = "tool"
                    self.tool_calling_fragments.append(tool_str)
                    has_changed = True

            # 模型回复
            if respond:
                respond_str = self._ensure_string(respond)
                if respond_str:
                    if (
                        self.current_stage != "respond"
                        and not think
                        and respond_str.strip()
                    ):
                        self.current_stage = "respond"
                    self.responding_buffer.append(respond_str)
                    has_changed = True

            # 限流刷新
            if has_changed and self.live:
                current_time = time.time()
                self.stats["total_updates"] += 1

                if current_time - self.last_refresh_time >= self.refresh_interval:
                    render_start = time.time()
                    try:
                        self.live.update(self._build_renderable(), refresh=True)
                    except Exception as e:
                        self.console.print(f"[red]X 实时更新失败: {e}[/red]")
                        fallback = Panel(
                            f"[yellow]正在处理数据... (错误: {e})[/yellow]",
                            border_style="yellow",
                        )
                        self.live.update(fallback, refresh=True)

                    render_time = time.time() - render_start
                    self.stats["total_renders"] += 1
                    self.stats["max_render_time"] = max(
                        self.stats["max_render_time"], render_time
                    )
                    if render_time > 0.1:
                        self.stats["slow_renders"] += 1
                        self.console.log(
                            f"! 渲染耗时 {render_time * 1000:.1f}ms (第 {self.stats['slow_renders']} 次慢渲染)"
                        )
                    self.last_refresh_time = current_time
                else:
                    try:
                        self.live.update(self._build_renderable(), refresh=False)
                    except Exception:
                        pass

        except KeyboardInterrupt:
            self.is_complete = True
            if self.live:
                try:
                    self.live.update(
                        Panel(
                            "[yellow]用户中断了流式输出[/yellow]",
                            border_style="yellow",
                            title="流式输出已中断",
                        ),
                        refresh=True,
                    )
                except Exception:
                    pass
            raise
        except Exception as e:
            self.console.print(f"\n[red]X __call__ 错误: {e}[/red]")
            raise

    def get_stats(self) -> Dict:
        return {
            "total_updates": self.stats["total_updates"],
            "total_renders": self.stats["total_renders"],
            "max_render_time_ms": round(self.stats["max_render_time"] * 1000, 2),
            "slow_renders": self.stats["slow_renders"],
            "buffer_sizes": {
                "thinking": len(self.thinking_buffer),
                "responding": len(self.responding_buffer),
                "tool_calling": len(self.tool_calling_fragments),
            },
        }
