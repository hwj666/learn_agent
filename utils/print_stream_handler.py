import sys
from typing import Optional, List, Dict


class PrintStreamHandler:
    # 🎨 定义标准 ANSI 颜色转义序列
    COLOR_RESET = "\033[0m"
    COLOR_DIM = "\033[2m"
    COLOR_BOLD = "\033[1m"

    # 各阶段专属颜色
    COLOR_SYSTEM = "\033[36m"  # 青色
    COLOR_GREY = "\033[90m"  # 灰色（亮黑色）
    COLOR_RESPOND = "\033[32m"  # 绿色
    COLOR_TOOL = "\033[35m"  # 品红

    def __init__(self):
        self.last_chunk_type: Optional[str] = None
        self.seen_tool_indices = set()
        self.tool_names_buffer: Dict[int, str] = {}

    async def __aenter__(self):
        # 建立通道时强制彻底重置所有中间状态
        self.last_chunk_type = None
        self.seen_tool_indices.clear()
        self.tool_names_buffer.clear()

        print(
            f"{self.COLOR_SYSTEM}🤖 [系统] 正在连接大模型并建立流式通道...{self.COLOR_RESET}",
            flush=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print(
            f"\n\n{self.COLOR_SYSTEM}🏁 [系统] 模型流式传输结束。\n{self.COLOR_RESET}",
            flush=True,
        )
        return False

    async def __call__(
        self, think: str, text: str, cleaned_tool_deltas: List[Dict], chunk_type: str
    ):
        if not chunk_type:
            return

        chunk_type_lower = chunk_type.lower()

        # 1. 极限防御：如果是工具调用状态但列表为空，静默退出
        if "tool" in chunk_type_lower and not cleaned_tool_deltas:
            return

        # 2. 状态机切换检测
        if chunk_type_lower != self.last_chunk_type:
            if "think" in chunk_type_lower or "reason" in chunk_type_lower:
                print(
                    f"\n\n{self.COLOR_GREY}{self.COLOR_DIM}{self.COLOR_BOLD}🧠 ============= [思考链开始] ============={self.COLOR_RESET}"
                )
                self.last_chunk_type = "thinking"
            elif "respond" in chunk_type_lower or "content" in chunk_type_lower:
                print(
                    f"\n\n{self.COLOR_RESPOND}{self.COLOR_BOLD}🤖 ============= [模型正式回复] ============={self.COLOR_RESET}"
                )
                self.last_chunk_type = "responding"
            elif "tool" in chunk_type_lower:
                print(
                    f"\n\n{self.COLOR_TOOL}{self.COLOR_BOLD}🛠️ ============= [并发工具调用流] ============={self.COLOR_RESET}"
                )
                self.last_chunk_type = "tool_calling"

        # 3. 提取有效载荷并追加对应的颜色前缀
        if self.last_chunk_type == "thinking" and think:
            payload = think.replace("\n\n", "\n")
            sys.stdout.write(
                f"{self.COLOR_GREY}{self.COLOR_DIM}{payload}{self.COLOR_RESET}"
            )
            sys.stdout.flush()

        elif self.last_chunk_type == "responding" and text:
            sys.stdout.write(f"{self.COLOR_RESPOND}{text}{self.COLOR_RESET}")
            sys.stdout.flush()

        elif self.last_chunk_type == "tool_calling" and cleaned_tool_deltas:
            for tool_delta in cleaned_tool_deltas:
                idx = tool_delta["index"]
                name = tool_delta["name"]
                args_delta = tool_delta["arguments"]

                # 🎯 调试断点（如需排查，可取消注释下面这行查看大模型流中到底有没有给 name）
                # print(f"\n[DEBUG] 当前帧数据: index={idx}, name={name}, args={args_delta}", flush=True)

                # 🎯 核心优化：确保如果是第一帧或者是全新激活的工具，必须有个名字或标签显示在最前面
                if idx not in self.seen_tool_indices:
                    self.seen_tool_indices.add(idx)
                    if name:
                        self.tool_names_buffer[idx] = name
                        display_name = name
                    else:
                        display_name = f"未知工具 #{idx}"
                    sys.stdout.write(
                        f"\n{self.COLOR_TOOL}{self.COLOR_BOLD}[{display_name}]{self.COLOR_RESET} "
                    )

                # 🎯 动态捕捉延迟到达的工具大名：如果第一帧没给名字，后续帧给名字了，立刻动态修正
                elif name and idx not in self.tool_names_buffer:
                    self.tool_names_buffer[idx] = name
                    sys.stdout.write(
                        f"\n{self.COLOR_TOOL}{self.COLOR_BOLD}[更新工具大名 -> {name}]{self.COLOR_RESET} "
                    )

                # 追加参数流碎片
                if args_delta:
                    sys.stdout.write(f"{self.COLOR_TOOL}{args_delta}{self.COLOR_RESET}")

            sys.stdout.flush()
