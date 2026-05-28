# 流式输出处理器（标准写法，无黑魔法）
class StreamPrintHandler:
    def __init__(self):
        self.thinking_started = False
        self.thinking_closed = False

    async def on_chunk(self, think: str, content: str, chunk_type: str):
        """
        彩色流式输出：思考流 / 回答流 / 工具调用流
        自动闭合、自动换行、不乱码、不重叠
        """
        # ====================== 思考流（蓝色）======================
        if chunk_type == "thinking":
            if not self.thinking_started:
                # 蓝色开头
                print("\n\033[1;34m🤔 思考中：\033[0m", end="", flush=True)
                self.thinking_started = True
                self.thinking_closed = False

            # 蓝色内容
            print(f"\033[34m{think}\033[0m", end="", flush=True)

        # ====================== 回答流（绿色）======================
        elif chunk_type == "responding":
            # 如果思考没闭合，自动闭合换行
            if self.thinking_started and not self.thinking_closed:
                print("\033[0m\n", flush=True)
                self.thinking_closed = True

            # 绿色输出回答
            print(f"\033[1;32m{content}\033[0m", end="", flush=True)

        # ====================== 工具调用（黄色高亮）======================
        elif chunk_type == "tool_calling":
            if self.thinking_started and not self.thinking_closed:
                print("\033[0m\n", flush=True)
                self.thinking_closed = True

            # 黄色工具参数
            print(f"\033[1;33m{content}\033[0m", end="", flush=True)