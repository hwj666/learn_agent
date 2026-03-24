import os
import json
from typing import Dict, Any, List, Callable
from dotenv import load_dotenv
from openai import OpenAI
from tool_manager import ToolManager
from concurrent.futures import ThreadPoolExecutor, as_completed

_ = load_dotenv()

class AgentCore:
    def __init__(self, api_key=None, base_url=None, model="qwen-max", system_prompt=""):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        self.model = model
        self.history = [{"role": "system", "content":system_prompt}]
        self.tool_manager = ToolManager()
        self.max_loops = 8

    def bind_tools(self, tools: List[Callable]):
        for func in tools:
            self.tool_manager.register_func(func)

    def run(self, prompt: str) -> str:
        print("\n" + "="*60)
        print(f"\033[1;34m🧑💻 用户：{prompt}\033[0m")
        print("="*60 + "\n")

        self.history.append({"role": "user", "content": prompt})

        for _ in range(self.max_loops):
            self._clean_empty_assistant()
            msg = self._stream_chat()

            if self._is_valid_msg(msg):
                self.history.append(msg)

            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    args = tc["function"]["arguments"]
                    print(f"\n\033[1;35m🔧 调用工具：{name}")
                    print(f"📥 参数：{args}\033[0m")
                results = self._exec_tools(msg["tool_calls"])
                # 显示工具返回结果
                for r in results:
                    print(f"\033[1;36m📤 工具返回：{r['content']}\033[0m")
                self.history.extend(results)
                print("\033[1;32m✅ 工具执行完毕\033[0m\n")
                continue

            content = msg.get("content", "").strip()
            if content:
                print("\n\033[1;32m✅ 最终回答：", content, "\033[0m\n")
                return content

            self.history.append({
                "role": "user",
                "content": "请直接给出明确答案，不要只思考不输出。"
            })

        return "⚠️ 模型未返回有效内容"

    def _stream_chat(self) -> Dict[str, Any]:
        kwargs = {
            "model": self.model,
            "messages": self.history,
            "tools": self.tool_manager.get_schemas() or None,
            "stream": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "presence_penalty": 0.1,  # 鼓励谈论新话题
            "frequency_penalty": 0.1  # 减少字词重复
        }

        if "qwen" in self.model.lower() or "deepseek" in self.model.lower():
            kwargs["extra_body"] = {
                "enable_thinking": True,
                "top_k": 20,    # 限制模型只从概率前 20 的 token 中采样
                "min_p": 0      # 最小概率阈值
            }

        response = self.client.chat.completions.create(**kwargs)

        content, reasoning, buf = "", "", {}
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            r = getattr(delta, "reasoning_content", None)
            if r:
                reasoning += r
                print(f"\033[90m{r}\033[0m", end="", flush=True)
                continue

            if delta.content:
                content += delta.content
                print(delta.content, end="", flush=True)
                continue

            if delta.tool_calls:
                self._merge_tool_calls(delta.tool_calls, buf)

        msg = {"role": "assistant"}
        if content.strip(): msg["content"] = content.strip()
        if reasoning.strip(): msg["reasoning_content"] = reasoning.strip()
        if buf: msg["tool_calls"] = list(buf.values())
        return msg

    def _exec_tools(self, tool_calls):
        results = []
        max_workers = min(len(tool_calls), 5)

        with ThreadPoolExecutor(max_workers) as executor:
            futures = {}
            for tc in tool_calls:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                try:
                    fut = executor.submit(self.tool_manager.execute, name, args)
                    futures[fut] = tc["id"]
                except Exception as e:
                    results.append({"role": "tool", "tool_call_id": tc["id"], "content": f"错误：{str(e)}"})

            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    res = fut.result(timeout=20)
                    results.append({"role": "tool", "tool_call_id": tid, "content": str(res)})
                except Exception as e:
                    results.append({"role": "tool", "tool_call_id": tid, "content": f"执行失败：{str(e)}"})
        return results

    def _merge_tool_calls(self, tcs, buf):
        for tc in tcs:
            i = tc.index
            if i not in buf:
                buf[i] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            if tc.id: buf[i]["id"] = tc.id
            if tc.function:
                if tc.function.name: buf[i]["function"]["name"] += tc.function.name
                if tc.function.arguments: buf[i]["function"]["arguments"] += tc.function.arguments

    def _clean_empty_assistant(self):
        self.history = [
            m for m in self.history
            if not (
                m["role"] == "assistant"
                and not m.get("content")
                and not m.get("tool_calls")
            )
        ]

    def _is_valid_msg(self, msg):
        if msg.get("role") != "assistant":
            return True
        return bool(msg.get("tool_calls") or msg.get("content", "").strip())





if __name__ == "__main__":
    from registry_tool import get_all_tools
    import code_tools # 仅出发自动注册
    # 初始化 Agent
    agent = AgentCore(model="qwen-max")

    # 🔥 自动注册所有调试工具
    agent.bind_tools(get_all_tools())

    # ===================== 测试指令（直接改这里）=====================
    agent.run("帮我查看当前项目有哪些Python文件")
    agent.run("帮我调试 bug.py 代码，检查语法错误，并修复问题")
    # agent.run("帮我分析当前代码结构")