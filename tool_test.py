from registry_tool import tool
import time
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    time.sleep(0.5)
    return f"{city}：晴天 24℃"

@tool
def calculate_add(a: int, b: int) -> int:
    """加法计算"""
    time.sleep(0.5)
    return a + b

@tool
def get_stock(code: str) -> str:
    """获取股票价格"""
    time.sleep(0.5)
    return f"{code}：186.5 元"


from agent_core import AgentCore
from registry_tool import get_all_tools

if __name__ == "__main__":
    agent = AgentCore(model="qwen-max")
    agent.bind_tools(get_all_tools())

    # 多轮对话
    agent.run("广州天气多少？")
    agent.run("1234 + 5678 等于多少？")
    agent.run("股票 000001 价格多少？")
    # agent.run("帮我总结一下")
    agent.run("请以 Markdown 表格形式总结以上所有问题的结果，包含‘问题类型’和‘结果’两列。")