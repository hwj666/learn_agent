import asyncio
from tools.loader import discover_and_load_tools  # 修复路径
from core.config import AppConfig
from agent.react_agent import ReactAgent

# 🔥 自动扫描并加载所有工具（系统 + 用户）
discover_and_load_tools(user_tools_dir="plugins")

# 加载配置
config = AppConfig.from_yaml("config.yaml")
agent_config = config.get_agent("simple_agent")
agent = ReactAgent(agent_config, session_id="3")

async def main():
    await agent.run("帮我在当前work目录下创建一个test.py文件，并实现一个快速排序算法，然后测试一下是否可以运行")

if __name__ == "__main__":
    asyncio.run(main())