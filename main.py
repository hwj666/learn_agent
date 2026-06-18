import asyncio
from tools.loader import discover_and_load_tools  # 修复路径
from core.config import AppConfig
from agent.react_agent import ReactAgent
from agent.plan_agent import DynamicPlanExecuteAgent

# 🔥 自动扫描并加载所有工具（系统 + 用户）
discover_and_load_tools(user_tools_dir="plugins")

# 加载配置
config = AppConfig.from_yaml("config.yaml")
agent_config = config.get_agent("simple_agent")
agent = DynamicPlanExecuteAgent(agent_config, session_id="3")


async def main():
    await agent.run("帮我运行一下./work/test.py文件，如果报错则进行修复后再次执行")


if __name__ == "__main__":
    asyncio.run(main())
