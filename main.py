import asyncio
from tools.loader import discover_and_load_tools
from core.config import AppConfig
from core.agent import create_react_agent

discover_and_load_tools()

config = AppConfig.from_yaml("config.yaml")
agent_config = config.get_agent("simple_agent")
agent = create_react_agent(agent_config, session_id="3")


async def main():
    await agent.run("帮我运行一下./work/test.py文件，如果报错则进行修复后再次执行")


if __name__ == "__main__":
    asyncio.run(main())
