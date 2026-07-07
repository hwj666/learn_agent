"""
运行示例
"""

import asyncio
import sys

from tools.loader import discover_and_load_tools
from schema.config import AppConfig
from core.agent import create_react_agent


async def main():
    discover_and_load_tools()

    config = AppConfig.from_yaml("config.yaml")
    agent_config = config.get_agent("simple_agent")
    agent = create_react_agent(agent_config, session_id="demo-1")

    query = "帮我写一个 hello world Python 脚本并执行"
    print(f"查询: {query}")
    print("=" * 50)

    result = await agent.run(query)
    print(f"\n结果:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
