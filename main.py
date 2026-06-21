import asyncio
import sys
from tools.loader import discover_and_load_tools
from core.config import AppConfig
from core.agent import create_react_agent, setup_trace_logging

discover_and_load_tools()

# Trace 日志配置
ENABLE_TRACE = "--trace" in sys.argv
if ENABLE_TRACE:
    setup_trace_logging("trace.log")
    print("[INFO] Trace 日志已启用，记录到 trace.log")

config = AppConfig.from_yaml("config.yaml")
agent_config = config.get_agent("simple_agent")
agent = create_react_agent(agent_config, session_id="3", trace_enabled=ENABLE_TRACE)


async def main():
    await agent.run("帮我运行一下./work/test.py文件，如果报错则进行修复后再次执行")


if __name__ == "__main__":
    asyncio.run(main())
