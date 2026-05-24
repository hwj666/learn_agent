import asyncio
from tools.loader import discover_and_load_tools  # 修复路径
from core.config import AppConfig
from agent.react_agent import ReactAgent

# 🔥 自动扫描并加载所有工具（系统 + 用户）
discover_and_load_tools(user_tools_dir="plugins")

# 加载配置
config = AppConfig.from_yaml("config.yaml")
model_config = config.get_agent_model("simple_agent")

# ✅ 修复：allowed_toolsets 必须传入 set 而不是 list
agent = ReactAgent(
    model_config=model_config,
    max_steps=10,
    allowed_toolsets={"manage_todo_list", "dev"}  # 这里必须是 set！
)

async def main():
    print("🚀 ReAct Agent 启动，支持并发调用！\n")

    # 并发执行多个任务（协程安全，你的框架完全支持）
    results = await asyncio.gather(
        agent.run("广州和上海这两个城市的天气分别是多少？"),
        agent.run("1234 + 5678 等于多少？"),
        # agent.run("股票 000001 价格多少？")
    )

    # 输出结果
    for i, result in enumerate(results, 1):
        print(f"\n✅ 任务 {i} 结果：")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())