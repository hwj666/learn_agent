# main.py
import asyncio
from common.exceptions import TimeoutFuseError

# 💡 外部使用唯一标准：所有 API 统一从顶级命名空间引入
from tracing import (
    AgentSession,
    trace_step,
    update_step_metadata,
    emit_stream_chunk,
    ConsoleJsonTransport,
)


# ==========================================
# 🛠️ 2. 原子工具组件层：只需关心打点，不碰 Stream 接口
# ==========================================
@trace_step("calculator_tool", log_args=True)
async def execute_calculation(formula: str) -> int:
    """普通的数值计算工具"""
    await asyncio.sleep(0.05)
    result = eval(formula)

    # ✅ 工具调用：只更新元数据，不涉及任何 stream 概念
    await update_step_metadata(status="success", result=result)
    return result


@trace_step("search_weather_tool")
async def fetch_weather_api(city: str) -> str:
    """天气查询工具（模拟偶发性超时挂掉）"""
    await asyncio.sleep(0.05)
    if city == "TimeoutCity":
        # 抛出业务异常，底层的 translators 会自动捕捉它
        raise TimeoutFuseError(
            "Weather API upstream response timeout", error_code="NET_408"
        )
    return "Sunny, 25°C"


# ==========================================
# 🤖 3. 大模型与 Agent 编排层：按需使用流式 API
# ==========================================
@trace_step("llm_generation")
async def call_llm_stream(prompt: str) -> str:
    """大模型流式调用步骤"""
    chunks = ["The answer ", "is based on ", "real-time data."]
    for chunk in chunks:
        await asyncio.sleep(0.02)
        # ✅ 流式调用：仅在大模型吐流的地方显式引入此 API
        await emit_stream_chunk(chunk)

    # 渲染结束，上报 Token 消耗
    await update_step_metadata(tokens=50, cost=0.001)
    return "".join(chunks)


@trace_step("agent_orchestrator")
async def run_agent_brain(user_query: str):
    """Agent 核心编排大脑"""
    print("[Agent Brain] 开始决策并拆解任务...")

    # 嵌套调用分支 1：执行计算工具
    calc_res = await execute_calculation("1 + 1")

    # 嵌套调用分支 2：执行天气查询工具
    try:
        weather_res = await fetch_weather_api("TimeoutCity")
    except TimeoutFuseError:
        # ✅ 业务降级：虽然工具失败了（并被 tracing 记录），但流程继续
        print("[Agent Brain] 捕获到工具超时，激活降级策略...")
        weather_res = "Weather data unavailable (Fallback)"

    # 嵌套调用分支 3：驱动 LLM 整合最终回答
    await call_llm_stream(prompt=f"Combine {calc_res} and {weather_res}")


# ==========================================
# 🚀 4. 请求最外层入口：拦截整个请求的生命周期
# ==========================================
async def handle_user_http_request():
    """模拟在 Web 框架（如 FastAPI）的最外层入口拦截当前请求"""
    transport = ConsoleJsonTransport()

    print("=== [Web 入口] 收到用户请求，启动大容器 ===")

    # 💡 核心：用大容器包裹整个工作流
    async with AgentSession(transport) as session:
        await run_agent_brain(user_query="Check weather and calculate 1+1")

        # ✅ 修正点：通过 exporter 获取账单快照
        snapshot = session.exporter.billing_snapshot()
        print(
            f"\n[Web 入口] 会话即将关闭，本次请求最终原子计账：总 Tokens={snapshot.tokens}, 总费用=${snapshot.cost:.4f}"
        )

    # 出了 async with，网络队列自动 flush，ContextVar 彻底恢复干净
    print("=== [Web 入口] 大容器已完美释放，请求结束 ===")


if __name__ == "__main__":
    asyncio.run(handle_user_http_request())
