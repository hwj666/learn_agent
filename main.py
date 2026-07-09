import asyncio
import json
import logging
import sys

# 💡 1. 干净地从全新四层拓扑包中引入对应的控制器与 Pydantic 契约模型
from schema.session import AgentSession
from schema.metadata import PlannerMetadata, SubStepMetadata

# 引入你的第三方基础设施资产
from core.openai_client import OpenAIClient
from tools.execute import ToolExecutor
from tools.loader import discover_and_load_tools
from schema.config import AppConfig
from agent.react_agent import ReActExecution


# =====================================================================
# 物理流式打印桩 (纯视觉高亮，不再需要任何繁琐的手动锁排查探测)
# =====================================================================
def probe_print(msg: str):
    sys.stdout.write(f"🔍 [PROBE] {msg}\n")
    sys.stdout.flush()


# 初始化加载基础资产
discover_and_load_tools()
config = AppConfig.from_yaml("config.yaml")
agent_config = config.get_agent("simple_agent")
client = OpenAIClient(agent_config.model_config)
executor = ToolExecutor(allowed_toolsets=agent_config.tool_set)

# 2. 这里的 ReActExecution 应使用我们之前适配重构的【单类 AgentSession 绑定版】
agent = ReActExecution(client, executor)


async def _execute_task(session: AgentSession, worker: ReActExecution):
    t_idx = 0
    t_desc = "run test.py file"

    # 🚀 改造点一：这里直接实例化新版的 Pydantic 元数据类（原 TaskMetadata 对应原子步骤）
    task_meta = SubStepMetadata(
        description=f"Executing subtask {t_idx}: {t_desc}", tool_name="subtask_runner"
    )

    probe_print("进入 _execute_task，准备直接以 with 语法切入 session.step 管道...")

    # 🚀 改造点二：旧版多类组合被一击必杀！直接通过单实例控制器的 step 触发完美看门狗
    # 自动在内部推导 parent_id（此时由于 Planner 已经退出控制栈，此节点的 parent_id 自动为 None 或者是 Root）
    with session.step(node_id=f"Task_{t_idx}", metadata=task_meta):
        task_meta.arguments = {"task_desc": t_desc}

        probe_print("session.step 管道已激活，开始调用微观 ReAct 引擎...")

        # 🚀 改造点三：直接将干净的单 session 对象与执行 id 跨线程/协程单向传递
        task_report = await worker.run(
            current_task_desc=t_desc,
            context_data={},  # 传空或传你底层持久化的长期字典
            session=session,
            execution_id=f"Task_{t_idx}_Exec_0",
        )

        probe_print(f"ReAct 引擎顺利返回，成果长度: {len(task_report or '')}。")
        task_meta.output_data = {"final_report": task_report}

    # 💡 出了 with 区间后，该 subtask 节点的耗时、COMPLETED 状态已在底层被安全自动冻结
    probe_print("准备同步累加 Token 消耗记账...")
    # 原本繁琐的手工合并 trace 文本流直接不需要了！因为全链路 trace_logs 已经在底层 RLock 下原子式推入 deque 环形区
    print(f"Task {t_idx} completed safely.")


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    user_query = "帮我运行一下./work/test.py文件，如果报错则进行修复后再次执行"

    # 🚀 改造点四：全局大闭环初始化。一个 Query 过来，仅初始化这一个独立的沙箱实例
    session = AgentSession(session_id="session_1001", timeout_limit=40.0)
    probe_print("AgentSession（三合一单类版）实例创建成功。")

    # 实例化规划层 Pydantic 契约
    planner_meta = PlannerMetadata(
        description="Analyzing query and planning subtasks", raw_user_query=user_query
    )

    probe_print("对 Planner 节点执行染色进入...")

    # 🚀 改造点五：旧版手动 add_root_node / register 刷新全部蒸发
    # 用最清爽的 with 语法包办单节点的诞生、状态流转和消亡
    with session.step(node_id="Planner_Root", metadata=planner_meta):
        await asyncio.sleep(0.02)  # 模拟长考
        planner_meta.planned_tasks = [
            "Check environment path",
            "Execute and debug test.py",
        ]

    probe_print("Planner 阶段自动封盘，准备触发子任务原子链条...")

    try:
        # 直接协程下发，由外层整体控制管道耗时阈值
        await asyncio.wait_for(_execute_task(session, agent), timeout=35.0)

        probe_print("子任务链条顺利落地，准备执行会话最后 close()...")
        session.close()  # 🏁 补齐状态，优雅退场
        probe_print("会话 close() 完成。")

    except asyncio.TimeoutError:
        probe_print(
            "🔥 【超时爆破】子任务管线在 35 秒内未能返回，触发看门狗级联中止强杀！"
        )
        session.close(exc_type=TimeoutError, exc_val="Pipeline stalled and exploded")
    except Exception as e:
        session.close(exc_type=type(e), exc_val=e)
        probe_print(f"Main workflow crashed: {e}")

    probe_print("准备调用 session.to_snapshot() 生成读写分离纯净大快照...")

    try:
        # 🚀 改造点六：旧版噩梦般的 session.to_dict() 与多线程死锁风险彻底不复存在！
        # 直接拿到不带任何物理对象指针的纯净 dict，无脑通过 json.dumps 转换
        final_dict = session.to_snapshot()
        probe_print("大快照字典生成完毕！准备执行 json.dumps()...")

        final_json = json.dumps(final_dict, indent=2, ensure_ascii=False)
        probe_print("json.dumps 序列化大功告成！正在向控制台倾倒资产...")

        print("\n" + "=" * 18 + " 📊 工业界扁平化 OTel 审计日志大快照 " + "=" * 18)
        print(final_json)
    except Exception as snap_err:
        probe_print(f"💥 快照序列化最终防线崩溃！错误详情: {snap_err}")


if __name__ == "__main__":
    asyncio.run(main())
