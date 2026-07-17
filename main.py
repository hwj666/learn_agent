import asyncio
import json
import logging
import sys
import os

# 💡 从基础设施引入完全体异步高可用日志工厂
from schema.logger import create_async_production_logger

# 🟢 核心对齐：统一引入运行时上下文大管家，彻底消灭散落的全局 ContextVar 变量
from schema.session.runtime import RuntimeContext

from schema.session.session import AgentSession
from schema.metadata import PlannerMetadata, SubStepMetadata
from core.openai_client import OpenAIClient
from tools.execute import ToolExecutor
from tools.loader import discover_and_load_tools
from schema.config import AppConfig
from agent.react_agent import ReActExecution


def probe_print(msg: str):
    sys.stdout.write(f"🔍 [PROBE] {msg}\n")
    sys.stdout.flush()


# =====================================================================
# 1. 初始化加载基础资产与引擎单例
# =====================================================================
try:
    discover_and_load_tools()
    config = AppConfig.from_yaml("config.yaml")
    agent_config = config.get_agent("simple_agent")
    client = OpenAIClient(agent_config.model_config)
    executor = ToolExecutor(allowed_toolsets=agent_config.tool_set)
    agent = ReActExecution(client, executor)
except Exception as init_err:
    probe_print(f"❌ 基础资产或配置文件加载失败: {init_err}")
    # 预防文件不存在时阻塞后续演示，进行安全降级兜底
    agent = None


# =====================================================================
# 2. 核心异步任务编排管线
# =====================================================================
async def _execute_task(session: AgentSession, worker: ReActExecution):
    t_idx = 0
    t_desc = "run test.py file"
    node_id = f"Task_{t_idx}"

    task_meta = SubStepMetadata(
        description=f"Executing subtask {t_idx}: {t_desc}", tool_name="subtask_runner"
    )

    probe_print("进入 _execute_task，激活大管家全局会话域...")

    # 🌟 核心对齐：利用上下文大管家，锁死当前协程的生命周期拓扑
    with RuntimeContext.guard_session(session.session_id):
        with session.step(node_id=node_id, metadata=task_meta):
            task_meta.arguments = {"task_desc": t_desc}

            probe_print("session.step 物理管道与日志天网已双向激活...")

            # 使用大管家代理的 Logger 打印进入节点的审计日志
            session.logger.info(f"开始执行微观节点任务: {t_desc}")

            if worker is None:
                probe_print(
                    "⚠️ ReAct 引擎未就绪（可能由于缺少 config.yaml），跳过模型执行步骤。"
                )
                task_report = "Mock Report: test.py executed and healed."
            else:
                task_report = await worker.run(
                    current_task_desc=t_desc,
                    context_data={},
                    session=session,
                    execution_id=f"Task_{t_idx}_Exec_0",
                )

            probe_print(f"ReAct 引擎顺利返回，成果长度: {len(task_report or '')}。")
            task_meta.output_data = {"final_report": task_report}
            session.update_metadata_stream(node_id=node_id, metadata=task_meta)

            session.logger.info("微观节点任务执行成功，元数据流已冲刷。")

        probe_print("准备同步累加 Token 消耗记账...")
        print(f"Task {t_idx} completed safely.")


# =====================================================================
# 3. 业务引擎主入口
# =====================================================================
async def main():
    user_query = "帮我运行一下./work/test.py文件，如果报错则进行修复后再次执行"
    probe_print(f"用户原始诉求: {user_query}")

    # 🟢 闭环 1：一键激活全局唯一的异步高可用落盘通道（内部包含物理现场打标机）
    logger, log_listener = create_async_production_logger(
        logger_name="AgentEngine", log_dir="logs", log_file_name="session_audit.log"
    )

    # 将高可用异步 Logger 注入到 Session 空间中
    session = AgentSession(session_id="session_1001", timeout_limit=40.0, logger=logger)

    try:
        # 稳稳进入全局双向锚定物理域，开始追踪
        with RuntimeContext.guard_session(session.session_id, trace_id="T-retry-001"):
            with RuntimeContext.guard_node("Task_0"):
                logger.info("启动 test.py 自愈修复全链路自动化管线...")

                # 投递执行管线，限时 35 秒强杀看门狗
                await asyncio.wait_for(_execute_task(session, agent), timeout=35.0)

        probe_print("子任务链条顺利落地，准备执行会话最后 close()...")
        # 🛡️ 触发优雅停机（Drain）：等待后台事件队列里的资产无损消费并落盘后再放行
        await session.close()
        probe_print("会话 close() 完成。")

    except asyncio.TimeoutError:
        probe_print(
            "🔥 【超时爆破】子任务管线在 35 秒内未能返回，触发看门狗级联中止强杀！"
        )
        await session.close(
            exc_type=TimeoutError, exc_val="Pipeline stalled and exploded"
        )
    except Exception as e:
        await session.close(exc_type=type(e), exc_val=e)
        probe_print(f"Main workflow crashed: {e}")

    probe_print("准备调用 session.to_snapshot() 生成读写分离纯净大快照...")

    try:
        # 🛡️ 生成 100% 纯净独立的 OTel 事件驱动大快照
        final_dict = session.to_snapshot()
        probe_print("大快照字典生成完毕！准备执行 json.dumps()...")
        final_json = json.dumps(final_dict, indent=2, ensure_ascii=False)
        print(
            "\n" + "=" * 18 + " 📊 工业界高级事件驱动 OTel 审计日志大快照 " + "=" * 18
        )
        print(final_json)
    except Exception as snap_err:
        probe_print(f"💥 快照序列化最终防线崩溃！错误详情: {snap_err}")

    # 🟢 闭环 2：程序完成整体退出前，优雅无损冲刷落盘内存队列残余磁盘 I/O 句柄
    if log_listener:
        log_listener.stop()
        print("\n🔒 后台日志落盘线程安全终止，句柄全部无损释放。")


if __name__ == "__main__":
    # 1. 禁言外部 HTTP/三方库等大面积噪声，防止污染控制台输出
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # 2. 🌟 核心修正：利用 asyncio 驱动异步大系统主脉搏，而不是走同步的死胡同
    asyncio.run(main())
