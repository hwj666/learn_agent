"""
系统工具
包含任务完成、环境信息等系统级工具
"""

from typing import Any, Dict
from pydantic import BaseModel, Field

from tools.base import BaseTool
from tools.registry import ToolRegistry


class TaskCompletedArgs(BaseModel):
    summary: str = Field(description="任务完成总结，简要描述已完成的工作内容")
    result: str = Field(default="", description="可选的最终结果或输出内容")


@ToolRegistry.register(name="task_completed", toolset="system")
class TaskCompletedTool(BaseTool[TaskCompletedArgs]):
    description = """
        任务完成声明工具。当所有子任务都已执行完毕，且目标已达成时调用此工具。
        调用此工具表示任务成功结束，请确保在调用前已完成所有必要的工作。

        参数说明：
        - summary: 必填，简要总结已完成的工作
        - result: 可选，最终输出结果
    """

    async def execute(self, ctx: Dict[str, Any], args: TaskCompletedArgs) -> str:
        return f"[OK] 任务已完成\n\n总结：{args.summary}\n\n结果：{args.result if args.result else '无额外输出'}"


class GetEnvInfoArgs(BaseModel):
    pass


@ToolRegistry.register(name="get_env_info", toolset="system")
class GetEnvInfoTool(BaseTool[GetEnvInfoArgs]):
    description = "获取当前环境信息，包括工作目录、工具集等"

    async def execute(self, ctx: Dict[str, Any], args: GetEnvInfoArgs) -> str:
        info = {
            "session_id": ctx.get("session_id", "unknown"),
            "agent_id": ctx.get("agent_id", "unknown"),
            "sandbox_read_dirs": ctx.get("sandbox_read_dirs", []),
            "sandbox_write_dirs": ctx.get("sandbox_write_dirs", []),
        }
        return f"环境信息：\n{info}"


class FinishTaskArgs(BaseModel):
    reason: str = Field(description="结束原因")


@ToolRegistry.register(name="finish_task", toolset="system")
class FinishTaskTool(BaseTool[FinishTaskArgs]):
    description = "结束当前任务，用于无法完成或需要用户介入的情况"

    async def execute(self, ctx: Dict[str, Any], args: FinishTaskArgs) -> str:
        return f"[STOP] 任务已结束\n\n原因：{args.reason}"
