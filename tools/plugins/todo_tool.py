from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field, field_validator
from tools.base import BaseTool
import json

class TodoItemArgs(BaseModel):
    id: Optional[str] = Field(default=None, description="待办ID，更新时必填")
    title: Optional[str] = Field(default=None, description="待办标题")
    status: Optional[str] = Field(
        default="pending",
        description="状态：pending/in_progress/completed/cancelled"
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {"pending", "in_progress", "completed", "cancelled"}
        if v not in valid:
            raise ValueError(f"状态必须是：{valid}")
        return v

class TodoToolArgs(BaseModel):
    todos: Optional[List[TodoItemArgs]] = Field(default=None, description="待办列表")
    merge: bool = Field(default=False, description="是否增量更新")

class TodoTool(BaseTool):
    name = "manage_todo_list"
    description = "待办事项管理工具：读取/新增/修改/删除"
    toolset = "todo"
    args_schema = TodoToolArgs

    async def execute(self, ctx: Dict[str, Any], args: TodoToolArgs) -> str:
        store = ctx.get("todo_store")
        if not store:
            return json.dumps({"success": False, "error": "TodoStore 未初始化"}, ensure_ascii=False)

        agent_id = ctx.get("agent_id")
        if not agent_id:
            return json.dumps({"success": False, "error": "未指定 agent_id"}, ensure_ascii=False)

        # 写入逻辑
        if args.todos is not None:
            raw = []
            for item in args.todos:
                d = item.model_dump(exclude_none=True)
                d["agent_id"] = agent_id
                raw.append(d)
            items = await store.write_by_agent(agent_id, raw, args.merge)
        else:
            items = await store.read_by_agent(agent_id)

        # 统计
        total = len(items)
        pending = sum(1 for i in items if i.get("status") == "pending")
        in_progress = sum(1 for i in items if i.get("status") == "in_progress")
        completed = sum(1 for i in items if i.get("status") == "completed")

        return json.dumps({
            "todos": items,
            "summary": {"total": total, "pending": pending, "in_progress": in_progress, "completed": completed},
            "success": True
        }, ensure_ascii=False)