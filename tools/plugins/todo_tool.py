import json
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from tools.base import BaseTool, BaseToolState
from tools.registry import ToolRegistry

class TodoState(BaseToolState):
    items: List[Dict[str, Any]] = Field(default_factory=list)

class TodoToolArgs(BaseModel):
    action: Literal["add", "update", "delete", "list"] = Field(
        description="操作类型：add(新增规划), update(修改状态或标题), delete(删除), list(仅查看列表)"
    )
    titles: Optional[List[str]] = Field(
        default=None, 
        description="待办标题列表（新增 add 时必填，支持传入包含多个步骤标题的数组，例如：['步骤1', '步骤2']）。修改时请留空。"
    )
    title: Optional[str] = Field(default=None, description="新的单个待办标题（仅在 update 修改单个标题时使用，或作为 add 时的单条兜底）")
    index: Optional[int] = Field(default=None, description="从1开始的序号（修改 update、删除 delete 时必填，由列表返回）")
    status: str = Field(
        default="pending",
        description="状态：pending(未开始)/in_progress(进行中)/completed(已完成)/cancelled(已取消)"
    )

    # 🌟 修复核心：更强力的状态清洗，完美防御空字符串 "" 或纯空格
    @field_validator("status", mode="before")
    @classmethod
    def clean_status(cls, v: Any) -> str:
        if isinstance(v, str):
            cleaned = v.strip()
            # 如果大模型传了空字符串 "" 或带有换行的空值，直接强行重置为安全的 "pending"
            if not cleaned:
                return "pending"
            
            valid_statuses = {"pending", "in_progress", "completed", "cancelled"}
            if cleaned in valid_statuses:
                return cleaned
        return "pending" 

@ToolRegistry.register(name="manager_todo", toolset="todo")
class TodoTool(BaseTool[TodoToolArgs]):
    description = """
        # 严格执行规范
        待办事项管理工具。在开始任何多步骤任务前，必须先使用 action='add' 规划步骤。
        # 核心禁令
        1. 新增(add)：严禁使用单数 `title`！必须将步骤写成数组传入 `titles`（例如：[\"步骤1\"])
        2. 修改(update)与删除(delete)：必须且只能传入 `index`（数字序号）！严禁不传 index!
        3. 状态(status)：必须是以下精准字面量之一：'pending', 'in_progress', 'completed', 'cancelled'。严禁带有空格或换行！
    """
    
    state_schema = TodoState

    async def execute(self, ctx: Dict[str, Any], args: TodoToolArgs) -> str:
        state: TodoState = ctx["tool_state"]

        # ==========================================================
        # 🌟 强力防呆机制：在业务最前端直接对 add 操作进行强制数据清洗
        # ==========================================================
        if args.action == "add":
            final_titles = []
            if args.titles and isinstance(args.titles, list):
                final_titles.extend([t for t in args.titles if t.strip()])
            if args.title and args.title.strip():
                final_titles.append(args.title.strip())
            
            # 裁剪大模型带出来的碎碎念尾巴
            cleaned_titles = []
            for t in final_titles:
                for stop_word in ["任务已完成", "当前状态", "若未完成"]:
                    if stop_word in t:
                        t = t.split(stop_word)
                cleaned_titles.append(t.strip(" ，。.,\n\t"))
            
            args.titles = [t for t in cleaned_titles if t]

        # ==========================================================
        # 业务核心逻辑执行区
        # ==========================================================
        if args.action == "add":
            if not args.titles:
                return json.dumps({
                    "success": False, 
                    "error": "【新增失败】未检测到有效的待办标题。请提供 titles 数组或 title 字符串。"
                }, ensure_ascii=False)
            
            for t in args.titles:
                state.items.append({"title": t, "status": args.status})

        elif args.action == "update":
            if args.index is None:
                return json.dumps({
                    "success": False, 
                    "error": "【参数缺失】更新失败。必须显式传入 index 参数（数字序号）。若不知道序号，请先执行 action='list' 查看。"
                }, ensure_ascii=False)
                
            idx = args.index - 1
            if 0 <= idx < len(state.items):
                if args.title is not None:
                    clean_title = args.title
                    for stop_word in ["（当前状态", "当前状态"]:
                        if stop_word in clean_title:
                            clean_title = clean_title.split(stop_word)
                    state.items[idx]["title"] = clean_title.strip()
                if args.status is not None:
                    state.items[idx]["status"] = args.status
            else:
                return json.dumps({
                    "success": False, 
                    "error": f"【序号无效】无法更新。输入的序号 {args.index} 超出范围，当前共有 {len(state.items)} 个待办项。"
                }, ensure_ascii=False)

        elif args.action == "delete":
            if args.index is None:
                return json.dumps({
                    "success": False, 
                    "error": "【参数缺失】删除失败。必须显式传入 index 参数（数字序号）。若不知道序号，请先执行 action='list' 查看。"
                }, ensure_ascii=False)
                
            idx = args.index - 1
            if 0 <= idx < len(state.items):
                state.items.pop(idx)
            else:
                return json.dumps({
                    "success": False, 
                    "error": f"【序号无效】无法删除。输入的序号 {args.index} 超出范围，当前共有 {len(state.items)} 个待办项。"
                }, ensure_ascii=False)

        # 组织标准输出反馈给大模型
        formatted_todos = [
            {"index": i + 1, "title": item["title"], "status": item.get("status", "pending")} 
            for i, item in enumerate(state.items)
        ]
        
        return json.dumps({
            "success": True,
            "action_executed": args.action,
            "todos": formatted_todos,
            "summary": {
                "total": len(state.items),
                "pending": sum(1 for x in state.items if x.get("status") == "pending"),
                "in_progress": sum(1 for x in state.items if x.get("status") == "in_progress"),
                "completed": sum(1 for x in state.items if x.get("status") == "completed"),
                "cancelled": sum(1 for x in state.items if x.get("status") == "cancelled"),
            }
        }, ensure_ascii=False)