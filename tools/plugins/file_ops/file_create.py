import os
import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from tools.base import BaseTool
from tools.registry import ToolRegistry


class FileCreateArgs(BaseModel):
    path: str = Field(description="要创建的新文件路径。支持相对路径。")
    content: str = Field(
        default="",
        description="新文件的初始代码或文本内容。如果想创建一个空文件，请传空字符串。",
    )
    overwrite: bool = Field(
        default=False,
        description="是否强制覆盖已存在的文件。默认为 False。如果文件已存在且你未显式声明为 True，工具会拒绝写入以保证安全。",
    )


@ToolRegistry.register(name="file_create", toolset="file_ops")
class FileCreateTool(BaseTool[FileCreateArgs]):
    description = (
        "【新建文件首选】在指定路径创建一个新文件并写入初始内容。\n"
        "【重要使用规范（小模型必读）】:\n"
        "1. 工具会自动为你递归创建所有不存在的父级文件夹，你只需要给出完整的最终文件路径即可。\n"
        "2. 安全防冲机制：如果要创建的文件已经存在，默认会报错拦截。如果你确实想用新内容彻底覆盖旧文件，请必须将 overwrite 参数显式设为 True。\n"
        "3. 如果你想对已有文件进行‘局部修改’，严禁使用本工具！请出门右转调用 patch_file_range 工具。"
    )
    toolset = "file_ops"

    async def execute(self, ctx: Dict[str, Any], args: FileCreateArgs) -> str:
        safe_path = ""
        try:
            workspace_dir = os.path.abspath(ctx.get("workspace_dir", "."))
            safe_path = os.path.abspath(os.path.join(workspace_dir, args.path))

            # 1. 严格的安全沙箱检查
            if not safe_path.startswith(workspace_dir):
                return "错误：禁止在工作区外部创建文件。"

            # 禁止把目录当文件创建
            if os.path.isdir(safe_path) or safe_path.endswith(("/", "\\")):
                return "错误：你指定的路径看起来是一个文件夹，本工具只能用于创建具体的文件。"

            # 2. 防意外覆盖拦截逻辑
            if os.path.exists(safe_path) and not args.overwrite:
                return (
                    f"错误：文件 '{args.path}' 已经存在！\n"
                    f"提示：为了防止你误用 file_create 毁掉已有代码，系统已自动拦截。\n"
                    f"- 如果你想【彻底重写】此文件，请重新调用并将 overwrite 参数设为 True。\n"
                    f"- 如果你只想【修改其中几行】，请使用 patch_file_range 工具。"
                )

            # 3. 智能生命线：全自动递归创建父目录，防止小模型因没建文件夹而报错
            parent_dir = os.path.dirname(safe_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            # 4. 换行符清洗与格式化：确保文件符合标准 POSIX 规范（以换行符结尾）
            content_to_write = args.content
            # 如果原项目已有其他文件，尽量对齐换行符，默认使用 LF
            line_ending = "\n"

            # 如果内容不为空，且结尾没有换行符，自动补上一个，防止文件尾部格式残缺
            if content_to_write and not content_to_write.endswith(("\n", "\r\n")):
                content_to_write += line_ending

            # 5. 原子性安全写入
            tmp_file = f"{safe_path}.tmp"
            with open(tmp_file, "w", encoding="utf-8", newline="") as f:
                f.write(content_to_write)
            os.replace(tmp_file, safe_path)

            action_str = (
                "覆盖并更新" if args.overwrite and os.path.exists(safe_path) else "创建"
            )
            return f"成功：已成功{action_str}文件 '{args.path}'，并初始化了其文本内容。"

        except Exception as e:
            if safe_path and os.path.exists(f"{safe_path}.tmp"):
                os.remove(f"{safe_path}.tmp")
            return f"创建文件失败，系统发生异常: {str(e)}"
