import os
import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from tools.base import BaseTool
from tools.registry import ToolRegistry


class FileExistsArgs(BaseModel):
    file_path: str = Field(
        ..., description="需要检查的文件的路径。例如：'./work/test.py'"
    )


@ToolRegistry.register(name="file_exists", toolset="file")
class FileExistsTool(BaseTool[FileExistsArgs]):
    description = (
        "专门用于判断某个特定的文件或文件夹在本地系统中是否存在。"
        "\n【⚠️ 严禁行为】: 如果你已经明确知道要找的文件名，严禁调用 list_dir 工具去遍历整个目录，必须优先调用本工具进行精准判断。"
    )

    # 🟢 修正：将 _run 改为 execute，并补全 ctx 参数以满足基类抽象方法要求
    async def execute(self, ctx: Dict[str, Any], args: FileExistsArgs) -> str:
        # 清洗路径字符串
        path = args.file_path.strip().strip("'\"")

        if os.path.exists(path):
            file_type = "目录" if os.path.isdir(path) else "文件"
            return f"存在: 路径 '{path}' 确实存在，它是一个{file_type}。"
        else:
            return f"不存在: 路径 '{path}' 在系统中不存在。"
