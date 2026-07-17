import os
import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from tools.base import BaseTool
from tools.registry import ToolRegistry


class FileViewArgs(BaseModel):
    path: str = Field(description="文件路径。支持绝对或相对路径。")
    start_line: int = Field(
        default=1,
        description="从第几行开始查看。从 1 开始数。如果填错、填负数或填了极大的数字，会自动纠正为 1。",
    )
    max_lines: int = Field(
        default=15,
        description="本次查看几行。默认 15 行，最多 30 行。小模型请勿看太多，防止遗忘上下文！",
    )

    # 💡 针对小模型乱传 start_line 的强行纠正校验器
    @field_validator("start_line", mode="before")
    @classmethod
    def clean_start_line(cls, v):
        try:
            val = int(v)
            # 如果是负数，或者超过了 10 万行的离谱数字（0.8B几乎不可能在处理10万行以上代码）
            if val < 1 or val > 100000:
                return 1
            return val
        except (ValueError, TypeError):
            return 1  # 如果模型传了字符串或其他奇葩类型，直接兜底为 1

    # 💡 针对小模型乱传 max_lines 的强行截断校验器
    @field_validator("max_lines", mode="before")
    @classmethod
    def clean_max_lines(cls, v):
        try:
            val = int(v)
            if val < 1:
                return 15  # 负数重置为默认值
            if val > 30:
                return 30  # 超过 30 行直接截断成 30
            return val
        except (ValueError, TypeError):
            return 15


@ToolRegistry.register(name="file_view", toolset="file_ops")
class FileViewTool(BaseTool[FileViewArgs]):
    description = (
        "【修改文件前必用】精确查看指定文件的行内容与行号。\n"
        "【小模型使用规范】:\n"
        "1. 在修改代码（调用 patch_file_range）之前，必须先用本工具确认代码精准行号和旧内容快照。\n"
        "2. 单次最多只能看 30 行。如果文件很长，请根据返回的总行数，分批改变 start_line 进行翻页查看。"
    )

    async def execute(self, ctx: Dict[str, Any], args: FileViewArgs) -> str:
        try:
            workspace_dir = os.path.abspath(ctx.get("workspace_dir", "."))
            safe_path = os.path.abspath(os.path.join(workspace_dir, args.path))
            if not safe_path.startswith(workspace_dir):
                return "错误：禁止查看工作区外的文件。"
            if not os.path.exists(safe_path):
                return f"错误：文件 '{args.path}' 不存在。请检查路径是否正确。"

            with open(safe_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            total_lines = len(lines)

            if total_lines == 0:
                return f"提示：文件 '{args.path}' 目前是一个完全空白的文件，总行数为 0。你可以直接调用修改工具从第 1 行开始写入。"

            cleaned_start_line = max(1, args.start_line)
            start_idx = cleaned_start_line - 1

            if start_idx >= total_lines:
                return f"错误：你指定的起始行号 {cleaned_start_line} 超过了文件的总行数（当前文件总共只有 {total_lines} 行）。请重新指定较小的 start_line。"

            actual_max = min(max(1, args.max_lines), 30)
            end_idx = min(total_lines, start_idx + actual_max)
            real_end_line = end_idx

            output = []
            for i, line in enumerate(lines[start_idx:end_idx]):
                current_line_num = start_idx + i + 1
                clean_line = line.rstrip("\r\n")
                output.append(f"[Line {current_line_num}] {clean_line}\n")

            header = f"【文件视图】正在查看 '{args.path}'（总共 {total_lines} 行）。当前展示第 {cleaned_start_line} 到 {real_end_line} 行：\n"
            if real_end_line < total_lines:
                header += f"提示：后续还有 {total_lines - real_end_line} 行未展示。如果需要看后面，请将 start_line 设为 {real_end_line + 1} 再次调用。\n"

            header += "--------------------------------------------------\n"
            return header + "".join(output)

        except Exception as e:
            return f"查看文件失败，系统异常：{str(e)}"
