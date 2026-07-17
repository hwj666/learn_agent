import os
import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from tools.base import BaseTool
from tools.registry import ToolRegistry


class PatchFileRangeArgs(BaseModel):
    path: str = Field(description="要修改的文件路径。")
    line_start: int = Field(description="起始行号（从 1 开始）。")
    line_end: int = Field(description="结束行号（包含这一行）。")
    old_content: str = Field(
        description="该区间内目前实际存在的旧代码。严禁传空字符串！"
    )
    new_content: str = Field(description="想替换成的新代码。")

    @model_validator(mode="after")
    def clean_and_fix_args(self) -> "PatchFileRangeArgs":
        current_work_dir = os.getcwd()
        if os.path.isabs(self.path):
            if self.path.startswith(current_work_dir):
                self.path = os.path.relpath(self.path, current_work_dir)
            else:
                self.path = self.path.lstrip("/")

        # 2. 拦截 line_start 的异常数字
        if self.line_start < 1 or self.line_start > 100000:
            self.line_start = 1

        if not self.old_content:
            old_lines_count = 1
        else:
            normalized_content = self.old_content.replace("\r\n", "\n")
            old_lines_count = normalized_content.count("\n") + (
                0 if normalized_content.endswith("\n") else 1
            )
        expected_end = self.line_start + max(1, old_lines_count) - 1

        if self.line_end != expected_end or self.line_end > 100000:
            self.line_end = expected_end

        return self


@ToolRegistry.register(name="patch_file_range", toolset="file_ops")
class PatchFileRangeTool(BaseTool[PatchFileRangeArgs]):
    description = (
        "【修改文件唯一首选】基于行号区间和旧代码快照进行安全无损的文件修改。\n"
        "【重要使用规范】:\n"
        "1. 调用前必须先使用 file_view 查看目标文件的精确行号。\n"
        "2. old_content 的字数和行数要适中（建议 3-10 行），既能保证文件内唯一，又能防止你产生漏字幻觉。\n"
        "3. 严禁传入整个文件作为 old_content！请只截取需要修改的那一小段上下文。\n"
        "4. 本工具自带智能纠偏，如果你数错了 5 行以内的行号，工具会自动为你校准并安全替换。"
    )
    toolset = "file_ops"

    async def execute(self, ctx: Dict[str, Any], args: PatchFileRangeArgs) -> str:
        safe_path = ""
        try:
            workspace_dir = os.path.abspath(ctx.get("workspace_dir", "."))
            safe_path = os.path.abspath(os.path.join(workspace_dir, args.path))
            if not safe_path.startswith(workspace_dir):
                return "错误：禁止修改工作区外的文件。"
            if not os.path.exists(safe_path):
                return f"错误：文件 '{args.path}' 不存在。"

            if args.line_start <= 0 or args.line_end < args.line_start:
                return f"错误：无效的行号范围 [{args.line_start}, {args.line_end}]。"

            with open(safe_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            total_lines = len(lines)
            line_ending = "\r\n" if (total_lines > 0 and "\r\n" in lines) else "\n"

            if total_lines == 0:
                if args.line_start == 1:
                    content_to_write = args.new_content
                    if content_to_write and not content_to_write.endswith(
                        ("\n", "\r\n")
                    ):
                        content_to_write += line_ending
                    with open(safe_path, "w", encoding="utf-8") as f:
                        f.write(content_to_write)
                    return f"成功：原文件为空白，已写入新内容。"
                return "错误：文件内容为空，但指定的起始行号不是 1。"

            def normalize(s: str) -> str:
                return re.sub(r"\s+", "", s)

            target_old_norm = normalize(args.old_content)
            if not target_old_norm:
                return "错误：old_content 不能为空，小模型必须提供旧代码片段以进行安全校验。"

            start_idx = args.line_start - 1
            end_idx = min(args.line_end, total_lines)
            matched = False

            # 阶段 A：范围扩容纠偏
            actual_slice_content = "".join(lines[start_idx:end_idx])
            if normalize(actual_slice_content) == target_old_norm:
                matched = True
            else:
                for offset in [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:
                    new_start = start_idx + offset
                    new_end = end_idx + offset
                    if new_start >= 0 and new_end <= total_lines:
                        fallback_content = "".join(lines[new_start:new_end])
                        if normalize(fallback_content) == target_old_norm:
                            start_idx, end_idx = new_start, new_end
                            matched = True
                            break

            # 阶段 B：全局唯一兜底
            if not matched:
                full_content = "".join(lines)
                matches = [
                    m.start()
                    for m in re.finditer(
                        re.escape(args.old_content.strip()), full_content
                    )
                ]

                if not matches:
                    old_lines_count = len(args.old_content.splitlines())
                    for i in range(total_lines - old_lines_count + 1):
                        slice_try = "".join(lines[i : i + old_lines_count])
                        if normalize(slice_try) == target_old_norm:
                            start_idx, end_idx = i, i + old_lines_count
                            matched = True
                            break
                elif len(matches) == 1:
                    char_idx = matches
                    start_idx = full_content[:char_idx].count("\n")
                    end_idx = start_idx + len(args.old_content.splitlines())
                    matched = True

            # 阶段 C：带探针 debug 导视输出（修复版）
            if not matched:
                view_start = max(0, start_idx - 2)
                view_end = min(total_lines, end_idx + 2)

                debug_lines = []
                for idx_dbg in range(view_start, view_end):
                    line_num_dbg = idx_dbg + 1
                    marker = "➡️ " if start_idx <= idx_dbg < end_idx else "   "
                    debug_lines.append(
                        f"{marker}[Line {line_num_dbg}] {lines[idx_dbg].rstrip('\r\n')}"
                    )

                current_view = "\n".join(debug_lines)

                return (
                    f"错误：安全校验失败！你提供的 old_content 在文件中指定的行号附近找不到。\n"
                    f'【你指定的行号附近（带 ➡️ 标记）实际内容为】:\n"""\n{current_view}\n"""\n'
                    f"请根据上方实际行号和代码，校准后重新提交。"
                )

            real_deleted_start = start_idx + 1
            real_deleted_end = end_idx

            if not args.new_content:
                normalized_new_lines = []
            else:
                raw_split = args.new_content.split("\n")
                if raw_split[-1] == "":
                    raw_split.pop()
                normalized_new_lines = [
                    line.rstrip("\r") + line_ending for line in raw_split
                ]

            lines[start_idx:end_idx] = normalized_new_lines

            tmp_file = f"{safe_path}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.writelines(lines)
            os.replace(tmp_file, safe_path)

            return f"成功：文件 {args.path} 修改成功。已自动纠偏，将原文件第 {real_deleted_start} 到 {real_deleted_end} 行的内容安全替换。"

        except Exception as e:
            if safe_path and os.path.exists(f"{safe_path}.tmp"):
                os.remove(f"{safe_path}.tmp")
            return f"修改失败，系统发生异常: {str(e)}"
