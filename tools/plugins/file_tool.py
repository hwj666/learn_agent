import os
import re
from typing import Dict, Any
from pydantic import BaseModel, Field

from tools.base import BaseTool
from tools.registry import ToolRegistry

# =====================================================================
# 1. FileViewTool (极致防爆：单次看最多 30 行)
# =====================================================================
class FileViewArgs(BaseModel):
    path: str = Field(description="文件路径。")
    start_line: int = Field(default=1, description="从第几行开始看。从 1 开始数。")
    max_lines: int = Field(default=15, description="看几行。默认 15 行，最多 30 行。别看太多！")

@ToolRegistry.register(name="file_view", toolset="file_ops")
class FileViewTool(BaseTool[FileViewArgs]):
    description = "看文件内容。单次看很少的行数，防止字数太多让你混乱。"

    async def execute(self, ctx: Dict[str, Any], args: FileViewArgs) -> str:
        try:
            workspace_dir = os.path.abspath(ctx.get("workspace_dir", "."))
            safe_path = os.path.abspath(os.path.join(workspace_dir, args.path))
            if not safe_path.startswith(workspace_dir): return "错误：不能看外面的文件。"
            if not os.path.exists(safe_path): return "错误：文件不存在。"

            with open(safe_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            total_lines = len(lines)
            start_idx = max(0, args.start_line - 1)
            # 严格限制 0.8B 视窗上限为 30 行
            actual_max = min(max(1, args.max_lines), 30)
            end_idx = min(total_lines, start_idx + actual_max)

            if start_idx >= total_lines:
                return f"错误：文件只有 {total_lines} 行，你数错了。"

            output = [f"{i + start_idx + 1}: {line}" for i, line in enumerate(lines[start_idx:end_idx])]
            return f"--- {args.path} ({args.start_line}-{end_idx}行/共{total_lines}行) ---\n" + "".join(output)
        except Exception as e:
            return f"看文件失败：{str(e)}"



class FileWriteAllArgs(BaseModel):
    path: str = Field(description="要写的文件路径。")
    content: str = Field(description="文件的全部新内容。直接覆盖写。")

@ToolRegistry.register(name="file_write_all", toolset="file_ops")
class FileWriteAllTool(BaseTool[FileWriteAllArgs]):
    description = "写新文件，或者把一个小文件全部换掉。直接覆盖，不用考虑旧内容。"
    toolset = "file_ops"

    async def execute(self, ctx: Dict[str, Any], args: FileWriteAllArgs) -> str:
        try:
            workspace_dir = os.path.abspath(ctx.get("workspace_dir", "."))
            safe_path = os.path.abspath(os.path.join(workspace_dir, args.path))
            if not safe_path.startswith(workspace_dir): return "错误：不能写外面的文件。"

            os.makedirs(os.path.dirname(safe_path), exist_ok=True)
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(args.content)
            return f"成功：文件 {args.path} 已经写好了。"
        except Exception as e:
            return f"写文件失败：{str(e)}"


# =====================================================================
# 3. FileSearchReplaceTool (0.8B 专属：带容错的正则弹性替换)
# =====================================================================
class FileSearchReplaceArgs(BaseModel):
    path: str = Field(description="要改的文件路径。")
    old_text: str = Field(description="你要改的那【1到3行】旧代码。必须在文件里是唯一的。")
    new_text: str = Field(description="你想换成的新代码。如果是删除，请传空字符串。")

@ToolRegistry.register(name="file_search_replace", toolset="file_ops")
class FileSearchReplaceTool(BaseTool[FileSearchReplaceArgs]):
    description = "修改、插入或删除文件里的某几行代码。通过旧代码找位置换成新代码。"
    toolset = "file_ops"

    async def execute(self, ctx: Dict[str, Any], args: FileSearchReplaceArgs) -> str:
        try:
            workspace_dir = os.path.abspath(ctx.get("workspace_dir", "."))
            safe_path = os.path.abspath(os.path.join(workspace_dir, args.path))
            if not os.path.exists(safe_path): return f"错误：文件 '{args.path}' 不存在。"
            with open(safe_path, "r", encoding="utf-8") as f:
                content = f.read()

            escaped_old = re.escape(args.old_text)
            # 修复：原代码 re.sub(r'\\ ', r'\s*', escaped_old) 会引发 bad escape \s 崩溃
            # 改进：使用更安全的字符串原生 replace 方法将转义后的空格替换为正则弹性空白占位符
            fuzzy_pattern = escaped_old.replace(r'\ ', r'\s*')
            
            try:
                matches = list(re.finditer(fuzzy_pattern, content))
            except Exception as re_err:
                # 极端后备方案：如果正则引擎还是报错，降级为纯文本绝对匹配
                count = content.count(args.old_text)
                if count == 1:
                    updated_content = content.replace(args.old_text, args.new_text)
                    with open(safe_path, "w", encoding="utf-8") as f:
                        f.write(updated_content)
                    return f"成功：文件 {args.path} 已通过降级匹配修改成功。"
                matches = []

            if len(matches) == 0:
                return (f"错误：没在文件里找到你写的 `old_text`。\n"
                        f"请先用 FileViewTool 重新看一眼再写。")
            if len(matches) > 1:
                return (f"错误：在文件里找到了 {len(matches)} 处相同的旧代码。\n"
                        f"只传一个 '{args.old_text}' 无法确定改哪一个。请让你的 `old_text` 包含更多上下邻近行的代码，确保唯一。")

            # 替换唯一的匹配项
            match = matches[0]
            updated_content = content[:match.start()] + args.new_text + content[match.end():]
            
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            return f"成功：文件 {args.path} 已经修改成功。"
        except Exception as e:
            return f"修改失败：{str(e)}"