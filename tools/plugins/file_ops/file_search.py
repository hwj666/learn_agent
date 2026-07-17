import os
import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from tools.base import BaseTool
from tools.registry import ToolRegistry


class SearchTextArgs(BaseModel):
    query: str = Field(
        description="要在文件中搜索的关键词、变量名、类名或函数定义片段。"
    )
    path: Optional[str] = Field(
        default=None,
        description="限制的搜索路径（支持文件或文件夹）。如果不传，工具会全局扫描整个项目。",
    )


@ToolRegistry.register(name="file_search_text", toolset="file_ops")
class FileSearchTextTool(BaseTool[SearchTextArgs]):
    description = (
        "【全局/局部找代码】在项目或指定路径的文件里，搜索包含特定关键词的行号和核心上下文。\n"
        "【重要提示】: 本工具返回的结果会包含匹配行及后续两行的代码快照，如果信息足够，你可以直接根据行号进行 patch_file_range 修改，无需重复调用 file_view。"
    )
    toolset = "file_ops"

    async def execute(self, ctx: Dict[str, Any], args: SearchTextArgs) -> str:
        try:
            workspace_dir = os.path.abspath(ctx.get("workspace_dir", "."))

            if args.path:
                search_target = os.path.abspath(os.path.join(workspace_dir, args.path))
                if not search_target.startswith(workspace_dir):
                    return "错误：不能搜索外部目录。"
                if not os.path.exists(search_target):
                    return f"错误：指定的搜索路径 '{args.path}' 不存在。"
            else:
                search_target = workspace_dir

            file_results = {}
            EXCLUDE_DIRS = {
                ".git",
                "__pycache__",
                "node_modules",
                ".venv",
                "dist",
                "build",
                ".idea",
                ".vscode",
                ".pytest_cache",
            }
            EXCLUDE_EXTS = (
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".ico",
                ".pyc",
                ".pdf",
                ".zip",
                ".tar.gz",
                ".tar",
                ".gz",
                ".exe",
                ".dll",
                ".so",
                ".lock",
            )

            if os.path.isfile(search_target):
                self._search_in_file(
                    search_target, workspace_dir, args.query, file_results
                )
            else:
                for root, dirs, files in os.walk(search_target):
                    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                    for file in files:
                        if file.endswith(EXCLUDE_EXTS):
                            continue
                        file_path = os.path.join(root, file)
                        self._search_in_file(
                            file_path, workspace_dir, args.query, file_results
                        )

            if not file_results:
                return f"提示：在指定范围内未找到任何包含 '{args.query}' 的代码。请尝试更换更通用的关键词。"

            output = [f"搜索 '{args.query}' 的聚合结果如下：\n"]
            total_matches_count = 0
            max_total_display = 15

            for rel_path, matches in file_results.items():
                if total_matches_count >= max_total_display:
                    output.append(
                        f"\n⚠️ 警告：匹配项过多，已截断后续文件的展示。请让你的搜索词（query）更精准一些。"
                    )
                    break

                output.append(f"\n📂 文件路径: {rel_path}")
                output.append("-" * 40)

                display_matches = matches[:5]
                for item in display_matches:
                    if total_matches_count >= max_total_display:
                        break

                    line_num = item["line_num"]
                    snippet = item["snippet"]

                    output.append(f"[第 {line_num} 行触发匹配] 👇")
                    output.append(snippet)
                    total_matches_count += 1

                if len(matches) > 5:
                    output.append(
                        f"  (... 该文件内还有 {len(matches) - 5} 处匹配已省略 ...)\n"
                    )

            return "".join(output)

        except Exception as e:
            return f"搜索失败，系统异常：{str(e)}"

    def _search_in_file(
        self, file_path: str, workspace_dir: str, query: str, file_results: dict
    ):
        try:
            if os.path.getsize(file_path) > 5 * 1024 * 1024:
                return

            rel_path = os.path.relpath(file_path, workspace_dir)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                if query in line:
                    context_lines = []
                    end_ctx_idx = min(len(lines), idx + 3)

                    for c_idx in range(idx, end_ctx_idx):
                        c_line_num = c_idx + 1
                        raw_line = lines[c_idx].rstrip("\r\n")
                        if len(raw_line) > 150:
                            raw_line = raw_line[:150] + "..."

                        flag = "🌟 MATCH -> " if c_idx == idx else "            "
                        context_lines.append(f"{flag}[Line {c_line_num}] {raw_line}")

                    snippet_str = "\n".join(context_lines) + "\n"

                    if rel_path not in file_results:
                        file_results[rel_path] = []
                    file_results[rel_path].append(
                        {"line_num": idx + 1, "snippet": snippet_str}
                    )
        except Exception:
            pass
