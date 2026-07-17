import os
import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from tools.base import BaseTool
from tools.registry import ToolRegistry


class ListDirTreeArgs(BaseModel):
    path: str = Field(
        default=".",
        description="要列出目录树的起始目标文件夹路径。默认为当前工作区根目录 '.'。",
    )
    max_depth: int = Field(
        default=2,
        description="目录树展示的最大嵌套深度。默认 2 层，最多允许 4 层。",
    )


@ToolRegistry.register(name="list_dir_tree", toolset="file_ops1")
class ListDirTreeTool(BaseTool[ListDirTreeArgs]):
    description = (
        "【项目全局导航】以可视化的树状图（Tree）形式列出指定目录下的文件夹和文件结构。"
    )
    toolset = "file_ops"

    EXCLUDE_DIRS = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "env",
        "venv",
        "dist",
        "build",
        ".idea",
        ".vscode",
        ".pytest_cache",
        ".next",
        ".nuxt",
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

    async def execute(self, ctx: Dict[str, Any], args: Any) -> str:
        try:
            if isinstance(args, dict):
                validated_args = ListDirTreeArgs(**args)
            elif isinstance(args, ListDirTreeArgs):
                validated_args = args
            else:
                return f"错误：不支持的参数契约类型 {type(args)}"

            workspace_dir = os.path.abspath(ctx.get("workspace_dir", "."))
            safe_target = os.path.abspath(
                os.path.join(workspace_dir, validated_args.path)
            )

            # 安全边界检查
            if not safe_target.startswith(workspace_dir):
                return "错误：禁止查看工作区外部的目录树。"
            if not os.path.exists(safe_target):
                return f"错误：指定的路径 '{validated_args.path}' 不存在。"
            if not os.path.isdir(safe_target):
                return (
                    f"错误：路径 '{validated_args.path}' 是一个文件。请使用 file_view。"
                )

            cleaned_depth = min(max(1, validated_args.max_depth), 4)

            # 🔒 修复点 1：显式初始化一个干净、局部隔离的空全局容器，不带任何默认参数污染
            raw_tree_nodes = []

            # 根节点标记
            root_name = (
                os.path.basename(safe_target) if os.path.basename(safe_target) else "."
            )
            raw_tree_nodes.append(f"📁 {root_name}")

            # 启动递归物理扫描
            self._build_tree(
                current_dir=safe_target,
                max_depth=cleaned_depth,
                current_depth=1,
                prefix="",
                tree_lines=raw_tree_nodes,
            )

            # 🔒 修复点 2：仅在最终出口进行一次性缝合，Header 绝对不参与内部递归循环
            final_output = (
                f"【项目目录树】当前展示路径 '{validated_args.path}' 的骨架结构（最大深度限制为 {cleaned_depth} 层）：\n"
                f"--------------------------------------------------\n"
                f"\n".join(raw_tree_nodes)
            )
            return final_output

        except Exception as e:
            return f"生成目录树失败，系统异常: {str(e)}"

    def _build_tree(
        self,
        current_dir: str,
        max_depth: int,
        current_depth: int,
        prefix: str,
        tree_lines: list,
    ):
        if current_depth > max_depth:
            return

        try:
            all_entries = os.listdir(current_dir)
        except Exception:
            return

        dirs = []
        files = []
        for entry in all_entries:
            full_path = os.path.join(current_dir, entry)
            if os.path.isdir(full_path):
                if entry not in self.EXCLUDE_DIRS:
                    dirs.append(entry)
            else:
                if not entry.endswith(self.EXCLUDE_EXTS):
                    files.append(entry)

        dirs.sort()
        files.sort()

        MAX_FILES_PER_DIR = 15
        total_files_count = len(files)
        if total_files_count > MAX_FILES_PER_DIR:
            files = files[:MAX_FILES_PER_DIR]
            has_omitted_files = True
        else:
            has_omitted_files = False

        entries_to_show = [(d, True) for d in dirs] + [(f, False) for f in files]
        total_entries = len(entries_to_show)

        for idx, (name, is_dir) in enumerate(entries_to_show):
            is_last = (idx == total_entries - 1) and not has_omitted_files

            connector = "└── " if is_last else "├── "
            icon = "📁 " if is_dir else "📄 "

            # 🔒 修复点 3：这里只 append 单纯美化后的行，没有任何外层 Header 字符串混入！
            tree_lines.append(f"{prefix}{connector}{icon}{name}")

            if is_dir:
                next_prefix = prefix + ("    " if is_last else "│   ")
                # 显式传递当前作用域的指针
                self._build_tree(
                    current_dir=os.path.join(current_dir, name),
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                    prefix=next_prefix,
                    tree_lines=tree_lines,
                )

        if has_omitted_files:
            tree_lines.append(
                f"{prefix}└── ... (还有 {total_files_count - MAX_FILES_PER_DIR} 个文件已省略)"
            )
