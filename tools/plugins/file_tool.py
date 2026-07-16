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
                ▪ "\n".join(raw_tree_nodes)

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