import os
import sys
import re
import platform
import shutil
import asyncio
import signal
from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field
from tools.base import BaseTool
from tools.registry import ToolRegistry


# 假设的参数类与基类，保持与你的系统一致
class RunCodeFileArgs(BaseModel):
    runtime: str = Field(
        ...,
        description=(
            "目标代码的编程语言简称。必须是小写字母，严禁包含任何可执行代码、参数或 `-c` 标志。"
            "有效示例: 'python', 'javascript', 'cpp', 'go', 'java'。"
            "错误示例: 'python -c ...'（绝对禁止）。"
        ),
    )
    file_path: str = Field(
        ..., description="目标代码文件的完整相对路径或绝对路径。例如: './work/test.py'"
    )
    args: str = Field(
        "",
        description="可选。如果脚本运行需要传递额外的命令行参数，写在这里。例如：'--mode train --epoch 10'",
    )


@ToolRegistry.register(name="run_code", toolset="bash")
class RunCodeTool(BaseTool[RunCodeFileArgs]):
    description = (
        "执行指定编程语言的代码文件。工具会根据文件路径和语言类型，"
        "自动调用对应的编译器或解释器（如 python, node, g++, gcc, go 等）进行编译和运行，并返回标准输出或错误信息。"
        "\n【⚠️ 严禁行为】: 严禁在 runtime 参数中夹带 '-c'、'import'、'subprocess' 或任何具体的 Shell 命令代码。它只能接收纯粹的语言简称。"
        "\n【支持语言简称】: javascript, typescript, python, go, ruby, php, perl, shellscript, powershell, c, cpp, java, rust"
        "\n【调用时机】: 当本地已经存在一个代码文件，且需要获取该文件的运行结果或验证其正确性时调用。"
    )

    # 1. 类属性声明
    is_win: bool = platform.system() == "Windows"

    # 🟢 优化：预编译 ANSI Escape 颜色过滤正则，避免在 clean 循环中重复解析
    _ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[.*?[a-zA-Z])")

    executor_map: dict = {
        "javascript": "node",
        "typescript": "ts-node",
        "python": "python -u",
        "go": "go run",
        "ruby": "ruby",
        "php": "php",
        "perl": "perl",
        "shellscript": "bash",
        "powershell": "powershell -ExecutionPolicy Bypass -File",
        "c": "gcc $fileName -o $fileNameWithoutExt.exe && $fileNameWithoutExt.exe"
        if platform.system() == "Windows"
        else "gcc $fileName -o $fileNameWithoutExt && ./$fileNameWithoutExt",
        "cpp": "g++ $fileName -o $fileNameWithoutExt.exe && $fileNameWithoutExt.exe"
        if platform.system() == "Windows"
        else "g++ $fileName -o $fileNameWithoutExt && ./$fileNameWithoutExt",
        "java": "javac $fileName && java $fileNameWithoutExt",
        "rust": "rustc $fileName && $fileNameWithoutExt.exe"
        if platform.system() == "Windows"
        else "rustc $fileName && ./$fileNameWithoutExt",
    }

    def _resolve_command(self, args: RunCodeFileArgs) -> Tuple[bool, str, str]:
        """2. 核心插值引擎：动态解析 Code Runner 占位符"""
        lang = args.runtime.lower().strip()
        if lang not in self.executor_map:
            return False, f"错误: Code Runner 暂未配置语言 '{lang}' 的执行命令。", ""

        raw_cmd = self.executor_map[lang]
        abs_path = os.path.abspath(args.file_path.strip().strip("'\""))
        file_dir = os.path.dirname(abs_path)
        file_name = os.path.basename(abs_path)
        file_name_without_ext, _ = os.path.splitext(file_name)

        compiled_cmd = (
            raw_cmd.replace("$fullFileName", abs_path)
            .replace("$dir", file_dir)
            .replace("$fileName", f'"{file_name}"' if self.is_win else f"'{file_name}'")
            .replace("$fileNameWithoutExt", file_name_without_ext)
        )

        if "$" not in raw_cmd:
            quoted_file = f'"{file_name}"' if self.is_win else f"'{file_name}'"
            compiled_cmd = f"{compiled_cmd} {quoted_file}"

        if args.args.strip():
            compiled_cmd += f" {args.args.strip()}"

        return True, compiled_cmd, file_dir

    async def execute(self, ctx: Dict[str, Any], args: RunCodeFileArgs) -> str:
        """3. 生命周期主控管线（安全隔离与防御力）"""
        is_safe, cmd, file_dir = self._resolve_command(args)
        if not is_safe:
            return cmd

        env = os.environ.copy()
        env.update(
            {
                "DEBIAN_FRONTEND": "noninteractive",
                "PAGER": "cat",
                "PYTHONUNBUFFERED": "1",
            }
        )

        kwargs = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "env": env,
            **(
                {"creationflags": 0x08000000}
                if self.is_win
                else (
                    {"process_group": 0}
                    if sys.version_info >= (3, 11)
                    else {"preexec_fn": os.setsid}
                )
            ),
        }

        shell = (
            ["powershell.exe", "-NonInteractive", "-Command"]
            if self.is_win
            else [shutil.which("bash") or "sh", "-c"]
        )
        process, timeout, max_bytes = None, 45.0, 50 * 1024

        try:
            # 🟢 修正：原本传参 shell[0], *shell[1:] 会破坏列表解包，导致找不到程序。现改为星号全展开。
            process = await asyncio.create_subprocess_exec(
                *shell, cmd, cwd=file_dir, **kwargs
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            trunc = len(stdout_bytes) > max_bytes or len(stderr_bytes) > max_bytes
            return self._diagnose(
                args.runtime,
                stdout_bytes[:max_bytes],
                stderr_bytes[:max_bytes],
                process.returncode,
                trunc,
            )

        except asyncio.TimeoutError:
            await self._cleanup(process)
            return f"错误: 代码运行超时（超过 {timeout} 秒）已被强制终止。请检查是否存在无限循环。"
        except Exception as e:
            await self._cleanup(process)
            return f"工具底层执行异常: {str(e)}"

    def _diagnose(
        self, runtime: str, out: bytes, err: bytes, code: int, trunc: bool
    ) -> str:
        """4. 智能多语言诊断器"""

        def clean(b: bytes) -> str:
            if not b:
                return ""
            # 支持动态多编码自动平滑解码
            for enc in [sys.getfilesystemencoding(), "utf-8", "gbk", "cp1252"]:
                try:
                    return "\n".join(
                        l.strip()
                        for l in self._ansi_escape.sub("", b.decode(enc)).splitlines()
                        if l.strip()
                    ).strip()
                except UnicodeDecodeError:
                    continue
            return b.decode("utf-8", errors="replace").strip()

        o_str, e_str = clean(out), clean(err)
        res = [f"[标准输出]:\n{o_str}"] if o_str else []

        if e_str:
            res.append(f"[标准错误]:\n{e_str}")

            lang = runtime.lower().strip()
            ai_tips = []

            # 各大语言诊断分析树
            if lang == "python":
                if "ModuleNotFoundError" in e_str or "ImportError" in e_str:
                    ai_tips.append(
                        "缺失第三方依赖库，请尝试运行 `pip install <模块名>` 进行安装。"
                    )
                elif "SyntaxError" in e_str:
                    ai_tips.append(
                        "存在语法错误，请检查是否混用了中英文标点、括号未闭合或缩进不正确。"
                    )
                elif "IndexError" in e_str:
                    ai_tips.append("数组/列表索引越界，请检查循环边界或空列表取值。")
            elif lang in ["javascript", "typescript"]:
                if "is not defined" in e_str:
                    ai_tips.append(
                        "使用了未定义的变量或函数，请检查拼写、作用域或 import/require 导入。"
                    )
                elif (
                    "Cannot read properties of undefined" in e_str
                    or "TypeError" in e_str
                ):
                    ai_tips.append(
                        "尝试读取了 null 或 undefined 的属性，请在调用前增加空值校验(?.操作符)。"
                    )
            elif lang in ["c", "cpp"]:
                if "was not declared in this scope" in e_str:
                    ai_tips.append(
                        "变量/函数未声明，请检查拼写或是否漏掉了对应的头文件（如 #include <iostream>）。"
                    )
                elif "undefined reference to" in e_str:
                    ai_tips.append(
                        "链接阶段失败，请检查函数实现是否存在，或者编译时是否漏掉了源文件/库链接。"
                    )
            elif lang == "go":
                if "imported and not used" in e_str:
                    ai_tips.append(
                        "Go 语言不允许存在未使用的包，请删除或注释掉未使用的 import 行。"
                    )
                elif "undefined:" in e_str:
                    ai_tips.append(
                        "未定义标识符，请检查变量名拼写，或多文件编译时未包含依赖文件。"
                    )
            elif lang == "java":
                if "cannot find symbol" in e_str:
                    ai_tips.append(
                        "找不到符号，请检查类名、变量名拼写是否正确，或是否漏掉了 import 语句。"
                    )
                elif "is public, should be declared in a file named" in e_str:
                    ai_tips.append(
                        "Java 主类名必须与文件名完全一致，请修改主类名或文件名使其匹配。"
                    )

            if ai_tips:
                res.append("💡 [AI 诊断提示]:")
                res.extend([f"  - {tip}" for tip in ai_tips])

        # C/C++ 内存核心错误无 stderr 时的兜底诊断
        if code != 0 and not e_str:
            lang = runtime.lower().strip()
            if lang in ["c", "cpp"] and code in [139, -11, 3221225477]:
                res.append(
                    "💡 [AI 诊断提示]: 进程无错误输出但异常退出。极可能触发了【段错误 (Segmentation Fault)】，"
                    "请检查是否存在野指针、内存越界、死循环导致栈溢出或空指针解引用。"
                )

        if trunc:
            res.append("\n⚠️ [提示]: 输出内容过大，后端已触发 50KB 内存防爆熔断保护。")
        if code != 0:
            res.append(f"❌ [执行状态]: 进程异常退出，状态码: {code}")

        return "\n".join(res) if res else "代码执行成功，无任何控制台输出。"

    async def _cleanup(self, process: asyncio.subprocess.Process) -> None:
        """5. 渐进式进程树安全强杀"""
        if not process or process.returncode is not None:
            return
        try:
            if self.is_win:
                k = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/F",
                    "/T",
                    "/PID",
                    str(process.pid),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(k.wait(), timeout=2.0)
            else:
                # 🟢 优化：加入 try 结构，防止在某些沙箱环境中 getpgid 失败导致进程无法强杀
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    await asyncio.sleep(0.2)
                    if process.returncode is None:
                        os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    process.terminate()
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
