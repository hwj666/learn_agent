import asyncio
import logging
import os
import platform
import re
import shutil
import signal
import sys
from typing import Dict, Any, Tuple, List
from pydantic import BaseModel, Field

from tools.base import BaseTool
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class RunCommandArgs(BaseModel):
    command: str = Field(
        description=(
            "要在终端中执行的纯粹单条生态命令（如: 'pip install requests' 或 'npm install'）。\n"
            "【🔥🔥 终极阉割执行禁令 - 违反必报错 🔥🔥】:\n"
            "1. ❌ 严禁包含 'cd ' 命令！禁止通过 && 拼装目录切换！系统会自动在当前正确工作区运行，无需你切换目录。\n"
            "2. ❌ 严禁手写 'grep', 'sed', 'awk', 'cat', 'find' 等文件读写过滤命令！。\n"
            "3. ❌ 严禁手写 Linux 特有的重定向符号 '2>&1' 和管道符（如 '| head', '| grep'）。请直接发送最纯粹的命令本体。\n"
            "5. 代码格式化与自动修复首选: `ruff check --fix <path>` 或 `ruff format <path>`。\n"
            "6. 任何安装包操作必须加上非交互式自动同意参数（例如: pip install xxx）。"
        )
    )
    timeout: int = Field(
        default=30,
        description="命令执行的超时时间（秒），防止常驻进程死循环或卡死，默认 30 秒",
    )


@ToolRegistry.register(name="run_command", toolset="bash")
class RunCommandTool(BaseTool[RunCommandArgs]):
    description = (
        "【万能命令执行器（已功能阉割）】在系统终端中执行指定的外部 CLI 命令（如安装依赖包、代码审查、版本控制、打包编译等）。\n"
        "❌ 严禁在此工具中拼装 cd 命令或使用任何读写/过滤文件的命令（如 grep, sed, cat）。"
    )

    def __init__(self):
        super().__init__()
        self.current_os = platform.system()
        self.is_windows = self.current_os == "Windows"

    def _audit_and_sanitize(self, cmd: str) -> Tuple[bool, str, str]:
        """第一步：高危命令审计与功能阉割硬拦截"""
        cmd_lower = cmd.lower().strip()

        # ==================== 1. 功能阉割硬拦截层 ====================
        # 拦截 1：强行阻断大模型在通用终端里瞎跳目录
        if "cd " in cmd_lower or cmd_lower.startswith("cd"):
            return (
                False,
                "错误: 该工具已被【功能阉割】。❌ 严禁在 run_command 中执行 cd 命令或拼装目录切换！\n"
                "请直接在 command 字段中填写你要运行的命令本体（例如: pip install requests），系统会自动在当前工作区运行。",
                "",
            )

        # 拦截 2：强行阻断模型用终端命令去读写和过滤文件（逼它去用专属的 Python 结构化工具）
        bad_file_cmds = ["grep", "sed", "awk", "cat", "find", "select-string", "type "]
        used_bad_cmds = [c for c in bad_file_cmds if re.search(rf"\b{c}\b", cmd_lower)]
        if used_bad_cmds:
            return (
                False,
                f"错误: 该工具已被【功能阉割】。❌ 严禁在终端中使用 {used_bad_cmds} 操作文件！\n"
                "1. 想搜索关键字/查找代码？请立即退出并改用 [search_code] 工具。\n"
                "2. 想看文件内容/前几行？请立即退出并改用 [view_file] 工具。\n"
                "3. 想修改/替换/重构代码？请立即退出并改用 [patch_file_range] 工具。",
                "",
            )

        # ==================== 2. 跨平台语法静默清洗 ====================
        clean_cmd = cmd
        if self.is_windows:
            # 自动剥离 Linux 特有的标准错误重定向（Windows 下直接报语法错误）
            clean_cmd = clean_cmd.replace("2>&1", "")
            # 自动剥离 Linux 独有的流处理管道符（| head, | tail, | less）
            if "| head" in clean_cmd or "| tail" in clean_cmd or "| less" in clean_cmd:
                clean_cmd = re.sub(r"\|\s*(head|tail|less).*$", "", clean_cmd)

        # ==================== 3. 原有高危安全拦截层 ====================
        cmd_cleaned = re.sub(r'["\']', "", clean_cmd.lower())
        cmd_cleaned = re.sub(r"\s+", " ", cmd_cleaned).strip()

        dangerous_pattern = (
            r"\b(rm\s+-[a-z]*r[a-z]*\s+([/~\*]|\.\.))|"  # rm -rf /, rm -r *
            r"\b(mkfs|mkfs\.\w+|format\s+[a-zA-Z]:|dd\s+if=|dd\s+of=|reboot|shutdown|init\s+0)\b|"
            r"(\s*>\s*([/\\]dev|[/\\]etc[/\\]|[/\\]system32))"  # 恶意写保护区
        )
        if re.search(dangerous_pattern, cmd_cleaned):
            return (
                False,
                "错误: 检测到高危系统删除、格式化或写保护区命令，已被系统安全层强制拦截。",
                "",
            )

        return True, "", clean_cmd.strip()

    def _build_process_kwargs(self) -> Dict[str, Any]:
        """第二步：跨平台非交互式环境变量与子进程底层安全配置"""
        env_copy = os.environ.copy()
        env_copy.update(
            {
                "DEBIAN_FRONTEND": "noninteractive",
                "PAGER": "cat",
                "PYTHONUNBUFFERED": "1",
            }
        )

        kwargs = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "env": env_copy,
        }

        if self.is_windows:
            # CREATE_NO_WINDOW: 隐藏控制台弹窗，静默运行
            kwargs["creationflags"] = 0x08000000
        else:
            if sys.version_info >= (3, 11):
                kwargs["process_group"] = 0
            else:
                kwargs["preexec_fn"] = os.setsid
        return kwargs

    async def _read_stream_safe(
        self, process: asyncio.subprocess.Process, timeout: float
    ) -> Tuple[bytes, bytes, bool]:
        """第三步：内存防爆流式读取（50KB 内存熔断保护）"""
        stdout_chunks: List[bytes] = []
        stderr_chunks: List[bytes] = []
        state = {"stdout_len": 0, "stderr_len": 0, "truncated": False}
        MAX_CAPTURE_BYTES = 50 * 1024  # 50KB 内存熔断限额

        async def read_stream(stream, chunks, len_key):
            try:
                while True:
                    chunk = await stream.read(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    state[len_key] += len(chunk)
                    if state[len_key] > MAX_CAPTURE_BYTES:
                        state["truncated"] = True
                        break
            except Exception:
                pass

        # 使用 wait_for 统一控制绝对超时
        await asyncio.wait_for(
            asyncio.gather(
                read_stream(process.stdout, stdout_chunks, "stdout_len"),
                read_stream(process.stderr, stderr_chunks, "stderr_len"),
            ),
            timeout=timeout,
        )

        return b"".join(stdout_chunks), b"".join(stderr_chunks), state["truncated"]

    def _process_and_diagnose(
        self,
        cmd: str,
        stdout_bytes: bytes,
        stderr_bytes: bytes,
        return_code: int,
        is_truncated: bool,
    ) -> str:
        """第四步：文本清洗、反幻觉诊断与大模型语义拼接"""

        def _decode(b: bytes) -> str:
            if not b:
                return ""
            encodings = list(
                dict.fromkeys(
                    filter(
                        None, [sys.getfilesystemencoding(), "utf-8", "gbk", "cp1252"]
                    )
                )
            )
            for enc in encodings:
                try:
                    return b.decode(enc).strip()
                except UnicodeDecodeError:
                    continue
            return b.decode(encodings[0], errors="replace").strip()

        def _clean_ansi(text: str) -> str:
            if not text:
                return ""
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[.*?[a-zA-Z])")
            lines = [
                line.strip()
                for line in ansi_escape.sub("", text).splitlines()
                if line.strip()
            ]
            return "\n".join(lines).strip()

        out_str = _clean_ansi(_decode(stdout_bytes))
        err_str = _clean_ansi(_decode(stderr_bytes))

        # 智能反幻觉诊断引导核心
        if err_str:
            if "command not found" in err_str and "python3" in cmd:
                err_str += "\n\n【智能排错提示】: 沙箱中 Python 唤醒词是 'python'，请勿使用 'python3'。若要运行 Python 文件，请立即退出并改用专属工具 [run_code_file]。"
            elif "command not found" in err_str or (
                self.is_windows
                and any(x in err_str for x in ["not recognized", "找不到"])
            ):
                match = re.search(
                    r"([^:\s]+):\s*(?:command not found|项识别为|not recognized)",
                    err_str,
                )
                missing_cmd = match.group(1) if match else "该工具"
                err_str += f"\n\n【智能排错提示】: 环境中找不到命令 '{missing_cmd}'。请确认命令拼写是否正确，或尝试通过系统包管理器先行安装。"

        # 最终组装
        result = []
        if out_str:
            result.append(f"[标准输出]:\n{out_str}")
        if err_str:
            result.append(f"[标准错误]:\n{err_str}")
        if is_truncated:
            result.append(
                "\n⚠️ [警告]: 输出内容过大，后端已启动流式截断（内存防爆熔断保护），请勿盲目运行打印海量日志的命令。"
            )
        if return_code != 0:
            result.append(f"❌ [执行状态]: 命令执行失败，非零异常状态码: {return_code}")

        return "\n".join(result) if result else "命令执行成功，无任何控制台输出。"

    async def execute(self, ctx: Dict[str, Any], args: RunCommandArgs) -> str:
        """主入口：编排整个命令执行生命周期（企业级强化版）"""
        if self.current_os not in ["Windows", "Linux", "Darwin"]:
            return f"错误: 当前运行环境 {self.current_os} 不受支持。"

        # 1. 安全拦截与审计
        original_command = args.command
        is_safe, audit_msg, clean_command = self._audit_and_sanitize(original_command)
        if not is_safe:
            return audit_msg

        timeout = max(1, min(args.timeout, 300))
        workspace_dir = os.path.abspath(ctx.get("workspace_dir", "."))
        process = None

        try:
            # 2. 跨平台进程参数装配
            kwargs = self._build_process_kwargs()

            if self.is_windows:
                process = await asyncio.create_subprocess_exec(
                    "powershell.exe",
                    "-NonInteractive",
                    "-Command",
                    clean_command,
                    cwd=workspace_dir,
                    **kwargs,
                )
            else:
                # Linux/Darwin: 注入 preexec_fn 使子进程独立成组，防止 killpg 误杀主程序
                shell = "bash" if shutil.which("bash") else "sh"
                process = await asyncio.create_subprocess_exec(
                    shell,
                    "-c",
                    clean_command,
                    cwd=workspace_dir,
                    preexec_fn=os.setsid,  # 【关键修复】创建新进程组
                    **kwargs,
                )

            # 3. 流式动态读取（内存防爆）
            stdout_bytes, stderr_bytes, is_truncated = await self._read_stream_safe(
                process, timeout
            )
            return_code = await process.wait()

            # 4. 后置处理与智能排错
            execution_result = self._process_and_diagnose(
                clean_command, stdout_bytes, stderr_bytes, return_code, is_truncated
            )

            # 【修复语法缩进】正确的返回逻辑
            if clean_command != original_command:
                prefix_hint = "【系统提示：检测到当前为 Windows 环境，后端已自动将不兼容的 Linux 重定向/管道符清洗调整】\n"
                return prefix_hint + execution_result
            return execution_result

        except asyncio.TimeoutError:
            await self._cleanup_process(process)
            return (
                f"错误: 命令执行超时（超过 {timeout} 秒），进程已被强行终止。\n"
                f"原因提示: 大模型你可能执行了常驻进程或触发了需要人工确认的交互式输入，请改用非交互式命令。"
            )
        except Exception as e:
            if process:
                await self._cleanup_process(process)
            return f"工具底层执行异常: {str(e)}"

    async def _cleanup_process(self, process: asyncio.subprocess.Process) -> None:
        """高可靠安全清理子进程及其整个进程树（防御强化版）"""
        if not process or process.returncode is not None:
            return

        pid = process.pid

        # --- Windows 清理逻辑 ---
        if self.is_windows:
            try:
                kill_proc = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/F",
                    "/T",
                    "/PID",
                    str(pid),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(kill_proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"⚠️ [进程清理] taskkill 强杀进程树超时(2s)，尝试直接降级强杀 PID: {pid}"
                )
                try:
                    process.kill()
                except Exception:
                    pass
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

        # --- Linux / Darwin 清理逻辑 ---
        else:
            try:
                # 获取进程组 ID。因为设置了 os.setsid，此时 pgid == pid
                pgid = os.getpgid(pid)
                if pgid == os.getgetpid():  # 防御性编程：绝对不允许等于当前主程序 PID
                    raise RuntimeError("进程组ID与主程序冲突，拒绝杀组")

                # 优雅终止进程组 (SIGTERM)
                os.killpg(pgid, signal.SIGTERM)

                # 异步轮询等待退出
                for _ in range(10):  # 略微延长等待窗口 (0.5秒)
                    if process.returncode is not None:
                        break
                    await asyncio.sleep(0.05)

                # 依然未退出，暴力强杀 (SIGKILL)
                if process.returncode is None:
                    os.killpg(pgid, signal.SIGKILL)

            except ProcessLookupError:
                logger.debug(f"ℹ️ [进程清理] 进程 {pid} 在清理前已提前退出")
            except Exception as e:
                logger.error(f"❌ [进程清理] Linux 进程组清理期间异常: {str(e)}")
                try:
                    process.kill()
                except Exception:
                    pass

        # --- 最终兜底：回收僵尸进程句柄 ---
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except Exception:
            pass
