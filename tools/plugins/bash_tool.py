import asyncio
import logging
import os
import platform
import sys
import shutil
import signal
import re
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, Field

# 假设这是你的基础类
from tools.base import BaseTool
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class RunCommandArgs(BaseModel):
    command: str = Field(
        description=(
            "要在终端中执行的命令。例如 Windows: 'dir', Linux: 'ls -la'。\n"
            "【极端重要执行规范】:\n"
            "1. 执行 Python 脚本时，必须统一使用 'python' 开头（当前环境无 'python3' 命令，切勿盲目脑补）。\n"
            "2. 代码格式化与自动修复(首选): `ruff check --fix <path>` 或 `ruff format <path>`"
            "3. 严禁盲目臆测环境中不存在的高级工具（如 wget, htop, tree），请优先使用系统自带原生的 curl, cat, ls 等。\n"
            "4. 任何安装包或系统更新操作必须加上非交互式自动同意参数（例如: apt-get install -y xxx, pip install xxx）。"
        )
    )
    timeout: int = Field(
        default=30,
        description="命令执行的超时时间（秒），防止常驻进程死循环或卡死，默认 30 秒",
    )


@ToolRegistry.register(name="run_command", toolset="bash")
class RunCommandTool(BaseTool[RunCommandArgs]):
    description = (
        "核心系统终端工具。可用于在 Windows(PowerShell) 或 Linux/macOS(Bash/Sh) 中执行自动化测试脚本、"
        "编译项目或查看系统环境状态。在使用前，你必须通过前置步骤明确当前运行的操作系统类型，严禁执行任何交互式命令。"
    )

    def __init__(self):
        self.current_os = platform.system()
        self.is_windows = self.current_os == "Windows"

    async def _cleanup_process(
        self, process: asyncio.subprocess.Process, is_windows: bool
    ) -> None:
        """
        高可靠安全清理子进程及其整个进程树
        1. 彻底解决 ProcessLookupError 异常二次穿透导致的句柄泄漏
        2. 为 Windows 端的 taskkill 引入绝对超时，杜绝协程死锁
        """
        if not process or process.returncode is not None:
            return

        pid = process.pid

        if is_windows:
            try:
                # 1. Windows 端强杀：为辅助进程引入 2 秒绝对超时，防止 taskkill 自身挂起引发死锁
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
                # 兜底捕获所有权限不足、命令找不到等异常，确保不穿透
                try:
                    process.kill()
                except Exception:
                    pass
        else:
            try:
                # 2. Linux/Unix 端强杀：进程组两阶段强杀
                pgid = os.getpgid(pid)

                # 第一阶段：温柔劝退
                os.killpg(pgid, signal.SIGTERM)

                # 引入快速轮询，在 0.2 秒内只要子进程退出了就立即往下走，无需白白 sleep 满 0.2 秒
                for _ in range(4):
                    if process.returncode is not None:
                        break
                    await asyncio.sleep(0.05)

                # 第二阶段：死不悔改则强制抹杀
                if process.returncode is None:
                    os.killpg(pgid, signal.SIGKILL)

            except ProcessLookupError:
                # 运行到中途时，如果进程已经提前死掉，os.getpgid 或 killpg 会抛出此异常
                logger.debug(f"ℹ️ [进程清理] 进程 {pid} 在清理前已提前退出")
            except Exception as e:
                logger.error(
                    f"❌ [进程清理] Linux 进程组清理期间发生未知异常: {str(e)}"
                )
                # 哪怕进程组报错，也要尝试单独强杀主 PID 兜底
                try:
                    process.kill()
                except Exception:
                    pass

        # ====================== 核心资源回收区 ======================
        # 无论前方哪个分支抛出什么奇葩异常，这里是终点站，必须通过 wait() 彻底收回底层文件句柄和 PID
        try:
            # 给予最终的进程状态冲刷，防止 asyncio 内部留下死循环的僵尸状态句柄
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            logger.critical(
                f"🚨 [严重警告] 进程 {pid} 拒绝响应 wait() 回收，可能已变成系统僵尸进程！"
            )
        except Exception:
            pass

    async def execute(self, ctx: Dict[str, Any], args: RunCommandArgs) -> str:
        """主入口：编排整个命令执行生命周期"""
        if self.current_os not in ["Windows", "Linux", "Darwin"]:
            return f"错误: 当前运行环境 {self.current_os} 不受支持。"

        # 1. 安全审计
        is_safe, audit_msg = self._audit_command(args.command)
        if not is_safe:
            return audit_msg

        timeout = max(1, min(args.timeout, 300))
        process = None

        try:
            # 2. 跨平台进程参数装配
            kwargs = self._build_process_kwargs()

            if self.is_windows:
                process = await asyncio.create_subprocess_exec(
                    "cmd.exe", "/c", args.command, **kwargs
                )
            else:
                shell = "bash" if shutil.which("bash") else "sh"
                process = await asyncio.create_subprocess_exec(
                    shell, "-c", args.command, **kwargs
                )

            # 3. 内存防爆：流式动态读取
            stdout_bytes, stderr_bytes, is_truncated = await self._read_stream_safe(
                process, timeout
            )
            return_code = await process.wait()

            # 4. 后置处理、清洗与智能排错
            return self._process_and_diagnose(
                args.command, stdout_bytes, stderr_bytes, return_code, is_truncated
            )

        except asyncio.TimeoutError:
            await self._cleanup_process(process, self.is_windows)
            return (
                f"错误: 命令执行超时（超过 {timeout} 秒），进程已被强行终止。\n"
                f"原因提示: 大模型你可能执行了常驻进程或触发了需要人工确认的交互式输入，请改用非交互式命令。"
            )
        except Exception as e:
            if process:
                await self._cleanup_process(process, self.is_windows)
            return f"工具底层执行异常: {str(e)}"

    def _audit_command(self, cmd: str) -> Tuple[bool, str]:
        """第一步：高危命令审计拦截"""
        cmd_cleaned = re.sub(r'["\']', "", cmd.lower())
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
            )
        return True, ""

    def _build_process_kwargs(self) -> Dict[str, Any]:
        """第二步：跨平台非交互式环境变量与进程配置"""
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
        """第三步：内存防爆流式读取（核心流控）"""
        stdout_chunks: List[bytes] = []
        stderr_chunks: List[bytes] = []
        state = {"stdout_len": 0, "stderr_len": 0, "truncated": False}
        MAX_CAPTURE_BYTES = 50 * 1024  # 50KB 内存熔断保护

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

        # 使用 wait_for 统一控制读取和进程退出的绝对超时
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

        # 智能排错引导核心
        if err_str:
            if "command not found" in err_str and "python3" in cmd:
                err_str += "\n\n【智能排错提示】: 沙箱中 Python 唤醒词是 'python'，请勿使用 'python3'。"
            elif self.is_windows and any(
                x in err_str for x in ["not recognized", "找不到"]
            ):
                bad_cmds = ["ls", "cat", "rm", "mkdir", "pwd"]
                used_bad = [c for c in bad_cmds if re.search(rf"\b{c}\b", cmd.lower())]
                if used_bad:
                    err_str += f"\n\n【智能排错提示】: 当前为 Windows 环境，无法运行 Linux 命令 '{used_bad}'。请改用平替命令（如 dir, type, del）。"
            elif "command not found" in err_str:
                match = re.search(r"([^:\s]+):\s*command not found", err_str)
                missing_cmd = match.group(1) if match else "该工具"
                err_str += f"\n\n【智能排错提示】: 环境中找不到命令 '{missing_cmd}'。请尝试寻找原生基础命令平替，或通过系统包管理器先行安装。"

        # 最终组装
        result = []
        if out_str:
            result.append(f"[标准输出]:\n{out_str}")
        if err_str:
            result.append(f"[标准错误]:\n{err_str}")
        if is_truncated:
            result.append(
                "\n⚠️ [警告]: 输出内容过大，后端已启动流式截断，请勿盲目 cat 大文件。"
            )
        if return_code != 0:
            result.append(f"❌ [执行状态]: 命令执行失败，非零异常状态码: {return_code}")

        return "\n".join(result) if result else "命令执行成功，无任何控制台输出。"

    async def _cleanup_process(self, process, is_windows):
        """辅助方法：强杀残留进程"""
        if not process:
            return
        try:
            if is_windows:
                process.terminate()
            else:
                import signal

                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
