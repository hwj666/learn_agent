import asyncio
import os
import sys
import platform
import shutil
import signal
import re
from typing import Dict, Any
from pydantic import BaseModel, Field

# 假设这是你的基础类
from tools.base import BaseTool


class RunCommandArgs(BaseModel):
    command: str = Field(
        description="要在终端中执行的命令。例如 Windows: 'dir', Linux: 'ls -la'"
    )
    timeout: int = Field(
        default=30,
        description="命令执行的超时时间（秒），防止死循环，默认 30 秒"
    )


class RunCommandTool(BaseTool[RunCommandArgs]):
    description = "用于在 Windows(PowerShell) 或 Linux/macOS(Bash/Sh) 中执行自动化测试脚本、编译项目或查看系统环境状态。"
    toolset = "bash"

    async def _cleanup_process(self, process: asyncio.subprocess.Process, is_windows: bool) -> None:
        """安全清理子进程及其所有子进程，防止句柄和进程泄漏"""
        if process.returncode is not None:
            return

        if is_windows:
            try:
                # Windows: 强行杀死整个进程树 (/T)
                kill_proc = await asyncio.create_subprocess_exec(
                    "taskkill", "/F", "/T", "/PID", str(process.pid),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await kill_proc.wait()
            except Exception:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        else:
            try:
                # Linux/Unix: 向进程组发送终止信号
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM)
                await asyncio.sleep(0.2)

                if process.returncode is None:
                    os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            except Exception:
                pass

        try:
            await process.wait()
        except Exception:
            pass
    async def execute(self, ctx: Dict[str, Any], args: RunCommandArgs) -> str:
        current_os = platform.system()
        is_windows = current_os == "Windows"

        if current_os not in ["Windows", "Linux", "Darwin"]:
            return f"错误: 当前运行环境 {current_os} 不受支持。"

        cmd = args.command
        timeout = max(1, min(args.timeout, 300))
        process = None

        # 高危命令拦截
        dangerous_pattern = (
            r"\b(rm\s+-(r|f|rf|fr)\s*[/~]|rm\s+--recursive|rm\s+--force|"
            r"mkfs|mkfs\.\w+|format\s+[a-zA-Z]:|dd\s+if=|dd\s+of=|"
            r":\s*>\s*[/\\]dev|chmod\s+-R\s+777|reboot|shutdown|init\s+0)\b"
        )
        if re.search(dangerous_pattern, cmd.lower()):
            return "错误: 检测到高危系统命令，已被系统拦截。"

        try:
            if is_windows:
                process = await asyncio.create_subprocess_exec(
                    "powershell.exe", "-NoProfile", "-Command", cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=0x08000000
                )
            else:
                shell_executable = "bash" if shutil.which("bash") else "sh"
                kwargs = {
                    "stdout": asyncio.subprocess.PIPE,
                    "stderr": asyncio.subprocess.PIPE
                }
                if sys.version_info >= (3, 11):
                    kwargs["process_group"] = 0
                else:
                    kwargs["preexec_fn"] = os.setsid

                process = await asyncio.create_subprocess_exec(
                    shell_executable, "-c", cmd, **kwargs
                )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                await self._cleanup_process(process, is_windows)
                return f"错误: 命令执行超时（超过 {timeout} 秒），已强行终止整个进程树。"

            # 解码函数
            def decode_output(output_bytes: bytes) -> str:
                if not output_bytes:
                    return ""
                encodings = [sys.getfilesystemencoding(), "utf-8", "gbk", "cp1252"]
                encodings = list(dict.fromkeys(filter(None, encodings)))
                for enc in encodings:
                    try:
                        return output_bytes.decode(enc).strip()
                    except UnicodeDecodeError:
                        continue
                return output_bytes.decode(encodings[0], errors="replace").strip()

            # ====================== 输出处理核心 ======================
            def process_output(text: str) -> str:
                if not text:
                    return ""
                # 移除 ANSI 颜色
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[.*?[a-zA-Z])')
                text = ansi_escape.sub('', text)
                # 清理空行
                lines = [line.strip() for line in text.splitlines()]
                lines = [line for line in lines if line]
                text = "\n".join(lines)
                # 超长截断
                max_len = 2000
                if len(text) > max_len:
                    text = text[:max_len] + "\n...（输出过长，已截断）"
                return text.strip()
            # ==========================================================

            out_str = process_output(decode_output(stdout))
            err_str = process_output(decode_output(stderr))
            return_code = process.returncode

            # 组装结果
            result = []
            if out_str:
                result.append(f"[标准输出]:\n{out_str}")
            if err_str:
                result.append(f"[标准错误]:\n{err_str}")
            if return_code != 0:
                result.append(f"[退出状态码]: {return_code}")

            return "\n".join(result) if result else "命令执行成功，无任何输出。"

        except Exception as e:
            if process:
                await self._cleanup_process(process, is_windows)
            return f"工具执行异常: {str(e)}"