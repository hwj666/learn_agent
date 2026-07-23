# tracing/infra/transport/file.py
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from aiofiles import open as aio_open
from aiofiles.os import rename

from tracing.transport.protocol import Transport


logger = logging.getLogger(__name__)


class FileTransport(Transport[dict]):
    """
    本地文件落盘传输器（降级专用）。

    特性：
    1. 异步写入，不阻塞事件循环。
    2. JSON Lines 格式（.jsonl），便于离线处理。
    3. 支持文件大小滚动（Rolling）。
    4. 写入失败时抛异常，交由上层 BatchExporter 处理。

    注意：
    - 此类不负责删除旧文件，建议配合 logrotate 或 cron job。
    - 此类不负责重放（replay），重放应由独立脚本完成。
    """

    def __init__(
        self,
        base_dir: str,
        *,
        filename_prefix: str = "events",
        max_size_mb: int = 100,
        rotate_on_startup: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.filename_prefix = filename_prefix
        self.max_size_bytes = max_size_mb * 1024 * 1024

        # 当前正在写入的文件
        self._current_path: Optional[Path] = None
        self._file = None  # aiofiles file handle
        self._lock = asyncio.Lock()

        # 启动时滚动旧文件（防止上次崩溃遗留的大文件）
        if rotate_on_startup:
            self._rotate_existing_files()

    # ==================================================================
    # Public API
    # ==================================================================

    async def send(self, batch: List[dict]) -> None:
        """
        将批量数据追加写入当前文件。
        """
        if not batch:
            return

        async with self._lock:
            try:
                await self._ensure_file()
                # JSON Lines 格式：每行一个 JSON
                lines = (json.dumps(obj, ensure_ascii=False) + "\n" for obj in batch)
                await self._file.write("".join(lines))
                await self._file.flush()
                await self._maybe_rotate()
            except OSError as e:
                # 磁盘满、权限不足等致命错误
                logger.critical(
                    f"Failed to write to disk: {e}. "
                    f"Dropping {len(batch)} events permanently."
                )
                raise  # 重新抛出，触发 BatchExporter 的 on_drop

    async def close(self) -> None:
        """关闭文件句柄（可选，用于优雅停机）"""
        async with self._lock:
            if self._file:
                await self._file.close()
                self._file = None

    # ==================================================================
    # Internal Logic
    # ==================================================================

    async def _ensure_file(self) -> None:
        """确保当前文件存在且可用"""
        if self._file is None or self._file.closed:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._current_path = self._generate_filename()
            self._file = await aio_open(self._current_path, mode="a", encoding="utf-8")
            logger.info(f"FileTransport opened new file: {self._current_path}")

    def _generate_filename(self) -> Path:
        """生成带时间戳的文件名"""
        import time

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return self.base_dir / f"{self.filename_prefix}_{timestamp}.jsonl"

    async def _maybe_rotate(self) -> None:
        """检查文件大小，必要时滚动"""
        if not self._current_path:
            return

        size = (await self._file.tell()) if self._file else 0
        if size >= self.max_size_bytes:
            await self._rotate()

    async def _rotate(self) -> None:
        """执行滚动：关闭当前文件，重命名为 .1（或更精确的归档名）"""
        if self._file:
            await self._file.close()
            self._file = None

        if self._current_path and self._current_path.exists():
            # 重命名为 .done，方便离线处理工具识别
            archive_path = self._current_path.with_suffix(".jsonl.done")
            await rename(self._current_path, archive_path)
            logger.info(f"Rotated file to: {archive_path}")

        self._current_path = None

    def _rotate_existing_files(self) -> None:
        """
        启动时的滚动逻辑。
        将上次遗留的未归档文件标记为 .done，防止重复写入。
        """
        for p in self.base_dir.glob(f"{self.filename_prefix}_*.jsonl"):
            if not p.name.endswith(".done"):
                archive_path = p.with_suffix(".jsonl.done")
                try:
                    os.rename(p, archive_path)
                    logger.info(f"Startup rotation: {p} -> {archive_path}")
                except OSError as e:
                    logger.warning(f"Failed to rotate existing file {p}: {e}")
