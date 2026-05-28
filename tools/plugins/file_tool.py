import os
import sys
import shutil
import time
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from tools.base import BaseTool

class FileEditArgs(BaseModel):
    file_path: str = Field(description="目标文件的绝对路径或相对路径")
    action: str = Field(description="编辑动作：'create'(创建), 'view'(查看), 'insert'(插入), 'delete'(删除内容/删除行), 'replace'(替换), 'remove_file'(删除文件), 'restore'(还原最近一次备份), 'write_all'(重写整个文件)")
    line_num: int = Field(default=-1, description="目标行号（从 1 开始）。若为 -1，'insert' 默认为末尾追加，'delete'/'replace' 默认为全文匹配")
    target: str = Field(default="", description="目标字符串（非指定行操作时，用于全文删除或被替换）")
    replacement: str = Field(default="", description="新内容（用于 create 初始化、insert 插入或 replace 替换的内容）")

class FileEditTool(BaseTool[FileEditArgs]):
    description = "用于对本地文件进行读写编辑，支持按行号或文本内容操作。支持动作：create 创建文件、view 查看内容、insert 插入内容、delete 删除行/文本、replace 替换内容、write_all 全量覆写、remove_file 删除文件、restore 还原最新备份。强制沙箱权限管控，禁止越权访问，限制单文件最大10MB，修改前自动备份，出错自动回滚，请勿同时传入行号与目标文本。"
    toolset = "file"
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 限制最大操作 10MB 文件，防止内存溢出

    def _verify_permission(self, target_path: str, is_write_action: bool, ctx: Dict[str, Any]) -> bool:
        """安全修复：解决符号链接绕过与大小写不敏感系统的路径解析漏洞"""
        try:
            # 1. 彻底解析真实路径，防御符号链接攻击
            abs_target_path = os.path.normpath(os.path.abspath(os.path.realpath(target_path)))
            
            # Windows/macOS 大小写不敏感，统一转小写安全比对
            is_case_insensitive = os.name == 'nt' or sys.platform == 'darwin'
            if is_case_insensitive:
                abs_target_path = abs_target_path.lower()

            # 标准化沙箱目录路径
            read_dirs = [os.path.normpath(os.path.abspath(os.path.realpath(d))) for d in ctx.get("sandbox_read_dirs", [])]
            write_dirs = [os.path.normpath(os.path.abspath(os.path.realpath(d))) for d in ctx.get("sandbox_write_dirs", [])]

            def is_sub_path(target: str, parent: str) -> bool:
                if is_case_insensitive:
                    parent = parent.lower()
                try:
                    return os.path.commonpath([target, parent]) == parent
                except ValueError:
                    return False

            # 写操作仅允许写入目录
            if is_write_action:
                return any(is_sub_path(abs_target_path, w_dir) for w_dir in write_dirs)
            # 读操作允许读/写目录
            else:
                return any(is_sub_path(abs_target_path, w_dir) for w_dir in write_dirs) or any(is_sub_path(abs_target_path, r_dir) for r_dir in read_dirs)
        except Exception:
            return False

    def _create_backup(self, file_path: str, ctx: Dict[str, Any]) -> str:
        """细节优化：引入时间戳后缀，避免连续操作覆盖备份"""
        if ctx.get("auto_backup", True) is False or not os.path.exists(file_path):
            return ""
        try:
            timestamp = time.strftime("%Y%m%d%H%M%S")
            backup_path = f"{file_path}.{timestamp}.bak"
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception:
            return ""

    def _get_latest_backup(self, file_path: str) -> str:
        """辅助方法：检索最近一次的文件备份"""
        dir_name = os.path.dirname(file_path) or "."
        base_name = os.path.basename(file_path)
        try:
            files = os.listdir(dir_name)
            backups = [os.path.join(dir_name, f) for f in files if f.startswith(base_name) and f.endswith(".bak")]
            if not backups:
                return ""
            # 按修改时间降序，取最新
            backups.sort(key=os.path.getmtime, reverse=True)
            return backups[0]
        except Exception:
            return ""

    async def execute(self, ctx: Dict[str, Any], args: FileEditArgs) -> str:
        args.action = args.action.lower()
        path = args.file_path
        line_idx = args.line_num - 1

        # 定义写操作集合
        write_actions = {"create", "insert", "delete", "replace", "remove_file", "restore", "write_all"}
        is_write = args.action in write_actions
        # 权限检查
        if not self._verify_permission(path, is_write, ctx):
            perm_type = "写" if is_write else "读"
            return f"安全错误：您对路径 '{path}' 没有【{perm_type}】权限，拒绝访问"
        # 安全校验：禁止同时指定行号和目标文本
        if args.line_num > 0 and args.target.strip():
            return "错误：不能同时指定 'line_num' 和 'target' 参数，请保持操作语义唯一"
        print(args.action)
        # 1. 还原备份
        if args.action == "restore":
            backup_path = self._get_latest_backup(path)
            if not backup_path or not os.path.exists(backup_path):
                return f"错误：未找到文件 '{path}' 的有效备份文件"
            try:
                shutil.copy2(backup_path, path)
                return f"成功从最新备份 {os.path.basename(backup_path)} 中恢复文件"
            except Exception as e:
                return f"从备份还原文件失败: {str(e)}"

        # 2. 创建文件
        if args.action == "create":
            if os.path.exists(path):
                return f"错误：文件 '{path}' 已存在，无法重复创建"
            
            dir_name = os.path.dirname(path)
            if dir_name and not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name, exist_ok=True)
                except Exception as e:
                    return f"创建目录失败: {str(e)}"
            try:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(args.replacement)
                return f"成功创建文件 '{path}'"
            except Exception as e:
                return f"文件创建失败: {str(e)}"

        # 非创建/还原操作，必须文件存在
        if not os.path.exists(path):
            return f"错误：文件 '{path}' 不存在，请先执行 'create' 操作"

        # 写操作自动备份
        backup_file = ""
        if args.action in {"insert", "delete", "replace", "remove_file", "write_all"}:
            backup_file = self._create_backup(path, ctx)

        # 3. 全量覆盖写入
        if args.action == "write_all":
            try:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(args.replacement)
                msg = f"成功：已使用新内容完全覆盖文件 '{path}'"
                if backup_file:
                    msg += f"（历史版本已自动备份至 {os.path.basename(backup_file)}）"
                return msg
            except Exception as e:
                if backup_file and os.path.exists(backup_file):
                    shutil.copy2(backup_file, path)
                return f"全量覆写文件失败: {str(e)}，已尝试从自动备份中恢复原文件"

        # 4. 物理删除文件
        if args.action == "remove_file":
            try:
                os.remove(path)
                msg = f"成功删除文件 '{path}'"
                if backup_file:
                    msg += f"（已自动备份至 {os.path.basename(backup_file)}）"
                return msg
            except Exception as e:
                return f"物理删除文件失败: {str(e)}"

        # 大文件保护
        try:
            file_size = os.path.getsize(path)
            if file_size > self.MAX_FILE_SIZE:
                return f"安全错误：文件大小超过限制（最大支持 {self.MAX_FILE_SIZE // 1024 // 1024}MB，当前 {file_size // 1024 // 1024}MB），拒绝操作"
        except Exception as e:
            return f"获取文件大小失败: {str(e)}"

        # 读取文件内容
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                lines = f.readlines()
        except Exception as e:
            return f"读取文件失败: {str(e)}"

        # 自动探测文件原生换行符
        detected_newline = "\n"
        if lines:
            first_line = lines[0]
            if first_line.endswith("\r\n"):
                detected_newline = "\r\n"
            elif first_line.endswith("\n"):
                detected_newline = "\n"

        # 5. 查看文件
        if args.action == "view":
            if args.line_num > 0:
                if 0 <= line_idx < len(lines):
                    return f"第 {args.line_num} 行: {lines[line_idx].rstrip()}"
                return f"错误：行号 {args.line_num} 超出文件范围（总计 {len(lines)} 行）"
            # 带行号展示全文
            return "".join(f"{i+1}: {line}" for i, line in enumerate(lines))

        # 6. 插入内容
        elif args.action == "insert":
            insert_content = args.replacement
            # 统一换行符
            if not insert_content.endswith(detected_newline):
                insert_content += detected_newline

            if args.line_num <= 0:
                # 尾部追加
                lines.append(insert_content)
            else:
                # 指定行插入，自动处理越界
                insert_pos = max(0, min(line_idx, len(lines)))
                lines.insert(insert_pos, insert_content)

        # 7. 删除内容/行
        elif args.action == "delete":
            if args.line_num > 0:
                if 0 <= line_idx < len(lines):
                    lines.pop(line_idx)
                else:
                    return f"错误：行号 {args.line_num} 超出文件范围（总计 {len(lines)} 行）"
            else:
                if not args.target:
                    return "错误：全文删除必须指定 'target' 目标字符串"
                # 过滤不包含目标字符串的行
                lines = [line for line in lines if args.target not in line]

        # 8. 替换内容
        elif args.action == "replace":
            if args.line_num > 0:
                if 0 <= line_idx < len(lines):
                    replace_content = args.replacement
                    if not replace_content.endswith(detected_newline):
                        replace_content += detected_newline
                    lines[line_idx] = replace_content
                else:
                    return f"错误：行号 {args.line_num} 超出文件范围（总计 {len(lines)} 行）"
            else:
                if not args.target:
                    return "错误：全文替换必须指定 'target' 目标字符串"
                # 全局替换
                lines = [line.replace(args.target, args.replacement) for line in lines]

        else:
            return f"错误：不支持的动作 '{args.action}'"

        # 写入文件
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.writelines(lines)
            msg = f"成功执行 {args.action} 操作，文件已保存"
            if backup_file:
                msg += f"（备份文件: {os.path.basename(backup_file)}）"
            return msg
        except Exception as e:
            # 失败自动回滚
            if backup_file and os.path.exists(backup_file):
                shutil.copy2(backup_file, path)
            return f"写入文件失败: {str(e)}，已尝试从自动备份中恢复原文件"


if __name__ == "__main__":
    file_edit_tool = FileEditTool()
    args =FileEditArgs(file_path="work/test.py",action="write_all",replacement="xxx")
