import glob
import subprocess
import ast
import time
from registry_tool import tool

@tool
def list_project_files(root_dir: str = ".") -> str:
    """列出项目中的 Python 代码文件"""
    try:
        files = glob.glob(f"{root_dir}/**/*.py", recursive=True)
        return "\n".join(files[:30]) if files else "无 Python 文件"
    except Exception:
        return "获取文件列表失败"

@tool
def read_code_file(file_path: str) -> str:
    """读取代码文件内容"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取失败：{str(e)}"

@tool
def write_code_file(file_path: str, content: str) -> str:
    """⚠️ 必须调用这个函数写入修复后的代码，自动覆盖原文件完成修复"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ 写入成功：{file_path}"
    except Exception as e:
        return f"写入失败：{str(e)}"

@tool
def run_python_code(file_path: str, timeout: int = 15) -> str:
    """运行 Python 文件，返回输出与错误堆栈"""
    try:
        result = subprocess.run(
            ["python", file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return f"📤 输出：\n{result.stdout}\n\n❌ 错误：\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "执行超时"
    except Exception as e:
        return f"执行异常：{str(e)}"

@tool
def check_python_syntax(file_path: str) -> str:
    """检查 Python 代码语法是否正确"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            ast.parse(f.read())
        return "✅ 语法合法"
    except SyntaxError as e:
        return f"❌ 语法错误：行{e.lineno} | {e.msg}"
    except Exception:
        return "语法检查失败"

@tool
def get_code_structure(file_path: str) -> str:
    """分析代码结构：类、函数、导入"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([a.name for a in node.names])
            if isinstance(node, ast.ImportFrom):
                imports.append(f"{node.module}")
        return f"类：{classes}\n函数：{functions}\n导入：{imports}"
    except Exception:
        return "分析结构失败"

@tool
def search_in_code(file_path: str, keyword: str) -> str:
    """在代码中搜索关键词，返回行号"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        res = [f"第{i}行：{line.strip()}" for i, line in enumerate(lines, 1) if keyword in line]
        return "\n".join(res[:20]) if res else "未找到关键词"
    except Exception:
        return "搜索失败"

@tool
def get_error_context(file_path: str, line_num: int, window: int = 3) -> str:
    """获取报错行的上下文代码"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        start = max(0, line_num - window - 1)
        end = min(len(lines), line_num + window)
        return "".join(lines[start:end])
    except Exception:
        return "获取上下文失败"

@tool
def create_backup_file(file_path: str) -> str:
    """修改前自动备份文件，防止代码丢失"""
    try:
        backup = f"{file_path}.backup.{int(time.time())}.py"
        with open(file_path, "r", encoding="utf-8") as f1, open(backup, "w", encoding="utf-8") as f2:
            f2.write(f1.read())
        return f"✅ 备份完成：{backup}"
    except Exception:
        return "备份失败"