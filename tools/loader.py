import importlib.util
import inspect
from pathlib import Path
import sys
import threading
from typing import Any, Optional, Set
from tools.base import BaseTool
from tools.registry import ToolRegistry

# 线程锁，确保多线程环境下模块加载安全
_load_lock = threading.Lock()

def _ensure_parent_modules(namespace_root: str, relative_parts: tuple, dir_path: Path) -> str:
    """
    自顶向下安全构建虚拟包结构，确保整条包路径存在于 sys.modules
    返回当前文件所在的父包名
    """
    current_ns = namespace_root
    current_path = dir_path

    # 确保根命名空间包存在
    if current_ns not in sys.modules:
        init_file = current_path / "__init__.py"
        if init_file.exists():
            spec = importlib.util.spec_from_file_location(current_ns, str(init_file))
        else:
            spec = importlib.util.spec_from_loader(current_ns, None, is_package=True)

        if spec:
            root_module = importlib.util.module_from_spec(spec)
            root_module.__path__ = [str(current_path)]
            root_module.__package__ = current_ns
            sys.modules[current_ns] = root_module

    # 逐层构建子包
    for part in relative_parts[:-1]:
        parent_ns = current_ns
        current_ns = f"{parent_ns}.{part}"
        current_path = current_path / part

        if current_ns not in sys.modules:
            sub_init = current_path / "__init__.py"
            if sub_init.exists():
                spec = importlib.util.spec_from_file_location(current_ns, str(sub_init))
            else:
                spec = importlib.util.spec_from_loader(current_ns, None, is_package=True)

            if spec:
                sub_mod = importlib.util.module_from_spec(spec)
                sub_mod.__path__ = [str(current_path)]
                sub_mod.__package__ = current_ns
                sys.modules[current_ns] = sub_mod

                # 挂载到父模块
                parent_mod = sys.modules.get(parent_ns)
                if parent_mod:
                    setattr(parent_mod, part, sub_mod)

    return current_ns

def _load_module_by_path(file_path: Path, module_name_key: str, parent_package: str, last_part: str) -> Optional[Any]:
    """
    从文件路径加载模块，并自动注册 BaseTool 子类
    """
    if module_name_key in sys.modules:
        return sys.modules[module_name_key]

    module = None
    try:
        spec = importlib.util.spec_from_file_location(module_name_key, str(file_path))
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        module.__package__ = parent_package
        sys.modules[module_name_key] = module

        # 执行模块
        spec.loader.exec_module(module)

        # 挂载到父包
        parent_mod = sys.modules.get(parent_package)
        if parent_mod:
            setattr(parent_mod, last_part, module)

        # 自动扫描并注册工具
        for _, obj in inspect.getmembers(module, inspect.isclass):
            try:
                if (
                    issubclass(obj, BaseTool)
                    and obj is not BaseTool
                    and not inspect.isabstract(obj)
                    and hasattr(obj, "name")
                    and hasattr(obj, "toolset")
                ):
                    ToolRegistry.register(obj)
            except Exception:
                continue

        return module

    except Exception as e:
        print(f"[加载失败] {file_path.name} -> {str(e)}")
        # 出错清理
        sys.modules.pop(module_name_key, None)
        parent_mod = sys.modules.get(parent_package)
        if parent_mod and hasattr(parent_mod, last_part):
            delattr(parent_mod, last_part)
        return None

def _scan_and_load_package(dir_path: Path, namespace_root: str) -> None:
    """递归扫描目录并加载所有 .py 文件，自动注册工具"""
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"[警告] 目录不存在: {dir_path}")
        return

    with _load_lock:
        visited_paths: Set[Path] = set()

        for path in dir_path.rglob("*.py"):
            try:
                real_path = path.resolve()
                if real_path in visited_paths:
                    continue
                visited_paths.add(real_path)
            except Exception:
                continue

            # 跳过 __init__.py 和隐藏文件
            if path.name == "__init__.py" or path.name.startswith("_") or path.name.startswith("."):
                continue

            relative_parts = path.relative_to(dir_path).with_suffix("").parts
            module_name = f"{namespace_root}.{'.'.join(relative_parts)}"

            try:
                parent_pkg = _ensure_parent_modules(namespace_root, relative_parts, dir_path)
                _load_module_by_path(real_path, module_name, parent_pkg, relative_parts[-1])
            except Exception as e:
                print(f"[错误] 处理文件 {path} 失败 -> {e}")

def discover_and_load_tools(user_tools_dir: Optional[str] = None) -> None:
    """
    一键自动加载所有工具
    - 系统内置工具：tools/plugins/
    - 用户自定义工具：传入 user_tools_dir 路径
    """
    current_file_dir = Path(__file__).resolve().parent
    system_plugins_dir = current_file_dir / "plugins"

    print(f"[信息] 加载系统插件: {system_plugins_dir}")
    _scan_and_load_package(system_plugins_dir, namespace_root="tools.plugins")

    if user_tools_dir:
        user_path = Path(user_tools_dir).resolve()
        print(f"[信息] 加载用户插件: {user_path}")
        _scan_and_load_package(user_path, namespace_root="user.plugins")