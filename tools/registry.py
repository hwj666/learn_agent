import sys
import threading
from typing import Dict, Set, Type, Callable
from .base import BaseTool


class ToolRegistry:
    _registry: Dict[str, Type[BaseTool]] = {}
    _register_lock = threading.Lock()

    @classmethod
    def register(
        cls, *, name: str, toolset: str
    ) -> Callable[[Type[BaseTool]], Type[BaseTool]]:
        """
        标准参数化工具注册装饰器（不兼容无参老代码，强制具名传参）。

        使用方式:
        @ToolRegistry.register(name="file_view", toolset="file_ops")
        class FileViewTool(BaseTool): ...
        """

        def decorator(tool_cls: Type[BaseTool]) -> Type[BaseTool]:
            # 强行动态绑定到类属性上，确保工具内部仍可通过 cls.name / self.toolset 正常访问
            tool_cls.name = name
            tool_cls.toolset = toolset

            with cls._register_lock:
                if name in cls._registry:
                    existing_cls = cls._registry[name]

                    if existing_cls is not tool_cls:
                        # 工业级同源幂等校验：处理不同 import 路径导致的重复加载
                        mod_existing = sys.modules.get(existing_cls.__module__)
                        mod_current = sys.modules.get(tool_cls.__module__)
                        file_existing = getattr(mod_existing, "__file__", None)
                        file_current = getattr(mod_current, "__file__", None)

                        is_same_file = (
                            file_existing
                            and file_current
                            and file_existing == file_current
                        )
                        is_same_classname = (
                            existing_cls.__qualname__ == tool_cls.__qualname__
                        )

                        if is_same_file and is_same_classname:
                            cls._registry[name] = tool_cls
                            print(
                                f"[UPDATE] 幂等更新工具：{name} (由于不同的导入路径重复加载)"
                            )
                            return tool_cls

                        raise RuntimeError(
                            f"工具命名冲突！工具名 '{name}' 已被 "
                            f"{existing_cls.__module__}.{existing_cls.__name__} 注册，"
                            f"无法被 {tool_cls.__module__}.{tool_cls.__name__} 重复覆盖。"
                        )

                cls._registry[name] = tool_cls

            print(f"[OK] 注册工具：{name} (toolset={toolset})")
            return tool_cls

        return decorator

    @classmethod
    def get_tool(cls, name: str) -> Type[BaseTool] | None:
        return cls._registry.get(name)

    @classmethod
    def get_tools_by_set(cls, toolsets: Set[str]) -> Dict[str, Type[BaseTool]]:
        return {
            name: t_cls
            for name, t_cls in cls._registry.items()
            if getattr(t_cls, "toolset", None) in toolsets
        }

    @classmethod
    def clear(cls) -> None:
        with cls._register_lock:
            cls._registry.clear()
