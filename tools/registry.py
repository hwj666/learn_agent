import threading
from typing import Dict, Set, Type
from tools.base import BaseTool

class ToolRegistry:
    """
    全局工具注册表（只管理类模板，不持有运行时实例）
    线程安全 + 防重名 + 支持装饰器注册 + 支持工具集过滤
    """
    
    _registry: Dict[str, Type[BaseTool]] = {}
    _register_lock = threading.Lock()

    @classmethod
    def register(cls, tool_cls: Type[BaseTool]) -> Type[BaseTool]:
        """
        注册工具类模板（支持装饰器语法 @ToolRegistry.register）
        """
        tool_name = getattr(tool_cls, "name", None)
        toolset = getattr(tool_cls, "toolset", None)
        
        if not tool_name or not toolset:
            raise ValueError(
                f"工具类 {tool_cls.__name__} 必须显式定义 'name' 和 'toolset' 属性"
            )
        
        with cls._register_lock:
            if tool_name in cls._registry:
                existing_cls = cls._registry[tool_name]
                if existing_cls is not tool_cls:
                    raise RuntimeError(
                        f"工具命名冲突！工具名 '{tool_name}' 已被 "
                        f"{existing_cls.__module__}.{existing_cls.__name__} 注册，"
                        f"无法被 {tool_cls.__module__}.{tool_cls.__name__} 重复覆盖。"
                    )
            cls._registry[tool_name] = tool_cls
        
        # 打印注册日志（可选）
        print(f"✅ 注册工具：{tool_name} (toolset={toolset})")
        return tool_cls

    @classmethod
    def get_tool(cls, name: str) -> Type[BaseTool] | None:
        """安全获取单个工具类模板"""
        return cls._registry.get(name)

    @classmethod
    def get_tools_by_set(cls, toolsets: Set[str]) -> Dict[str, Type[BaseTool]]:
        """根据工具集名称过滤出符合条件的工具映射字典"""
        return {
            name: t_cls for name, t_cls in cls._registry.items()
            if getattr(t_cls, "toolset", None) in toolsets
        }

    @classmethod
    def clear(cls) -> None:
        """清空注册表（专供单元测试重置状态使用）"""
        with cls._register_lock:
            cls._registry.clear()