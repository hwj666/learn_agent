from abc import ABC, abstractmethod
import copy
from typing import Any, Dict, Type, TypeVar,Generic, get_args
from pydantic import BaseModel

ArgsType = TypeVar("T", bound=BaseModel)
class BaseTool(Generic[ArgsType], ABC):
    """所有自定义工具的硬核抽象基类"""
    name: str = ""          # 工具全局唯一标识
    description: str = ""   # 工具功能描述（供LLM识别）
    toolset: str = ""       # 工具所属集合标签
    args_schema: Type[ArgsType] # 显式绑定的参数结构体类

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # 💡 兼容策略 1：如果子类已经手动定义了 args_schema（如写了内部类或手动赋值），则不要去覆盖它
        if "args_schema" in cls.__dict__:
            return

        # 💡 兼容策略 2：如果子类通过泛型继承（如 BaseTool[GetWeatherArgs]），则动态提取
        if hasattr(cls, "__orig_bases__"):
            for base in cls.__orig_bases__:
                args = get_args(base)
                # ✅ 彻底修复元组 Bug：get_args 返回的是元组，取第一个元素 args[0]
                if args and isinstance(args, tuple) and issubclass(args[0], BaseModel):
                    cls.args_schema = args[0]
                    break
    
    def __setattr__(self, name, value):
        # 允许在初始化期间（__init__）赋值
        if not getattr(self, "_initialized", False):
            super().__setattr__(name, value)
            return
            
        # 🚀 运行时锁死：如果已经初始化完毕，任何在 _run 里的赋值行为都会被拦截！
        raise AttributeError(
            f"❌ 警告：工具基类已被设置为【无状态/只读】模式！"
            f"禁止在运行时修改 self.{name}。请将状态存入局部变量或通过 ctx 传递。"
        )
    
    @abstractmethod
    async def execute(self, ctx: Dict[str, Any], args: ArgsType) -> Any:
        """工具核心执行逻辑。ctx 传递运行时上下文状态"""
        pass

    @classmethod
    def to_schema(cls) -> Dict[str, Any]:
        """
        全自动生成 OpenAI 兼容的 Tool Schema（彻底解决 $defs 与 $ref 不兼容问题）
        """
        if not cls.name or not cls.description:
            raise ValueError(f"工具 {cls.__name__} 必须定义 name 和 description")

        schema: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
            }
        }

        if cls.args_schema and issubclass(cls.args_schema, BaseModel):
            # 1. 拿到 Pydantic 原生带有 $defs 的 schema
            pydantic_schema = cls.args_schema.model_json_schema()
            pydantic_schema.pop("title", None)  # 移除无关的 title 干扰大模型

            # 2. 如果存在嵌套定义的 $defs，启动内联替换机制 [1]
            if "$defs" in pydantic_schema:
                defs = pydantic_schema.pop("$defs")
                
                def inline_refs(obj: Any) -> Any:
                    """递归将所有 $ref 替换为真正的对象定义"""
                    if isinstance(obj, dict):
                        if "$ref" in obj:
                            # 提取引用名称，例如 "#/$defs/TodoItemArgs" -> "TodoItemArgs"
                            ref_name = obj["$ref"].split("/")[-1]
                            # 现场深拷贝一份定义过来替换掉 $ref [1]
                            ref_schema = copy.deepcopy(defs[ref_name])
                            ref_schema.pop("title", None)  # 同样移除嵌套对象的 title
                            return inline_refs(ref_schema)
                        return {k: inline_refs(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [inline_refs(item) for item in obj]
                    return obj

                # 执行内联展开
                pydantic_schema = inline_refs(pydantic_schema)

            schema["function"]["parameters"] = pydantic_schema
        else:
            schema["function"]["parameters"] = {
                "type": "object",
                "properties": {}
            }

        return schema