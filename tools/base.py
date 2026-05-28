from abc import ABC, abstractmethod
import copy
from typing import Any, Dict, Type, TypeVar, Generic, get_args
from pydantic import BaseModel

# 泛型约束：所有工具参数必须是 Pydantic Model
ArgsType = TypeVar("ArgsType", bound=BaseModel)

class BaseTool(Generic[ArgsType], ABC):
    """
    LLM 函数调用工具 抽象基类
    特性：无状态安全、自动Schema生成、泛型参数绑定、嵌套结构兼容
    """
    # 类属性（子类必须定义）
    name: str = ""
    description: str = ""
    toolset: str = ""
    args_schema: Type[ArgsType]

    def __init_subclass__(cls, **kwargs):
        """子类创建时自动初始化：自动补全name、自动提取泛型参数"""
        super().__init_subclass__(**kwargs)

        # 自动生成工具名（不指定则用类名小写）
        if not cls.name:
            cls.name = cls.__name__.lower()

        # 如果子类已手动定义 args_schema，不覆盖
        if "args_schema" in cls.__dict__:
            return

        # 从泛型 BaseTool[XXXArgs] 自动提取参数类
        for base in cls.__orig_bases__:
            type_args = get_args(base)
            if type_args and issubclass(type_args[0], BaseModel):
                cls.args_schema = type_args[0]
                break

    def __init__(self):
        """初始化标记：用于 __setattr__ 防运行时修改"""
        self._initialized = True

    def __setattr__(self, name, value):
        """
        🔒 无状态保护：初始化后禁止修改 self 属性
        防止 LLM 工具出现状态污染、线程不安全问题
        """
        if not hasattr(self, "_initialized"):
            super().__setattr__(name, value)
            return

        raise AttributeError(
            f"工具是【无状态只读】的！禁止运行时修改 self.{name}\n"
            "请使用局部变量或 ctx 上下文传递状态。"
        )

    @abstractmethod
    async def execute(self, ctx: Dict[str, Any], args: ArgsType) -> Any:
        """
        工具执行入口（必须实现）
        :param ctx: 全局上下文（会话、用户信息、日志等）
        :param args: 类型安全的参数对象
        """
        pass

    @classmethod
    def to_schema(cls) -> Dict[str, Any]:
        """
        🔥 核心：生成 OpenAI 100% 兼容的函数调用 Schema
        自动解决嵌套模型 $ref/$defs 不兼容问题
        """
        if not cls.name or not cls.description:
            raise ValueError(f"工具 {cls.__name__} 必须配置 name 和 description")

        schema = {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
            }
        }

        # 生成参数结构
        if cls.args_schema and issubclass(cls.args_schema, BaseModel):
            pydantic_schema = cls.args_schema.model_json_schema()
            pydantic_schema.pop("title", None)

            # 递归展开嵌套定义（解决 $ref 不兼容）
            if "$defs" in pydantic_schema:
                defs = pydantic_schema.pop("$defs")

                def inline_refs(obj: Any) -> Any:
                    if isinstance(obj, dict):
                        if "$ref" in obj:
                            ref_name = obj["$ref"].split("/")[-1]
                            ref_obj = copy.deepcopy(defs[ref_name])
                            ref_obj.pop("title", None)
                            return inline_refs(ref_obj)
                        return {k: inline_refs(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [inline_refs(i) for i in obj]
                    return obj

                pydantic_schema = inline_refs(pydantic_schema)

            schema["function"]["parameters"] = pydantic_schema
        else:
            schema["function"]["parameters"] = {
                "type": "object",
                "properties": {}
            }

        return schema