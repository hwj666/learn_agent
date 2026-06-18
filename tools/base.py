from abc import ABC, abstractmethod
import copy
from typing import Any, Dict, Type, TypeVar, Generic, get_args
from pydantic import BaseModel

class BaseToolState(BaseModel):
    pass

class EmptyState(BaseToolState):
    pass

ArgsType = TypeVar("T", bound=BaseModel)

class BaseTool(Generic[ArgsType], ABC):
    """
    LLM 函数调用工具 抽象基类
    配合参数化强约束装饰器使用，去除冗余的 self-name 补全
    """
    # 这些属性由 @ToolRegistry.register(name=..., toolset=...) 在类创建后强行注入
    name: str
    toolset: str
    
    # 允许子类覆盖或保持默认
    description: str = ""
    args_schema: Type[ArgsType]
    state_schema: Type[BaseToolState] = EmptyState

    def __init_subclass__(cls, **kwargs):
        """专门负责做泛型提取，不越权处理 name 和 toolset"""
        super().__init_subclass__(**kwargs)

        if "args_schema" in cls.__dict__:
            return

        # 核心：自动提取泛型 BaseTool[XXXArgs] 的参数模型
        for base in cls.__orig_bases__:
            type_args = get_args(base)
            if type_args and issubclass(type_args[0], BaseModel):
                cls.args_schema = type_args[0]
                break

    def __init__(self):
        self._initialized = True

    def __setattr__(self, name, value):
        """🔒 无状态保护：防运行时污染"""
        if not hasattr(self, "_initialized"):
            super().__setattr__(name, value)
            return
        raise AttributeError(
            f"工具是【无状态只读】的！禁止运行时修改 self.{name}\n"
            "请使用局部变量或 ctx 上下文传递状态。"
        )

    @abstractmethod
    async def execute(self, ctx: Dict[str, Any], args: ArgsType) -> Any:
        pass

    @classmethod
    def to_schema(cls) -> Dict[str, Any]:
        """🔥 生成 OpenAI 100% 兼容的 Schema，支持嵌套模型展开"""
        # 注意：此处校验确保装饰器一定执行过并注入了有效的 name 与类自带的 description
        if not getattr(cls, "name", None) or not cls.description:
            raise ValueError(f"工具 {cls.__name__} 必须配置有效的 name 且 description 不能为空")

        schema = {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
            }
        }

        if cls.args_schema and issubclass(cls.args_schema, BaseModel):
            pydantic_schema = cls.args_schema.model_json_schema()
            pydantic_schema.pop("title", None)

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