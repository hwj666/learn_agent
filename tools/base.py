import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar, Generic, get_args, Set
from pydantic import BaseModel

ArgsType = TypeVar("T", bound=BaseModel)


class BaseTool(Generic[ArgsType], ABC):
    """
    工业级工具基类（无状态、强类型、LLM 友好）。

    特性：
    1. 支持定义抽象中间层（如 BaseSQLTool），仅最终叶子类需具象化。
    2. 运行时禁止修改实例属性（无状态防御）。
    3. 自动解析泛型参数，支持复杂的多层继承链。
    4. 输出极致净化的 JSON Schema（去 $ref、防自引用、保留核心格式约束）。

    注意：
    - `name` 和 `toolset` 由 `@ToolRegistry.register` 装饰器注入，无需在类中定义。
    """

    name: str
    toolset: str
    description: str = ""
    args_schema: Type[ArgsType]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # 1. 判定是否为抽象类（仅检查抽象方法，不关心 name/toolset）
        is_abstract = bool(getattr(cls, "__abstractmethods__", False))

        # 2. 如果子类已显式定义 args_schema，跳过自动推导
        if "args_schema" in cls.__dict__:
            return

        # 3. 递归向父类链路追溯泛型契约
        def _find_args_schema(current_cls):
            if hasattr(current_cls, "__orig_bases__"):
                for base in current_cls.__orig_bases__:
                    type_args = get_args(base)
                    if (
                        type_args
                        and isinstance(type_args[0], type)
                        and issubclass(type_args[0], BaseModel)
                    ):
                        return type_args[0]
            for base_cls in current_cls.__bases__:
                if base_cls is not object:
                    res = _find_args_schema(base_cls)
                    if res:
                        return res
            return None

        found_schema = _find_args_schema(cls)
        if found_schema:
            cls.args_schema = found_schema
        elif not is_abstract:
            # 仅在非抽象类且无法推断泛型时抛出错误
            raise TypeError(
                f"工具类 {cls.__name__} 无法推断泛型参数 T (ArgsType)。\n"
                f"请显式定义 args_schema 或检查继承链是否包含 Generic[ArgsType]。"
            )

    def __init__(self):
        # 标记为已初始化，此后禁止 __setattr__ 修改业务属性
        self._initialized = True

    def __setattr__(self, name: str, value: Any):
        if not hasattr(self, "_initialized"):
            super().__setattr__(name, value)
            return

        # 允许框架层修改内部属性（如 _initialized 或其他框架私有字段）
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        raise AttributeError(
            f"工具是【无状态只读】的！禁止运行时修改 self.{name}\n"
            f"请使用局部变量或 ctx 上下文传递状态。"
        )

    @abstractmethod
    def execute(self, ctx: Dict[str, Any], args: ArgsType) -> Any:
        """
        执行工具逻辑。

        :param ctx: 上下文（如 trace_id, user_id, logger 等）
        :param args: 由 args_schema 校验后的强类型参数
        :return: 任意可被 JSON 序列化的结果
        """
        raise NotImplementedError

    @classmethod
    def to_schema(cls) -> Dict[str, Any]:
        """
        生成 LLM Function Calling 所需的 Schema。
        经过深度清洗，最小化 Token 消耗，保留精准对齐指导。
        """
        # 2. 强制校验：确保工具已被 Registry 正确注册（name 和 toolset 已注入）
        if not hasattr(cls, "name") or not cls.name:
            raise ValueError(
                f"工具 {cls.__name__} 未设置 'name'。\n"
                f"请使用 @ToolRegistry.register(name=..., toolset=...) 装饰器进行注册。"
            )
        if not hasattr(cls, "toolset") or not cls.toolset:
            raise ValueError(
                f"工具 {cls.__name__} 未设置 'toolset'。\n"
                f"请使用 @ToolRegistry.register(name=..., toolset=...) 装饰器进行注册。"
            )
        if not cls.description:
            raise ValueError(
                f"工具 {cls.__name__} 的 description 不能为空，这对 LLM 理解工具至关重要。"
            )

        schema = {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }

        if getattr(cls, "args_schema", None) and issubclass(cls.args_schema, BaseModel):
            pydantic_schema = cls.args_schema.model_json_schema()
            defs = pydantic_schema.pop("$defs", {})

            # 内存感知的内联展开（防止循环引用）
            def inline_refs(obj: Any, seen_refs: Set[str]) -> Any:
                if isinstance(obj, dict):
                    # 处理 anyOf, oneOf, allOf (LLM 兼容性关键)
                    for key in ("anyOf", "oneOf", "allOf"):
                        if key in obj:
                            obj[key] = [
                                inline_refs(item, seen_refs) for item in obj[key]
                            ]

                    # 处理 $ref 引用
                    if "$ref" in obj:
                        ref_name = obj["$ref"].split("/")[-1]
                        # 循环引用截断（返回空对象，避免 LLM 困惑）
                        if ref_name in seen_refs:
                            return {}
                        if ref_name in defs:
                            ref_obj = copy.deepcopy(defs[ref_name])
                            return inline_refs(ref_obj, seen_refs | {ref_name})
                        return {}

                    # 🚀 精准 Token 优化：消灭大模型不关注的描述性元数据
                    obj.pop("title", None)
                    obj.pop("examples", None)
                    obj.pop("$schema", None)
                    # additionalProperties 常引发 LLM 幻觉，予以删除
                    obj.pop("additionalProperties", None)

                    # 💡 保留 default, pattern, format, description
                    # 这些是约束 LLM 行为、降低幻觉率的关键信号
                    return {k: inline_refs(v, seen_refs) for k, v in obj.items()}

                elif isinstance(obj, list):
                    return [inline_refs(i, seen_refs) for i in obj]
                return obj

            cleaned_schema = inline_refs(pydantic_schema, set())
            schema["function"]["parameters"] = cleaned_schema

        return schema
