import inspect
from typing import Dict, Any, List, Callable

from pydantic import create_model
from pydantic.fields import FieldInfo
from pydantic.type_adapter import TypeAdapter

class ToolManager:
    def __init__(self):
        self.tools_map: Dict[str, Callable] = {}
        self.schemas: List[Dict[str, Any]] = []

    def register_func(self, func: Callable):
        func_name = func.__name__
        docstring = inspect.getdoc(func) or f"工具 {func_name}"
        sig = inspect.signature(func)
        type_hints = inspect.get_annotations(func)

        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            annotation = type_hints.get(param_name, Any)
            default = param.default

            if isinstance(default, FieldInfo):
                fields[param_name] = (annotation, default)
            elif default is inspect.Parameter.empty:
                fields[param_name] = (annotation, ...)
            else:
                fields[param_name] = (annotation, default)

        try:
            ParamModel = create_model(f"{func_name}_Params", **fields)
            schema = TypeAdapter(ParamModel).json_schema()
        except Exception as e:
            raise RuntimeError(f"工具 {func_name} 生成失败: {e}") from e

        properties = {k: v.copy() for k, v in schema.get("properties", {}).items()}
        for p in properties.values():
            p.pop("title", None)

        tool_def = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": docstring.strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": schema.get("required", []),
                    "additionalProperties": False
                }
            }
        }

        self.tools_map[func_name] = func
        self.schemas.append(tool_def)

    def get_schemas(self):
        return self.schemas.copy()

    def get_tool(self, name: str) -> Callable:
        if name not in self.tools_map:
            raise KeyError(f"工具不存在: {name}")
        return self.tools_map[name]

    def execute(self, name: str, arguments: Dict):
        return self.get_tool(name)(**arguments)