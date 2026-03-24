from typing import Dict, Any, List, Callable

_TOOL_REGISTRY: List[Callable] = []

def tool(func: Callable) -> Callable:
    _TOOL_REGISTRY.append(func)
    return func

def get_all_tools() -> List[Callable]:
    return _TOOL_REGISTRY.copy()

def list_tool_names() -> list:
    return [f.__name__ for f in _TOOL_REGISTRY]