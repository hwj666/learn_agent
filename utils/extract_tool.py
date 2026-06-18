import json
import logging
import re
from typing import Dict, List, Optional, Any, Union
import uuid

logger = logging.getLogger(__name__)


def extract_implicit_tool_calls(
    content: str, tools: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]:
    """从非结构化纯文本中提取 ToolCall 数组。

    Args:
        content: 大模型返回的纯文本内容（可能包含 markdown json 块或孤立的 JSON 结构）
        tools: 当前轮次可用的工具定义列表，用于在模型漏写工具名时进行推断

    Returns:
        标准化的 ToolCall 字典列表
    """
    extracted_calls: List[Dict[str, Any]] = []

    if not content or not content.strip():
        return extracted_calls

    # 1. 尝试提取 JSON 数据
    json_data = _extract_json_data(content)
    if not json_data:
        return extracted_calls

    # 2. 尝试解析 JSON
    parsed_calls = _parse_json_data(json_data)
    if not parsed_calls:
        return extracted_calls

    # 3. 规范化每个调用
    for call_data in parsed_calls:
        normalized_call = _normalize_tool_call(call_data, tools)
        if normalized_call:
            extracted_calls.append(normalized_call)

    return extracted_calls


def _extract_json_data(content: str) -> Optional[str]:
    """从文本中提取 JSON 字符串。"""
    # 先尝试匹配 Markdown JSON 代码块
    json_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE
    )
    if json_match:
        return json_match.group(1).strip()

    # 再尝试匹配独立的 JSON 对象（包括嵌套的）
    # 使用栈来匹配配对的 {} 或 []
    depth = 0
    start = -1
    for i, char in enumerate(content):
        if char in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0 and start != -1:
                # 检查提取的字符串是否是有效的 JSON
                json_str = content[start : i + 1]
                try:
                    json.loads(json_str)  # 验证是否是有效 JSON
                    return json_str
                except json.JSONDecodeError:
                    continue

    return None


def _parse_json_data(json_str: str) -> List[Dict]:
    """解析 JSON 字符串为调用列表。"""
    try:
        parsed = json.loads(json_str)

        # 统一转为列表格式
        if isinstance(parsed, dict):
            return [parsed]
        elif isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        else:
            logger.warning(f"[Tool提取器] 解析的 JSON 不是对象或数组: {type(parsed)}")
            return []

    except json.JSONDecodeError as e:
        logger.warning(f"[Tool提取器] JSON 解析失败: {e}")
        return []


def _normalize_tool_call(
    call_data: Dict, tools: Optional[List[Dict]] = None
) -> Optional[Dict[str, Any]]:
    """规范化单个工具调用。"""
    if not isinstance(call_data, dict):
        return None

    # 情况 A: 标准格式 {"name": "...", "arguments": ...}
    if "name" in call_data and "arguments" in call_data:
        tool_name = str(call_data["name"])

        # 处理 arguments
        if isinstance(call_data["arguments"], str):
            # 已经是字符串，验证是否为有效 JSON
            try:
                json.loads(call_data["arguments"])
                tool_args = call_data["arguments"]
            except json.JSONDecodeError:
                # 如果不是有效 JSON，则视为纯字符串参数
                tool_args = json.dumps({"input": call_data["arguments"]})
        else:
            # 非字符串参数，序列化为 JSON
            tool_args = json.dumps(call_data["arguments"], ensure_ascii=False)

    # 情况 B: 非标准格式，需要推断工具名
    else:
        # 尝试从 tools 中推断工具名
        tool_name = _infer_tool_name(call_data, tools)

        # 将整个字典作为参数
        tool_args = json.dumps(call_data, ensure_ascii=False)

    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "name": tool_name,
        "arguments": tool_args,
    }


def _infer_tool_name(call_data: Dict, tools: Optional[List[Dict]] = None) -> str:
    """推断最可能匹配的工具名。"""
    if not tools or not isinstance(tools, list):
        return "unknown_tool"

    # 尝试根据参数结构匹配工具
    call_param_keys = set(call_data.keys())

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        # 支持不同的工具定义格式
        func_def = tool.get("function", {}) if "function" in tool else tool

        if not isinstance(func_def, dict):
            continue

        tool_name = func_def.get("name")
        tool_params = func_def.get("parameters", {})
        required_params = set(tool_params.get("required", []))

        # 检查必需参数是否都在调用中存在
        if required_params.issubset(call_param_keys):
            return tool_name or "unknown_tool"

    # 如果没有匹配的，返回第一个工具的名称
    for tool in tools:
        if isinstance(tool, dict):
            func_def = tool.get("function", {}) if "function" in tool else tool
            if isinstance(func_def, dict) and "name" in func_def:
                return func_def["name"]

    return "unknown_tool"


# 保持对原有函数的兼容性
__all__ = ["extract_implicit_tool_calls"]
