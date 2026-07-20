import json
import logging
import re
from typing import Dict, List, Optional, Any, Set, Union
import uuid

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _clean_json_string(json_str: str) -> str:
    """清洗 JSON 字符串，移除 Markdown 标记和常见 LLM 错误"""
    cleaned = json_str.strip()

    # 移除 Markdown 代码块
    if cleaned.startswith("```"):
        cleaned = cleaned[7:] if cleaned.startswith("```json") else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # 移除 trailing commas（LLM 常见问题）
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", cleaned)

    return cleaned


def parse_llm_json_arguments(
    arguments: Optional[Union[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """高性能防 ReDoS 的大模型参数解析器（独立纯函数）"""
    if not arguments:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        raise ValueError("Arguments must be a string or dictionary.")

    cleaned = _clean_json_string(arguments)

    # 尝试标准解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 兜底策略：通过花括号边界截取
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            candidate = cleaned[start : end + 1]
            candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("Failed to extract valid JSON payload from arguments.")


def _extract_json_candidates(text: str) -> List[str]:
    """提取所有可能的 JSON 字符串候选，按长度降序排列"""
    candidates: List[str] = []

    # 1. 尝试 Markdown JSON 代码块
    for match in _JSON_BLOCK_RE.finditer(text):
        candidates.append(match.group(1).strip())

    # 2. 使用栈匹配独立的 JSON 对象（支持嵌套）
    depth = 0
    start = -1
    for i, char in enumerate(text):
        if char in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    candidates.append(candidate)
                except json.JSONDecodeError:
                    pass

    # 按长度降序排列，优先尝试最长的（通常最完整）
    return sorted(set(candidates), key=len, reverse=True)


def _infer_tool_name(
    call_data: Dict[str, Any], tools: List[Dict[str, Any]], param_keys: Set[str]
) -> Optional[str]:
    """推断工具名称，基于参数匹配度评分"""
    if not tools:
        return None

    best_score = -1
    best_name = None

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        func_def = tool.get("function", {}) if "function" in tool else tool
        if not isinstance(func_def, dict):
            continue

        tool_name = func_def.get("name")
        if not tool_name:
            continue

        tool_params = func_def.get("parameters", {})
        if not isinstance(tool_params, dict):
            continue

        properties = set(tool_params.get("properties", {}).keys())
        required = set(tool_params.get("required", []))

        # 计算匹配度
        matching_required = len(required & param_keys)
        matching_total = len(properties & param_keys)

        # 必须满足所有必需参数
        if required and not required.issubset(param_keys):
            continue

        # 评分：匹配的参数越多越好，但避免空参数工具（除非是唯一选择）
        score = matching_total * 10 + matching_required * 5
        if properties and matching_total > 0:
            score += 100  # 有参数匹配的优先

        if score > best_score:
            best_score = score
            best_name = tool_name

    return best_name or "unknown_tool"


def extract_implicit_tool_calls(
    content: str, tools: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """从非结构化纯文本中提取 ToolCall 数组"""
    extracted_calls: List[Dict[str, Any]] = []

    if not content or not content.strip():
        return extracted_calls

    # 提取所有 JSON 候选
    json_candidates = _extract_json_candidates(content)
    if not json_candidates:
        return extracted_calls

    # 尝试解析每个候选
    for json_str in json_candidates:
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        # 统一转为列表格式
        items = (
            [parsed]
            if isinstance(parsed, dict)
            else (parsed if isinstance(parsed, list) else [])
        )

        for item in items:
            if not isinstance(item, dict):
                continue

            # 标准化调用格式
            normalized_call = _normalize_tool_call(item, tools)
            if normalized_call:
                extracted_calls.append(normalized_call)

        # 如果找到了有效的调用，可以提前退出（优先使用最长的 JSON）
        if extracted_calls:
            break

    return extracted_calls


def _normalize_tool_call(
    call_data: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """规范化单个工具调用"""
    if not isinstance(call_data, dict):
        return None

    param_keys = set(call_data.keys())

    # 情况 A: 标准格式 {"name": "...", "arguments": ...}
    if "name" in call_data and "arguments" in call_data:
        tool_name = str(call_data["name"])
        tool_args = call_data["arguments"]
    else:
        # 情况 B: 非标准格式，需要推断工具名
        tool_name = _infer_tool_name(call_data, tools or [], param_keys)
        tool_args = call_data

    # 序列化参数
    if isinstance(tool_args, str):
        try:
            json.loads(tool_args)  # 验证是否为有效 JSON
        except json.JSONDecodeError:
            tool_args = json.dumps({"input": tool_args}, ensure_ascii=False)
    else:
        tool_args = json.dumps(tool_args, ensure_ascii=False)

    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "name": tool_name,
        "arguments": tool_args,
    }


__all__ = ["extract_implicit_tool_calls", "parse_llm_json_arguments"]
