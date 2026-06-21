"""
评估器
用于评估 Agent 执行质量、自我反思、结果验证
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from core.message import LLMMessage
from tools.execute import ToolExecutor


class Evaluator(ABC):
    """
    评估器接口
    
    用于：
    - 评估工具调用结果是否正确
    - 判断任务是否真正完成
    - 生成改进建议
    - 自我反思
    """

    def __init__(self, executor: ToolExecutor, client=None):
        self.executor = executor
        self.client = client
        self.logger = logging.getLogger("Evaluator")

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        history: List[LLMMessage],
        result: str,
    ) -> Dict[str, Any]:
        """
        评估执行结果
        
        Returns:
            dict with keys:
                - success: bool - 是否成功
                - confidence: float - 置信度 0-1
                - feedback: str - 反馈意见
                - suggestions: list - 改进建议
        """
        pass

    @abstractmethod
    async def reflect(
        self,
        query: str,
        history: List[LLMMessage],
        result: str,
    ) -> str:
        """
        自我反思
        
        Returns:
            str - 反思内容
        """
        pass


class SimpleEvaluator(Evaluator):
    """
    简单评估器
    
    基于关键词匹配和规则进行评估
    """

    FAIL_KEYWORDS = ("error", "fail", "失败", "错误", "exception", "无法")
    SUCCESS_KEYWORDS = ("成功", "完成", "已完成", "done", "success")

    async def evaluate(
        self,
        query: str,
        history: List[LLMMessage],
        result: str,
    ) -> Dict[str, Any]:
        result_lower = result.lower()

        has_fail = any(k in result_lower for k in self.FAIL_KEYWORDS)
        has_success = any(k in result_lower for k in self.SUCCESS_KEYWORDS)

        if has_fail:
            return {
                "success": False,
                "confidence": 0.8,
                "feedback": "执行过程中遇到错误",
                "suggestions": ["检查工具调用参数", "重试失败的操作"],
            }

        if has_success:
            return {
                "success": True,
                "confidence": 0.7,
                "feedback": "任务已完成",
                "suggestions": [],
            }

        return {
            "success": True,
            "confidence": 0.5,
            "feedback": "结果不确定",
            "suggestions": ["验证结果正确性"],
        }

    async def reflect(
        self,
        query: str,
        history: List[LLMMessage],
        result: str,
    ) -> str:
        return f"执行完成。查询: {query[:50]}... 结果: {result[:100]}..."


class LlmEvaluator(Evaluator):
    """
    LLM 评估器
    
    使用大模型进行智能评估和反思
    """

    EVAL_PROMPT = """你是一个评估专家，请评估以下智能体的执行结果：

查询：{query}
结果：{result}

请评估：
1. 是否成功完成任务？
2. 置信度（0-1）
3. 反馈意见
4. 改进建议（如果有）

请以 JSON 格式输出：
{{
    "success": bool,
    "confidence": float,
    "feedback": "str",
    "suggestions": ["str"]
}}
"""

    REFLECT_PROMPT = """你是一个智能体反思助手，请分析以下执行过程：

查询：{query}
历史记录：
{history}
结果：{result}

请反思：
1. 执行过程中有哪些决策是合理的？
2. 哪些地方可以改进？
3. 下次遇到类似问题会怎么做？
"""

    async def evaluate(
        self,
        query: str,
        history: List[LLMMessage],
        result: str,
    ) -> Dict[str, Any]:
        messages = [
            LLMMessage.system("你是一个评估专家"),
            LLMMessage.user(self.EVAL_PROMPT.format(query=query, result=result)),
        ]

        resp = await self.client.chat(messages=messages)

        try:
            return json.loads(resp.content or "{}")
        except Exception:
            return {
                "success": True,
                "confidence": 0.5,
                "feedback": resp.content or "评估结果解析失败",
                "suggestions": [],
            }

    async def reflect(
        self,
        query: str,
        history: List[LLMMessage],
        result: str,
    ) -> str:
        history_str = "\n".join(f"{msg.role}: {msg.content[:100]}" for msg in history[-10:])
        messages = [
            LLMMessage.system("你是一个智能体反思助手"),
            LLMMessage.user(self.REFLECT_PROMPT.format(
                query=query,
                history=history_str,
                result=result,
            )),
        ]

        resp = await self.client.chat(messages=messages)
        return resp.content or ""
