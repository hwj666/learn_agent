"""
批评工具
用于自我反思、结果验证、改进建议等
"""
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from tools.base import BaseTool
from tools.registry import ToolRegistry


class ReflectArgs(BaseModel):
    query: str = Field(
        description="原始查询"
    )
    history: str = Field(
        description="执行历史记录"
    )
    result: str = Field(
        description="当前结果"
    )


@ToolRegistry.register(name="reflect", toolset="critic")
class ReflectTool(BaseTool[ReflectArgs]):
    description = """
        自我反思工具。用于分析执行过程，找出问题和改进点。
        
        参数：
        - query: 原始查询
        - history: 执行历史
        - result: 当前结果
    """

    async def execute(self, ctx: Dict[str, Any], args: ReflectArgs) -> str:
        return f"🔍 反思结果：\n\n查询：{args.query}\n\n结果：{args.result}\n\n建议：请根据执行历史分析是否需要改进。"


class ValidateResultArgs(BaseModel):
    expected: str = Field(
        description="期望结果描述"
    )
    actual: str = Field(
        description="实际结果"
    )


@ToolRegistry.register(name="validate_result", toolset="critic")
class ValidateResultTool(BaseTool[ValidateResultArgs]):
    description = """
        结果验证工具。用于验证实际结果是否符合期望。
        
        参数：
        - expected: 期望结果描述
        - actual: 实际结果
    """

    async def execute(self, ctx: Dict[str, Any], args: ValidateResultArgs) -> str:
        if args.expected.lower() in args.actual.lower() or args.actual.lower() in args.expected.lower():
            return f"✅ 验证通过\n\n期望：{args.expected}\n实际：{args.actual}"
        return f"⚠️ 验证失败\n\n期望：{args.expected}\n实际：{args.actual}"


class AskForFeedbackArgs(BaseModel):
    question: str = Field(
        description="询问用户的问题"
    )


@ToolRegistry.register(name="ask_for_feedback", toolset="critic")
class AskForFeedbackTool(BaseTool[AskForFeedbackArgs]):
    description = """
        请求用户反馈工具。当需要用户确认或提供更多信息时使用。
        
        参数：
        - question: 询问用户的问题
    """

    async def execute(self, ctx: Dict[str, Any], args: AskForFeedbackArgs) -> str:
        return f"📢 需要用户反馈：\n\n{args.question}"


class SuggestImprovementArgs(BaseModel):
    problem: str = Field(
        description="问题描述"
    )
    current_approach: str = Field(
        description="当前方法"
    )


@ToolRegistry.register(name="suggest_improvement", toolset="critic")
class SuggestImprovementTool(BaseTool[SuggestImprovementArgs]):
    description = """
        改进建议工具。用于分析问题并提供改进方案。
        
        参数：
        - problem: 问题描述
        - current_approach: 当前方法
    """

    async def execute(self, ctx: Dict[str, Any], args: SuggestImprovementArgs) -> str:
        return f"💡 改进建议：\n\n问题：{args.problem}\n\n当前方法：{args.current_approach}\n\n建议：请分析当前方法的不足并提出改进方案。"
