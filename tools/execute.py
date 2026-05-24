import json
from typing import Any, Dict, List, Set
from pydantic import ValidationError
from tools.registry import ToolRegistry
from core.message import LLMMessage, LLMMessageBuilder, ToolCall

class ToolExecutor:
    """
    运行时工具箱：无状态设计，现场实例化，绝对线程/协程安全
    每个 Agent 持有自己的 ToolExecutor 实例，实现权限隔离
    """

    def __init__(self, allowed_toolsets: Set[str] | None = None) -> None:
        """锁定当前 Agent 的权限范围，默认空集合"""
        self.allowed_toolsets = allowed_toolsets or set()

    def get_schemas(self) -> List[Dict[str, Any]]:
        """获取当前 Agent 有权使用的所有工具的 LLM Schema 声明"""
        tool_classes = ToolRegistry.get_tools_by_set(self.allowed_toolsets)
        return [cls.to_schema() for cls in tool_classes.values()]

    async def execute(self, tool_call: ToolCall, ctx: Dict[str, Any]) -> LLMMessage:
        """执行单个工具（无状态函数式调用）"""

        # ------------------------------
        # 1. 路由与权限强校验
        # ------------------------------
        tool_cls = ToolRegistry.get_tool(tool_call.name)
        if not tool_cls or tool_cls.toolset not in self.allowed_toolsets:
            return LLMMessageBuilder.tool(
                tool_call.id,
                f"❌ 权限拒绝：未在当前 Agent 授权集中找到工具 `{tool_call.name}`"
            )

        # ------------------------------
        # 2. 安全拦截：必须携带合法 agent_id
        # ------------------------------
        if not ctx or "agent_id" not in ctx:
            return LLMMessageBuilder.tool(
                tool_call.id,
                "❌ 安全拒绝：未识别有效的 Agent 运行时上下文（agent_id 缺失）"
            )

        try:
            # ------------------------------
            # 3. 无状态实例化（协程安全核心）
            # ------------------------------
            tool_instance = tool_cls()

            # ------------------------------
            # 4. 超强鲁棒性：清洗 LLM 脏 JSON 数据
            # ------------------------------
            cleaned_args = tool_call.arguments.strip() if tool_call.arguments else "{}"

            # 清洗 ```json ... ``` 格式
            if cleaned_args.startswith("```"):
                if cleaned_args.startswith("```json"):
                    cleaned_args = cleaned_args[7:]  # 去掉 ```json
                else:
                    cleaned_args = cleaned_args[3:]  # 去掉 ```
                # 从右侧截断 ```
                if cleaned_args.endswith("```"):
                    cleaned_args = cleaned_args[:-3]
                cleaned_args = cleaned_args.strip()

            # 解析 JSON
            arguments_dict = json.loads(cleaned_args) if cleaned_args else {}

            # ------------------------------
            # 5. Pydantic 强校验
            # ------------------------------
            validated_args = tool_instance.args_schema.model_validate(arguments_dict)

            # ------------------------------
            # 6. 执行工具（异步）
            # ------------------------------
            result = await tool_instance.execute(ctx, validated_args)

            # 返回工具结果（自动转字符串，LLM 友好）
            return LLMMessageBuilder.tool(
                tool_call.id,
                str(result)
            )

        # ------------------------------
        # 异常捕获体系
        # ------------------------------
        except json.JSONDecodeError:
            return LLMMessageBuilder.tool(
                tool_call.id,
                "❌ 参数解析失败：输入不是合法的 JSON 格式"
            )

        except ValidationError as e:
            # 生成 LLM 能看懂的错误信息
            error_msgs = []
            for err in e.errors():
                field_path = ".".join(map(str, err["loc"]))
                error_msgs.append(f"字段 `{field_path}`: {err['msg']}")
            return LLMMessageBuilder.tool(
                tool_call.id,
                f"❌ 参数校验失败: {'; '.join(error_msgs)}"
            )

        except Exception as e:
            # 兜底保护，防止单个工具崩溃整个 Agent
            return LLMMessageBuilder.tool(
                tool_call.id,
                f"❌ 工具执行异常: {str(e)}"
            )