import logging
from typing import List, Dict, Set, Any, Tuple, Optional

from schema.message import LLMMessage, ToolCall
from tools.registry import ToolRegistry

logger = logging.getLogger("AgentGateway")


class ToolRouter:
    """
    【企业级标准：工具网关安全路由层】
    核心职责：
    1. 事前（LLM推理前）：根据安全上下文，动态分发允许大模型感知和使用的工具 JSON Schema。
    2. 事中（工具执行前）：执行业务鉴权防篡改、自动化依赖注入（DI）、敏感操作人工审批熔断，最终洗净任务交付发动机。
    """

    def __init__(self, executor: Any) -> None:
        """
        初始化工具网关
        :param executor: 共享发动机层（ToolExecutor 实例），负责底层的并发调度、超时控制和安全沙箱执行
        """
        self.executor = executor

    def get_schemas_for_user(self, ctx: "AgentContext") -> List[Dict[str, Any]]:
        """
        【事前控制】动态获取当前用户/Agent 准许使用的所有工具 Schema 传给 LLM
        :param ctx: 统一的智能体安全与资源上下文
        :return: 经过排序的、OpenAI 格式的标准 JSON Schema 列表
        """
        # 从全局注册中心根据当前上下文的权限集合过滤工具类
        tool_classes = ToolRegistry.get_tools_by_set(ctx.allowed_toolsets)
        # 排序并序列化为 Schema（排序能确保 LLM 接收到的 Prompt 具有确定性，提高缓存命中率）
        return [cls.to_schema() for _, cls in sorted(tool_classes.items())]

    async def dispatch(
        self, tool_calls: List["ToolCall"], ctx: "AgentContext", timeout: float = 30.0
    ) -> List["LLMMessage"]:
        """
        【事中控制】接收大模型的原始调用指令，处理鉴权、注入依赖后，驱动底层发动机
        :param tool_calls: 大模型单次吐出的原始工具调用指令列表（支持 Batch 批量并发）
        :param ctx: 统一的上下文容器，内含当前用户的 allowed_toolsets 权限及 DB/Redis 依赖资源
        :param timeout: 单次工具执行的整体最大超时限制（秒）
        :return: 包含网关拦截回执与真实执行回执的统一消息列表
        """
        if not tool_calls:
            return []

        resolved_calls: List[Tuple[ToolCall, Any]] = []
        rejected_messages: List["LLMMessage"] = []

        for tool_call in tool_calls:
            try:
                # 从注册中心检索工具类的元数据
                tool_cls = ToolRegistry.get_tool(tool_call.name)

                # 1. 动态权限拦截（深度防御：防止大模型幻觉、历史上下文污染、Prompt 注入攻击引发的越权）
                if not tool_cls or tool_cls.toolset not in ctx.allowed_toolsets:
                    logger.warning(
                        f"[Gateway] 权限拦截：用户 `{ctx.user_id}` 企图触发未授权工具 `{tool_call.name}`"
                    )
                    rejected_messages.append(
                        LLMMessage.tool(
                            tool_call.id,
                            content=f"❌ 权限拒绝：当前策略未获准使用工具 `{tool_call.name}`。",
                        )
                    )
                    continue

                # 2. 核心拦截：支持 Human-in-the-Loop (人工审批熔断机制)
                if getattr(tool_cls, "require_approval", False):
                    logger.info(
                        f"[Gateway] 触发人工审批：工具 `{tool_call.name}` 属于敏感操作，进入挂起状态。"
                    )
                    # 抛出结构化控制流异常，由上层 Agent 状态机捕获并持久化当前会话，等待人工 Callback 信号
                    raise RequireApprovalException(
                        message=f"操作确认：执行工具 `{tool_call.name}` 需要人工审批。",
                        tool_call=tool_call,
                        context=ctx,
                    )

                # 3. 实例化工具（确保每个指令拥有完全独立的无状态实例，避免多协程并发状态污染）
                tool_instance = tool_cls()

                # 4. 万能依赖注入（Dependency Injection）
                # 自动匹配工具声明的内部变量，并从上下文中动态挂载对应的基础设施（如数据库连接池、Redis客户端等）
                for dep_name, dep_value in ctx.dependencies.items():
                    if hasattr(tool_instance, dep_name):
                        setattr(tool_instance, dep_name, dep_value)

                # 5. 运行时上下文注入
                # 允许自定义工具在执行内部逻辑时，能够无缝感知当前操作的用户ID、租户ID或多轮对话变量
                if hasattr(tool_instance, "runtime_ctx"):
                    tool_instance.runtime_ctx = ctx

                # 收集洗净、合规且准备就绪的任务
                resolved_calls.append((tool_call, tool_instance))

            except RequireApprovalException:
                # 审批拦截异常直接向上抛出，中断当前的 dispatch 流，交由业务层进行状态机挂起
                raise
            except Exception as e:
                # 全链路异常保护：捕获工具在网关初始化或注入阶段的未知异常（如工具定义有 Bug），保障网关服务永不崩溃
                logger.exception(
                    f"[Gateway] 工具 `{tool_call.name}` 在网关洗净阶段发生未知错误"
                )
                rejected_messages.append(
                    LLMMessage.tool(
                        tool_call.id,
                        content=f"❌ 错误：工具初始化失败，技术细节: {str(e)}",
                    )
                )

        # 6. 把洗干净、解耦好、注入完合规依赖的任务，一次性交给共享发动机异步并行执行
        executed_messages = []
        if resolved_calls:
            executed_messages = await self.executor.execute(
                resolved_calls, timeout=timeout
            )

        # 7. 线性合并：网关层直接拦截的业务错误回执 + 发动机层真实执行后的物理回执
        return rejected_messages + executed_messages
