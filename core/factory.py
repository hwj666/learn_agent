"""
策略工厂
根据配置创建不同的 Agent 策略
"""
import logging
from typing import Dict, Type, Optional

from core.policy import ExecutionPolicy, ReactPolicy
from core.config import AgentConfig


logger = logging.getLogger("PolicyFactory")


class PolicyFactory:
    """
    策略工厂

    注册并创建不同的执行策略
    """

    _policies: Dict[str, Type[ExecutionPolicy]] = {
        "react": ReactPolicy,
    }

    @classmethod
    def register(cls, name: str, policy_class: Type[ExecutionPolicy]) -> None:
        """注册策略"""
        cls._policies[name] = policy_class
        logger.info(f"注册策略: {name} -> {policy_class.__name__}")

    @classmethod
    def create(
        cls,
        name: str,
        config: AgentConfig,
        **kwargs
    ) -> ExecutionPolicy:
        """创建策略实例"""
        if name not in cls._policies:
            available = ", ".join(cls._policies.keys())
            raise ValueError(f"未知策略: {name}，可用策略: {available}")

        policy_class = cls._policies[name]
        return policy_class(**kwargs)

    @classmethod
    def get_policy_names(cls) -> list:
        """获取所有已注册策略名称"""
        return list(cls._policies.keys())


# 预注册内置策略
PolicyFactory.register("react", ReactPolicy)

# 注册 PlanPolicy（延迟导入避免循环依赖）
try:
    from agents.plan_policy import PlanPolicy
    PolicyFactory.register("plan", PlanPolicy)
except ImportError:
    pass
