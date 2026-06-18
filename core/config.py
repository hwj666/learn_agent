from dataclasses import dataclass, field
from typing import Dict, Set
import yaml


@dataclass
class ProviderConfig:
    """服务商配置（OpenAI / DeepSeek / 阿里云 等）"""

    base_url: str
    api_key: str
    provider: str


@dataclass
class ModelConfig:
    """模型参数配置"""

    model_provider: str
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # 运行时自动注入，不参与序列化/构造
    provider: ProviderConfig = field(init=False, default=None)


@dataclass
class AgentConfig:
    """Agent 智能体配置"""

    model: str
    max_steps: int
    tool_set: Set[str] = field(default_factory=set)

    # 运行时自动注入
    model_config: ModelConfig = field(init=False, default=None)

    def __post_init__(self):
        """YAML list → set 自动转换"""
        if self.tool_set is None:
            self.tool_set = set()
        elif isinstance(self.tool_set, list):
            self.tool_set = set(self.tool_set)


@dataclass
class AppConfig:
    """全局应用配置（自动织入依赖关系）"""

    agents: Dict[str, AgentConfig]
    model_providers: Dict[str, ProviderConfig]
    models: Dict[str, ModelConfig]

    def __post_init__(self):
        """配置加载完成后，自动绑定对象关系"""
        # 1. Model → Provider
        for model_key, model_cfg in self.models.items():
            provider_id = model_cfg.model_provider
            if provider_id not in self.model_providers:
                raise ValueError(f"模型 {model_key} 引用了不存在的服务商 {provider_id}")
            model_cfg.provider = self.model_providers[provider_id]

        # 2. Agent → Model
        for agent_name, agent_cfg in self.agents.items():
            if agent_cfg.model not in self.models:
                raise ValueError(
                    f"智能体 {agent_name} 引用了不存在的模型 {agent_cfg.model}"
                )
            agent_cfg.model_config = self.models[agent_cfg.model]

    @classmethod
    def from_yaml(cls, file_path: str) -> "AppConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            agents={
                name: AgentConfig(**cfg) for name, cfg in data.get("agents", {}).items()
            },
            model_providers={
                name: ProviderConfig(**cfg)
                for name, cfg in data.get("model_providers", {}).items()
            },
            models={
                name: ModelConfig(**cfg) for name, cfg in data.get("models", {}).items()
            },
        )

    def get_agent(self, agent_name: str) -> AgentConfig:
        if agent_name not in self.agents:
            raise KeyError(f"Agent '{agent_name}' 不存在")
        return self.agents[agent_name]
