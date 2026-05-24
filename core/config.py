from dataclasses import dataclass, field
from typing import List, Dict, Optional
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
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int

    # 运行时自动注入，不参与序列化/构造
    provider: ProviderConfig = field(init=False, default=None)


@dataclass
class AgentConfig:
    """Agent 智能体配置"""
    model: str                  # 绑定的模型 ID（如 gpt35_1）
    max_steps: int              # ReAct 最大思考步数
    tool_set: List[str] = field(default_factory=list)  # 允许使用的工具集


@dataclass
class AppConfig:
    """全局应用配置（自动织入依赖关系）"""
    agents: Dict[str, AgentConfig]
    model_providers: Dict[str, ProviderConfig]
    models: Dict[str, ModelConfig]

    def __post_init__(self):
        """配置加载完成后，自动绑定对象关系"""
        # 1. 把 Provider 注入到每个 Model 中
        for model_key, model_cfg in self.models.items():
            provider_id = model_cfg.model_provider
            if provider_id not in self.model_providers:
                raise ValueError(
                    f"模型 {model_key} 引用了不存在的服务商 {provider_id}"
                )
            model_cfg.provider = self.model_providers[provider_id]

        # 2. 校验 Agent 绑定的 Model 是否存在
        for agent_name, agent_cfg in self.agents.items():
            if agent_cfg.model not in self.models:
                raise ValueError(
                    f"智能体 {agent_name} 引用了不存在的模型 {agent_cfg.model}"
                )

    @classmethod
    def from_yaml(cls, file_path: str) -> "AppConfig":
        """从 YAML 文件加载配置"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            agents={
                name: AgentConfig(**cfg)
                for name, cfg in data.get("agents", {}).items()
            },
            model_providers={
                name: ProviderConfig(**cfg)
                for name, cfg in data.get("model_providers", {}).items()
            },
            models={
                name: ModelConfig(**cfg)
                for name, cfg in data.get("models", {}).items()
            },
        )

    def get_agent_model(self, agent_name: str) -> ModelConfig:
        """根据 Agent 名称获取完整 Model 配置（含 provider）"""
        if agent_name not in self.agents:
            raise KeyError(f"Agent '{agent_name}' 不存在")
        agent = self.agents[agent_name]
        return self.models[agent.model]