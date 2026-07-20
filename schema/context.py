from __future__ import annotations
from typing import Any, Dict, Optional, Set
from types import MappingProxyType
from pydantic import BaseModel, Field, PrivateAttr


class AgentContext(BaseModel):
    """🎒 终极务实派：全链路显式穿透上下文（写时复制机制，天然支持完美并发）"""

    session_id: str
    tenant_id: str
    user_id: str
    trace_id: str

    # 全局共享的长寿命依赖（如数据库、LLM 客户端），保持指针引用
    dependencies: Dict[str, Any] = Field(default_factory=dict)
    allowed_toolsets: Set[str] = Field(default_factory=set)

    # 业务数据和变量：强制设为受保护属性，严禁研发直接改写
    _payload: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _vars: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, **data: Any):
        """覆盖初始化函数：确保传入的 payload 和 vars 能正确落入私有属性池中"""
        initial_payload = data.pop("payload", {})
        initial_vars = data.pop("vars", {})
        super().__init__(**data)
        self._payload = initial_payload
        self._vars = initial_vars

    class Config:
        arbitrary_types_allowed = True
        exclude = {"_payload", "_vars"}

    @property
    def payload(self) -> MappingProxyType:
        """👁️ 只读暴露：研发在任何地方都能安全读取数据，防止上下文被意外篡改"""
        return MappingProxyType(self._payload)

    @property
    def vars(self) -> MappingProxyType:
        """👁️ 只读暴露：运行时动态变量池"""
        return MappingProxyType(self._vars)

    def update_payload(self, **kwargs: Any) -> None:
        """✍️ 唯一的 payload 更新入口"""
        self._payload.update(kwargs)

    def update_vars(self, **kwargs: Any) -> None:
        """✍️ 唯一的动态变量更新入口"""
        self._vars.update(kwargs)

    def fork(self, **overrides: Any) -> "AgentContext":
        """🌿 创建子上下文，用于 Planner → SubTask / Turn → Turn 的写时复制"""
        forked = AgentContext(
            session_id=self.session_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            trace_id=overrides.get("trace_id", self.trace_id),
            dependencies=self.dependencies,
            allowed_toolsets=self.allowed_toolsets.copy(),
            payload=self._payload.copy(),
            vars=self._vars.copy(),
        )
        for key, value in overrides.items():
            if key not in ("payload", "vars"):
                setattr(forked, key, value)
        return forked
