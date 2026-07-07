import sys
import time
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import List, Optional, Set, Any

from .node import NodeRecord, SessionStatus, NodeStatus
from .session import SessionContext


@dataclass
class ExecutionContext:
    """
    【单次执行上下文】 - 工人视角
    （RAII + 并发隔离 + 状态栅栏硬托底 · 最终无死角版）

    设计哲学：
    1. Worker 只负责“干活”和“汇报”，绝不决策全局命运。
    2. 异常只在发生时捕获，绝不吞噬，也不依赖隐式线程帧。
    3. 局部超时只杀死自己，全局超时才拉响大盘警报。
    """

    execution_id: str
    _session: SessionContext
    parent_node: Optional[NodeRecord] = None
    timeout_limit: float = 15.0

    # 物理硬管制属性
    local_deadline: float = field(init=False)
    local_fingerprints: Set[str] = field(default_factory=set, init=False)
    active_node_stack: List[NodeRecord] = field(default_factory=list, init=False)

    # 顶层异常隔离栅栏（仅 __exit__ 使用，阻断跨栈污染）
    _active_exception: Optional[tuple] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """RAII 初始化：即刻锁定局部时钟死线"""
        now = time.time()
        self.local_deadline = min(
            now + self.timeout_limit,
            self._session.global_deadline,
        )

    @property
    def session_view(self) -> Any:
        """实时拉取全局只读快照（无缓存、无过期）"""
        return self._session.get_readonly_view()

    @property
    def reporter(self) -> Any:
        """动态路由到会话的安全上报中介"""
        return self._session.get_reporter()

    # =========================================================================
    # 单步切面（核心发动机）
    # =========================================================================
    @contextmanager
    def step(
        self,
        node_id: str,
        metadata: Any,
        parent_node_override: Optional[NodeRecord] = None,
    ):
        """
        全系统唯一的图节点拓扑血缘与生命周期切面托管器。

        语义保证：
        - 无论发生什么，节点状态最终必被染色（RUNNING -> COMPLETED/FAILED）。
        - 绝不因 sys.exc_info() 残留导致误判。
        - 支持嵌套调用，自动构建树状血缘。
        """
        self.check_budget()

        node = NodeRecord(node_id=node_id, metadata=metadata)

        # 1. 动态自适应血缘 Edge 绑定
        # 优先级：显式指定 > 栈顶（子节点） > 根节点（父节点）
        actual_parent = parent_node_override
        if not actual_parent:
            actual_parent = (
                self.active_node_stack[-1]
                if self.active_node_stack
                else self.parent_node
            )

        self.active_node_stack.append(node)

        if actual_parent:
            with actual_parent._lock:
                actual_parent.children.append(node)

        self.reporter.update_node_status(node, NodeStatus.RUNNING)

        landed_normally = False
        try:
            yield node
            # 只有完好无损地走出 with 业务块，才标记为真阳性落地
            landed_normally = True
        finally:
            # RAII 对称性精确出栈（防止脏栈干扰后续逻辑）
            if self.active_node_stack and self.active_node_stack[-1] is node:
                self.active_node_stack.pop()

            # 状态染色：三级分支，逻辑完备且无歧义
            if landed_normally:
                # 分支 A：纯净正常落地，原子染绿
                self.reporter.update_node_status(node, NodeStatus.COMPLETED)
            else:
                # 分支 B/C：非正常落地，启动异常屏障判定
                exc_type, exc_val, _ = sys.exc_info()
                if exc_type is not None:
                    # 分支 B：真实捕捉到当前 step 内部业务抛出的未平仓崩溃
                    self.reporter.update_node_status(
                        node,
                        NodeStatus.FAILED,
                        f"{exc_type.__name__}: {exc_val}",
                    )
                else:
                    # 分支 C：GeneratorExit 或被上层 Task 取消/强制掐断
                    self.reporter.update_node_status(
                        node,
                        NodeStatus.FAILED,
                        "[Forced Abort] Interrupted by external control flow.",
                    )

    # =========================================================================
    # 顶层托管（地毯式轰炸清理）
    # =========================================================================
    def __enter__(self):
        """进入 Worker 上下文，激活父节点"""
        if self.parent_node:
            self.reporter.update_node_status(self.parent_node, NodeStatus.RUNNING)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        顶层异常自愈清理（RAII 级联消杀屏障）。

        职责：
        1. 封印顶层异常，防止其污染线程帧。
        2. 清理所有残留的“活跃流产子节点”。
        3. 根据 Worker 自身命运沉降父节点。
        """
        # 封印异常（仅用于级联消杀，不用于业务判断）
        if exc_type is not None:
            self._active_exception = (exc_type, exc_val, exc_tb)

        final_status = NodeStatus.FAILED if exc_type else NodeStatus.COMPLETED
        master_reason = f"{exc_type.__name__}: {exc_val}" if exc_type else ""

        # 1. 清理残留子节点（它们属于被动流产）
        while self.active_node_stack:
            stale_node = self.active_node_stack.pop()
            abort_reason = (
                f"[Cascaded Abort] Forced interrupted by context teardown. "
                f"Master reason: {master_reason}"
                if master_reason
                else "[Cascaded Abort]"
            )
            self.reporter.update_node_status(
                stale_node, NodeStatus.FAILED, abort_reason
            )

        # 2. 清理主承载父节点（它随 Worker 的命运沉降）
        if self.parent_node:
            self.reporter.update_node_status(
                self.parent_node, final_status, master_reason
            )

        # 3. 坚决不吞异常，让上层引擎（Event Loop / Scheduler）最终收卷
        return False

    # =========================================================================
    # 安全自检底座（硬管制）
    # =========================================================================
    def check_budget(self) -> None:
        """
        轻量级高频埋点自检。

        语义规则：
        - 全局死线爆了 -> 上报 Session TIMEOUT
        - 局部死线爆了 -> 仅抛异常，不污染全局
        - Token 耗尽 -> 上报 Session FAILED
        """
        view = self._session.get_readonly_view()

        # 1. 全局终态拦截（Session 已死，禁止新工作）
        if view.status in (SessionStatus.FAILED, SessionStatus.TIMEOUT):
            raise RuntimeError("Execution abandoned: Global session already terminated")

        now = time.time()

        # 2. 超时判定（精细划分）
        if view.is_expired:
            # 全局时钟到期：拉响大盘警报
            self.reporter.report_session_failure(
                SessionStatus.TIMEOUT,
                f"Global session timeout detected in execution {self.execution_id}",
            )
            raise TimeoutError("Global deadline exceeded")

        if now > self.local_deadline:
            # 局部时钟到期：仅自杀，不连坐
            raise TimeoutError(
                f"Local worker time quota exceeded ({self.timeout_limit}s reached)"
            )

        # 3. Token 预算拦截
        if view.total_tokens >= view.max_token_budget:
            self.reporter.report_session_failure(
                SessionStatus.FAILED,
                "Global token budget exhausted",
            )
            raise RuntimeError("Token budget exhausted")

    def consume_tokens(self, prompt: int, completion: int) -> None:
        """上报 Token 消耗（会计接口）"""
        self.reporter.report_token_cost(prompt + completion)

    def check_and_record_local_fingerprint(self, fp: str) -> bool:
        """
        本地指纹去重。
        返回 True 表示命中重复，False 表示首次记录。
        """
        if fp in self.local_fingerprints or self._session.has_fingerprint(fp):
            return True
        self.local_fingerprints.add(fp)
        self.reporter.report_fingerprint(fp)
        return False
