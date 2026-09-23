"""M4 ToolProvider —— 所有工具来源的统一入口。

这是 P0 阶段最重要的一个抽象。内置工具、MCP Hub(M20)、Skill 自带工具
都实现这个协议；M4 只认 Provider，**不知道 MCP 是什么**。

为什么必须在 P0 就定下来：现在写是十几行接口；等 P2 接 MCP 时再回头改，
意味着已写好的工具注册、权限判定、成本记账要全部重构一遍。
—— ARCHITECTURE.md 建设顺序里「早做便宜、晚做很贵」差距最大的一处。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class PermissionLevel(StrEnum):
    """按副作用分级。权限闸门(M5)直接挂在等级上。"""

    READ = "L-read"  # 只读、无成本                   → 自动放行
    COMPUTE = "L-compute"  # 产资产、花钱             → 预算内放行
    WRITE = "L-write"  # 改本地状态、可回滚           → 自动放行
    EXTERNAL = "L-external"  # 不可逆外部动作         → 强制人工确认


class ToolMeta(BaseModel):
    """目录级元数据。常驻上下文，约 30 token/个。

    完整 schema 通过 get_schema() 按需展开（约 300-800 token/个）。
    60 个工具：目录 1.8K vs 全量 30K，差 17 倍 —— 这就是两级披露的意义。
    """

    name: str  # 已含命名空间的最终名
    summary: str  # 一句话
    permission: PermissionLevel = PermissionLevel.EXTERNAL
    provider: str = ""
    # 花钱的工具按这个分类计次（image / video / audio）。Cost Guard 的次数口径
    # 靠它：媒体生成按次计费不按 token，目录里没填单价时次数照样拦得住。
    # 留空 = 不计次（本地 ffmpeg 合成这类 L-compute 不花钱，就留空）。
    cost_kind: str = ""
    # 批量工具：这个参数名指向的数组有几项，就算几次调用。
    # 不设的话 gen_videos 一次生成 20 段只会被记成 1 次，预算护栏等于没有
    # （2026-09-19 加批量并发时发现）。
    cost_units_arg: str = ""
    # 单次调用的调度器超时（秒）。0 = 用调度器默认（120s）。
    # 批量渲染这类一次调用跑几十分钟的工具必须显式给足，
    # 否则渲染没跑完先被调度器杀了。
    timeout: float = 0
    # 结果进上下文的截断上限（字符）。0 = 调度器默认（4000）。
    # 列表类工具（list_assets）要给大一点：612 份资产被截到前几十条，
    # 模型会以为剩下的不存在，然后编 id。
    max_result_chars: int = 0
    # 预算护栏问人的标记（仅闸门在超限询问时临时置上；budget_ask=True 时
    # summary 里带原因）。budget_money=True 是金额口径 —— auto 模式可以
    # 自动放行次数护栏，金额护栏永远问人。
    budget_ask: bool = False
    budget_money: bool = False


class ToolResult(BaseModel):
    ok: bool = True
    content: str = ""
    asset_ref: str | None = None  # 完整内容的指针（M12），上下文只带引用
    truncated: bool = False
    error: str | None = None
    duration_ms: int = 0

    # 挂起信号：工具要求主循环停下来等人（如 request_review）。
    # 这是 agentic 形态下 HITL 的实现方式 —— 人审不再是图上的固定节点，
    # 而是模型**自己决定何时**调用的一个 function。
    suspend: bool = False
    suspend_payload: dict[str, Any] = Field(default_factory=dict)

    # 给**程序**看的结构化附加信息（不进上下文）。例：媒体生成失败时
    # {"retryable": False, "task_id": "..."} —— 调用方据此决定能不能原样重提，
    # 不再靠在错误文本里找「网络」「HTTP 5」这种字眼猜（2026-09-23 审查：猜错了就是
    # 把还在服务端跑、已经计费的任务又提交一遍）。
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_message_content(self) -> str:
        if not self.ok:
            return f"[工具执行失败] {self.error}"
        body = self.content
        if self.truncated and self.asset_ref:
            body += f"\n\n[内容已截断，完整内容见 asset_id: {self.asset_ref}]"
        return body


class ProviderHealth(BaseModel):
    ok: bool = True
    detail: str = ""


@runtime_checkable
class ToolProvider(Protocol):
    """工具提供方协议。

    实现方：
      · BuiltinProvider     —— 内置工具（P0）
      · McpHubProvider      —— 外部 MCP server（P2，M20）
      · SkillToolProvider   —— Skill 自带工具（P3，M7）
    """

    name: str
    namespaced: bool  # True 时工具名加 {name}__ 前缀。MCP 必须 True，内置 False

    async def list_tools(self) -> list[ToolMeta]: ...

    async def get_schema(self, tool: str) -> dict[str, Any]:
        """返回 OpenAI function-calling 格式的完整定义。"""
        ...

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult: ...

    async def health(self) -> ProviderHealth: ...


class ToolSpec(BaseModel):
    """内置工具的声明式定义。"""

    name: str
    summary: str
    permission: PermissionLevel
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    description: str = ""
    cost_kind: str = ""  # 见 ToolMeta.cost_kind
    cost_units_arg: str = ""  # 见 ToolMeta.cost_units_arg
    timeout: float = 0  # 见 ToolMeta.timeout
    max_result_chars: int = 0  # 见 ToolMeta.max_result_chars

    def meta(self, provider: str) -> ToolMeta:
        """目录级元数据。各 Provider 的 list_tools() 用它，别再手拼一遍字段 ——
        之前 12 处各拼各的，加一个字段就要改 12 处，漏一处该字段就静默丢失。"""
        return ToolMeta(
            name=self.name,
            summary=self.summary,
            permission=self.permission,
            provider=provider,
            cost_kind=self.cost_kind,
            cost_units_arg=self.cost_units_arg,
            timeout=self.timeout,
            max_result_chars=self.max_result_chars,
        )

    def to_openai(self, full_name: str) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": full_name,
                "description": self.description or self.summary,
                "parameters": self.parameters,
            },
        }
