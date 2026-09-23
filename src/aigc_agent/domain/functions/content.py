"""内容生产 functions —— agentic 形态下的领域能力面。

每个生产环节暴露成一个 function，模型自己决定调哪个、什么时候调、调几次。
不需要提前把流程画成图。

但**所有 function 都写同一套 Asset 底座**，所以血缘、版本、单步重跑、
成本归因一样不少 —— 图从「必须先画的输入」变成「执行痕迹自动形成的输出」。

`request_review` 是这套设计的关键：人审不再是图上的固定节点，而是模型
自己判断「这个该让人看一眼」时调用的一个 function。它会挂起主循环。

2026-09-17：`list_assets` 从"全吐"改成带过滤和分页，另加 `find_episode`。
之前 612 份资产一次全吐、被截到前几十条，模型以为其余不存在，转头去
search_library 逐集找了 34 次，还编出不存在的 id。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore, AssetType

_TYPE_BY_KIND = {
    "outline": AssetType.OUTLINE,
    "script": AssetType.SCRIPT,
    "storyboard": AssetType.STORYBOARD,
    "copy": AssetType.TEXT,
}

DEFAULT_LIST_LIMIT = 30


READ_PAGE = 12_000  # read_asset / 参考文档一页多少字

# 短内容里出现这些字样，说明模型没写正文、只写了个指代（抄回来的「省略」「同上」）
STUB_HINTS = ("此处省略", "（略）", "(略)", "……省略", "...省略", "同上", "见上文", "见前文",
              "内容同前", "全文略")


def _stub_problem(content: str) -> str:
    """存稿内容是不是空壳（占位字样 + 很短）。是就返回拒收原因。

    折叠占位符「<N 字已折叠>」在参数层和资产库两道都拦了；这里补它的变体 ——
    模型换个说法（「此处省略…」「同上」）照样能把一份空壳存成「第 3 集剧本」
    （2026-09-23 审查）。只拦**短**内容，正文里偶尔出现这几个字不误伤。
    """
    s = (content or "").strip()
    if len(s) < 300 and any(h in s for h in STUB_HINTS):
        return (
            f"内容只有 {len(s)} 个字，还写着「省略 / 同上」之类的指代 —— 这不是正文，没有保存。"
            "把完整内容写出来；太长就 fs_write 分块写本地文件再 fs_import；"
            "要沿用已存的内容就传它的资产 id，别存一份空壳"
        )
    return ""


def page_of(text: str, offset: int, call_head: str) -> str:
    """取一页，并在结尾说清楚还剩多少、下一页怎么读。call_head 形如 read_asset(asset_id="x"。"""
    total = len(text)
    start = max(0, min(int(offset or 0), total))
    end = min(total, start + READ_PAGE)
    body = text[start:end]
    if start == 0 and end >= total:
        return body
    if start >= total:
        return f"[offset={offset} 已经超过末尾：共 {total:,} 字，前面都读过了]"
    head = f"[第 {start + 1:,}–{end:,} 字 / 共 {total:,} 字]\n"
    tail = (
        f"\n\n[还有 {total - end:,} 字没读：{call_head}, offset={end}) 继续]"
        if end < total
        else "\n\n[读完了]"
    )
    return head + body + tail


class ContentFunctions:
    """内容领域的 ToolProvider。

    P2 接生图/TTS/剪辑时，在这里加 function 即可 —— 不动 harness，
    也不需要改任何图定义（因为根本没有图定义）。
    """

    name = "content"
    namespaced = False

    def __init__(self, store: AssetStore, local: Any = None) -> None:
        self.store = store
        # 本地素材索引（LocalMaterials，2026-09-20）：find_episode 连产物目录里这一集的文件一起报。
        # None = 只查资产库（测试 / 没有产物目录时）
        self.local = local
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    # ---------- 声明 ----------

    def _build(self) -> None:
        self._add(
            "save_draft",
            "把你写好的内容存为一份资产，返回资产 id",
            PermissionLevel.WRITE,
            {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "正文全文"},
                    "kind": {
                        "type": "string",
                        "enum": ["outline", "copy", "script", "storyboard"],
                        "description": "内容类型",
                    },
                    "summary": {"type": "string", "description": "一句话摘要，40 字内"},
                    "parent_id": {
                        "type": "string",
                        "description": "若基于某份资产改写，填它的 id，用于建立血缘",
                    },
                    "episode": {
                        "type": "integer",
                        "description": "剧本按集存时填集号 —— 按集流水管线按它分批接续下游环节",
                    },
                },
                "required": ["content", "kind"],
            },
            description=(
                "存内容并拿到 id。**每产出一个版本就存一次**——存过才能被人审、"
                "被对比、被回退。改写已有资产时务必填 parent_id，否则血缘断了。"
                "短剧逐集剧本务必填 episode，流水线靠它按批接续分镜/提示词/渲染。"
            ),
        )

        self._add(
            "read_asset",
            "按 id 读回一份资产的内容（长的分页读：offset 往后翻）",
            PermissionLevel.READ,
            {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "offset": {
                        "type": "integer",
                        "description": "从第几个字开始读，默认 0；上一页结尾会写下一页的 offset",
                    },
                },
                "required": ["asset_id"],
            },
            description=(
                "上下文里只有资产的 id 和摘要，需要全文时用这个取回。一次最多 "
                f"{READ_PAGE:,} 字；更长的（整集分镜、资产库、多集剧本）结尾会写「还有 N 字，"
                "offset=… 继续」—— **要改写长资产就先读全**，只读到一半就改写会把后半截丢掉。"
            ),
            max_result_chars=READ_PAGE + 400,
        )

        self._add(
            "list_assets",
            "列资产（id / 类型 / 版本 / 摘要），可按类型、集号、创建者、关键词过滤，默认最新 30 条",
            PermissionLevel.READ,
            {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [t.value for t in AssetType],
                        "description": (
                            "只要某一类：script / outline / storyboard / text / image / video…"
                        ),
                    },
                    "episode": {
                        "type": "integer",
                        "description": "只要第几集的（剧本/提示词/片段）",
                    },
                    "creator": {
                        "type": "string",
                        "description": "创建者前缀，如 tool:drama_shots / model / human",
                    },
                    "contains": {"type": "string", "description": "摘要里包含的关键词"},
                    "limit": {"type": "integer", "description": "最多几条，默认 30，最多 200"},
                    "offset": {"type": "integer", "description": "翻页：跳过前几条"},
                    "oldest_first": {"type": "boolean", "description": "默认最新在前"},
                    "all_projects": {
                        "type": "boolean",
                        "description": "连别的项目（别的产物目录）的也列，默认只列当前项目",
                    },
                },
            },
            description=(
                "找资产用这个，**不要用 search_library 找自己刚存的东西**。"
                "默认只列**当前项目**（当前产物目录）的、最新 30 条；资产多了要加 type / episode / "
                "contains 过滤或翻页，没列出来不代表不存在。找某一集的剧本直接用 find_episode。"
                "用户要拿别的项目（别的剧）的东西时才传 all_projects=true。"
            ),
            max_result_chars=12_000,
        )

        self._add(
            "find_episode",
            "按集号找这一集的全部产物（剧本 / 分镜提示词 / 视频片段）各取最新版，"
            "连产物目录里这一集的本地文件一起列",
            PermissionLevel.READ,
            {
                "type": "object",
                "properties": {"episode": {"type": "integer", "description": "集号"}},
                "required": ["episode"],
            },
            description=(
                "写第 N 集之前回看前一集、修订某一集、核对哪几集还没写、"
                "找用户电脑上这一集的素材，都用它。返回各类产物的最新版 id（正文用 read_asset 取）"
                "和产物目录里文件名带这一集集号的本地文件（路径可直接给 fs_read / view_image / "
                "view_video）。资产库没记录不等于没有——看本地文件那一段。"
            ),
            max_result_chars=12_000,
        )

        self._add(
            "compare_assets",
            "并排对比两份资产，用于给人看候选之间的差异",
            PermissionLevel.READ,
            {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a", "b"],
            },
        )

        self._add(
            "request_review",
            "把候选交给人做决定，并**暂停**，等人回复后再继续",
            PermissionLevel.WRITE,
            {
                "type": "object",
                "properties": {
                    "asset_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要给人看的资产 id，多个即为候选",
                    },
                    "question": {"type": "string", "description": "你想问人什么"},
                    "stage": {
                        "type": "string",
                        "description": "当前处在哪个环节，如 大纲/正文/配图",
                    },
                    "major": {
                        "type": "boolean",
                        "description": (
                            "是不是大节点：true = 即使 /auto 自动模式也停下来等人拍板一次"
                            "（剧本方向定稿、开始生图/生视频之前）；false = /auto 下自动采纳。"
                            "不填则按 stage 里的关键词猜。"
                        ),
                    },
                },
                "required": ["asset_ids", "question"],
            },
            description=(
                "在关键节点主动请人拍板：定了大纲、出了正文、要发布之前。"
                "调用后你会停下来，人的决策（采纳/打回及理由）会作为本次调用的"
                "返回值给你。**不要在没存资产之前调它。** 一次迭代只能提一次。"
            ),
        )

    def _add(
        self,
        name: str,
        summary: str,
        permission: PermissionLevel,
        params: dict[str, Any],
        description: str = "",
        max_result_chars: int = 0,
    ) -> None:
        self._specs[name] = ToolSpec(
            name=name,
            summary=summary,
            permission=permission,
            parameters=params,
            description=description or summary,
            max_result_chars=max_result_chars,
        )

    # ---------- ToolProvider 协议 ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail=f"{len(self._specs)} 个内容 function")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            result = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    # ---------- 实现 ----------

    async def _fn_save_draft(
        self,
        content: str,
        kind: str,
        summary: str = "",
        parent_id: str = "",
        episode: int = 0,
    ) -> ToolResult:
        gen_params: dict[str, Any] = {"episode": episode} if episode else {}
        stub = _stub_problem(content)
        if stub:
            return ToolResult(ok=False, error=stub)
        if parent_id:
            asset = self.store.revise(parent_id, content, summary=summary, creator="model")
            if gen_params:
                asset.gen_params.update(gen_params)
                self.store.put(asset)
        else:
            asset = self.store.create(
                content,
                type_=_TYPE_BY_KIND.get(kind, AssetType.TEXT),
                summary=summary,
                creator="model",
                gen_params=gen_params,
            )
        return ToolResult(
            content=f"已存为 {asset.id}（{kind} v{asset.version}）：{asset.summary}",
            asset_ref=asset.id,
        )

    async def _fn_read_asset(self, asset_id: str, offset: int = 0) -> ToolResult:
        """分页读。2026-09-23 审查：之前不分页、被调度器截到 4000 字，截断提示还指回同一个
        工具 —— 超过 4000 字的资产模型永远读不全，改写后存回去，后半截就静默丢了。"""
        text = self.store.content(asset_id)
        return ToolResult(
            content=page_of(text, offset, f'read_asset(asset_id="{asset_id}"'),
            asset_ref=asset_id,
        )

    async def _fn_list_assets(
        self,
        type: str = "",  # noqa: A002 — 对齐参数名
        episode: int = 0,
        creator: str = "",
        contains: str = "",
        limit: int = DEFAULT_LIST_LIMIT,
        offset: int = 0,
        oldest_first: bool = False,
        all_projects: bool = False,
    ) -> ToolResult:
        type_ = None
        if type:
            try:
                type_ = AssetType(type)
            except ValueError:
                return ToolResult(
                    ok=False,
                    error=f"未知类型 {type!r}。可选：{', '.join(t.value for t in AssetType)}",
                )
        items = self.store.find(
            type_=type_,
            episode=int(episode or 0),
            creator=creator,
            contains=contains,
            newest_first=not oldest_first,
            project="*" if all_projects else None,
        )
        total = len(items)
        scope = "全部项目" if all_projects or not self.store.project else "当前项目"
        if not total:
            if not (type or episode or creator or contains):
                return ToolResult(content=f"{scope}还没有任何资产")
            return ToolResult(
                content=f"{scope}没有匹配的资产（整个资产库共 {len(self.store)} 份，"
                "换个过滤条件，或 all_projects=true 看别的项目）"
            )
        limit = max(1, min(int(limit or DEFAULT_LIST_LIMIT), 200))
        offset = max(0, int(offset or 0))
        page = items[offset : offset + limit]
        head = f"{scope}匹配 {total} 份，显示第 {offset + 1}–{offset + len(page)} 条"
        if offset + len(page) < total:
            head += f"（还有 {total - offset - len(page)} 条，offset={offset + len(page)} 翻页）"
        lines = [head]
        for a in page:
            ep = self.store.episode_of(a)
            tag = f" 第{ep}集" if ep else ""
            lines.append(f"{a.brief()}{tag} · {a.creator or '—'}")
        return ToolResult(content="\n".join(lines))

    async def _local_episode(self, n: int) -> str:
        """产物目录里这一集的本地文件（文件名带集号的）。目录扫描是磁盘 IO，丢线程。"""
        if self.local is None:
            return ""
        try:
            index = await asyncio.to_thread(self.local.index)
            return index.render_episode(n)
        except OSError:
            return ""

    async def _fn_find_episode(self, episode: int) -> ToolResult:
        n = int(episode)
        found = self.store.episode_assets(n)
        local = await self._local_episode(n)
        if not found and not local:
            done = self.store.episodes_done()
            hint = f"已有剧本的集：{_ranges(done)}" if done else "还没有任何一集的剧本"
            return ToolResult(content=f"第 {n} 集还没有任何产物（资产库和产物目录都没有）。{hint}")
        lines: list[str] = []
        if found:
            lines.append(f"第 {n} 集的产物（各取最新版）：")
            for label, a in found.items():
                lines.append(f"- {label}：{a.brief()} · {a.creator}")
        else:
            done = self.store.episodes_done()
            hint = f"（资产库里有剧本的集：{_ranges(done)}）" if done else ""
            lines.append(f"资产库里没有第 {n} 集的记录{hint}，但产物目录里有它的文件，见下。")
        if local:
            lines.append(local)
        ref = next(iter(found.values())).id if found else None
        return ToolResult(content="\n".join(lines), asset_ref=ref)

    async def _fn_compare_assets(self, a: str, b: str) -> ToolResult:
        x, y = self.store.get(a), self.store.get(b)
        return ToolResult(
            content=(
                f"--- {x.brief()} ---\n{self.store.content(a)}\n\n"
                f"--- {y.brief()} ---\n{self.store.content(b)}"
            )
        )

    async def _fn_request_review(
        self,
        asset_ids: list[str],
        question: str,
        stage: str = "",
        major: bool | None = None,
    ) -> ToolResult:
        missing = [i for i in asset_ids if not self.store.has(i)]
        if missing:
            return ToolResult(
                ok=False,
                error=f"这些资产不存在：{missing}。先用 save_draft 存下来再请人审。",
            )
        preview = "\n".join(self.store.get(i).brief() for i in asset_ids)
        payload: dict[str, Any] = {
            "assets": asset_ids,
            "question": question,
            "stage": stage,
            "target": stage,
        }
        if isinstance(major, bool):
            payload["major"] = major
        return ToolResult(
            content=f"已提交给人审阅（{stage or '未标注环节'}）：{question}\n{preview}",
            suspend=True,  # ← 挂起主循环，等人的决策
            suspend_payload=payload,
        )


def _ranges(nums: list[int]) -> str:
    """[1,2,3,5,6,9] → '1–3, 5–6, 9'。"""
    if not nums:
        return "无"
    out: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}–{prev}" if start != prev else str(start))
        start = prev = n
    out.append(f"{start}–{prev}" if start != prev else str(start))
    return ", ".join(out)
