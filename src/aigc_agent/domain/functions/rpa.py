"""RPA functions —— 浏览器采集小红书 / 抖音。

**权限设成 L-external**：它会真的开浏览器、用你的登录态访问平台。
和「调个 API 读数据」不是一个性质，每次都该让你知道。

三个能力，由浅入深：
  xhs_collect      只读列表卡片（快，但拿不到详情和分享链接）
  douyin_hot_rpa   抓热榜页词条
  browse_and_copy  **真实点进去 + 复制分享链接 + 抓详情**，完全不依赖数据接口
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore, AssetType
from ..rpa.browser import BrowserSession, have_playwright
from ..rpa.collectors import collect_douyin_hot, collect_xiaohongshu
from ..rpa.humanize import load_pace
from ..rpa.interact import browse_and_capture, grant_clipboard


class RpaFunctions:
    name = "rpa"
    namespaced = False
    disclosure = "full"

    def __init__(self, store: AssetStore, workspace: Path) -> None:
        self.store = store
        self.profile = workspace / "rpa" / "profile"

    @property
    def _specs(self) -> dict[str, ToolSpec]:
        return {
            "browse_and_copy": ToolSpec(
                name="browse_and_copy",
                summary="真实点开内容、复制分享链接、抓详情（不走任何数据接口）",
                permission=PermissionLevel.EXTERNAL,
                description=(
                    "**全程真实点击**：打开列表 → 逐个点进详情 → 点分享 → 复制链接 → 退回。\n"
                    "同时抓详情页上肉眼可见的互动数据和热评，所以**不依赖任何数据接口**。\n"
                    "拿到的链接可直接喂给 fetch_douyin 做深度拆解，"
                    "或配合 douyin-viral-analyzer 这个 skill 写归因报告。\n"
                    "节奏是拟人的，会比较慢 —— 这是有意的，保护账号。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "site": {"type": "string", "enum": ["douyin", "xhs"]},
                        "keyword": {
                            "type": "string",
                            "description": (
                                "小红书：搜索词，留空走发现页。"
                                "抖音：搜索页在自动化下不加载内容，恒走热榜页，此参数无效"
                            ),
                        },
                        "count": {"type": "integer", "description": "点开几条，默认 5"},
                        "pace": {
                            "type": "string",
                            "enum": ["default", "cautious", "brisk"],
                            "description": "行为节奏，默认 default",
                        },
                    },
                    "required": ["site"],
                },
            ),
            "xhs_collect": ToolSpec(
                name="xhs_collect",
                summary="只读小红书列表卡片（快，但没有详情和分享链接）",
                permission=PermissionLevel.EXTERNAL,
                description=(
                    "只抠列表页的标题/作者/赞数，**不点进去**，所以快。\n"
                    "要分享链接和详情数据请用 browse_and_copy。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "搜索词，留空抓推荐流"},
                        "limit": {"type": "integer", "description": "抓几条，默认 20"},
                        "scrolls": {"type": "integer", "description": "滚动次数，默认 3"},
                        "pace": {
                            "type": "string",
                            "enum": ["default", "cautious", "brisk"],
                            "description": "行为节奏",
                        },
                    },
                },
            ),
            "douyin_hot_rpa": ToolSpec(
                name="douyin_hot_rpa",
                summary="用浏览器抓抖音热榜页词条（全站热榜；带 keyword 只保留相关的）",
                permission=PermissionLevel.EXTERNAL,
                description=(
                    "只读热榜词条，是**全站热榜**，不按关键词搜 —— "
                    "带 keyword 时只保留和它相关的条目，"
                    "一条都不相关就如实报。按关键词找热点用 douyin_hot_list(keyword=…)。"
                    "要点进视频拿链接请用 browse_and_copy。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "默认 20"},
                        "keyword": {"type": "string", "description": "只保留和它相关的条目"},
                    },
                },
            ),
        }

    # ---------- ToolProvider ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        if not have_playwright():
            return ProviderHealth(ok=False, detail="未装 playwright，RPA 不可用")
        logged = (self.profile / "Default").exists()
        return ProviderHealth(
            ok=True,
            detail="浏览器就绪" + ("（有登录态）" if logged else "（首次需扫码登录）"),
        )

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            r = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            r = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        r.duration_ms = int((time.perf_counter() - started) * 1000)
        return r

    # ---------- 实现 ----------

    async def _fn_browse_and_copy(
        self, site: str, keyword: str = "", count: int = 5, pace: str = "default"
    ) -> ToolResult:
        if not have_playwright():
            return ToolResult(ok=False, error="未装 playwright")
        if site not in ("douyin", "xhs"):
            return ToolResult(ok=False, error=f"site 只能是 douyin 或 xhs，收到 {site!r}")

        p = load_pace("" if pace in ("", "default") else pace)
        url, cards, origin = _entry(site, keyword)

        async with BrowserSession(self.profile) as s:
            await grant_clipboard(s.context, origin)
            await s.goto(url, settle_ms=3500)
            caps, err = await browse_and_capture(s, site, cards, count, p)

        if err:
            return ToolResult(ok=False, error=err)

        body = "\n".join(c.brief(i) for i, c in enumerate(caps, 1))
        links = [c.best_link for c in caps]
        site_cn = "抖音" if site == "douyin" else "小红书"
        scope = ""
        if site == "douyin" and keyword:
            # 抖音侧只能点全站热榜（搜索页在自动化下不加载）。之前结果里不说，模型把无关的热榜
            # 内容当成「和关键词相关的热点」写进简报（2026-09-24 审查）
            related = [str(i) for i, c in enumerate(caps, 1) if _relevant(c.title, keyword)]
            scope = "·全站热榜"
            head = f"注意：抖音侧点的是全站热榜，没有按「{keyword}」搜索；" + (
                f"和它相关的只有第 {'、'.join(related)} 条，其余别当出处"
                if related
                else "一条都不相关，别当出处 —— 按关键词找用 "
                f"douyin_hot_list(keyword=\"{keyword}\")"
            )
            body = head + "\n" + body
        asset = self.store.create(
            body,
            type_=AssetType.TEXT,
            summary=f"{site_cn}·真实点击{scope}·{len(caps)}条",
            creator="tool:browse_and_copy",
            gen_params={"site": site, "keyword": keyword, "links": links},
        )
        hint = (
            '\n\n想深挖某条：fetch_douyin(ref="<上面的链接>")'
            if site == "douyin"
            else "\n\n小红书详情已抓在上面，可直接用于选题分析"
        )
        return ToolResult(content=body + hint, asset_ref=asset.id)

    async def _fn_xhs_collect(
        self, keyword: str = "", limit: int = 20, scrolls: int = 3, pace: str = "default"
    ) -> ToolResult:
        if not have_playwright():
            return ToolResult(ok=False, error="未装 playwright。跑：pip install playwright")
        p = load_pace("" if pace in ("", "default") else pace)
        async with BrowserSession(self.profile) as s:
            posts, err = await collect_xiaohongshu(s, keyword, limit, scrolls, p)
        if err:
            return ToolResult(ok=False, error=err)
        return self._pack(posts, f"小红书·{keyword or '发现页'}", "xhs_collect", keyword=keyword)

    async def _fn_douyin_hot_rpa(self, limit: int = 20, keyword: str = "") -> ToolResult:
        if not have_playwright():
            return ToolResult(ok=False, error="未装 playwright")
        async with BrowserSession(self.profile) as s:
            # 带关键词时多抓一些再筛
            posts, err = await collect_douyin_hot(s, max(limit, 50) if keyword else limit,
                                                  load_pace())
        if err:
            return ToolResult(ok=False, error=err)
        if keyword:
            # 2026-09-23 审查：之前 keyword 没往下传，全站热榜被当成「相关热点」喂给简报，
            # 规划模型硬套无关热点当出处
            hits = [p for p in posts if _relevant(p.title, keyword)][:limit]
            if not hits:
                return ToolResult(
                    ok=False,
                    error=(
                        f"抖音热榜 {len(posts)} 条里没有和「{keyword}」相关的"
                        "（RPA 只能抓全站热榜）。"
                        f"按关键词找用 douyin_hot_list(keyword=\"{keyword}\") 或 "
                        f"xhs_collect(keyword=\"{keyword}\")"
                    ),
                )
            return self._pack(hits, f"抖音热榜（RPA）·与「{keyword}」相关", "douyin_hot_rpa",
                              keyword=keyword)
        return self._pack(posts, "抖音热榜（RPA）·全站", "douyin_hot_rpa")

    def _pack(self, posts: list[Any], title: str, tool: str, keyword: str = "") -> ToolResult:
        body = "\n".join(p.line(i) for i, p in enumerate(posts, 1))
        asset = self.store.create(
            body,
            type_=AssetType.TEXT,
            summary=f"{title}·{len(posts)}条",
            creator=f"tool:{tool}",
            gen_params={"keyword": keyword} if keyword else {},
        )
        return ToolResult(content=f"{title}（{len(posts)} 条）\n{body}", asset_ref=asset.id)


def _relevant(title: str, keyword: str) -> bool:
    """热榜词条和关键词沾不沾边：整词出现，或关键词里任意两个相邻的字出现（中文没有空格分词）。"""
    t = (title or "").lower()
    k = "".join((keyword or "").lower().split())
    if not t or not k:
        return False
    if k in t:
        return True
    return any(k[i : i + 2] in t for i in range(len(k) - 1))


def _entry(site: str, keyword: str) -> tuple[str, list[str], str]:
    """入口页 + 可点卡片的候选选择器 + 剪贴板授权用的 origin。"""
    if site == "douyin":
        # 实测 2026-09-11：搜索页的 scroll-list 在自动化浏览器下**恒为空**，
        # 滚动多轮也不加载。热榜页正常（20 个 a[href*="/video/"]），
        # 所以 keyword 只用于事后筛选，入口统一走热榜。
        url = "https://www.douyin.com/hot"
        cards = [
            'a[href*="/video/"]',
            '[data-e2e="feed-video-container"] a',
            "div[class*=video-card] a",
        ]
        return url, cards, "https://www.douyin.com"

    url = (
        f"https://www.xiaohongshu.com/search_result?keyword={keyword}"
        if keyword
        else "https://www.xiaohongshu.com/explore"
    )
    cards = [
        "section.note-item a.cover",
        'a[href*="/explore/"]',
        "section[class*=note] a",
        "div.note-item a",
    ]
    return url, cards, "https://www.xiaohongshu.com"
