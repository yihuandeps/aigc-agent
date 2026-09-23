"""真实点击 —— 点进内容、复制分享链接、抓详情。

和 collectors.py 的区别：那个只读列表卡片，这个**真的点进去**，
拿到分享链接和详情页数据，然后交给拆解 skill。

取链接有三条路，按可靠性降序，前面拿不到就退到后面：
  1. 分享面板里的「复制链接」→ 读剪贴板（拿到的是短链，最接近手动操作）
  2. 分享面板里直接显示的链接文本 → 读 DOM
  3. 详情页地址栏 URL → 一定有，但没有短链的追踪参数

第 3 条兜底，所以**只要点进去了就一定有链接**，不会空手而归。
"""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .humanize import Pace, SessionBudget, act_pause, move_like_reading

# 分享按钮的候选选择器，两个平台都在改版，多给几个
_SHARE_BTN = {
    # 实测 2026-09-11：抖音详情是 modal，分享按钮是 video-player-share
    # （不是 video-share，少写 -player 就永远找不到）
    "douyin": [
        '[data-e2e="video-player-share"]',
        '[class*=share] [class*=icon]',
        "span[class*=share]",
    ],
    "xhs": [
        '[class*=share-wrapper]',
        '[class*=share] use[*|href*="share"]',
        'span[class*=share]',
        '[id*=share]',
    ],
}

# 分享面板里「复制链接」的候选
_COPY_BTN = [
    "text=复制链接",
    "text=复制口令",
    '[class*=copy]',
    'li:has-text("复制链接")',
    'div:has-text("复制链接")',
]

# 实测得到的详情页选择器
_DESC_SEL = {
    "douyin": ['[data-e2e="video-desc"]', "[class*=video-info-detail]"],
    "xhs": ["#detail-title", "[class*=note-content] [class*=title]", "h1"],
}
_AUTHOR_SEL = {
    "douyin": ['[data-e2e="feed-video-nickname"]', '[data-e2e="video-author-info"] span'],
    "xhs": ["[class*=author] [class*=name]", "span.username"],
}

_DY_ID = re.compile(r"(?:/video/|modal_id=)(\d{15,25})")
_XHS_ID = re.compile(r"/(?:explore|discovery/item)/([0-9a-f]{16,})")
_SHORT = re.compile(r"https?://v\.douyin\.com/[\w-]+/?|https?://xhslink\.com/\S+")


@dataclass
class Captured:
    """点进去之后拿到的一条内容。"""

    platform: str
    url: str = ""  # 详情页地址，一定有
    share_link: str = ""  # 分享短链，可能拿不到
    content_id: str = ""
    title: str = ""
    author: str = ""
    stats: dict[str, str] = field(default_factory=dict)
    comments: list[str] = field(default_factory=list)
    how: str = ""  # 链接是怎么拿到的，便于排查

    @property
    def best_link(self) -> str:
        return self.share_link or self.url

    def brief(self, i: int) -> str:
        lines = [f"{i:2}. {self.title[:48] or '(无标题)'}"]
        if self.author:
            lines.append(f"     @{self.author}")
        if self.stats:
            lines.append("     " + " · ".join(f"{k} {v}" for k, v in self.stats.items()))
        lines.append(f"     {self.best_link}   [{self.how}]")
        if self.comments:
            lines.append(f"     热评：{self.comments[0][:40]}")
        return "\n".join(lines)


async def grant_clipboard(context: Any, origin: str) -> None:
    """授予剪贴板权限。不给的话读剪贴板会静默失败。"""
    try:
        await context.grant_permissions(["clipboard-read", "clipboard-write"], origin=origin)
    except Exception:  # noqa: BLE001 — 某些浏览器通道不支持，退到读 DOM
        pass


async def read_clipboard(page: Any) -> str:
    try:
        return (await page.evaluate("navigator.clipboard.readText()")) or ""
    except Exception:  # noqa: BLE001
        return ""


async def _click_first(page: Any, selectors: list[str], pace: Pace) -> bool:
    """挨个试选择器，点到为止。

    同样不预先移鼠标 —— 见 open_and_capture 里那段 A/B 实测说明。
    保留的拟人成分是点击前后的随机停顿。
    """
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if not el:
                continue
            await el.scroll_into_view_if_needed(timeout=3000)
            await asyncio.sleep(random.uniform(0.25, 0.8))
            await el.click(timeout=4000)
            await act_pause(pace)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


async def copy_share_link(page: Any, platform: str, pace: Pace) -> tuple[str, str]:
    """点分享 → 点复制链接 → 读剪贴板。返回 (链接, 来源说明)。"""
    if not await _click_first(page, _SHARE_BTN.get(platform, []), pace):
        return "", "没找到分享按钮"

    await asyncio.sleep(random.uniform(0.8, 2.0))  # 等面板弹出

    if await _click_first(page, _COPY_BTN, pace):
        link = await read_clipboard(page)
        m = _SHORT.search(link)
        if m:
            return m.group(0), "分享面板·剪贴板"
        if link.startswith("http"):
            return link.split()[0], "分享面板·剪贴板"

    # 剪贴板没拿到，看看面板里有没有直接显示链接
    try:
        text = await page.inner_text("body")
        m = _SHORT.search(text)
        if m:
            return m.group(0), "分享面板·页面文本"
    except Exception:  # noqa: BLE001
        pass
    return "", "分享面板打开了但没取到链接"


async def _grab_text(page: Any, selectors: list[str]) -> str:
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                t = " ".join((await el.inner_text()).split())
                if t:
                    return t[:100]
        except Exception:  # noqa: BLE001
            continue
    return ""


async def _grab_stats(page: Any, platform: str) -> dict[str, str]:
    """抓详情页上肉眼可见的互动数据。"""
    out: dict[str, str] = {}
    pairs = (
        [
            ("赞", '[data-e2e="video-player-digg"]'),
            ("评", '[data-e2e="feed-comment-icon"]'),
            ("藏", '[data-e2e="video-player-collect"]'),
            ("转", '[data-e2e="video-player-share"]'),
        ]
        if platform == "douyin"
        else [
            ("赞", '[class*=like] [class*=count], [id*=like] span'),
            ("藏", '[class*=collect] [class*=count]'),
            ("评", '[class*=chat] [class*=count], [class*=comment] [class*=count]'),
        ]
    )
    for label, sel in pairs:
        try:
            el = await page.query_selector(sel)
            if el:
                v = (await el.inner_text()).strip()
                if v and v not in ("", "0"):
                    out[label] = " ".join(v.split())[:12]
        except Exception:  # noqa: BLE001
            continue
    return out


async def _grab_comments(page: Any, platform: str, limit: int = 5) -> list[str]:
    """抓可见的热评。评论区是归因校验的关键（见拆解 skill 的归因铁律）。"""
    sels = (
        ['[data-e2e="comment-item"]', '[class*=comment-item]', '[class*=CommentItem]']
        if platform == "douyin"
        else ['[class*=comment-item]', '[class*=parent-comment]', '[class*=CommentItem]']
    )
    for sel in sels:
        try:
            nodes = await page.query_selector_all(sel)
        except Exception:  # noqa: BLE001
            continue
        if not nodes:
            continue
        out = []
        for n in nodes[:limit]:
            try:
                t = " ".join((await n.inner_text()).split())
            except Exception:  # noqa: BLE001
                continue
            if t and len(t) > 1:
                out.append(t[:120])
        if out:
            return out
    return []


async def open_and_capture(
    session: Any, card: Any, platform: str, pace: Pace, with_detail: bool = True
) -> Captured | None:
    """点开一张卡片，抓链接与详情，然后退回列表。"""
    page = await session.page()
    before = page.url

    # ⚠️ 点击**之前不要把鼠标移到卡片上**。
    # A/B 实测 2026-09-11（抖音热榜页，同一批卡片、同样坐标）：
    #     纯 click()                → ✓ URL 变了
    #     mouse.move + click()      → ✗ 没反应
    #     mouse.move + 坐标 click() → ✗ 没反应
    # 悬停会触发卡片的预览层，后续点击落在预览层上而不是链接本身。
    # 拟人化不能以破坏功能为代价 —— 鼠标移动保留在浏览/阅读阶段（见 humanize），
    # 那里才是它真正起作用的地方；点击这一下移不移对风控意义不大。
    try:
        await card.scroll_into_view_if_needed(timeout=4000)
        await asyncio.sleep(random.uniform(0.3, 1.0))
        await card.click(timeout=6000)
    except Exception:  # noqa: BLE001
        return None

    # 等详情打开。**必须验证 URL 真的变了** —— 点空了却当成功，
    # 会记下一条只有列表页地址的空壳数据。
    opened = False
    for _ in range(8):
        await asyncio.sleep(random.uniform(0.6, 1.2))
        if page.url != before:
            opened = True
            break
    if not opened:
        return None

    await move_like_reading(page, pace)
    url = page.url
    cap = Captured(platform=platform, url=url, how="详情页地址")

    m = (_DY_ID if platform == "douyin" else _XHS_ID).search(url)
    if m:
        cap.content_id = m.group(1)

    # 试着走一遍真实的分享 → 复制链接
    link, how = await copy_share_link(page, platform, pace)
    if link:
        cap.share_link, cap.how = link, how
    else:
        cap.how = f"详情页地址（{how}）"

    if with_detail:
        cap.title = await _grab_text(page, _DESC_SEL.get(platform, []))
        cap.author = await _grab_text(page, _AUTHOR_SEL.get(platform, []))
        if not cap.title:
            # modal 模式下页面 title 不变（一直是「抖音热点榜…」），
            # 所以只在 DOM 抠不到时才退回它
            try:
                cap.title = (await page.title()).split(" - ")[0][:80]
            except Exception:  # noqa: BLE001
                pass
        cap.stats = await _grab_stats(page, platform)
        cap.comments = await _grab_comments(page, platform)

    # 退回列表。抖音详情是 modal，Escape 关掉即可，比 go_back 更贴近真实操作，
    # 也不会丢掉列表已加载的内容。Escape 不管用才退而求其次。
    try:
        await page.keyboard.press("Escape")
        await asyncio.sleep(random.uniform(0.6, 1.5))
        if page.url != before:
            await page.go_back(timeout=15000)
            await asyncio.sleep(random.uniform(1.0, 2.4))
    except Exception:  # noqa: BLE001
        pass

    return cap


async def wait_for_cards(
    page: Any, selectors: list[str], timeout: float = 25.0  # noqa: ASYNC109
) -> tuple[list[Any], str]:
    """轮询等卡片出现，返回 (卡片, 命中的选择器)。

    不用 asyncio.timeout 包在外面（ruff ASYNC109 会建议那么做）：超时在这里
    是**正常结果不是异常** —— 页面没出卡片要返回空列表让调用方换选择器/换入口，
    抛 TimeoutError 会把「这个选择器没命中」和「浏览器挂了」混成同一件事。

    **不要用固定 settle 时间**：实测抖音热榜页 3.5s 时一个链接都没有，
    再等 3s 就有 20 个。页面加载快慢不定，固定等待要么白等要么等不够。
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        for sel in selectors:
            try:
                found = await page.query_selector_all(sel)
            except Exception:  # noqa: BLE001
                continue
            if len(found) >= 2:
                return found, sel
        await asyncio.sleep(1.5)
    return [], ""


async def browse_and_capture(
    session: Any,
    platform: str,
    card_selectors: list[str],
    count: int,
    pace: Pace,
) -> tuple[list[Captured], str]:
    """在列表页逐个点开内容，采集链接与详情。"""
    page = await session.page()
    budget = SessionBudget(pace)

    cards, hit_sel = await wait_for_cards(page, card_selectors)
    if not cards:
        return [], (
            "等了 25 秒列表页仍没有可点的内容卡片。"
            "可能是页面结构变了，或平台对自动化浏览器限制了内容加载。"
        )

    out: list[Captured] = []
    for idx in range(min(count, len(cards))):
        stop = budget.exhausted()
        if stop:
            return out, "" if out else f"未采到内容（{stop}）"

        # 每轮重新取卡片：点进去再退回来，之前的元素引用会失效。
        # 退回后页面要重新渲染，所以同样得等，不能直接查。
        fresh, _ = await wait_for_cards(page, [hit_sel, *card_selectors], timeout=15.0)
        if len(fresh) <= idx:
            break

        cap = await open_and_capture(session, fresh[idx], platform, pace)
        if cap and cap.best_link:
            out.append(cap)
            budget.items += 1
        await act_pause(pace, budget)

    if not out:
        return [], "点开了但一条链接都没取到"
    return out, ""
