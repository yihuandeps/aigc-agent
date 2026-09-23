"""平台采集器 —— 从页面里抠出热点内容。

**选择器一定会失效**，两个平台都在频繁改版。所以：
  · 每个字段给多个候选选择器，逐个试
  · 抠不到就返回空，让上层如实告诉用户「页面结构变了」
  · **绝不返回编造的占位数据** —— 假热点比没热点更糟
"""

from __future__ import annotations

import re
from typing import Any

from .browser import BrowserSession, Post
from .humanize import Pace, SessionBudget, act_pause, move_like_reading, page_cooldown

XHS_EXPLORE = "https://www.xiaohongshu.com/explore"
XHS_SEARCH = "https://www.xiaohongshu.com/search_result?keyword={kw}"
DY_HOT = "https://www.douyin.com/hot"
DY_SEARCH = "https://www.douyin.com/search/{kw}"

# 笔记卡片的候选选择器，按命中概率排序
_XHS_CARD = ["section.note-item", "div.note-item", "a.cover", "section[class*=note]"]
_XHS_TITLE = ["span[class*=title]", "div.footer .title", "a.title", "[class*=title]"]
_XHS_AUTHOR = ["span.name", "div.author .name", "[class*=author] [class*=name]"]
_XHS_LIKE = ["span.count", "[class*=like] span", "[class*=count]"]


async def _text(node: Any, selectors: list[str]) -> str:
    for sel in selectors:
        try:
            el = await node.query_selector(sel)
            if el:
                t = (await el.inner_text()).strip()
                if t:
                    return " ".join(t.split())
        except Exception:  # noqa: BLE001
            continue
    return ""


async def collect_xiaohongshu(
    session: BrowserSession,
    keyword: str = "",
    limit: int = 20,
    scrolls: int = 3,
    pace: Pace | None = None,
) -> tuple[list[Post], str]:
    """采集小红书。keyword 为空则抓发现页推荐流。

    小红书是 TikHub 免费额度不覆盖的平台（全部 402），所以这里是唯一通路。
    """
    pace = pace or Pace()
    budget = SessionBudget(pace)

    url = XHS_SEARCH.format(kw=keyword) if keyword else XHS_EXPLORE
    await page_cooldown(pace)  # 打开页面前先歇一下，别一上来就冲
    page = await session.goto(url)
    await move_like_reading(page, pace)

    content = await page.content()
    if any(h in content for h in ("扫码登录", "登录后查看", "立即登录")):
        return [], "小红书要求登录。用 `agent rpa login xhs` 扫码，登录态会记住。"

    await session.scroll(page, scrolls, pace)

    cards: list[Any] = []
    for sel in _XHS_CARD:
        cards = await page.query_selector_all(sel)
        if len(cards) >= 3:
            break
    if not cards:
        return [], "没抠到笔记卡片 —— 页面结构大概率变了，需要更新选择器"

    posts: list[Post] = []
    seen: set[str] = set()
    for c in cards[: limit * 2]:
        stop = budget.exhausted()
        if stop:
            return posts, "" if posts else f"未采到内容（{stop}）"
        title = await _text(c, _XHS_TITLE)
        if not title:
            continue
        href = ""
        try:
            a = await c.query_selector("a[href*='/explore/'], a[href*='/search_result/']")
            if a:
                href = await a.get_attribute("href") or ""
        except Exception:  # noqa: BLE001
            pass
        if href.startswith("/"):
            href = "https://www.xiaohongshu.com" + href
        nid = _xhs_id(href)
        if nid and nid in seen:
            continue
        if nid:
            seen.add(nid)
        posts.append(
            Post(
                platform="xiaohongshu",
                title=title,
                url=href,
                post_id=nid,
                author=await _text(c, _XHS_AUTHOR),
                likes=await _text(c, _XHS_LIKE),
            )
        )
        budget.items += 1
        if len(posts) >= limit:
            break
        # 每抠一条歇一下 —— 连续抠几十条不停是很典型的机器行为
        await act_pause(pace, budget)

    if not posts:
        return [], "卡片抠到了但标题全是空 —— 选择器要更新"
    return posts, ""


async def collect_douyin_hot(
    session: BrowserSession, limit: int = 20, pace: Pace | None = None
) -> tuple[list[Post], str]:
    """采集抖音热榜页。

    注意：抖音热榜**优先走 TikHub API**（douyin_hot_list），稳定得多。
    这条 RPA 路径只在 API 不可用时兜底，或者你要看登录态下的个性化榜单。
    """
    pace = pace or Pace()
    await page_cooldown(pace)
    page = await session.goto(DY_HOT)
    await move_like_reading(page, pace)
    await session.scroll(page, 2, pace)

    # 注意：别用 li[class*=item] 这类泛选择器 —— 实测会命中左侧导航菜单，
    # 抓回来一堆「发布视频/视频管理/作品数据」。认视频链接最稳。
    posts: list[Post] = []
    for sel in ['a[href*="/video/"]', 'div[class*=hot-list] li', 'div[class*=rank] li']:
        nodes = await page.query_selector_all(sel)
        if len(nodes) < 3:
            continue
        for n in nodes[:limit]:
            try:
                t = " ".join((await n.inner_text()).split())
                href = await n.get_attribute("href") or ""
            except Exception:  # noqa: BLE001
                continue
            if not t or len(t) < 4:  # 太短的多半是图标/角标
                continue
            hot = ""
            m = re.search(r"(\d+(?:\.\d+)?[万亿]?)$", t)
            if m:
                hot = m.group(1)
                t = t[: m.start()].strip()
            if href.startswith("/"):
                href = "https://www.douyin.com" + href
            posts.append(Post(platform="douyin", title=t[:60], likes=hot, url=href))
        if posts:
            break

    if not posts:
        return [], "没抠到热榜条目 —— 建议改用 douyin_hot_list（走 API，更稳）"
    return posts, ""


def _xhs_id(href: str) -> str:
    m = re.search(r"/(?:explore|discovery/item|search_result)/([0-9a-f]{16,})", href)
    return m.group(1) if m else ""
