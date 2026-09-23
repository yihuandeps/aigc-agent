"""热点选题 functions —— 让内容长在真实热度上，而不是凭空编。

**为什么这层重要**：没有它，"做一条科技热点视频"里的"热点"只能靠模型
按常识编，产出必然空泛。接上真实热榜之后，选题、文案、分镜才有根。

数据来源分工（实测 2026-09-11）：

| 平台 | 来源 | 状态 |
|---|---|---|
| 抖音热榜 | TikHub `billboard/fetch_hot_total_list` | ✅ 免费额度可用 |
| 抖音搜索 | TikHub `search/fetch_video_search_v3` | ✅ |
| 小红书 | TikHub 全部端点 | ❌ **402 需付费余额**，走 RPA |

抖音榜单必须用 `type=range`（按日期区间）。`type=snapshot` 那条路实测
永远返回空数组 —— 参数合法、HTTP 200、objs 为 []，很容易被当成"没热点"。
"""

from __future__ import annotations

import datetime as dt
import json
import time
from typing import Any

import httpx2 as httpx

from ...harness.model.media import default_proxy
from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore, AssetType

BASE = "https://api.tikhub.io/api/v1"
TIMEOUT = 60.0


class HotspotFunctions:
    name = "hotspot"
    namespaced = False
    disclosure = "full"

    def __init__(self, store: AssetStore, api_key: str) -> None:
        self.store = store
        self.api_key = api_key
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    def _build(self) -> None:
        self._specs["douyin_hot_list"] = ToolSpec(
            name="douyin_hot_list",
            summary="拉抖音实时热榜（真实热度数据）",
            permission=PermissionLevel.COMPUTE,
            description=(
                "做任何「热点」内容之前**先调这个**，不要凭印象编热点。\n"
                "返回热点词 + 热度值 + 分类标签。category 可筛分类（如 科技/娱乐/社会）。\n"
                "拿到热点词后，用 douyin_search_videos 看这个词下面实际在火什么样的视频。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "取几条，默认 20"},
                    "days": {"type": "integer", "description": "回溯天数，默认 1（今天）"},
                    "keyword": {"type": "string", "description": "按热词搜索（传给接口），可省略"},
                    "category": {
                        "type": "string",
                        "description": "按分类筛，如 科技/体育/财经/军事。本地过滤，可省略",
                    },
                },
            },
        )

        self._specs["douyin_search_videos"] = ToolSpec(
            name="douyin_search_videos",
            summary="按关键词搜抖音视频（⚠️ 需 TikHub 付费余额）",
            permission=PermissionLevel.COMPUTE,
            description=(
                "⚠️ **该端点不接受免费额度，需 TikHub 充值**，没余额会直接报错。\n"
                "拿到热点词之后用这个看**实际在火的视频长什么样**：标题怎么写、"
                "数据多少、什么角度切入。这比只看一个热点词有用得多。\n"
                "挑出高赞的几条，再用 fetch_douyin 做深度拆解。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "count": {"type": "integer", "description": "默认 10"},
                    "sort": {
                        "type": "string",
                        "enum": ["综合", "最新", "最多点赞"],
                        "description": "默认最多点赞",
                    },
                },
                "required": ["keyword"],
            },
        )

    # ---------- ToolProvider ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        if not self.api_key:
            return ProviderHealth(ok=False, detail="未配 TIKHUB_API_KEY，热榜不可用")
        return ProviderHealth(ok=True, detail="抖音热榜/搜索可用（小红书需付费余额，走 RPA）")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            r = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            r = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        r.duration_ms = int((time.perf_counter() - started) * 1000)
        return r

    # ---------- HTTP ----------

    async def _get(self, path: str, params: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if not self.api_key:
            return {}, "未配 TIKHUB_API_KEY"
        async with httpx.AsyncClient(timeout=TIMEOUT, proxy=default_proxy()) as c:
            resp = await c.get(
                f"{BASE}{path}", params=params,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            return {}, f"HTTP {resp.status_code}，响应不是 JSON"

        if resp.status_code == 402:
            return {}, "TikHub 余额不足：该端点不接受免费额度，需充值"
        if resp.status_code >= 400:
            detail = data.get("detail") or {}
            msg = detail.get("message_zh") or detail.get("message") or str(data)[:150]
            return {}, f"HTTP {resp.status_code}：{msg}"
        return data, ""

    async def _post(self, path: str, body: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if not self.api_key:
            return {}, "未配 TIKHUB_API_KEY"
        async with httpx.AsyncClient(timeout=TIMEOUT, proxy=default_proxy()) as c:
            resp = await c.post(
                f"{BASE}{path}", json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            return {}, f"HTTP {resp.status_code}，响应不是 JSON"
        if resp.status_code == 402:
            return {}, "TikHub 余额不足：该端点需付费余额"
        if resp.status_code >= 400:
            detail = data.get("detail") or {}
            msg = detail.get("message_zh") or detail.get("message") or str(data)[:150]
            return {}, f"HTTP {resp.status_code}：{msg}"
        return data, ""

    # ---------- 实现 ----------

    async def _fn_douyin_hot_list(
        self, count: int = 20, days: int = 1, keyword: str = "", category: str = ""
    ) -> ToolResult:
        today = dt.date.today()
        # 至少回溯 1 天：start==end（只查今天）实测恒返回空数组，
        # 榜单是按区间聚合的，单日边界拿不到数据。
        start = (today - dt.timedelta(days=max(1, int(days or 1)))).strftime("%Y%m%d")
        params = {
            "page": 1,
            # 分类是**本地过滤**（接口的 keyword 只搜热词标题，搜"科技"一条都匹配
            # 不到），所以筛分类时先拉满一页再筛，否则筛完常常一条不剩。
            "page_size": 50 if category else max(1, min(int(count or 20), 50)),
            # 必须用 range —— snapshot 模式实测恒返回空数组
            "type": "range",
            "start_date": start,
            "end_date": today.strftime("%Y%m%d"),
        }
        if keyword:
            params["keyword"] = keyword

        data, err = await self._get("/douyin/billboard/fetch_hot_total_list", params)
        if err:
            return ToolResult(ok=False, error=err)

        objs = ((data.get("data") or {}).get("data") or {}).get("objs") or []
        if not objs:
            hint = "去掉 keyword 再试" if keyword else "把 days 调大一点"
            return ToolResult(
                ok=False,
                error=f"热榜返回空（{start}~{params['end_date']}）。{hint}。",
            )

        if category:
            objs = [o for o in objs if category in _tag(o)]
            if not objs:
                return ToolResult(
                    ok=False,
                    error=f"热榜里没有 [{category}] 分类的词条，换个分类或去掉它再试。",
                )
            objs = objs[: max(1, min(int(count or 20), 50))]

        lines = []
        for i, o in enumerate(objs, 1):
            word = o.get("word") or o.get("sentence") or "?"
            hot = o.get("hot_score") or o.get("hot_value") or 0
            tag = _tag(o)
            lines.append(f"{i:2}. {word}  热度 {_num(hot)}" + (f"  [{tag}]" if tag else ""))

        body = "\n".join(lines)
        asset = self.store.create(
            body,
            type_=AssetType.TEXT,
            summary=f"抖音热榜·{today.strftime('%m-%d')}·{len(objs)}条",
            creator="tool:douyin_hot_list",
            gen_params={
                "start": start,
                "end": params["end_date"],
                "keyword": keyword,
                "category": category,
            },
        )
        return ToolResult(
            content=f"抖音热榜（{start}~{params['end_date']}，{len(objs)} 条）\n{body}",
            asset_ref=asset.id,
        )

    async def _fn_douyin_search_videos(
        self, keyword: str, count: int = 10, sort: str = "最多点赞"
    ) -> ToolResult:
        sort_map = {"综合": "0", "最新": "1", "最多点赞": "2"}
        data, err = await self._post(
            "/douyin/search/fetch_video_search_v3",
            {
                "keyword": keyword,
                "count": max(1, min(int(count or 10), 30)),
                "offset": 0,
                "sort_type": sort_map.get(sort, "2"),
                "publish_time": "0",
            },
        )
        if err:
            return ToolResult(ok=False, error=err)

        vids = _collect_videos(data)
        if not vids:
            return ToolResult(ok=False, error=f"「{keyword}」没搜到视频，换个词试试")

        lines = []
        for i, v in enumerate(vids[: int(count or 10)], 1):
            st = v.get("statistics") or {}
            lines.append(
                f"{i:2}. {str(v.get('desc') or '')[:44]}\n"
                f"     赞 {_num(st.get('digg_count'))} 评 {_num(st.get('comment_count'))} "
                f"藏 {_num(st.get('collect_count'))} 转 {_num(st.get('share_count'))}"
                f"  id={v.get('aweme_id')}"
            )

        body = "\n".join(lines)
        asset = self.store.create(
            body,
            type_=AssetType.TEXT,
            summary=f"搜索·{keyword}·{len(lines)}条",
            creator="tool:douyin_search_videos",
            gen_params={"keyword": keyword, "sort": sort},
        )
        return ToolResult(
            content=(
                f"「{keyword}」搜索结果（按{sort}，{len(lines)} 条）\n{body}\n\n"
                f"想深挖某条：fetch_douyin(ref=\"<aweme_id>\")"
            ),
            asset_ref=asset.id,
        )


def _collect_videos(payload: Any) -> list[dict[str, Any]]:
    """从嵌套响应里捞视频条目。各版本接口结构不同，认字段不认路径。"""
    out: list[dict[str, Any]] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            if "aweme_id" in o and ("desc" in o or "statistics" in o):
                out.append(o)
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(payload)
    return out


def _num(n: Any) -> str:
    try:
        v = int(n or 0)
    except (TypeError, ValueError):
        return str(n or "-")
    if v >= 10000:
        return f"{v / 10000:.1f}w"
    return str(v)


def dump(obj: Any) -> str:  # pragma: no cover - 调试用
    return json.dumps(obj, ensure_ascii=False, indent=2)[:2000]


def _tag(o: dict) -> str:
    return str(o.get("sentence_tag_name") or o.get("label") or "")
