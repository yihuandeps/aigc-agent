"""联网找可用素材（2026-09-18，抖音分支的第二条素材路径）。

用户定的规则：镜头需要实拍/真实画面时，① 向用户要，② 联网抓可用素材。
"可用"在这里是硬约束 —— 只接**明确可商用**的素材站（Pexels / Pixabay，两家都免费、
无需署名），抖音/小红书上别人的视频只拿来分析，不进成片。

工具：
  stock_media_search  按关键词搜视频/图片，返回可下载的候选（带时长、尺寸、作者、授权）
  fetch_stock_media   下载某条候选到 workspace/materials/online/ 并登记成资产（授权信息记进资产）
  fetch_media_url     下载任意链接（授权 unknown，结果里明确提醒需人确认权利）
  web_fetch           抓一个网页的正文文字（补事实用）

需要的 key（.env）：PEXELS_API_KEY（pexels.com/api 免费申请）、
PIXABAY_API_KEY（pixabay.com/api/docs）。
"""

from __future__ import annotations

import html
import mimetypes
import os
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
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
from ..media import ffmpeg
from ..media.naming import safe_name

HttpGet = Callable[[str, dict[str, str], dict[str, Any]], Awaitable[tuple[int, Any, str]]]
Downloader = Callable[[str, Path], Awaitable[tuple[bool, str]]]

LICENSES = {
    "pexels": "Pexels License（可免费商用，无需署名）",
    "pixabay": "Pixabay Content License（可免费商用，无需署名）",
}


async def _default_get(
    url: str, headers: dict[str, str], params: dict[str, Any]
) -> tuple[int, Any, str]:
    async with httpx.AsyncClient(
        timeout=30.0, proxy=default_proxy(), follow_redirects=True, headers=headers
    ) as c:
        resp = await c.get(url, params=params)
    data: Any = None
    if "json" in (resp.headers.get("content-type") or ""):
        try:
            data = resp.json()
        except ValueError:
            data = None
    return resp.status_code, data, resp.text


def _orientation_ok(width: int, height: int, orientation: str) -> bool:
    """素材方向对不对得上。量不出尺寸的放过（交给合成时裁切铺满）。"""
    if not width or not height or orientation not in ("portrait", "landscape", "square"):
        return True
    if orientation == "portrait":
        return height > width
    if orientation == "landscape":
        return width > height
    return abs(width - height) <= max(width, height) * 0.1


def _pixels(f: dict[str, Any]) -> int:
    return int(f.get("width") or 0) * int(f.get("height") or 0)


def strip_html(text: str, max_chars: int = 6000) -> tuple[str, str]:
    """网页 → (标题, 正文文字)。去脚本/样式/标签，合并空白。"""
    t = text or ""
    m = re.search(r"<title[^>]*>(.*?)</title>", t, re.I | re.S)
    title = html.unescape(re.sub(r"\s+", " ", m.group(1))).strip() if m else ""
    t = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", t, flags=re.I | re.S)
    t = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</h\d>|</tr>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    lines = [re.sub(r"[ \t　]+", " ", ln).strip() for ln in t.splitlines()]
    body = "\n".join(ln for ln in lines if ln)
    return title, body[:max_chars]


class MaterialFunctions:
    name = "materials"
    namespaced = False

    def __init__(
        self,
        store: AssetStore,
        workspace: Path,
        http_get: HttpGet | None = None,
        downloader: Downloader | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.store = store
        # 项目画幅（/ratio）对应的方向：没传 orientation 时用它；空 = 竖屏
        self.default_orientation = ""
        self.root = Path(workspace) / "materials" / "online"
        self._get = http_get or _default_get
        self._download = downloader or ffmpeg.download
        self._env = env if env is not None else os.environ  # type: ignore[assignment]
        self._found: dict[str, dict[str, Any]] = {}  # 搜索结果句柄 → 下载信息
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    # ---------- 提权 ----------

    def permission_for(
        self, tool: str, args: dict[str, Any]
    ) -> tuple[PermissionLevel, str] | None:
        """注册表的提权钩子：给任意链接下来的素材填授权说明时，当场问人。

        填了授权，版权状态就从 unknown 变成 licensed，机审放它进成片 —— 而 license 是模型
        自己就能填的参数（2026-09-26 审查）。授权是真是假只有人知道。不填（记为 unknown）
        不用问。"""
        if tool != "fetch_media_url":
            return None
        lic = str(args.get("license") or "").strip()
        if not lic or lic.lower().startswith("unknown") or "来源不明" in lic:
            return None
        return (
            PermissionLevel.EXTERNAL,
            f"把这条素材的授权记为「{lic[:50]}」：记上之后机审按已授权处理、可以进成片 —— "
            "授权是不是真的只有你知道，需要你确认",
        )

    # ---------- 声明 ----------

    def _build(self) -> None:
        self._specs["stock_media_search"] = ToolSpec(
            name="stock_media_search",
            summary="在可商用素材站（Pexels / Pixabay）搜视频或图片，返回可下载的候选",
            permission=PermissionLevel.READ,
            description=(
                "镜头需要真实画面、用户又没有素材时用。关键词用英文效果好得多。"
                "返回带句柄的候选列表（时长、尺寸、作者、授权），挑一条交给 fetch_stock_media。"
                "**别人的抖音/小红书视频不能当素材**，只能分析。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "英文关键词，如 city night traffic"},
                    "kind": {
                        "type": "string",
                        "enum": ["video", "image"],
                        "description": "默认 video",
                    },
                    "count": {"type": "integer", "description": "每个站取几条，默认 5"},
                    "orientation": {
                        "type": "string",
                        "enum": ["portrait", "landscape", "square", "any"],
                        "description": "留空按项目画幅（用户用 /ratio 选的；默认竖屏 portrait）",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["auto", "pexels", "pixabay"],
                        "description": "默认 auto：配了 key 的站都搜",
                    },
                },
                "required": ["query"],
            },
            max_result_chars=8000,
        )
        self._specs["fetch_stock_media"] = ToolSpec(
            name="fetch_stock_media",
            summary="下载 stock_media_search 里的某条候选，登记成资产（授权信息随资产保存）",
            permission=PermissionLevel.WRITE,
            parameters={
                "type": "object",
                "properties": {
                    "handle": {"type": "string", "description": "候选句柄，如 pexels:v:12345"},
                    "name": {"type": "string", "description": "文件名/资产摘要，可省略"},
                },
                "required": ["handle"],
            },
        )
        self._specs["fetch_media_url"] = ToolSpec(
            name="fetch_media_url",
            summary="下载任意图片/视频链接并登记成资产（授权未知，需用户确认可用）",
            permission=PermissionLevel.WRITE,
            description=(
                "只在用户明确给了链接、或链接来自用户自己的账号/官方物料时用。授权记为 unknown。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "name": {"type": "string", "description": "文件名/资产摘要，可省略"},
                    "license": {
                        "type": "string",
                        "description": "用户说明过的授权（如「用户自己拍的」「品牌方官方物料」），"
                        "可省略 = 记为 unknown。填了会当场问用户确认",
                    },
                },
                "required": ["url"],
            },
        )
        self._specs["web_fetch"] = ToolSpec(
            name="web_fetch",
            summary="抓一个网页的正文文字（补事实、看新闻原文）",
            permission=PermissionLevel.READ,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "description": "最多返回多少字，默认 6000"},
                },
                "required": ["url"],
            },
            max_result_chars=16_000,
        )

    # ---------- ToolProvider ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        keys = [s for s in ("pexels", "pixabay") if self._key(s)]
        if not keys:
            return ProviderHealth(
                ok=False, detail="没配 PEXELS_API_KEY / PIXABAY_API_KEY，素材站不可用"
            )
        return ProviderHealth(ok=True, detail="素材站：" + "、".join(keys))

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            result = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    def _key(self, source: str) -> str:
        return str(self._env.get(f"{source.upper()}_API_KEY") or "").strip()

    # ---------- 搜索 ----------

    async def _fn_stock_media_search(
        self,
        query: str,
        kind: str = "video",
        count: int = 5,
        orientation: str = "",
        source: str = "auto",
    ) -> ToolResult:
        orientation = orientation or self.default_orientation or "portrait"
        if not query.strip():
            return ToolResult(ok=False, error="query 是空的")
        kind = "image" if kind == "image" else "video"
        n = max(1, min(int(count or 5), 20))
        sources = ["pexels", "pixabay"] if source in ("", "auto") else [source]
        sources = [s for s in sources if self._key(s)]
        if not sources:
            return ToolResult(
                ok=False,
                error=(
                    "没配素材站的 key。.env 里加 PEXELS_API_KEY（pexels.com/api 免费申请）"
                    "或 PIXABAY_API_KEY（pixabay.com/api/docs），重启后可用"
                ),
            )
        items: list[dict[str, Any]] = []
        errors: list[str] = []
        for s in sources:
            try:
                found, err = await (
                    self._search_pexels(query, kind, n, orientation)
                    if s == "pexels"
                    else self._search_pixabay(query, kind, n, orientation)
                )
            except Exception as e:  # noqa: BLE001
                found, err = [], f"{type(e).__name__}: {e}"
            if err:
                errors.append(f"{s}：{err}")
            items += found
        if not items:
            return ToolResult(
                ok=False, error="没搜到可用素材" + ("；" + "；".join(errors) if errors else "")
            )
        lines = [f"「{query}」{'视频' if kind == 'video' else '图片'}候选 {len(items)} 条："]
        for it in items:
            self._found[it["handle"]] = it
            dur = f"{it['duration']:.0f}s · " if it.get("duration") else ""
            lines.append(
                f"  {it['handle']} · {dur}{it['width']}x{it['height']} · by {it['author']} · "
                f"{it['license']}\n      {it['page']}"
            )
        lines.append("挑一条：fetch_stock_media(handle=\"…\")；都不合适就换关键词再搜。")
        if errors:
            lines.append("⚠ " + "；".join(errors))
        return ToolResult(content="\n".join(lines))

    async def _search_pexels(
        self, query: str, kind: str, n: int, orientation: str
    ) -> tuple[list[dict[str, Any]], str]:
        headers = {"Authorization": self._key("pexels")}
        params: dict[str, Any] = {"query": query, "per_page": n}
        if orientation in ("portrait", "landscape", "square"):
            params["orientation"] = orientation
        url = "https://api.pexels.com/videos/search" if kind == "video" else "https://api.pexels.com/v1/search"
        status, data, text = await self._get(url, headers, params)
        if status != 200 or not isinstance(data, dict):
            return [], f"HTTP {status} {text[:120]}"
        out: list[dict[str, Any]] = []
        if kind == "video":
            for v in data.get("videos") or []:
                files = [
                    f for f in (v.get("video_files") or [])
                    if str(f.get("file_type") or "").endswith("mp4")
                    and int(f.get("height") or 0) <= 1920
                ]
                if not files:
                    continue
                best = max(files, key=_pixels)
                out.append({
                    "handle": f"pexels:v:{v.get('id')}", "kind": "video", "source": "pexels",
                    "url": best.get("link"), "ext": ".mp4",
                    "duration": float(v.get("duration") or 0),
                    "width": int(best.get("width") or 0), "height": int(best.get("height") or 0),
                    "author": str((v.get("user") or {}).get("name") or ""),
                    "page": str(v.get("url") or ""),
                    "license": LICENSES["pexels"], "query": query,
                })
        else:
            for p in data.get("photos") or []:
                src = p.get("src") or {}
                out.append({
                    "handle": f"pexels:p:{p.get('id')}", "kind": "image", "source": "pexels",
                    "url": src.get("large2x") or src.get("original"), "ext": ".jpg", "duration": 0,
                    "width": int(p.get("width") or 0), "height": int(p.get("height") or 0),
                    "author": str(p.get("photographer") or ""), "page": str(p.get("url") or ""),
                    "license": LICENSES["pexels"], "query": query,
                })
        return out, ""

    async def _search_pixabay(
        self, query: str, kind: str, n: int, orientation: str
    ) -> tuple[list[dict[str, Any]], str]:
        params: dict[str, Any] = {
            "key": self._key("pixabay"), "q": query, "per_page": max(3, n), "safesearch": "true",
        }
        if kind == "video":
            url = "https://pixabay.com/api/videos/"
            # 视频接口不支持 orientation：多要一些，拿回来按宽高自己筛（2026-09-23 审查：横屏
            # 素材混进竖屏成片，第 1 镜是横屏时整片都成了横屏）
            if orientation in ("portrait", "landscape", "square"):
                params["per_page"] = max(20, n * 4)
        else:
            url = "https://pixabay.com/api/"
            params["image_type"] = "photo"
            if orientation in ("portrait", "landscape"):
                params["orientation"] = "vertical" if orientation == "portrait" else "horizontal"
        status, data, text = await self._get(url, {}, params)
        if status != 200 or not isinstance(data, dict):
            return [], f"HTTP {status} {text[:120]}"
        out: list[dict[str, Any]] = []
        for h in data.get("hits") or []:
            if len(out) >= n:
                break
            if kind == "video":
                vids = h.get("videos") or {}
                f = vids.get("large") or vids.get("medium") or vids.get("small") or {}
                if not f.get("url"):
                    continue
                if not _orientation_ok(int(f.get("width") or 0), int(f.get("height") or 0),
                                       orientation):
                    continue
                out.append({
                    "handle": f"pixabay:v:{h.get('id')}", "kind": "video", "source": "pixabay",
                    "url": f.get("url"), "ext": ".mp4", "duration": float(h.get("duration") or 0),
                    "width": int(f.get("width") or 0), "height": int(f.get("height") or 0),
                    "author": str(h.get("user") or ""), "page": str(h.get("pageURL") or ""),
                    "license": LICENSES["pixabay"], "query": query,
                })
            else:
                if not h.get("largeImageURL"):
                    continue
                out.append({
                    "handle": f"pixabay:p:{h.get('id')}", "kind": "image", "source": "pixabay",
                    "url": h.get("largeImageURL"), "ext": ".jpg", "duration": 0,
                    "width": int(h.get("imageWidth") or 0),
                    "height": int(h.get("imageHeight") or 0),
                    "author": str(h.get("user") or ""), "page": str(h.get("pageURL") or ""),
                    "license": LICENSES["pixabay"], "query": query,
                })
        return out, ""

    # ---------- 下载登记 ----------

    async def _fn_fetch_stock_media(self, handle: str, name: str = "") -> ToolResult:
        it = self._found.get(handle.strip())
        if it is None:
            if handle.startswith(("http://", "https://")):
                return await self._fn_fetch_media_url(handle, name)
            return ToolResult(ok=False, error=f"不认识 {handle!r}，先 stock_media_search 再挑")
        base = safe_name(name or f"{it['query']}_{it['handle'].replace(':', '_')}")
        target = self.root / it["source"] / f"{base}{it['ext']}"
        ok, why = await self._download(it["url"], target)
        if not ok:
            return ToolResult(ok=False, error=f"下载失败：{why}")
        a = self._register(
            target, it["kind"], name or f"{it['query']}·{it['source']}",
            {"source": it["source"], "license": it["license"], "author": it["author"],
             "source_url": it["page"], "query": it["query"], "duration": it.get("duration", 0)},
        )
        lic = f"授权：{it['license']}，作者 {it['author']}"
        return ToolResult(content=f"已下载并登记为资产 {a.id}：{target}\n{lic}", asset_ref=a.id)

    async def _fn_fetch_media_url(self, url: str, name: str = "", license: str = "") -> ToolResult:  # noqa: A002
        if not url.startswith(("http://", "https://")):
            return ToolResult(ok=False, error="url 要以 http(s):// 开头")
        ext = Path(url.split("?")[0]).suffix.lower() or ".bin"
        kind = "image" if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif") else "video"
        base = safe_name(name or Path(url.split("?")[0]).stem or "素材")
        target = self.root / "url" / f"{base}{ext}"
        ok, why = await self._download(url, target)
        if not ok:
            return ToolResult(ok=False, error=f"下载失败：{why}")
        lic = license or "unknown（来源不明，用进成片前请确认权利）"
        meta = {"source": "url", "license": lic, "source_url": url}
        a = self._register(target, kind, name or base, meta)
        return ToolResult(
            content=f"已下载并登记为资产 {a.id}：{target}\n⚠ 授权：{lic}", asset_ref=a.id
        )

    def _register(self, path: Path, kind: str, summary: str, meta: dict[str, Any]) -> Any:
        type_ = AssetType.IMAGE if kind == "image" else AssetType.VIDEO
        a = self.store.create(
            "", type_=type_, summary=summary, creator="tool:fetch_stock_media",
            gen_params={"local": str(path), **meta},
        )
        a.uri = str(path)
        fallback = "image/jpeg" if kind == "image" else "video/mp4"
        a.mime = mimetypes.guess_type(path.name)[0] or fallback
        return self.store.put(a)

    # ---------- 网页 ----------

    async def _fn_web_fetch(self, url: str, max_chars: int = 6000) -> ToolResult:
        if not url.startswith(("http://", "https://")):
            return ToolResult(ok=False, error="url 要以 http(s):// 开头")
        status, data, text = await self._get(url, {"User-Agent": "Mozilla/5.0"}, {})
        if status >= 400:
            return ToolResult(ok=False, error=f"HTTP {status}")
        if isinstance(data, (dict, list)):
            body = str(data)[: max(500, int(max_chars or 6000))]
            return ToolResult(content=f"{url}\n{body}")
        title, body = strip_html(text, max(500, int(max_chars or 6000)))
        return ToolResult(content=f"{title or url}\n{body}")
