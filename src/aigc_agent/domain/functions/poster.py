"""海报 / 图文卡片 function（M11 图文形态里的本地一步）。

  make_poster   把标题（和要点）叠到底图上，产出一张 PNG 资产。底图可以是 gen_image 的结果、
                素材库的图，或者不给底图用渐变。本地渲染，不花钱，不出网。

为什么不让生图模型直接画字：中文它画不对，改一个字就要重生成一次。本地叠字改字免费、
字体统一，AIGC 角标也顺手打上（平台要求画面显式标识）。
底图先用本地副本（产物目录里的 / Agent 在 blobs/ 自留的）；都没有才去抓远端临时外链，
抓下来记在 gen_params.blob。uri 不改 —— 改成本地路径，这张图以后当参考图会被当成
「不是公网链接」拒掉（2026-09-23 审查）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore, AssetType, local_copy
from ..media import poster as P
from ..media.ffmpeg import download

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


class PosterFunctions:
    name = "poster"
    namespaced = False
    disclosure = "full"

    def __init__(self, assets: AssetStore) -> None:
        self.assets = assets
        templates = "；".join(f"{k}：{v}" for k, v in P.TEMPLATES.items())
        self._specs = {
            "make_poster": ToolSpec(
                name="make_poster",
                summary="把标题/要点叠到底图上做成海报或图文卡片（本地渲染，不花钱）",
                permission=PermissionLevel.WRITE,
                description=(
                    "图文笔记的封面和内页用它出：先 gen_image 出一张**不带文字**的底图，"
                    "再用这个工具叠标题；改字免费，不用重新生图。不给底图就用渐变底。"
                    f"模板：{templates}。右上角自动打 AIGC 角标。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "标题，封面 ≤14 字为宜"},
                        "subtitle": {
                            "type": "string",
                            "description": "副标题或要点，多条用换行分隔（card 模板当正文）",
                        },
                        "template": {"type": "string", "enum": list(P.TEMPLATES)},
                        "aspect_ratio": {"type": "string", "enum": list(P.SIZES)},
                        "background_asset_id": {
                            "type": "string",
                            "description": "底图资产 id（image 类型）；留空用渐变底",
                        },
                        "brand": {"type": "string", "description": "账号名 / 水印，如 @野趣研究所"},
                        "accent": {"type": "string", "description": "强调色 #rrggbb，默认橙红"},
                        "aigc_label": {
                            "type": "string",
                            "description": "角标文字，默认「AI 生成」；传空串不打",
                        },
                        "summary": {"type": "string", "description": "这张图是干什么用的"},
                        "parent_id": {
                            "type": "string",
                            "description": "所属正文/方案资产 id，用于血缘",
                        },
                    },
                    "required": ["title"],
                },
            ),
        }

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        font = P.find_font()
        name = font.name if font else "默认（无中文）"
        return ProviderHealth(ok=True, detail=f"字体 {name} · 模板 {', '.join(P.TEMPLATES)}")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            result = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    # ---------- 实现 ----------

    async def _fn_make_poster(
        self,
        title: str,
        subtitle: str = "",
        template: str = "clean",
        aspect_ratio: str = "3:4",
        background_asset_id: str = "",
        brand: str = "",
        accent: str = "",
        aigc_label: str = "AI 生成",
        summary: str = "",
        parent_id: str = "",
    ) -> ToolResult:
        parents: list[str] = []
        background: bytes | None = None
        if background_asset_id:
            background = await self._background(background_asset_id)
            parents.append(background_asset_id)
        if parent_id and parent_id not in parents:
            parents.append(parent_id)

        spec = P.PosterSpec(
            title=(title or "").strip(),
            subtitle=subtitle or "",
            template=template or "clean",
            aspect_ratio=aspect_ratio or "3:4",
            brand=brand or "",
            accent=P.parse_color(accent or "", P.DEFAULT_ACCENT),
            background=background,
            aigc_label=aigc_label if aigc_label is not None else "AI 生成",
        )
        r = P.render(spec)
        asset = self.assets.create_blob(
            r.png,
            ".png",
            type_=AssetType.IMAGE,
            mime="image/png",
            summary=summary or f"海报：{spec.title[:20]}",
            parents=parents,
            creator="tool:make_poster",
            gen_params={
                "template": r.template,
                "aspect_ratio": spec.aspect_ratio,
                "title": spec.title,
                "subtitle": spec.subtitle,
                "brand": spec.brand,
                "font": r.font,
                "title_lines": r.title_lines,
                "background": background_asset_id,
                "warnings": r.warnings,
            },
        )
        note = (
            f"海报已生成 {asset.id} → {asset.uri}"
            f"（{r.width}×{r.height} · {r.template} · 标题 {r.title_lines} 行 · 字体 {r.font}）"
        )
        if r.warnings:
            note += "\n注意：" + "；".join(r.warnings)
        return ToolResult(content=note, asset_ref=asset.id)

    async def _background(self, asset_id: str) -> bytes:
        a = self.assets.get(asset_id)
        if a.type is not AssetType.IMAGE:
            raise ValueError(f"{asset_id} 是 {a.type.value} 资产，底图得是 image")
        # 之前不看本地副本、直接重下远端链接：生成超过约 24 小时链接就过期，必然失败
        local = local_copy(a)
        if local is not None:
            return local.read_bytes()
        uri = a.uri or ""
        if uri.startswith(("http://", "https://")):
            ext = Path(urlparse(uri).path).suffix.lower()
            target = self.assets.blob_dir / f"{a.id}{ext if ext in _IMAGE_EXT else '.png'}"
            ok, err = await download(uri, target)
            if not ok:
                raise RuntimeError(
                    f"底图 {asset_id} 没有本地副本，远端链接也下载失败"
                    f"（生成的链接约 24 小时过期）：{err}"
                )
            # 抓下来的记在 blob（local_copy 认它）；uri 保持远端链接，参考图链还要用
            a.gen_params = {**a.gen_params, "blob": str(target)}
            self.assets.put(a)
            return target.read_bytes()
        raise FileNotFoundError(f"{asset_id} 没有可用的图片文件：{uri or '（无 uri）'}")
