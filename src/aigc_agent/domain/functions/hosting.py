"""托管 functions（2026-09-19）—— 把本地素材变成生成接口能用的参考链接。

  hosting_status  看配了哪种托管、能不能用
  host_file       本地文件 → 登记成资产 + 上传拿链接（之后可作 gen_image / gen_video 参考）
  host_asset      已登记的资产（有本地副本）→ 上传拿链接 / 刷新过期链接
"""

from __future__ import annotations

import time
from typing import Any

from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore
from ..media.hosting import Hosting


class HostingFunctions:
    name = "hosting"
    namespaced = False

    def __init__(self, store: AssetStore, hosting: Hosting, files: Any = None) -> None:
        self.store = store
        self.hosting = hosting
        self.files = files  # FileFunctions：本地路径走同一套边界与登记
        self._specs = {
            "hosting_status": ToolSpec(
                name="hosting_status",
                summary="看本地素材托管配了没有（本地图片/视频要当参考必须先能上传）",
                permission=PermissionLevel.READ,
                parameters={"type": "object", "properties": {}},
            ),
            "host_file": ToolSpec(
                name="host_file",
                summary="把本地图片/视频登记成资产并上传到你的存储，拿到可作参考的公网链接",
                permission=PermissionLevel.WRITE,
                description=(
                    "用户要用自己电脑上的图/视频当参考（角色脸、产品、场景）时用。"
                    "返回资产 id 和链接；短剧里用 drama_use_local_ref 直接放进参考图包更省事。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "本地文件路径"},
                        "summary": {"type": "string", "description": "一句话说明，可省略"},
                    },
                    "required": ["path"],
                },
            ),
            "host_asset": ToolSpec(
                name="host_asset",
                summary="给已登记的资产上传本地副本拿公网链接（也用来刷新过期链接）",
                permission=PermissionLevel.WRITE,
                parameters={
                    "type": "object",
                    "properties": {
                        "asset_id": {"type": "string"},
                        "force": {
                            "type": "boolean",
                            "description": "链接还新鲜也重新上传，默认 false",
                        },
                    },
                    "required": ["asset_id"],
                },
            ),
        }

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=self.hosting.enabled, detail=self.hosting.brief())

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            result = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    async def _fn_hosting_status(self) -> ToolResult:
        return ToolResult(content=f"素材托管：{self.hosting.brief()}")

    async def _fn_host_file(self, path: str, summary: str = "") -> ToolResult:
        if not self.hosting.enabled:
            return ToolResult(ok=False, error=self.hosting.brief())
        if self.files is None:
            return ToolResult(ok=False, error="没接文件工具，登记不了本地文件")
        r = await self.files.invoke("fs_import", {"path": path, "summary": summary})
        if not r.ok:
            return r
        return await self._fn_host_asset(r.asset_ref or "")

    async def _fn_host_asset(self, asset_id: str, force: bool = False) -> ToolResult:
        try:
            a = self.store.get(asset_id)
        except KeyError:
            return ToolResult(ok=False, error=f"没有资产 {asset_id}")
        url, err = await self.hosting.ensure_asset(self.store, a, 20, force=force)
        if err and not url:
            return ToolResult(ok=False, error=err)
        note = f"（{err}）" if err else ""
        return ToolResult(content=f"资产 {a.id} 的参考链接：{url}{note}", asset_ref=a.id)
