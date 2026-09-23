# ruff: noqa: ASYNC240 — 同 media/ffmpeg.py：文件操作紧邻子进程，串行且小量
"""成片 functions —— 把零件装成能交付的视频。

前面的生成能力只能产零件：视频模型单次最长 8–12 秒，配音和字幕都是独立资产。
这里负责最后一公里：下载落盘 → 拼接 → 混音 → 烧字幕 → 导出。

全是确定性操作，**不过模型**。模型负责决策顺序，这里负责执行。
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
from ..assets.store import AssetStore, AssetType, local_copy
from ..media import ffmpeg
from ..pipeline.cutting import describe, plan_cuts

_EXT = {AssetType.VIDEO: ".mp4", AssetType.AUDIO: ".mp3", AssetType.IMAGE: ".png"}


class VideoEditFunctions:
    name = "edit"
    namespaced = False
    disclosure = "full"

    def __init__(self, store: AssetStore, workspace: Path, prefs: Any = None) -> None:
        self.store = store
        self.workspace = workspace
        # 产物目录偏好（OutputPrefs）：挂上后成片缺省导出到它的 exports/
        self.prefs = prefs
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    def _export_dir(self, out_dir: str) -> Path:
        """成片导出目录：显式传参 > 用户产物目录 > workspace/exports（旧默认）。"""
        if out_dir:
            return Path(out_dir)
        if self.prefs is not None:
            return self.prefs.dir_for("exports")
        return self.workspace / "exports"

    def _build(self) -> None:
        self._specs["fetch_asset_file"] = ToolSpec(
            name="fetch_asset_file",
            summary="把远端资产（生成接口返回的外链）下载到本地",
            permission=PermissionLevel.WRITE,
            description=(
                "生成接口返回的是**临时外链**，能存多久没保证。要归档或要送进剪辑，"
                "必须先下载。已经在本地的资产会直接跳过。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "asset_ids": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["asset_ids"],
            },
        )

        self._specs["compose_video"] = ToolSpec(
            name="compose_video",
            summary="把多段视频拼接，可选配音与字幕，导出成片",
            permission=PermissionLevel.COMPUTE,
            timeout=3600,  # 整剧几百段的拼接 + 烧字幕，ffmpeg 要跑很久
            description=(
                "**这是出成片的那一步。** 视频模型单次最长 8–12 秒，"
                "30 秒成片必须先分段生成，再用本函数按顺序拼起来。\n"
                "clips 按你要的播放顺序传。audio_id 传配音、subtitle_id 传 srt 字幕，都可省略。\n"
                "外链资产会自动先下载。out_dir 指定导出目录（如用户要求存到某个盘）。"
                """
**要快节奏就传 max_cut_seconds**：素材照旧按模型原生长度生成，
切碎在这一步做，不用多生成几段。"""
            ),
            parameters={
                "type": "object",
                "properties": {
                    "clips": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "视频资产 id，按播放顺序",
                    },
                    "audio_id": {"type": "string", "description": "配音资产 id，可省略"},
                    "subtitle_id": {"type": "string", "description": "srt 字幕资产 id，可省略"},
                    "out_dir": {"type": "string", "description": "导出目录，省略则存 workspace"},
                    "filename": {"type": "string", "description": "文件名，如 科技热点_30s.mp4"},
                    "max_cut_seconds": {
                        "type": "number",
                        "description": (
                            "单个镜头最长几秒。填了就启用快切：把素材切成密集短镜头，"
                            "相邻两刀不来自同一段素材。0 或省略 = 不切，整段顺序拼。"
                        ),
                    },
                    "min_cut_seconds": {
                        "type": "number",
                        "description": "单刀最短几秒，默认 1.2。太短会闪。",
                    },
                    "total_seconds": {
                        "type": "number",
                        "description": (
                            "成片目标时长。**有配音时忽略它、以配音长度为准**，"
                            "否则画面短于旁白会把旁白截断。没配音时才用这个值。"
                        ),
                    },
                },
                "required": ["clips"],
            },
        )

        self._specs["probe_media"] = ToolSpec(
            name="probe_media",
            summary="查看媒体资产的时长、分辨率、帧率、有无音轨",
            permission=PermissionLevel.READ,
            description="拼接前核对各段是否一致，以及总时长够不够。",
            parameters={
                "type": "object",
                "properties": {"asset_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["asset_ids"],
            },
        )

    # ---------- ToolProvider ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        ok = ffmpeg.have_ffmpeg()
        return ProviderHealth(ok=ok, detail="ffmpeg 可用" if ok else "缺 ffmpeg/ffprobe")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            r = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            r = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        r.duration_ms = int((time.perf_counter() - started) * 1000)
        return r

    # ---------- 内部 ----------

    async def _localize(self, asset_id: str) -> tuple[Path | None, str]:
        """确保资产有本地文件，返回 (路径, 错误)。

        **不改 uri**。远端 URL 是后续镜头的参考链（前序片段、音色锚点）要用的：
        之前下载完把 uri 换成本地路径，拼完第 1 集后锚点片段传给模型的就是
        `E:\\…\\blobs\\as_xxx.mp4`，被接口 400 拒掉（2026-09-18 实测）。
        本地路径记在 gen_params["local"]，和 gen_image / gen_video 的落盘副本一个口径。
        """
        a = self.store.get(asset_id)
        lc = local_copy(a)
        if lc is not None:
            return lc, ""
        uri = a.uri or ""
        if uri and not uri.startswith(("http://", "https://")):
            p = Path(uri)
            return (p, "") if p.exists() else (None, f"{asset_id} 的文件不在了：{uri}")
        if not uri:
            return None, f"{asset_id} 没有文件（可能是纯文本资产）"

        target = self.workspace / "assets" / "blobs" / f"{asset_id}{_EXT.get(a.type, '.bin')}"
        if target.exists():
            a.gen_params["local"] = str(target)
            self.store.put(a)
            return target, ""
        ok, err = await ffmpeg.download(uri, target)
        if not ok:
            return None, f"{asset_id} 下载失败：{err}"
        a.gen_params["local"] = str(target)
        self.store.put(a)
        return target, ""

    # ---------- 实现 ----------

    async def _fn_fetch_asset_file(self, asset_ids: list[str]) -> ToolResult:
        lines, failed = [], []
        for aid in asset_ids:
            p, err = await self._localize(aid)
            if p:
                lines.append(f"✓ {aid} → {p}（{p.stat().st_size / 1024 / 1024:.1f}MB）")
            else:
                failed.append(f"✗ {aid}: {err}")
        return ToolResult(
            ok=bool(lines),
            content="\n".join(lines + failed),
            error="\n".join(failed) if failed and not lines else None,
        )

    async def _fn_probe_media(self, asset_ids: list[str]) -> ToolResult:
        lines = []
        total = 0.0
        for aid in asset_ids:
            p, err = await self._localize(aid)
            if not p:
                lines.append(f"{aid}: {err}")
                continue
            info = await ffmpeg.probe(p)
            total += info.duration
            lines.append(f"{aid}  {info.brief}")
        lines.append(f"— 合计时长 {total:.1f}s")
        return ToolResult(content="\n".join(lines))

    async def _fn_compose_video(
        self,
        clips: list[str],
        audio_id: str = "",
        subtitle_id: str = "",
        out_dir: str = "",
        filename: str = "",
        max_cut_seconds: float = 0.0,
        min_cut_seconds: float = 1.2,
        total_seconds: float = 0.0,
    ) -> ToolResult:
        if not ffmpeg.have_ffmpeg():
            return ToolResult(ok=False, error="环境里没有 ffmpeg/ffprobe，无法合成")
        if not clips:
            return ToolResult(ok=False, error="clips 为空，至少要一段视频")

        work = self.workspace / "compose" / f"c{int(time.time())}"
        work.mkdir(parents=True, exist_ok=True)

        # 1. 落盘
        paths: list[Path] = []
        for aid in clips:
            p, err = await self._localize(aid)
            if not p:
                return ToolResult(ok=False, error=err)
            paths.append(p)

        # 2. 配音先落盘 —— 成片时长要对齐口播，剪之前就得知道旁白多长
        apath: Path | None = None
        if audio_id:
            apath, err = await self._localize(audio_id)
            if not apath:
                return ToolResult(ok=False, error=err)

        # 3. 拼接（可选快切）
        stage = work / "concat.mp4"
        steps: list[str] = []
        if max_cut_seconds and max_cut_seconds > 0:
            lens = [(await ffmpeg.probe(pth)).duration or 0.0 for pth in paths]
            # 有配音时**以配音长度为准**，而不是 total_seconds。
            # 画面短于旁白的话，mux_audio 会按画面长度把音频截掉 ——
            # 旁白直接断在半句上。文案按 30 秒写，TTS 实际渲染出来可能是
            # 28 秒也可能是 35 秒，只有对齐音频才不会切词。
            voice_len = (await ffmpeg.probe(apath)).duration if apath else 0.0
            total = voice_len or total_seconds or sum(lens)
            cuts = plan_cuts(lens, total, max_cut_seconds, min_cut_seconds)
            if not cuts:
                return ToolResult(ok=False, error="素材时长探测不到，排不出剪辑表")
            ok, err = await ffmpeg.concat_cuts(
                paths, [(c.clip, c.start, c.dur) for c in cuts], stage
            )
            if not ok:
                return ToolResult(ok=False, error=f"快切拼接失败：{err}")
            steps.append(f"{len(paths)} 段素材剪成 {describe(cuts)}")
        else:
            ok, err = await ffmpeg.concat(paths, stage)
            if not ok:
                return ToolResult(ok=False, error=f"拼接失败：{err}")
            steps.append(f"拼接 {len(paths)} 段")

        # 4. 混音
        if apath:
            ap = apath
            muxed = work / "muxed.mp4"
            ok, err = await ffmpeg.mux_audio(stage, ap, muxed)
            if not ok:
                return ToolResult(ok=False, error=f"混音失败：{err}")
            stage = muxed
            steps.append("配音")

        # 5. 字幕
        if subtitle_id:
            srt = work / "sub.srt"
            srt.write_text(self.store.content(subtitle_id), encoding="utf-8")
            burned = work / "subbed.mp4"
            ok, err = await ffmpeg.burn_subtitle(stage, srt, burned)
            if ok:
                stage = burned
                steps.append("烧字幕")
            else:
                # 字幕失败不该让整个成片作废，降级交付并如实说明
                steps.append(f"字幕失败已跳过（{err[:80]}）")

        # 6. 导出
        target_dir = self._export_dir(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        name = _safe_name(filename) or f"video_{int(time.time())}.mp4"
        final = target_dir / name
        final.write_bytes(stage.read_bytes())

        info = await ffmpeg.probe(final)
        asset = self.store.create(
            "",
            type_=AssetType.VIDEO,
            summary=f"成片·{name}",
            parents=(
                clips
                + ([audio_id] if audio_id else [])
                + ([subtitle_id] if subtitle_id else [])
            ),
            creator="tool:compose_video",
            gen_params={"clips": clips, "steps": steps},
        )
        asset.uri = str(final)
        asset.mime = "video/mp4"
        self.store.put(asset)

        return ToolResult(
            content=(
                f"成片已导出：{final}\n"
                f"{info.brief} · {final.stat().st_size / 1024 / 1024:.1f}MB\n"
                f"步骤：{' → '.join(steps)}\n"
                f"资产 {asset.id}"
            ),
            asset_ref=asset.id,
        )


def _safe_name(name: str) -> str:
    """清掉路径成分，防止写到目标目录之外。"""
    if not name:
        return ""
    base = Path(name).name
    cleaned = "".join(c if (c.isalnum() or c in "._- （）()") else "_" for c in base)
    cleaned = cleaned.strip("_")[:100]
    return cleaned if cleaned.endswith(".mp4") else f"{cleaned}.mp4"
