"""看图 / 看视频（2026-09-18，用户要的）—— 模型能"读"图片和视频的内容。

之前模型对图片和视频只能看到 id、摘要、文件大小：渲出来的片段有没有变脸、用户给的
素材里是什么、参考图对不对，全得人来看。这里把视觉能力做成两个工具：

  view_image  一张图 → 视觉模型描述（主体、人物、场景、光线、画面文字、生成瑕疵），或回答问题
  view_video  一段视频 → 按时间均匀抽帧（带时间戳）→ 视觉模型按时间顺序描述；可选转写音轨

输入统一接受：本地路径（受 config/filesystem.yaml 边界约束）、资产 id、http(s) 链接。
图片走 data URL（大图先用 ffmpeg 缩到 1600 宽），视频抽帧缩到 640 宽 —— 帧多用 detail=low，
不然 token 直接爆掉，而看内容要的是"发生了什么"，不是像素细节。
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import shutil
import tempfile
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

VISION_ROLE = "vision"
MAX_FRAMES = 24
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
_SHRINK_OVER = 3 * 1024 * 1024  # 图片超过 3MB 先缩

IMAGE_PROMPT = (
    "请仔细看这张图，用中文如实描述，不要编造看不到的东西：\n"
    "1) 画面主体与人物：外貌、年龄感、表情、服装、姿态\n"
    "2) 场景与道具\n"
    "3) 光线、色调、构图与镜头感\n"
    "4) 画面上出现的任何文字、字幕、水印、Logo（原样抄录，没有就说没有）\n"
    "5) 如果像 AI 生成的，指出明显瑕疵（手指、文字、透视、皮肤质感、对称感等）"
)

VIDEO_PROMPT = (
    "下面是同一段视频按时间顺序均匀抽出的关键帧，每帧标了时间点。请用中文如实描述，"
    "不要编造帧与帧之间没看到的内容：\n"
    "1) 按时间顺序说发生了什么（分镜级：谁在哪里做什么，动作和情绪怎么推进）\n"
    "2) 出场人物：外貌、服装，前后帧是否是同一个人、有没有变脸或服装突变\n"
    "3) 场景、光线、运镜（推拉摇移/手持/固定）\n"
    "4) 画面上出现的任何字幕、文字、水印（原样抄录，并说明在第几帧/什么位置；没有就说没有）\n"
    "5) 明显的生成瑕疵或穿帮（肢体异常、闪烁、物体消失、口型与情绪不符等）"
)

SYSTEM = "你是视觉分析助手：只描述真正看到的，看不清就说看不清，不要脑补。"


def _mmss(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


class VisionFunctions:
    """看图/看视频的 ToolProvider。路径边界复用 FileFunctions（同一份白名单）。"""

    name = "vision"
    namespaced = False

    def __init__(
        self,
        gateway: Any,
        store: AssetStore,
        files: Any = None,
        registry: Any = None,
        role: str = VISION_ROLE,
    ) -> None:
        self.gateway = gateway
        self.store = store
        self.files = files  # FileFunctions：用它的 resolve() 守边界；None = 不限
        self.registry = registry  # 转写音轨要调 transcribe 工具
        self.role = role
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    # ---------- 声明 ----------

    def _build(self) -> None:
        source_p = {
            "type": "string",
            "description": "本地路径 / 资产 id（as_…）/ http(s) 链接",
        }
        self._specs["view_image"] = ToolSpec(
            name="view_image",
            summary="看一张图片：描述内容、人物、场景、画面文字、生成瑕疵，或回答关于它的问题",
            permission=PermissionLevel.COMPUTE,
            # 不设 cost_kind：看图走文本网关，花费由 COST 事件记（闸门再记一次就重了）
            description=(
                "把图片交给视觉模型看。用途：核对参考图/角色图对不对、看用户给的素材是什么、"
                "检查生成图有没有文字水印或瑕疵。question 留空就全面描述。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": source_p,
                    "question": {"type": "string", "description": "想问的具体问题，可省略"},
                    "detail": {
                        "type": "string",
                        "enum": ["high", "low"],
                        "description": "看细节用 high（默认），只要大概用 low 省 token",
                    },
                },
                "required": ["source"],
            },
            max_result_chars=12_000,
        )
        self._specs["view_video"] = ToolSpec(
            name="view_video",
            summary="看一段视频：抽帧后按时间顺序描述内容、人物一致性、字幕/文字、穿帮，可选转写音轨",
            permission=PermissionLevel.COMPUTE,
            # 不设 cost_kind：同 view_image
            description=(
                "把视频按时间均匀抽帧交给视觉模型看。用途：审刚渲出来的片段（变脸？字幕？穿帮？）、"
                "了解用户素材讲了什么。frames 默认 8，最多 24；transcribe=true 会把音轨转写"
                "成文字一起给模型（多一次转写费）。question 留空就全面描述。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": source_p,
                    "question": {"type": "string", "description": "想问的具体问题，可省略"},
                    "frames": {"type": "integer", "description": "抽几帧，默认 8，最多 24"},
                    "transcribe": {"type": "boolean", "description": "是否转写音轨，默认 false"},
                },
                "required": ["source"],
            },
            timeout=600,
            max_result_chars=16_000,
        )

    # ---------- ToolProvider 协议 ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        ok = self.gateway is not None
        return ProviderHealth(ok=ok, detail="看图/看视频就绪" if ok else "没有文本网关")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            result = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    # ---------- 定位来源 ----------

    def _locate(self, source: str) -> tuple[Path | None, str, Any, str]:
        """来源 → (本地路径, 远端 url, 资产, 错误)。三者只会有一个有值（资产可附带）。"""
        s = (source or "").strip()
        if not s:
            return None, "", None, "source 是空的"
        if s.startswith("as_"):
            try:
                a = self.store.get(s)
            except KeyError:
                near = self.store.nearest(s) if hasattr(self.store, "nearest") else ""
                return None, "", None, f"没有资产 {s}" + (f"；相近的有 {near}" if near else "")
            lc = local_copy(a)
            if lc is not None:
                return lc, "", a, ""
            uri = a.uri or ""
            if uri.startswith(("http://", "https://")):
                return None, uri, a, ""
            if uri and Path(uri).exists():
                return Path(uri), "", a, ""
            return None, "", a, f"资产 {s} 没有可读的文件（既没本地副本也没链接）"
        if s.startswith(("http://", "https://")):
            return None, s, None, ""
        if self.files is not None:
            p, err = self.files.resolve(s, must_exist=True)
            if err:
                return None, "", None, err
            return p, "", None, ""
        p = Path(s).expanduser()
        return (p, "", None, "") if p.exists() else (None, "", None, f"{s} 不存在")

    async def _chat(self, messages: list[dict[str, Any]]) -> tuple[str, str]:
        if self.gateway is None:
            return "", "没有文本网关，看不了图"
        try:
            resp = await self.gateway.chat(self.role, messages)
        except KeyError:
            return "", f"models.yaml 没配 {self.role} 角色"
        except Exception as e:  # noqa: BLE001
            return "", f"视觉模型调用失败：{type(e).__name__}: {e}"
        return (resp.text or "").strip(), ""

    # ---------- 图片 ----------

    async def _fn_view_image(
        self, source: str, question: str = "", detail: str = "high"
    ) -> ToolResult:
        path, url, asset, err = self._locate(source)
        if err:
            return ToolResult(ok=False, error=err)
        if path is not None:
            if path.suffix.lower() in _VIDEO_EXT:
                return ToolResult(ok=False, error=f"{path.name} 是视频，用 view_video")
            url, err = await self._image_data_url(path)
            if err:
                return ToolResult(ok=False, error=err)
        text = IMAGE_PROMPT + (f"\n\n另外请回答：{question}" if question else "")
        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": url, "detail": "low" if detail == "low" else "high"},
                    },
                ],
            },
        ]
        answer, err = await self._chat(messages)
        if err:
            return ToolResult(ok=False, error=err)
        label = asset.id if asset else (path.name if path else url)
        return ToolResult(content=f"【{label}】\n{answer}")

    async def _image_data_url(self, path: Path) -> tuple[str, str]:
        data = await asyncio.to_thread(path.read_bytes)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        if len(data) > _SHRINK_OVER:
            small = await self._shrink(path)
            if small:
                data, mime = small, "image/jpeg"
        if not data:
            return "", f"{path.name} 是空文件"
        return f"data:{mime};base64," + base64.b64encode(data).decode(), ""

    async def _shrink(self, path: Path, width: int = 1600) -> bytes:
        """大图用 ffmpeg 缩一下再传，缩不了就原样传。"""
        tmp = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="view_"))
        try:
            out = tmp / "small.jpg"
            code, _ = await ffmpeg.run(
                ["ffmpeg", "-y", "-i", str(path), "-vf", f"scale='min({width},iw)':-2",
                 "-q:v", "4", str(out)]
            )
            if code == 0:
                return await asyncio.to_thread(out.read_bytes)
            return b""
        except Exception:  # noqa: BLE001
            return b""
        finally:
            await asyncio.to_thread(shutil.rmtree, tmp, True)

    # ---------- 视频 ----------

    async def _fn_view_video(
        self, source: str, question: str = "", frames: int = 8, transcribe: bool = False
    ) -> ToolResult:
        path, url, asset, err = self._locate(source)
        if err:
            return ToolResult(ok=False, error=err)
        tmp_dl: Path | None = None
        if path is None:
            # 远端视频先下到临时目录（抽帧要本地文件）
            tmp_dl = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="view_"))
            target = tmp_dl / "clip.mp4"
            ok, why = await ffmpeg.download(url, target)
            if not ok:
                await asyncio.to_thread(shutil.rmtree, tmp_dl, True)
                return ToolResult(ok=False, error=f"下载视频失败：{why}")
            path = target
        try:
            if path.suffix.lower() in _IMAGE_EXT:
                return ToolResult(ok=False, error=f"{path.name} 是图片，用 view_image")
            n = max(1, min(int(frames or 8), MAX_FRAMES))
            shots, info = await self._video_frames(path, n)
            if not shots:
                return ToolResult(
                    ok=False,
                    error=f"抽不出画面帧：{path.name}（检查 ffmpeg 是否可用、文件是否完整）",
                )
            transcript = ""
            if transcribe:
                transcript = await self._transcript(path, asset)
        finally:
            if tmp_dl is not None:
                await asyncio.to_thread(shutil.rmtree, tmp_dl, True)

        meta = f"总长 {info.duration:.1f}s"
        if getattr(info, "width", 0):
            meta += f"，{info.width}x{info.height}"
        meta += "，有音轨" if getattr(info, "has_audio", False) else "，无音轨"
        stamps = "；".join(f"第{i + 1}帧 @ {_mmss(t)}" for i, (t, _) in enumerate(shots))
        text = f"{VIDEO_PROMPT}\n\n视频信息：{meta}，共抽 {len(shots)} 帧：{stamps}"
        if transcript:
            text += f"\n\n音轨转写（供对照口型与内容）：\n{transcript[:4000]}"
        if question:
            text += f"\n\n另外请回答：{question}"
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for t, data in shots:
            content.append({"type": "text", "text": f"第 {_mmss(t)} 帧："})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(data).decode(),
                    "detail": "low",
                },
            })
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        answer, err = await self._chat(messages)
        if err:
            return ToolResult(ok=False, error=err)
        label = asset.id if asset else path.name
        head = f"【{label}】{meta}，看了 {len(shots)} 帧"
        if transcribe and not transcript:
            head += "（音轨没转写成功）"
        return ToolResult(content=f"{head}\n{answer}")

    async def _video_frames(
        self, path: Path, count: int
    ) -> tuple[list[tuple[float, bytes]], Any]:
        """按时间均匀抽 count 帧，返回 [(秒, jpg 字节)] 与 probe 信息。"""
        info = await ffmpeg.probe(path)
        dur = info.duration or 0.0
        if dur <= 0:
            return [], info
        tmp = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="view_"))
        shots: list[tuple[float, bytes]] = []
        try:
            for i in range(count):
                t = dur * (i + 0.5) / count  # 避开首尾黑场/淡出
                f = tmp / f"f{i:02d}.jpg"
                code, _ = await ffmpeg.run(
                    ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                     "-vf", "scale=640:-2", "-q:v", "5", str(f)]
                )
                if code == 0:
                    data = await asyncio.to_thread(f.read_bytes)
                    if data:
                        shots.append((t, data))
        except Exception:  # noqa: BLE001
            pass
        finally:
            await asyncio.to_thread(shutil.rmtree, tmp, True)
        return shots, info

    async def _extract_audio(self, path: Path) -> bytes:
        tmp = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="view_"))
        try:
            out = tmp / "audio.mp3"
            code, _ = await ffmpeg.run(
                ["ffmpeg", "-y", "-i", str(path), "-vn", "-c:a", "libmp3lame", "-q:a", "5",
                 str(out)]
            )
            return await asyncio.to_thread(out.read_bytes) if code == 0 else b""
        except Exception:  # noqa: BLE001
            return b""
        finally:
            await asyncio.to_thread(shutil.rmtree, tmp, True)

    async def _transcript(self, path: Path, asset: Any) -> str:
        """音轨 → transcribe 工具 → 文本。做不了就返回空串（结果里会标出来）。"""
        if self.registry is None:
            return ""
        audio = await self._extract_audio(path)
        if not audio:
            return ""
        tmp_asset = self.store.create_blob(
            audio, ".mp3", type_=AssetType.AUDIO, mime="audio/mpeg",
            summary=f"音轨·{asset.id if asset else path.name}",
            parents=[asset.id] if asset else [], creator="tool:view_video",
        )
        try:
            r = await self.registry.invoke(
                "transcribe", {"asset_id": tmp_asset.id, "format": "text"}
            )
        except Exception:  # noqa: BLE001
            return ""
        return (r.content or "").strip() if r.ok else ""
