"""抖音素材获取 functions —— 配合 `douyin-viral-analyzer` skill。

这是 Skill 与 Function 分工的一个实例（见 ARCHITECTURE.md §0）：
  · skills/douyin-viral-analyzer.md  —— **怎么拆**（方法论、归因铁律、报告模板）
  · 这里                             —— **拿什么来拆**（素材、数据、抽帧）

原 skill 自带的转写脚本没有移植：agent 已有 `transcribe` function（whisper-1，
可直出 srt），留两套转写是浪费，也会让血缘分叉。

脚本本身是 vendor 代码，放在 vendor/douyin/ 下按子进程调用，不改写
—— 它处理的是抖音分享口令解析、TikHub、yt-dlp 回退这些易变的外部细节，
保持原样便于日后从上游同步。
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
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

VENDOR = Path(__file__).parent / "vendor" / "douyin"
FETCH_SCRIPT = VENDOR / "dy_fetch.py"
RENDER_SCRIPT = VENDOR / "render_pdf.py"

# 抓取可能要下载视频 + 抽几十帧 + 调外部 API
FETCH_TIMEOUT = 600.0
RENDER_TIMEOUT = 120.0


class DouyinFunctions:
    name = "douyin"
    namespaced = False
    disclosure = "full"

    def __init__(self, store: AssetStore, workspace: Path) -> None:
        self.store = store
        self.workspace = workspace
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    # ---------- 声明 ----------

    def _build(self) -> None:
        self._specs["fetch_douyin"] = ToolSpec(
            name="fetch_douyin",
            summary="抓取抖音视频素材：无水印视频、公开数据、Top 评论、双轨抽帧、音频统计",
            permission=PermissionLevel.COMPUTE,
            description=(
                "一步拿全拆解所需素材。ref 可以是分享口令、短链、视频页链接、aweme_id，"
                "也可以是本地视频文件路径。\n"
                "返回素材包摘要（公开数据与比例、时长分辨率、场景切换时间点、静音段、"
                "高赞评论、缺失项）以及抽帧清单。\n"
                "**抽帧密度按视频时长自动选，不要试图调低**——漏一帧关键信息，归因就偏一次。\n"
                "拿到后用 transcribe 做口播转写（format 选 srt）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "抖音分享口令/链接/aweme_id，或本地视频文件路径",
                    },
                    "name": {
                        "type": "string",
                        "description": "给这次素材包起个短名，便于归档，如 tianshouyuan",
                    },
                },
                "required": ["ref"],
            },
        )

        self._specs["read_frames"] = ToolSpec(
            name="read_frames",
            summary="列出某次素材包的抽帧清单与时间戳",
            permission=PermissionLevel.READ,
            description=(
                "按时间顺序列出帧文件与对应秒数。"
                "**先看 first 和 last**（开场定钩子、结尾定记忆点），"
                "再按时间顺序看 uniform（清晰静态画面，辨认内容/字幕/表情），"
                "scene 用来数节奏（切换瞬间，多为运动模糊）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pack_id": {
                        "type": "string",
                        "description": "fetch_douyin 返回的素材包资产 id",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["all", "uniform", "scene", "key"],
                        "description": "key 只给首尾两帧",
                    },
                },
                "required": ["pack_id"],
            },
        )

        self._specs["render_douyin_report"] = ToolSpec(
            name="render_douyin_report",
            summary="把拆解报告 markdown 渲染成带排版的 PDF",
            permission=PermissionLevel.COMPUTE,
            description=(
                "最终交付物是 PDF，不是 markdown。传报告资产 id，套用报告样式渲染。\n"
                "**渲染前先自查报告已脱敏**：不能出现任何工具名、接口名、资产 id、调用花费。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "report_id": {"type": "string", "description": "报告 markdown 的资产 id"},
                    "filename": {
                        "type": "string",
                        "description": "输出文件名，建议 <账号>_<视频简称>_<日期>.pdf",
                    },
                },
                "required": ["report_id"],
            },
        )

    def _add_ok(self) -> bool:
        return FETCH_SCRIPT.exists()

    # ---------- ToolProvider 协议 ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        if not self._add_ok():
            return ProviderHealth(ok=False, detail=f"缺少 {FETCH_SCRIPT.name}")
        has_key = bool(os.environ.get("TIKHUB_API_KEY"))
        return ProviderHealth(
            ok=True,
            detail=(
                "素材抓取可用"
                + ("（外部数据源已配）" if has_key else "（未配数据源，仅本地视频/基础解析）")
            ),
        )

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            result = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    # ---------- 实现 ----------

    async def _fn_fetch_douyin(self, ref: str, name: str = "") -> ToolResult:
        if not self._add_ok():
            return ToolResult(ok=False, error=f"素材抓取脚本缺失：{FETCH_SCRIPT}")

        slug = _slug(name) or f"pack_{int(time.time())}"
        out_dir = self.workspace / "douyin" / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        code, stdout, stderr = await _run(
            [sys.executable, str(FETCH_SCRIPT), ref, "--out", str(out_dir)],
            timeout=FETCH_TIMEOUT,
        )

        summary_path = out_dir / "summary.md"
        if not summary_path.exists():
            tail = (stderr or stdout or "").strip()[-600:]
            return ToolResult(
                ok=False,
                error=f"抓取失败（exit {code}）。若是链接抓不到，按手动清单向用户要素材。\n{tail}",
            )

        # noqa: ASYNC240 — 小文件读取，紧随子进程结束，不值得开线程
        summary = summary_path.read_text(encoding="utf-8")  # noqa: ASYNC240
        meta = _read_json(out_dir / "meta.json")
        frames = _read_json(out_dir / "frames" / "index.json")

        # 素材包本体：内容就是摘要，细节按需再取
        pack = self.store.create(
            summary,
            type_=AssetType.TEXT,
            summary=f"抖音素材包·{meta.get('video_id') or slug}",
            creator="tool:fetch_douyin",
            gen_params={
                "ref": ref[:120],
                "out_dir": str(out_dir),
                "video_id": meta.get("video_id", ""),
                "frames": {k: len(v) for k, v in frames.items() if isinstance(v, list)},
            },
        )

        # 视频文件另存为资产，转写要用它
        video_asset_id = ""
        video_file = meta.get("video_file")
        if video_file and Path(video_file).exists():  # noqa: ASYNC240 — 一次 stat
            va = self.store.create(
                "",
                type_=AssetType.VIDEO,
                summary=f"原片·{meta.get('video_id') or slug}",
                parents=[pack.id],
                creator="tool:fetch_douyin",
            )
            va.uri = str(video_file)
            va.mime = "video/mp4"
            self.store.put(va)
            video_asset_id = va.id

        missing = "⚠️ 仍需人工补充" in summary
        head = [
            f"素材包 {pack.id} → {out_dir}",
            f"原片资产 {video_asset_id}（转写用它）" if video_asset_id else "（无视频文件）",
            f"抽帧 {sum(len(v) for v in frames.values() if isinstance(v, list))} 张",
        ]
        if missing:
            head.append("摘要末尾有「仍需人工补充」，按手动清单一次问一项")

        return ToolResult(
            content="\n".join(head) + "\n\n---\n" + summary,
            asset_ref=pack.id,
            truncated=len(summary) > 4000,
        )

    async def _fn_read_frames(self, pack_id: str, kind: str = "all") -> ToolResult:
        pack = self.store.get(pack_id)
        out_dir = Path(pack.gen_params.get("out_dir", ""))
        index = _read_json(out_dir / "frames" / "index.json")
        if not index:
            return ToolResult(ok=False, error=f"{pack_id} 没有抽帧数据")

        lines: list[str] = []

        def emit(label: str, entry: Any) -> None:
            if isinstance(entry, dict):
                lines.append(f"{label}  {entry.get('t', '?')}s  {entry.get('file', '')}")
            elif entry:
                lines.append(f"{label}  {entry}")

        emit("first", index.get("first"))
        emit("last", index.get("last"))
        if kind in ("all", "uniform"):
            for e in index.get("uniform") or []:
                emit("uniform", e)
        if kind in ("all", "scene"):
            for e in index.get("scene") or []:
                emit("scene", e)

        header = (
            f"fps={index.get('fps')} 场景阈值={index.get('scene_threshold')} · "
            f"目录 {out_dir / 'frames'}\n"
            "先读 first/last，再按时间顺序读 uniform；scene 用来数节奏。"
        )
        return ToolResult(content=header + "\n\n" + "\n".join(lines))

    async def _fn_render_douyin_report(self, report_id: str, filename: str = "") -> ToolResult:
        if not RENDER_SCRIPT.exists():
            return ToolResult(ok=False, error=f"渲染脚本缺失：{RENDER_SCRIPT}")

        md = self.store.content(report_id)
        if not md.strip():
            return ToolResult(ok=False, error=f"{report_id} 是空的")
        # 渲染脚本用同一个解释器跑，它 import 的 Python-Markdown 不在核心依赖里：没装时
        # 子进程只会留下一段 ImportError 堆栈，这里先说清楚怎么装
        if importlib.util.find_spec("markdown") is None:
            return ToolResult(
                ok=False,
                error="导出 PDF 需要 Python-Markdown：pip install markdown"
                "（或安装 .[douyin] 依赖组），本机还要有 Chrome 或 Edge",
            )

        reports = self.workspace / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        md_path = reports / f"{report_id}.md"
        md_path.write_text(md, encoding="utf-8")
        pdf_path = reports / (_slug(filename, keep_ext=True) or f"{report_id}.pdf")

        code, stdout, stderr = await _run(
            [sys.executable, str(RENDER_SCRIPT), str(md_path), str(pdf_path)],
            timeout=RENDER_TIMEOUT,
        )
        if not pdf_path.exists():
            tail = (stderr or stdout or "").strip()[-400:]
            return ToolResult(ok=False, error=f"渲染失败（exit {code}）：{tail}")

        asset = self.store.create(
            "",
            type_=AssetType.TEXT,
            summary=f"拆解报告 PDF·{pdf_path.name}",
            parents=[report_id],
            creator="tool:render_douyin_report",
        )
        asset.uri = str(pdf_path)
        asset.mime = "application/pdf"
        self.store.put(asset)

        size_kb = pdf_path.stat().st_size / 1024
        return ToolResult(
            content=f"已生成报告：{pdf_path}（{size_kb:.0f}KB）",
            asset_ref=asset.id,
        )


# ---------------------------------------------------------------- 辅助


async def _run(cmd: list[str], timeout: float) -> tuple[int, str, str]:  # noqa: ASYNC109
    """跑子进程。超时或崩溃都归一成 (code, stdout, stderr)，不抛出去打断 loop。"""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", f"超时（>{timeout:.0f}s）"
    return (
        proc.returncode or 0,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — 产物缺失或损坏都当作没有
        return {}
    return data if isinstance(data, dict) else {}


def _slug(name: str, keep_ext: bool = False) -> str:
    """清掉路径分隔符和非法字符，防止越出 workspace。"""
    if not name:
        return ""
    base = Path(name).name  # 丢掉任何目录成分
    keep = "._-" if keep_ext else "_-"
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in base)
    return cleaned.strip("_")[:80]
