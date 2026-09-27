"""M16 Distribution —— 各平台格式适配，产出**待发布包**。

发布方式定了（ARCHITECTURE 开放问题 #3）：**只产待发布包，由人上传。**
真正的发布动作是不可逆的 L-external，这版不接平台 API；人上传完用
mark_published() 把链接记回来，M17 的数据回流靠这条链接对上号。

一个包 = workspace/releases/<platform>_<slug>_<ts>/：
    content.md     正文原样（机审不改字，格式适配也不改正文）
    upload.md      给人看的上传清单：标题、话题、封面、AIGC 标识怎么打、机审风险、待改项
    manifest.json  机器读：平台、资产血缘、机审结果、AIGC 隐式标识、发布状态
    media/         本地媒体文件的副本；外链只记 URL

格式适配的边界：
  · 话题标签超上限 → 保留前 N 个并注明（这是格式，不是内容）
  · 标题超长 / 正文超长 / 类型不支持 → **列为问题让人改**，不静默截断
  · 机审有 block → 拒绝打包，除非人给出 override_reason（记进 manifest）
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..assets.store import Asset, AssetStore, AssetType, local_copy, rights_of
from ..compliance import ComplianceReport, report_to_params


class PlatformSpec(BaseModel):
    key: str
    name: str = ""
    kinds: list[str] = Field(default_factory=lambda: ["text"])
    title_max: int = 100
    body_max: int = 100_000
    tags_max: int = 10
    tag_prefix: str = "#"
    video: dict[str, Any] = Field(default_factory=dict)
    cover: dict[str, Any] = Field(default_factory=dict)
    aigc_label: str = ""

    def brief(self) -> str:
        kinds = "/".join(self.kinds)
        video = ""
        if self.video:
            video = (
                f" · 视频 {self.video.get('aspect_ratio', '')}"
                f" ≤{self.video.get('max_seconds', '?')}s"
            )
        return (
            f"{self.key}（{self.name}）：{kinds} · 标题≤{self.title_max} · 正文≤{self.body_max} · "
            f"话题≤{self.tags_max}{video}"
        )


class PlatformCatalog(BaseModel):
    specs: dict[str, PlatformSpec] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> PlatformCatalog:
        path = Path(path)
        if not path.exists():
            return cls(specs={"generic": PlatformSpec(key="generic", name="通用")})
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        specs = {}
        for key, spec in (raw.get("platforms") or {}).items():
            specs[str(key)] = PlatformSpec(key=str(key), **(spec or {}))
        if "generic" not in specs:
            specs["generic"] = PlatformSpec(key="generic", name="通用")
        return cls(specs=specs)

    def get(self, key: str) -> PlatformSpec | None:
        return self.specs.get(key)

    def render(self) -> str:
        return "\n".join(f"- {s.brief()}" for s in self.specs.values())


class Manifest(BaseModel):
    package_id: str
    platform: str
    kind: str = "text"
    title: str = ""
    tags: list[str] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)
    content_asset: str = ""
    media_assets: list[str] = Field(default_factory=list)
    cover_asset: str = ""
    report_asset: str = ""
    compliance: dict[str, Any] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    aigc: dict[str, Any] = Field(default_factory=dict)
    lineage: list[str] = Field(default_factory=list)
    override_reason: str = ""
    status: str = "ready"  # ready | published
    published_url: str = ""
    published_at: float = 0.0
    created_at: float = Field(default_factory=time.time)


def _slug(text: str, limit: int = 24) -> str:
    cleaned = "".join(c if (c.isalnum() or c in "_-") else "_" for c in text.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:limit] or "untitled"


_LABEL_HINTS = ("AI 生成", "AI生成", "AIGC", "人工智能生成")


def _first_line(content: str) -> str:
    """没给标题时取正文第一行。跳过 AIGC 标识那一行 —— 它是合规要求，不是标题。"""
    for line in content.splitlines():
        s = line.strip().lstrip("#").strip()
        if s and not any(h in s for h in _LABEL_HINTS):
            return s
    return ""


def _normalize_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for t in tags:
        s = str(t).strip().lstrip("#").strip()
        if s and s not in out:
            out.append(s)
    return out


class Packager:
    def __init__(self, releases_dir: Path, catalog: PlatformCatalog, assets: AssetStore) -> None:
        self.dir = releases_dir
        self.catalog = catalog
        self.assets = assets

    # ---------- 打包 ----------

    def build(
        self,
        content_asset_id: str,
        platform: str,
        *,
        title: str = "",
        tags: list[str] | None = None,
        media_asset_ids: list[str] | None = None,
        cover_asset_id: str = "",
        report: ComplianceReport | None = None,
        report_asset_id: str = "",
        override_reason: str = "",
    ) -> tuple[Asset, Manifest]:
        spec = self.catalog.get(platform)
        if spec is None:
            avail = ", ".join(self.catalog.specs)
            raise ValueError(f"未知平台 {platform!r}。可用：{avail}")
        content_asset = self.assets.get(content_asset_id)
        content = self.assets.content(content_asset_id)

        if report is not None and not report.passed and not override_reason.strip():
            raise PermissionError(
                f"机审有 {len(report.blocks)} 处 block 级问题，改完再打包；"
                "确要带病发布必须填 override_reason，它会记进 manifest。"
            )

        media = [self.assets.get(i) for i in (media_asset_ids or [])]
        kind = _kind_of(media)
        issues: list[str] = []
        if kind not in spec.kinds:
            issues.append(
                f"[block] {spec.name} 不支持 {kind} 类内容（支持：{'/'.join(spec.kinds)}）"
            )

        title = (title or _first_line(content))[:200]
        if len(title) > spec.title_max:
            issues.append(f"[block] 标题 {len(title)} 字，超过 {spec.name} 上限 {spec.title_max}")
        if len(content) > spec.body_max:
            issues.append(f"[block] 正文 {len(content)} 字，超过 {spec.name} 上限 {spec.body_max}")

        tags = _normalize_tags(tags or [])
        if spec.tags_max == 0 and tags:
            issues.append(f"[info] {spec.name} 不支持话题标签，已忽略 {len(tags)} 个")
            tags = []
        elif len(tags) > spec.tags_max:
            issues.append(
                f"[info] 话题 {len(tags)} 个，超过上限 {spec.tags_max}，只保留前 {spec.tags_max} 个"
            )
            tags = tags[: spec.tags_max]

        hard = [i for i in issues if i.startswith("[block]")]
        if hard and not override_reason.strip():
            raise ValueError("格式不合规，先改：\n" + "\n".join(hard))

        # ---- 落盘 ----
        pkg_id = f"pkg_{int(time.time())}_{_slug(title, 12)}"
        folder = self.dir / f"{platform}_{_slug(title)}_{int(time.time())}"
        folder.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        (folder / "content.md").write_text(content, encoding="utf-8")
        files["content"] = "content.md"

        generated = _is_generated(content_asset) or any(_is_generated(a) for a in media)
        generators = sorted(
            {a.creator for a in [content_asset, *media] if a.creator.startswith("model:")}
        )
        mark = json.dumps(
            {"AIGC": generated, "producer": "aigc-agent", "content_asset": content_asset_id,
             "generators": generators},
            ensure_ascii=False,
        )

        media_dir = folder / "media"
        media_refs: list[str] = []
        for a in media:
            # 本地副本优先（产物目录 / Agent 自留的 blobs）：生成媒体的 uri 多是会过期的链接，
            # 之前只看 uri，本地成片根本没进包（2026-09-23 审查）
            src = local_copy(a)
            uri = a.uri or ""
            if src is not None:
                media_dir.mkdir(exist_ok=True)
                dest = media_dir / f"{a.id}{src.suffix}"
                # 隐式标识写进文件本身：之前只在 manifest.json 里，实际上传的 mp4 不带
                if not (generated and _embed_mark(src, dest, mark)):
                    shutil.copy2(src, dest)
                    if generated and dest.suffix.lower() in (".mp4", ".png", ".jpg", ".jpeg"):
                        issues.append(
                            f"[info] 没能把 AIGC 隐式标识写进 {dest.name}（已记在 manifest）"
                        )
                files[a.id] = f"media/{dest.name}"
                media_refs.append(f"{a.id} → media/{dest.name}")
            else:
                files[a.id] = uri
                media_refs.append(f"{a.id} → {uri or '（无文件）'}")
                if uri.startswith(("http://", "https://")):
                    issues.append(
                        f"[warn] {a.id} 只有远端链接（约 24 小时过期），包里没有文件"
                        " —— 先下载到本地"
                    )
        aigc = {
            "generated": generated,
            "generators": generators,
            "explicit_label": "本内容由 AI 生成" if generated else "",
            "platform_note": spec.aigc_label,
            # 隐式标识：元数据里的机器可读标记（《标识办法》要求的另一半）
            "implicit": {
                "producer": "aigc-agent",
                "content_asset": content_asset_id,
                "generators": generators,
                "created_at": time.time(),
            },
        }

        manifest = Manifest(
            package_id=pkg_id,
            platform=platform,
            kind=kind,
            title=title,
            tags=tags,
            files=files,
            content_asset=content_asset_id,
            media_assets=[a.id for a in media],
            cover_asset=cover_asset_id,
            report_asset=report_asset_id,
            compliance=report_to_params(report) if report is not None else {},
            issues=issues,
            aigc=aigc,
            lineage=[a.id for a in self.assets.lineage(content_asset_id)],
            override_reason=override_reason.strip(),
        )
        upload = _render_upload(manifest, spec, report, media_refs)
        (folder / "upload.md").write_text(upload, encoding="utf-8")
        files["upload"] = "upload.md"
        (folder / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        files["manifest"] = "manifest.json"

        asset = self.assets.create(
            upload,
            type_=AssetType.PACKAGE,
            summary=f"待发布包·{spec.name}·{title[:20]}",
            parents=[content_asset_id, *[a.id for a in media]]
            + ([report_asset_id] if report_asset_id else [])
            + ([cover_asset_id] if cover_asset_id else []),
            creator="tool:build_release_package",
            gen_params={"manifest": manifest.model_dump(mode="json"), "folder": str(folder)},
        )
        asset.uri = str(folder)
        asset.mime = "inode/directory"
        self.assets.put(asset)
        return asset, manifest

    # ---------- 发布回填 ----------

    def mark_published(self, package_asset_id: str, url: str, by: str = "human") -> Manifest:
        asset = self.assets.get(package_asset_id)
        manifest = self.manifest_of(asset)
        manifest.status = "published"
        manifest.published_url = url.strip()
        manifest.published_at = time.time()
        asset.gen_params["manifest"] = manifest.model_dump(mode="json")
        asset.gen_params["published_by"] = by
        self.assets.put(asset)
        folder = Path(asset.uri or "")
        if folder.is_dir():
            (folder / "manifest.json").write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )
        return manifest

    def manifest_of(self, asset: Asset) -> Manifest:
        raw = asset.gen_params.get("manifest")
        if not raw:
            raise ValueError(f"{asset.id} 不是待发布包（没有 manifest）")
        return Manifest.model_validate(raw)

    def packages(self) -> list[tuple[Asset, Manifest]]:
        out = []
        for a in self.assets.find(type_=AssetType.PACKAGE, newest_first=False):
            if a.gen_params.get("manifest"):
                out.append((a, self.manifest_of(a)))
        return out


def _embed_mark(src: Path, dest: Path, mark: str) -> bool:
    """把 AIGC 隐式标识写进文件元数据，写成功返回 True（dest 已生成）。
    mp4：ffmpeg 流拷贝加 comment；png：文本块；jpg：注释段。其余格式不处理。"""
    suffix = src.suffix.lower()
    try:
        if suffix in (".mp4", ".mov", ".m4v"):
            import subprocess

            r = subprocess.run(  # noqa: S603, S607 — 固定命令
                ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0", "-c", "copy",
                 "-metadata", f"comment={mark}", "-metadata", "description=AIGC", str(dest)],
                capture_output=True, timeout=180,
            )
            return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0
        if suffix in (".png", ".jpg", ".jpeg"):
            from PIL import Image, PngImagePlugin

            with Image.open(src) as img:
                if suffix == ".png":
                    info = PngImagePlugin.PngInfo()
                    info.add_text("AIGC", mark)
                    img.save(dest, pnginfo=info)
                else:
                    img.save(dest, quality=95, comment=mark.encode("utf-8"))
            return dest.exists()
    except Exception:  # noqa: BLE001 — 写不进去就退回原样拷贝，manifest 里仍有记录
        dest.unlink(missing_ok=True)
        return False
    return False


def _kind_of(media: list[Asset]) -> str:
    if any(a.type is AssetType.VIDEO for a in media):
        return "video"
    if any(a.type is AssetType.IMAGE for a in media):
        return "image"
    return "text"


def _is_generated(a: Asset) -> bool:
    """和 rights_of 同一口径：fetch_douyin / fetch_media_url 抓的别人的原片不是 AI 生成的，
    不该写进隐式标识（2026-09-24 审查）。"""
    return rights_of(a) == "generated"


def _render_upload(
    m: Manifest, spec: PlatformSpec, report: ComplianceReport | None, media_refs: list[str]
) -> str:
    tags = " ".join(f"{spec.tag_prefix}{t}" for t in m.tags) if m.tags else "（无）"
    lines = [
        f"# 待发布包 · {spec.name}",
        "",
        f"- 包 id：{m.package_id}",
        f"- 类型：{m.kind}",
        f"- 标题：{m.title}",
        f"- 话题：{tags}",
        f"- 封面：{m.cover_asset or '（未指定）'}",
        f"- 正文：content.md（{len(m.files)} 个文件）",
    ]
    if media_refs:
        lines.append("- 媒体：")
        lines += [f"    - {r}" for r in media_refs]
    lines += ["", "## AIGC 标识（必做）"]
    if m.aigc.get("generated"):
        lines.append(
            f"- 显式：{m.aigc.get('explicit_label')} —— {spec.aigc_label or '在正文或简介注明'}"
        )
        gens = ", ".join(m.aigc.get("generators") or []) or "未知"
        lines.append(f"- 隐式：manifest.json 的 aigc.implicit 已写入（生成方 {gens}）")
    else:
        lines.append("- 内容非 AI 生成，无需标识")
    lines += ["", "## 机审"]
    if report is None:
        lines.append("- 未做机审（打包时没有传报告）。发布前请跑 check_compliance。")
    else:
        c = report.counts()
        verdict = "通过" if report.passed else "未通过"
        lines.append(f"- {verdict} · block {c['block']} · warn {c['warn']}")
        for f in sorted(report.findings, key=lambda x: x.severity.value):
            lines.append(f"    - [{f.severity.value}] {f.label}：{f.message}")
    if m.override_reason:
        lines.append(f"- ⚠ 带病发布，人工放行理由：{m.override_reason}")
    lines += ["", "## 待改项"]
    lines += [f"- {i}" for i in m.issues] or ["- 无"]
    lines += [
        "",
        "## 上传后",
        "- 把发布链接记回来：agent release published <包资产id> --url <链接>",
        "- 有数据后录入：agent analytics record <包资产id> --views … --likes …",
    ]
    return "\n".join(lines)
