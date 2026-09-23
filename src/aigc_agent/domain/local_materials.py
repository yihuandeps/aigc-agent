"""本地素材索引（2026-09-20）—— 产物目录（和 filesystem.yaml 里的 material_dirs）有什么，按集归组。

用户的真实现场：E:\\内容测试 里 texts/ 有第 2–37 集剧本、videos/ 有 ep1–ep10 的分段、exports/
有 1–10 集成片，模型却回答「剧本只写到第 5 集，6–10 集没有剧本」——它只查资产库
（find_episode / list_assets），从没看过磁盘；而资产库里那批剧本又是 outline 类型、没有集号，
两边都对不上，用户看着满满一个文件夹被告知「没有」。

这里做三件事：
  · 扫描：只扫产物目录与显式配置的素材目录（不扫整个盘），按文件名认集号
    （第6集 / ep6_ / EP06 / 第06集-03_2场 / S1E06 …）、类型（文本 / 文档 / 图 / 视频 / 音频 / 字幕）
    和文件名里的资产 id（as_…）
  · 摘要：一段话，每轮 pin 进 pre_input —— 目录在哪、各子目录多少文件、覆盖哪些集
  · 按集 / 类型 / 关键词查：find_episode 和 find_materials 用它，路径可直接给 fs_read / view_*

带 TTL + 目录 mtime 签名的缓存：几百个文件一轮扫一次是毫秒级；max_files 兜底，
防止有人把整个 E:/ 配成素材目录。
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOCAL_PIN = "local_materials"

KIND_BY_EXT: dict[str, str] = {
    **{e: "text" for e in (".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".csv", ".tsv",
                           ".fountain", ".rst", ".xml", ".html", ".htm")},
    **{e: "doc" for e in (".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".pdf", ".wps")},
    **{e: "image" for e in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")},
    **{e: "video" for e in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".ts")},
    **{e: "audio" for e in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma")},
    ".srt": "subtitle",
    ".vtt": "subtitle",
    ".ass": "subtitle",
}
KINDS = ("text", "doc", "image", "video", "audio", "subtitle", "other")
_KIND_CN = {
    "text": "文本", "doc": "文档", "image": "图片", "video": "视频",
    "audio": "音频", "subtitle": "字幕", "other": "其他",
}

_EP_PATTERNS = (
    re.compile(r"第\s*(\d{1,3})\s*集"),
    re.compile(r"(?<![a-z0-9])ep(?:isode)?[\s_\-]?(\d{1,3})(?![0-9])", re.I),
    re.compile(r"(?<![a-z0-9])s\d{1,2}e(\d{1,3})(?![0-9])", re.I),
)
_ASSET_ID = re.compile(r"as_[0-9a-f]{10}")
_SKIP_DIRS = {"trash", "__pycache__", "node_modules", ".venv", ".git"}
_NUM = re.compile(r"(\d+)")


def episode_in_name(name: str) -> int:
    """文件名 → 集号；认不出来是 0。"""
    for pat in _EP_PATTERNS:
        m = pat.search(name or "")
        if m:
            return int(m.group(1))
    return 0


def kind_of(path: Path) -> str:
    return KIND_BY_EXT.get(path.suffix.lower(), "other")


def _norm(p: Path) -> str:
    return str(p).replace("\\", "/")


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{n}B"


def _stamp(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _natural(s: str) -> tuple[Any, ...]:
    return tuple(int(t) if t.isdigit() else t.lower() for t in _NUM.split(s))


def ranges(nums: list[int]) -> str:
    """[1,2,3,5,6,9] → '1–3, 5–6, 9'。"""
    if not nums:
        return "无"
    out: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}–{prev}" if start != prev else str(start))
        start = prev = n
    out.append(f"{start}–{prev}" if start != prev else str(start))
    return ", ".join(out)


@dataclass(frozen=True)
class LocalFile:
    path: Path
    root: Path
    kind: str
    size: int
    mtime: float
    episode: int
    asset_id: str

    @property
    def rel(self) -> str:
        try:
            return _norm(self.path.relative_to(self.root))
        except ValueError:
            return _norm(self.path)

    @property
    def folder(self) -> str:
        parts = self.rel.split("/")
        return parts[0] if len(parts) > 1 else "."

    def line(self, absolute: bool = False) -> str:
        head = _norm(self.path) if absolute else self.rel
        tail = f"　资产 {self.asset_id}" if self.asset_id else ""
        return f"{head}　{_human(self.size)}　{_stamp(self.mtime)}{tail}"


@dataclass
class LocalIndex:
    roots: list[Path]
    output_root: Path | None = None  # 产物目录（相对路径按它解析）；其余是 material_dirs
    files: list[LocalFile] = field(default_factory=list)
    truncated: bool = False
    scanned_at: float = field(default_factory=time.time)

    def __len__(self) -> int:
        return len(self.files)

    def by_episode(self, episode: int) -> list[LocalFile]:
        return self._sorted(f for f in self.files if f.episode == episode)

    def episodes(self, kind: str = "", folder: str = "") -> list[int]:
        eps = {
            f.episode
            for f in self.files
            if f.episode and (not kind or f.kind == kind) and (not folder or f.folder == folder)
        }
        return sorted(eps)

    def filter(
        self, episode: int = 0, kind: str = "", query: str = "", folder: str = ""
    ) -> list[LocalFile]:
        q = (query or "").strip().lower()
        out = []
        for f in self.files:
            if episode and f.episode != episode:
                continue
            if kind and kind != "all" and f.kind != kind:
                continue
            if folder and f.folder != folder.strip("/").strip():
                continue
            if q and q not in f.rel.lower():
                continue
            out.append(f)
        return self._sorted(out)

    @staticmethod
    def _sorted(items: Any) -> list[LocalFile]:
        return sorted(items, key=lambda f: (_natural(f.folder), _natural(f.rel)))

    # ---------- 渲染 ----------

    def summary(self) -> str:
        """每轮 pin 的一段话；目录里什么都没有就返回空串（不 pin）。"""
        if not self.files:
            return ""
        lines = ["## 本地素材（产物目录自动扫描，每轮更新）"]
        for root in self.roots:
            mine = [f for f in self.files if f.root == root]
            if not mine:
                continue
            label = "产物目录" if root == self.output_root else "素材目录"
            lines.append(f"{label} {_norm(root)} 共 {len(mine)} 个文件：{self._folders(mine)}")
        if self.truncated:
            lines.append("（文件太多，只索引了前面一部分；用 find_materials 加条件查）")
        lines.append(
            "这些文件和资产库是同一批东西（文件名里的 as_… 就是资产 id），"
            "资产库查不到的先在这里找。"
            "看某一集用 find_episode(N) 或 find_materials(episode=N)；读文本 fs_read（支持 docx），"
            "看图 view_image，看视频 view_video；相对路径直接可用。"
        )
        return "\n".join(lines)

    @staticmethod
    def _folders(files: list[LocalFile]) -> str:
        groups: dict[str, list[LocalFile]] = {}
        for f in files:
            groups.setdefault(f.folder, []).append(f)
        parts: list[str] = []
        for name in sorted(groups, key=_natural)[:10]:
            fs = groups[name]
            kinds: dict[str, int] = {}
            for f in fs:
                kinds[f.kind] = kinds.get(f.kind, 0) + 1
            top = max(kinds, key=lambda k: kinds[k])
            eps = sorted({f.episode for f in fs if f.episode})
            desc = _KIND_CN.get(top, top)
            if eps:
                desc += f"：第 {ranges(eps)} 集"
            shown = "根目录" if name == "." else name
            parts.append(f"{shown} {len(fs)}（{desc}）")
        if len(groups) > 10:
            parts.append(f"…另有 {len(groups) - 10} 个子目录")
        return " · ".join(parts)

    def inventory(self) -> str:
        """开工前的清点：这个文件夹**有什么、没什么**。

        2026-09-22 用户定：内容都在本地跑，开工之前先把当前文件夹看一遍 ——
        「没有」比「有」更要紧：他见过模型对着满满一个文件夹说"第 6 集没有剧本"。
        """
        root = self.output_root
        where = _norm(root) if root else "（未设置）"
        if not self.files:
            return f"产物目录 {where}：**空的**，这次生成的东西都会落在这里。"
        mine = [f for f in self.files if f.root == root] if root else list(self.files)
        lines = [f"产物目录 {where}：{len(mine)} 个文件"]
        have: list[str] = []
        miss: list[str] = []
        for kind in KINDS:
            fs = [f for f in mine if f.kind == kind]
            if not fs:
                if kind in ("text", "image", "video"):
                    miss.append(_KIND_CN[kind])
                continue
            eps = sorted({f.episode for f in fs if f.episode})
            tail = f"（第 {ranges(eps)} 集）" if eps else ""
            have.append(f"{_KIND_CN[kind]} {len(fs)}{tail}")
        if have:
            lines.append("  有：" + " · ".join(have))
        if miss:
            lines.append("  没有：" + " / ".join(miss))
        others = [f for f in self.files if not root or f.root != root]
        if others:
            lines.append(f"  另有素材目录 {len({f.root for f in others})} 个、{len(others)} 个文件")
        return "\n".join(lines)

    def render_episode(self, episode: int, limit: int = 30) -> str:
        files = self.by_episode(episode)
        if not files:
            return ""
        root = self.output_root
        head = (
            f"第 {episode} 集的本地文件（{len(files)} 个；相对产物目录 {_norm(root)}，"
            "可直接传给 fs_read / view_image / view_video / fs_import）："
            if root
            else f"第 {episode} 集的本地文件（{len(files)} 个）："
        )
        lines = [head]
        for f in files[:limit]:
            lines.append("- " + f.line(absolute=f.root != root))
        if len(files) > limit:
            lines.append(
                f"…还有 {len(files) - limit} 个，"
                f"用 find_materials(episode={episode}, kind=…) 看全部"
            )
        return "\n".join(lines)

    def render(self, files: list[LocalFile], limit: int = 60, offset: int = 0) -> str:
        root = self.output_root
        page = files[offset : offset + limit]
        span = f"，显示第 {offset + 1}–{offset + len(page)} 个" if len(files) > len(page) else ""
        where = f"（相对产物目录 {_norm(root)}）" if root else ""
        lines = [f"命中 {len(files)} 个文件{span}{where}："]
        for f in page:
            ep = f"第{f.episode}集 · " if f.episode else ""
            lines.append(f"- {ep}{f.line(absolute=f.root != root)}")
        if offset + len(page) < len(files):
            rest = len(files) - offset - len(page)
            lines.append(f"…还有 {rest} 个，offset={offset + len(page)} 翻页")
        return "\n".join(lines)


class LocalMaterials:
    """产物目录 + 素材目录的索引器。output 是 OutputPrefs（.root 会被 /out 改）或直接给路径。"""

    def __init__(
        self,
        output: Any = None,
        extra_dirs: list[Path] | tuple[Path, ...] = (),
        max_files: int = 4000,
        max_depth: int = 4,
        ttl: float = 3.0,
    ) -> None:
        self._output = output
        self.extra_dirs = [Path(d).expanduser() for d in extra_dirs]
        self.max_files = max_files
        self.max_depth = max_depth
        self.ttl = ttl
        self._cache: LocalIndex | None = None
        self._sig: tuple[Any, ...] = ()

    @property
    def output_root(self) -> Path | None:
        o = self._output
        if o is None:
            return None
        if isinstance(o, (str, Path)):
            return Path(o)
        root = getattr(o, "root", None)
        return Path(root) if root else None

    def roots(self) -> list[Path]:
        out: list[Path] = []
        for r in [self.output_root, *self.extra_dirs]:
            if r is None:
                continue
            try:
                rr = Path(r).expanduser().resolve()
            except OSError:
                continue
            if rr not in out and rr.is_dir():
                out.append(rr)
        return out

    # ---------- 缓存 ----------

    def _signature(self, roots: list[Path]) -> tuple[Any, ...]:
        sig: list[Any] = []
        for r in roots:
            try:
                sig.append((str(r), r.stat().st_mtime_ns))
                with os.scandir(r) as it:
                    for e in it:
                        if e.is_dir(follow_symlinks=False) and not e.name.startswith("."):
                            # 用 os.stat 而不是 e.stat()：Windows 上 scandir 给的是父目录索引里
                            # 缓存的时间戳，子目录刚加了文件要过几十毫秒才更新，会漏掉新文件
                            sig.append((e.name, os.stat(e.path).st_mtime_ns))
            except OSError:
                sig.append((str(r), None))
        return tuple(sig)

    def index(self, force: bool = False) -> LocalIndex:
        roots = self.roots()
        sig = self._signature(roots)
        cached = self._cache
        # 两个条件都满足才复用：签名没变（目录没增删文件）且没过 ttl。
        # 只看签名不够 —— NTFS 改文件内容不动目录 mtime，刚写完的文件还会延迟几毫秒才更新；
        # ttl 到了就重扫一遍，几百个文件是毫秒级的事。
        if (
            not force
            and cached is not None
            and sig == self._sig
            and time.time() - cached.scanned_at < max(self.ttl, 0)
        ):
            return cached
        out = self.output_root
        try:
            out = out.expanduser().resolve() if out is not None else None
        except OSError:
            out = None
        idx = self._scan(roots, out if out in roots else None)
        # 签名在扫描之后再取一次：NTFS 对刚写完的文件会延迟更新目录 mtime，
        # 扫描前取的签名可能在扫描中途就过期，下一轮又白扫一遍
        self._cache, self._sig = idx, self._signature(roots)
        return idx

    def summary(self) -> str:
        return self.index().summary()

    # ---------- 扫描 ----------

    def _scan(self, roots: list[Path], output_root: Path | None = None) -> LocalIndex:
        idx = LocalIndex(roots=list(roots), output_root=output_root)
        files: list[LocalFile] = []
        for root in roots:
            base_depth = len(root.parts)
            for dirpath, dirnames, filenames in os.walk(root):
                here = Path(dirpath)
                depth = len(here.parts) - base_depth
                dirnames[:] = sorted(
                    d for d in dirnames
                    if not d.startswith(".") and d.lower() not in _SKIP_DIRS
                ) if depth < self.max_depth else []
                for name in sorted(filenames):
                    if name.startswith(".") or name.lower().endswith((".tmp", ".part")):
                        continue
                    p = here / name
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    m = _ASSET_ID.search(name)
                    files.append(
                        LocalFile(
                            path=p,
                            root=root,
                            kind=kind_of(p),
                            size=st.st_size,
                            mtime=st.st_mtime,
                            episode=episode_in_name(name),
                            asset_id=m.group(0) if m else "",
                        )
                    )
                    if len(files) >= self.max_files:
                        idx.truncated = True
                        idx.files = files
                        return idx
        idx.files = files
        return idx
