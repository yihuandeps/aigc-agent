"""本地文件系统 functions（2026-09-18，用户要的）—— 让模型能看、能读、能写本机文件。

之前模型只能碰 AssetStore 里的东西：用户电脑上的素材、剧本 docx 旁边的 txt、
剪辑软件导出的片段，模型全都够不着，只能让人手动搬进来。这里给一组 fs_* 工具。

**边界**（config/filesystem.yaml）：
  · roots  能碰的目录白名单。项目目录、workspace、产物目录永远在；其余在配置里加
  · deny   永远不碰的模式：凭据（.env / *.pem / *.key / 私钥目录 …）、.venv、.git、
    系统目录。_HARD_DENY 写死在代码里，配置只能往上加、删不掉
  · 所有写操作可回滚：覆盖前把旧文件备份到 workspace/trash/，删除 = 移到 trash/，
    从不真删 —— 所以它们是 L-write 级（自动放行），不是 L-external
  · **例外：改 Agent 自己**（代码、配置、skill、台账、记忆、资产库）要人确认
    （permission_for 按参数把这次调用提到 L-external）。2026-09-23 审查：之前模型不经
    确认就能改 models.yaml 的预算、往 mcp_servers.yaml 加任意命令（下次启动就执行）、
    写一个最高优先级的 skill、清零台账 —— 等于它能改自己的规则

**阻塞 I/O 全部丢到线程**：目录扫描和大文件读会卡事件循环，批量渲染的进度事件
就会断流；所以 _fn_* 只做参数整理，真正的读写在 *_sync 里跑在 to_thread。
"""

from __future__ import annotations

import asyncio
import fnmatch
import mimetypes
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ...harness.tools.provider import (
    PermissionLevel,
    ProviderHealth,
    ToolMeta,
    ToolResult,
    ToolSpec,
)
from ..assets.store import AssetStore, AssetType, local_copy
from ..documents import DOC_EXT, UNSUPPORTED_EXT, extract_text
from ..local_materials import ranges as index_ranges
from ..media import ffmpeg

TEXT_EXT = {
    ".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".csv", ".tsv", ".srt", ".vtt",
    ".ass", ".xml", ".html", ".htm", ".py", ".js", ".ts", ".css", ".toml", ".ini", ".cfg",
    ".log", ".rst", ".tex", ".fountain",
}
_TYPE_BY_EXT: dict[str, AssetType] = {
    **{e: AssetType.IMAGE for e in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")},
    **{e: AssetType.VIDEO for e in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")},
    **{e: AssetType.AUDIO for e in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")},
    ".srt": AssetType.SUBTITLE,
    ".vtt": AssetType.SUBTITLE,
}
_ASSET_EXT: dict[AssetType, str] = {
    AssetType.IMAGE: ".png",
    AssetType.VIDEO: ".mp4",
    AssetType.AUDIO: ".mp3",
    AssetType.SUBTITLE: ".srt",
}
_DEFAULT_DENY = [
    "**/.env", "**/.env.*", "**/*.pem", "**/*.key", "**/id_rsa*", "**/*.pfx",
    "**/.venv/**", "**/.git/**", "**/__pycache__/**", "**/node_modules/**",
    "C:/Windows/**", "C:/Program Files/**", "C:/Program Files (x86)/**",
]
# 永远生效、配置删不掉的拒绝规则（2026-09-23 审查）：配置里写了 deny 列表时，
# 上面的默认值会被整个替换掉 —— 凭据不能靠配置记得写。fs_read 读到的东西会原样
# 发给文本模型（第三方），私钥、登录凭据、浏览器数据一旦读出来就收不回。
_HARD_DENY = [
    *_DEFAULT_DENY,
    "**/.ssh/**", "**/.gnupg/**", "**/.aws/**", "**/.azure/**", "**/.kube/**",
    "**/.docker/**", "**/.config/gcloud/**", "**/.claude/**",
    "**/.git-credentials", "**/.netrc", "**/_netrc", "**/.npmrc", "**/.pypirc",
    "**/*credential*", "**/*.kdbx", "**/*.p12", "**/*.ppk",
    "**/id_ed25519*", "**/id_ecdsa*", "**/id_dsa*",
    # 应用配置与浏览器数据（登录态、Cookie、保存的密码）。AppData/Local/Temp 与 Programs 不拦 ——
    # 临时文件和装在那里的程序（用户的 agent.cmd 就在 Programs 下）是正常要碰的
    "**/AppData/Roaming/**", "**/AppData/LocalLow/**", "**/User Data/**",
    "**/AppData/Local/Microsoft/**", "**/AppData/Local/Google/**", "**/AppData/Local/Packages/**",
]
# workspace 里 Agent 自己的状态（写它们 = 改 Agent 的账本/记忆/资产库）
_STATE_DIRS = {
    "assets": "资产库",
    "costs": "成本台账",
    "memory": "记忆与会话快照",
    "logs": "会话日志",
    "skills_history": "skill 版本记录",
    "trash": "回收站",
}
# 写操作工具 → 哪些参数是「会被改动的路径」
_WRITE_TARGETS: dict[str, tuple[str, ...]] = {
    "fs_write": ("path",),
    "fs_mkdir": ("path",),
    "fs_move": ("src", "dst"),
    "fs_copy": ("dst",),
    "fs_delete": ("path",),
    "fs_export": ("path",),
}


@dataclass
class FsPolicy:
    """能碰哪些目录、永远不碰哪些、单次读多大。"""

    roots: list[Path] = field(default_factory=list)
    deny: list[str] = field(default_factory=lambda: list(_HARD_DENY))
    max_read_bytes: int = 2_000_000
    max_list: int = 500
    max_search_files: int = 5000
    # 素材目录（2026-09-20）：除产物目录外还要当「已有素材」自动索引的文件夹，自动算进 roots
    material_dirs: list[Path] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> FsPolicy:
        p = Path(path)
        if not p.exists():
            return cls()
        raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        roots = [Path(str(r)).expanduser() for r in (raw.get("roots") or []) if str(r).strip()]
        # 配置只能往上加：写了 deny 也不会把凭据规则挤掉
        deny = list(dict.fromkeys([str(d) for d in (raw.get("deny") or [])] + _HARD_DENY))
        materials = [
            Path(str(r)).expanduser() for r in (raw.get("material_dirs") or []) if str(r).strip()
        ]
        return cls(
            roots=roots,
            deny=deny,
            max_read_bytes=int(raw.get("max_read_bytes") or 2_000_000),
            max_list=int(raw.get("max_list") or 500),
            max_search_files=int(raw.get("max_search_files") or 5000),
            material_dirs=materials,
        )


def _norm(p: Path) -> str:
    return str(p).replace("\\", "/")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n}B"


def _stamp(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def decode_text(data: bytes) -> tuple[str, str] | None:
    """utf-8（含 BOM）→ gbk。带 NUL 或都解不出来当二进制，返回 None。"""
    if b"\x00" in data[:8192]:
        return None
    for enc in ("utf-8-sig", "gbk"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return None


class FileFunctions:
    """本地文件系统的 ToolProvider：fs_roots / fs_list / fs_info / fs_read / fs_search /
    fs_write / fs_mkdir / fs_move / fs_copy / fs_delete / fs_import / fs_export。"""

    name = "files"
    namespaced = False

    def __init__(
        self,
        store: AssetStore,
        workspace: Path,
        output_root: Any = None,
        policy: FsPolicy | None = None,
        project_root: Path | None = None,
        local: Any = None,
    ) -> None:
        self.store = store
        self.workspace = Path(workspace)
        # OutputPrefs（.root 会被 CLI 改）或直接给 Path；每次解析时现取
        self._output = output_root
        self.policy = policy or FsPolicy()
        # 本地素材索引（LocalMaterials，2026-09-20）：find_materials 查它；None = 工具报「没配」
        self.local = local
        self.project_root = Path(project_root) if project_root else self.workspace.parent
        self.trash = self.workspace / "trash"
        self._specs: dict[str, ToolSpec] = {}
        self._build()

    # ---------- 边界 ----------

    @property
    def output_root(self) -> Path:
        o = self._output
        if o is None:
            return self.workspace / "output"
        if isinstance(o, (str, Path)):
            return Path(o)
        root = getattr(o, "root", None)  # OutputPrefs.root，CLI 改了产物目录这里跟着变
        return Path(root) if root else self.workspace / "output"

    def roots(self) -> list[Path]:
        fixed = [self.project_root, self.workspace, self.output_root]
        out: list[Path] = []
        for r in fixed + list(self.policy.roots) + list(self.policy.material_dirs):
            try:
                rr = Path(r).expanduser().resolve()
            except OSError:
                continue
            if rr not in out:
                out.append(rr)
        return out

    def resolve(self, raw: str, must_exist: bool = False) -> tuple[Path | None, str]:
        """字符串 → 白名单内的绝对路径。相对路径按产物目录解析。返回 (路径, 错误)。"""
        s = (raw or "").strip().strip('"').strip("'")
        if not s:
            return None, "路径是空的"
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = self.output_root / p
        try:
            p = p.resolve()
        except OSError as e:
            return None, f"路径解析失败：{e}"
        if not any(p == r or p.is_relative_to(r) for r in self.roots()):
            allowed = "、".join(_norm(r) for r in self.roots())
            return None, (
                f"{_norm(p)} 不在允许访问的目录内（{allowed}）。"
                "要访问它，在 config/filesystem.yaml 的 roots 里加上所在目录。"
            )
        if self.denied(p):
            return None, f"{_norm(p)} 命中禁止访问规则（凭据/虚拟环境/系统目录），不读不写。"
        if must_exist and not p.exists():
            return None, f"{_norm(p)} 不存在"
        return p, ""

    def denied(self, p: Path) -> bool:
        s = _norm(p).lower()
        extra = [d for d in _HARD_DENY if d not in self.policy.deny]  # 手搓的 FsPolicy 也兜住
        for pat in list(self.policy.deny) + extra:
            q = pat.replace("\\", "/").lower()
            if fnmatch.fnmatch(s, q) or fnmatch.fnmatch(s, q.removeprefix("**/")):
                return True
        return False

    def protected(self, p: Path) -> str:
        """写到这里等于改 Agent 自己？是就返回是什么（给人看的），否则空串。

        产物目录永远不算（生成的东西本来就该写那儿，哪怕它恰好在项目里）；
        workspace 里只有 Agent 的状态目录算；项目目录里 workspace 以外的都算
        （代码、配置、skill、测试、脚本、文档）。
        """
        try:
            p = p.resolve()
            project = self.project_root.resolve()
            ws = self.workspace.resolve()
            out = self.output_root.resolve()
        except OSError:
            return ""
        if out != project and (p == out or p.is_relative_to(out)):
            return ""
        for sub, label in _STATE_DIRS.items():
            base = ws / sub
            if p == base or p.is_relative_to(base):
                return f"Agent 的{label}"
        if (p == project or p.is_relative_to(project)) and not (p == ws or p.is_relative_to(ws)):
            return "Agent 的代码 / 配置 / skill"
        return ""

    def permission_for(self, tool: str, args: dict[str, Any]) -> tuple[PermissionLevel, str] | None:
        """注册表的提权钩子：写操作落在 Agent 自己的文件上 → 这次按 L-external 过闸门。"""
        keys = _WRITE_TARGETS.get(tool)
        if not keys:
            return None
        hits: list[str] = []
        for k in keys:
            raw = str(args.get(k) or "").strip()
            if not raw:
                continue
            p, _ = self.resolve(raw)
            if p is None:
                continue  # 越界 / 命中 deny 的由工具自己拒，这里不用提权
            what = self.protected(p)
            if what:
                hits.append(f"{what}：{_norm(p)}")
        if not hits:
            return None
        return (
            PermissionLevel.EXTERNAL,
            "要改动 " + "；".join(hits[:2]) + " —— 这是 Agent 自己的文件，需要你确认",
        )

    def _trash_path(self, p: Path) -> Path:
        folder = self.trash / time.strftime("%Y%m%d-%H%M%S")
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / p.name
        n = 2
        while target.exists():
            target = folder / f"{p.stem}-{n}{p.suffix}"
            n += 1
        return target

    # ---------- 声明 ----------

    def _add(
        self,
        name: str,
        summary: str,
        permission: PermissionLevel,
        params: dict[str, Any],
        description: str = "",
        max_result_chars: int = 0,
    ) -> None:
        self._specs[name] = ToolSpec(
            name=name,
            summary=summary,
            permission=permission,
            parameters=params,
            description=description,
            max_result_chars=max_result_chars,
        )

    def _build(self) -> None:
        path_p = {"type": "string", "description": "绝对路径；相对路径按产物目录解析"}
        overwrite_p = {"type": "boolean", "description": "目标已存在时是否覆盖，默认 false"}
        self._add(
            "fs_roots",
            "看本地文件工具能访问哪些目录、哪些永远不碰",
            PermissionLevel.READ,
            {"type": "object", "properties": {}},
            description="不确定某个路径能不能访问时先看这个。产物目录、workspace、项目目录永远可访问。",
        )
        self._add(
            "fs_list",
            "列目录：子目录与文件（大小、修改时间），可按通配符过滤、可递归",
            PermissionLevel.READ,
            {
                "type": "object",
                "properties": {
                    "path": path_p,
                    "pattern": {"type": "string", "description": "通配符，如 *.mp4、第01集*"},
                    "recursive": {"type": "boolean", "description": "递归子目录，默认 false"},
                    "limit": {"type": "integer", "description": "最多列几条，默认 200"},
                    "offset": {"type": "integer", "description": "翻页：跳过前几条"},
                },
                "required": ["path"],
            },
            description="找文件先列目录。条目多时用 pattern 过滤或翻页，没列出来不代表不存在。",
            max_result_chars=16_000,
        )
        self._add(
            "fs_info",
            "看一个路径的信息：存不存在、文件还是目录、大小、修改时间、类型",
            PermissionLevel.READ,
            {"type": "object", "properties": {"path": path_p}, "required": ["path"]},
        )
        self._add(
            "fs_read",
            "读文本文件内容（自动识别 UTF-8 / GBK；docx / pptx / xlsx 自动抽文字），大文件可分段",
            PermissionLevel.READ,
            {
                "type": "object",
                "properties": {
                    "path": path_p,
                    "offset": {"type": "integer", "description": "从第几个字开始，默认 0"},
                    "max_chars": {"type": "integer", "description": "最多读几个字，默认 20000"},
                },
                "required": ["path"],
            },
            description=(
                "读文本（剧本、字幕、配置、csv…）和 Office 文档（docx / pptx / xlsx 抽出纯文字；"
                ".doc / .pdf 读不了，会提示用户另存）。图片/视频读不出文字，会返回文件信息 ——"
                "看内容用 view_image / view_video；要进流水线先 fs_import 成资产。"
            ),
            max_result_chars=40_000,
        )
        self._add(
            "fs_search",
            "在目录下的文本文件里搜关键词或正则，返回 文件:行号: 内容",
            PermissionLevel.READ,
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "搜索的目录（绝对路径）"},
                    "query": {"type": "string", "description": "关键词（不区分大小写）或正则"},
                    "glob": {
                        "type": "string",
                        "description": "只搜哪些文件，如 *.txt，默认全部文本",
                    },
                    "regex": {"type": "boolean", "description": "query 按正则解释，默认 false"},
                    "limit": {"type": "integer", "description": "最多返回几处，默认 50"},
                },
                "required": ["path", "query"],
            },
            max_result_chars=16_000,
        )
        self._add(
            "fs_write",
            "写文本文件：新建 / 覆盖 / 追加（覆盖前旧文件自动备份到回收目录）",
            PermissionLevel.WRITE,
            {
                "type": "object",
                "properties": {
                    "path": path_p,
                    "content": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["create", "overwrite", "append"],
                        "description": "create（默认，已存在则报错）/ overwrite / append",
                    },
                },
                "required": ["path", "content"],
            },
            description="父目录不存在会自动建。覆盖前旧版本存到 workspace/trash/，可找回。",
        )
        self._add(
            "fs_mkdir",
            "建目录（含多级）",
            PermissionLevel.WRITE,
            {"type": "object", "properties": {"path": path_p}, "required": ["path"]},
        )
        self._add(
            "fs_move",
            "移动或重命名文件/目录",
            PermissionLevel.WRITE,
            {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string", "description": "目标路径（含新名字）"},
                    "overwrite": overwrite_p,
                },
                "required": ["src", "dst"],
            },
        )
        self._add(
            "fs_copy",
            "复制文件或目录",
            PermissionLevel.WRITE,
            {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "overwrite": overwrite_p,
                },
                "required": ["src", "dst"],
            },
        )
        self._add(
            "fs_delete",
            "删除文件/目录 —— 实际是移到 workspace/trash/ 回收目录，可找回",
            PermissionLevel.WRITE,
            {"type": "object", "properties": {"path": path_p}, "required": ["path"]},
            description="从不真删。要彻底删除请用户自己清空 workspace/trash/。",
        )
        self._add(
            "fs_import",
            "把本地文件登记成资产（图/视频/音频/字幕/文本），返回资产 id",
            PermissionLevel.WRITE,
            {
                "type": "object",
                "properties": {
                    "path": path_p,
                    "summary": {"type": "string", "description": "一句话说明，默认用文件名"},
                    "kind": {
                        "type": "string",
                        "enum": ["auto", "image", "video", "audio", "subtitle", "script", "text"],
                        "description": "资产类型，默认按扩展名判断",
                    },
                },
                "required": ["path"],
            },
            description=(
                "用户自己的素材要进流水线（拼接、当参考、当剧本）先登记成资产。"
                "媒体文件不复制，资产记原路径；文本文件内容存进资产。"
            ),
        )
        self._add(
            "fs_export",
            "把资产导出成本地文件（文本写出 / 媒体复制或下载）",
            PermissionLevel.WRITE,
            {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "path": {
                        "type": "string",
                        "description": "目标文件路径，或目录（用资产摘要起名）",
                    },
                    "overwrite": {"type": "boolean", "description": "默认 false"},
                },
                "required": ["asset_id", "path"],
            },
        )
        self._add(
            "find_materials",
            "找本地素材：按集号 / 类型 / 关键词列出产物目录（和素材目录）里的文件，"
            "路径可直接给 fs_read / view_image / view_video / fs_import",
            PermissionLevel.READ,
            {
                "type": "object",
                "properties": {
                    "episode": {
                        "type": "integer",
                        "description": "只要第几集的（按文件名里的 第N集 / epN 认）",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["all", "text", "doc", "image", "video", "audio", "subtitle"],
                        "description": "类型，默认 all；text = txt/md/srt 等，doc = docx/pdf 等",
                    },
                    "query": {
                        "type": "string", "description": "路径里包含的关键词（不区分大小写）"
                    },
                    "folder": {
                        "type": "string",
                        "description": "只看某个子目录，如 texts / videos / exports",
                    },
                    "limit": {"type": "integer", "description": "最多列几条，默认 60"},
                    "offset": {"type": "integer", "description": "翻页：跳过前几条"},
                },
            },
            description=(
                "用户说「已有的素材 / 本地文件 / 之前生成的 / 文件夹里的」先用它，不要只查资产库："
                "产物目录里的剧本 .md、图片、分段视频、整集成片都在这里。不传条件 = 目录总览。"
                "要看别的文件夹用 fs_list；要把文件夹加进索引，"
                "在 config/filesystem.yaml 的 material_dirs 里配。"
            ),
            max_result_chars=16_000,
        )

    # ---------- ToolProvider 协议 ----------

    async def list_tools(self) -> list[ToolMeta]:
        return [s.meta(self.name) for s in self._specs.values()]

    async def get_schema(self, tool: str) -> dict[str, Any]:
        return self._specs[tool].to_openai(tool)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail=f"{len(self.roots())} 个可访问目录")

    async def invoke(self, tool: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            result = await getattr(self, f"_fn_{tool}")(**args)
        except Exception as e:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    # ---------- 只读 ----------

    async def _fn_fs_roots(self) -> ToolResult:
        lines = ["可访问的目录："] + [f"  {_norm(r)}" for r in self.roots()]
        lines.append("永远不碰：" + "、".join(self.policy.deny))
        lines.append(f"相对路径按产物目录解析：{_norm(self.output_root)}")
        lines.append(f"删除/覆盖的旧文件在：{_norm(self.trash)}")
        if self.policy.material_dirs:
            dirs = "、".join(_norm(Path(d)) for d in self.policy.material_dirs)
            lines.append(f"额外索引的素材目录（material_dirs）：{dirs}")
        return ToolResult(content="\n".join(lines))

    async def _fn_find_materials(
        self,
        episode: int = 0,
        kind: str = "all",
        query: str = "",
        folder: str = "",
        limit: int = 60,
        offset: int = 0,
    ) -> ToolResult:
        if self.local is None:
            return ToolResult(
                ok=False, error="没有本地素材索引（装配时没传 LocalMaterials），用 fs_list 看目录"
            )
        kind = (kind or "all").strip().lower()
        if kind not in ("all", "text", "doc", "image", "video", "audio", "subtitle", "other"):
            return ToolResult(ok=False, error=f"kind 不认识：{kind!r}")
        try:
            index = await asyncio.to_thread(self.local.index)
        except OSError as e:
            return ToolResult(ok=False, error=f"扫描产物目录失败：{e}")
        roots = "、".join(_norm(r) for r in index.roots)
        if not index.files:
            where = roots or "（产物目录还不存在）"
            return ToolResult(content=f"产物目录里还没有任何文件：{where}")
        hits = index.filter(
            episode=int(episode or 0), kind=kind, query=query or "", folder=folder or ""
        )
        if not hits:
            eps = index.episodes()
            span = f"有文件的集：{index_ranges(eps)}" if eps else "文件名里都没有集号"
            return ToolResult(
                content=f"没有匹配的文件（索引了 {len(index)} 个；{span}）。"
                "换个条件，或用 fs_list 直接看目录。"
            )
        limit = max(1, min(int(limit or 60), 300))
        body = index.render(hits, limit=limit, offset=max(0, int(offset or 0)))
        head = f"索引范围：{roots}（共 {len(index)} 个文件"
        head += "，只索引了前面一部分）" if index.truncated else "）"
        return ToolResult(content=f"{head}\n{body}")

    async def _fn_fs_list(
        self, path: str, pattern: str = "", recursive: bool = False, limit: int = 200,
        offset: int = 0,
    ) -> ToolResult:
        p, err = self.resolve(path, must_exist=True)
        if err:
            return ToolResult(ok=False, error=err)
        if not p.is_dir():
            return ToolResult(ok=False, error=f"{_norm(p)} 不是目录")
        limit = max(1, min(int(limit or 200), self.policy.max_list))
        return await asyncio.to_thread(
            self._list_sync, p, pattern, recursive, limit, max(0, offset)
        )

    def _list_sync(
        self, p: Path, pattern: str, recursive: bool, limit: int, offset: int
    ) -> ToolResult:
        entries: list[tuple[bool, Path, int, float]] = []
        walker = p.rglob(pattern or "*") if recursive else p.glob(pattern or "*")
        scanned = 0
        for child in walker:
            scanned += 1
            if scanned > 50_000:
                break
            if self.denied(child):
                continue
            try:
                st = child.stat()
            except OSError:
                continue
            entries.append((child.is_dir(), child, st.st_size, st.st_mtime))
        entries.sort(key=lambda e: (not e[0], str(e[1]).lower()))
        total = len(entries)
        page = entries[offset : offset + limit]
        span = f"（第 {offset + 1}–{offset + len(page)} 项）" if total > len(page) else ""
        lines = [f"{_norm(p)}　共 {total} 项{span}"]
        for is_dir, child, size, mtime in page:
            rel = _norm(child.relative_to(p)) if recursive else child.name
            if is_dir:
                lines.append(f"  [目录] {rel}/")
            else:
                lines.append(f"  {rel}　{_human(size)}　{_stamp(mtime)}")
        if total > offset + len(page):
            rest = total - offset - len(page)
            lines.append(f"  …还有 {rest} 项，用 offset={offset + len(page)} 翻页或加 pattern")
        return ToolResult(content="\n".join(lines))

    async def _fn_fs_info(self, path: str) -> ToolResult:
        p, err = self.resolve(path)
        if err:
            return ToolResult(ok=False, error=err)
        return await asyncio.to_thread(self._info_sync, p)

    def _info_sync(self, p: Path) -> ToolResult:
        if not p.exists():
            return ToolResult(content=f"{_norm(p)}：不存在")
        st = p.stat()
        if p.is_dir():
            try:
                n = sum(1 for _ in p.iterdir())
            except OSError:
                n = -1
            return ToolResult(content=f"{_norm(p)}：目录，{n} 项，修改于 {_stamp(st.st_mtime)}")
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        kind = _TYPE_BY_EXT.get(p.suffix.lower())
        tag = f"，资产类型 {kind.value}" if kind else ""
        when = _stamp(st.st_mtime)
        return ToolResult(
            content=f"{_norm(p)}：文件，{_human(st.st_size)}，{mime}{tag}，修改于 {when}"
        )

    async def _fn_fs_read(self, path: str, offset: int = 0, max_chars: int = 20_000) -> ToolResult:
        p, err = self.resolve(path, must_exist=True)
        if err:
            return ToolResult(ok=False, error=err)
        if p.is_dir():
            return ToolResult(ok=False, error=f"{_norm(p)} 是目录，用 fs_list")
        return await asyncio.to_thread(
            self._read_sync, p, max(0, int(offset or 0)), int(max_chars or 20_000)
        )

    def _text_of(self, p: Path) -> tuple[str, str] | None:
        """文件 → (文本, 编码/格式)。纯文本按 UTF-8 / GBK 解；docx / pptx / xlsx 抽文字；
        其余二进制返回 None。读不了的文档格式（.doc / .pdf）抛 ValueError，信息给用户看。"""
        ext = p.suffix.lower()
        if ext in DOC_EXT or ext in UNSUPPORTED_EXT:
            return extract_text(p)
        return decode_text(p.read_bytes())

    def _read_sync(self, p: Path, offset: int, max_chars: int) -> ToolResult:
        size = p.stat().st_size
        if size > self.policy.max_read_bytes:
            cap = _human(self.policy.max_read_bytes)
            return ToolResult(
                ok=False, error=f"{_norm(p)} 有 {_human(size)}，超过单次读取上限 {cap}"
            )
        try:
            decoded = self._text_of(p)
        except ValueError as e:
            return ToolResult(ok=False, error=str(e))
        if decoded is None:
            info = self._info_sync(p).content
            return ToolResult(
                content=f"{info}\n（二进制文件，读不出文字。看图片用 view_image、看视频用 "
                "view_video；要进流水线先 fs_import）"
            )
        text, enc = decoded
        piece = text[offset : offset + max(1, max_chars)]
        how = f"{enc} 抽出的文字" if enc in ("docx", "pptx", "xlsx") else enc
        head = f"{_norm(p)}（{how}，共 {len(text)} 字，第 {offset + 1}–{offset + len(piece)} 字）"
        rest = len(text) - offset - len(piece)
        more = f"\n…还有 {rest} 字，用 offset={offset + len(piece)} 继续" if rest > 0 else ""
        return ToolResult(content=f"{head}\n{piece}{more}")

    async def _fn_fs_search(
        self, path: str, query: str, glob: str = "", regex: bool = False, limit: int = 50
    ) -> ToolResult:
        p, err = self.resolve(path, must_exist=True)
        if err:
            return ToolResult(ok=False, error=err)
        if not p.is_dir():
            return ToolResult(ok=False, error=f"{_norm(p)} 不是目录")
        if not query:
            return ToolResult(ok=False, error="query 是空的")
        try:
            pat = re.compile(query if regex else re.escape(query), re.I)
        except re.error as e:
            return ToolResult(ok=False, error=f"正则不合法：{e}")
        cap = max(1, min(int(limit or 50), 500))
        return await asyncio.to_thread(self._search_sync, p, pat, glob or "*", cap)

    def _search_sync(self, p: Path, pat: re.Pattern[str], glob: str, limit: int) -> ToolResult:
        hits: list[str] = []
        scanned = 0
        started = time.perf_counter()
        for f in p.rglob(glob):
            if scanned >= self.policy.max_search_files or time.perf_counter() - started > 20:
                break
            if not f.is_file() or self.denied(f):
                continue
            ext = f.suffix.lower()
            if ext not in TEXT_EXT and ext not in DOC_EXT and glob == "*":
                continue
            try:
                if f.stat().st_size > self.policy.max_read_bytes:
                    continue
                decoded = self._text_of(f)
            except (OSError, ValueError):
                continue
            scanned += 1
            if decoded is None:
                continue
            for no, line in enumerate(decoded[0].splitlines(), 1):
                if pat.search(line):
                    hits.append(f"{_norm(f.relative_to(p))}:{no}: {line.strip()[:200]}")
                    if len(hits) >= limit:
                        break
            if len(hits) >= limit:
                break
        head = f"在 {_norm(p)} 扫了 {scanned} 个文本文件，命中 {len(hits)} 处"
        if len(hits) >= limit:
            head += "（已到上限，缩小范围再搜）"
        return ToolResult(content=head + ("\n" + "\n".join(hits) if hits else ""))

    # ---------- 写 ----------

    async def _fn_fs_write(self, path: str, content: str, mode: str = "create") -> ToolResult:
        p, err = self.resolve(path)
        if err:
            return ToolResult(ok=False, error=err)
        if mode not in ("create", "overwrite", "append"):
            return ToolResult(
                ok=False, error=f"mode 只能是 create / overwrite / append，不是 {mode!r}"
            )
        return await asyncio.to_thread(self._write_sync, p, content, mode)

    def _write_sync(self, p: Path, content: str, mode: str) -> ToolResult:
        if p.is_dir():
            return ToolResult(ok=False, error=f"{_norm(p)} 是目录")
        note = ""
        if p.exists():
            if mode == "create":
                return ToolResult(
                    ok=False,
                    error=f"{_norm(p)} 已存在。要覆盖传 mode=overwrite，要追加传 mode=append",
                )
            if mode == "overwrite":
                backup = self._trash_path(p)
                shutil.copy2(p, backup)
                note = f"（旧版本已备份到 {_norm(backup)}）"
        p.parent.mkdir(parents=True, exist_ok=True)
        flag = "a" if mode == "append" and p.exists() else "w"
        with p.open(flag, encoding="utf-8", newline="") as fh:
            fh.write(content)
        verb = "追加到" if mode == "append" else "写入"
        return ToolResult(content=f"已{verb} {_norm(p)}，{len(content)} 字{note}")

    async def _fn_fs_mkdir(self, path: str) -> ToolResult:
        p, err = self.resolve(path)
        if err:
            return ToolResult(ok=False, error=err)
        await asyncio.to_thread(p.mkdir, parents=True, exist_ok=True)
        return ToolResult(content=f"目录就绪：{_norm(p)}")

    async def _fn_fs_move(self, src: str, dst: str, overwrite: bool = False) -> ToolResult:
        return await self._transfer(src, dst, overwrite, move=True)

    async def _fn_fs_copy(self, src: str, dst: str, overwrite: bool = False) -> ToolResult:
        return await self._transfer(src, dst, overwrite, move=False)

    async def _transfer(self, src: str, dst: str, overwrite: bool, move: bool) -> ToolResult:
        s, err = self.resolve(src, must_exist=True)
        if err:
            return ToolResult(ok=False, error=err)
        d, err = self.resolve(dst)
        if err:
            return ToolResult(ok=False, error=err)
        return await asyncio.to_thread(self._transfer_sync, s, d, overwrite, move)

    def _transfer_sync(self, s: Path, d: Path, overwrite: bool, move: bool) -> ToolResult:
        if d.is_dir() and not s.is_dir():
            d = d / s.name
        note = ""
        if d.exists():
            if not overwrite:
                return ToolResult(
                    ok=False, error=f"目标已存在：{_norm(d)}（要覆盖传 overwrite=true）"
                )
            backup = self._trash_path(d)
            shutil.move(str(d), str(backup))
            note = f"（原目标已移到 {_norm(backup)}）"
        d.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(s), str(d))
            verb = "已移动"
        elif s.is_dir():
            shutil.copytree(s, d)
            verb = "已复制目录"
        else:
            shutil.copy2(s, d)
            verb = "已复制"
        return ToolResult(content=f"{verb} {_norm(s)} → {_norm(d)}{note}")

    async def _fn_fs_delete(self, path: str) -> ToolResult:
        p, err = self.resolve(path, must_exist=True)
        if err:
            return ToolResult(ok=False, error=err)
        if p in self.roots():
            return ToolResult(ok=False, error="不能删除根目录本身")
        return await asyncio.to_thread(self._delete_sync, p)

    def _delete_sync(self, p: Path) -> ToolResult:
        target = self._trash_path(p)
        shutil.move(str(p), str(target))
        return ToolResult(
            content=f"已移到回收目录：{_norm(p)} → {_norm(target)}（要彻底删除请清空 trash）"
        )

    # ---------- 资产互通 ----------

    async def _fn_fs_import(self, path: str, summary: str = "", kind: str = "auto") -> ToolResult:
        p, err = self.resolve(path, must_exist=True)
        if err:
            return ToolResult(ok=False, error=err)
        if p.is_dir():
            return ToolResult(ok=False, error=f"{_norm(p)} 是目录，一次登记一个文件")
        return await asyncio.to_thread(self._import_sync, p, summary, kind)

    def _import_sync(self, p: Path, summary: str, kind: str) -> ToolResult:
        ext = p.suffix.lower()
        if kind and kind != "auto":
            named = {"script": AssetType.SCRIPT, "text": AssetType.TEXT}
            type_ = named.get(kind) or AssetType(kind)
        else:
            type_ = _TYPE_BY_EXT.get(ext) or (
                AssetType.TEXT if ext in TEXT_EXT or ext in DOC_EXT else None
            )
        if type_ is None:
            return ToolResult(ok=False, error=f"认不出 {p.name} 的类型，用 kind 指定")
        label = summary or p.name
        if type_ in (AssetType.TEXT, AssetType.SCRIPT, AssetType.SUBTITLE):
            size = p.stat().st_size
            if size > self.policy.max_read_bytes:
                return ToolResult(ok=False, error=f"{p.name} 有 {_human(size)}，太大了")
            try:
                decoded = self._text_of(p)
            except ValueError as e:
                return ToolResult(ok=False, error=str(e))
            if decoded is None:
                return ToolResult(ok=False, error=f"{p.name} 不是文本文件")
            a = self.store.create(
                decoded[0], type_=type_, summary=label, creator="human:import",
                gen_params={"source": str(p)},
            )
            return ToolResult(
                content=f"已登记为资产 {a.id}（{type_.value}，{len(decoded[0])} 字）",
                asset_ref=a.id,
            )
        a = self.store.create(
            "", type_=type_, summary=label, creator="human:import",
            gen_params={"source": str(p), "local": str(p)},
        )
        a.uri = str(p)
        a.mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        self.store.put(a)
        size = _human(p.stat().st_size)
        return ToolResult(
            content=f"已登记为资产 {a.id}（{type_.value}，{size}，原文件不动）", asset_ref=a.id
        )

    async def _fn_fs_export(self, asset_id: str, path: str, overwrite: bool = False) -> ToolResult:
        try:
            a = self.store.get(asset_id)
        except KeyError:
            near = self.store.nearest(asset_id) if hasattr(self.store, "nearest") else ""
            hint = f"；相近的有 {near}" if near else ""
            return ToolResult(ok=False, error=f"没有资产 {asset_id}{hint}")
        d, err = self.resolve(path)
        if err:
            return ToolResult(ok=False, error=err)
        media = a.type in _ASSET_EXT
        if await asyncio.to_thread(d.is_dir):
            stem = re.sub(r'[\\/:*?"<>|\r\n]+', "_", (a.summary or a.id))[:60] or a.id
            d = d / f"{stem}{_ASSET_EXT.get(a.type, '.md')}"
        if await asyncio.to_thread(d.exists) and not overwrite:
            return ToolResult(ok=False, error=f"目标已存在：{_norm(d)}（要覆盖传 overwrite=true）")
        if not media:
            try:
                text = self.store.content(asset_id)
            except KeyError:
                return ToolResult(ok=False, error=f"{asset_id} 没有可导出的文本")
            mode = "overwrite" if overwrite else "create"
            await asyncio.to_thread(self._write_sync, d, text, mode)
            return ToolResult(content=f"已导出 {asset_id} → {_norm(d)}（{len(text)} 字）")
        lc = local_copy(a)
        src = str(lc) if lc is not None else (a.uri or "")
        if src and not src.startswith(("http://", "https://")):
            return await asyncio.to_thread(self._transfer_sync, Path(src), d, overwrite, False)
        if not src:
            return ToolResult(ok=False, error=f"{asset_id} 没有文件")
        ok, why = await ffmpeg.download(src, d)
        if not ok:
            return ToolResult(ok=False, error=f"下载失败：{why}")
        return ToolResult(content=f"已下载 {asset_id} → {_norm(d)}")
