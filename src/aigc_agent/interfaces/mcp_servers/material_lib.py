"""素材库 MCP server —— 本项目接入的第一个 MCP server。

自己写而不是接现成的，原因见 ARCHITECTURE.md 开放问题 #8：内部素材库没有
现成 server；而且它正好把 M20 的四个难点都过一遍 —— 命名空间、默认 L-external
+ 逐工具降级、描述包裹、stdio 生命周期与熔断。

**独立进程，本包里只 import domain/sensitive_paths.py**（拒绝表，只用标准库；
和主进程 fs_* 共用一份，免得两边各抄各的漏规则）。通过

    python -m aigc_agent.interfaces.mcp_servers.material_lib

启动（声明在 config/mcp_servers.yaml），根目录取环境变量 MATERIAL_LIB_ROOT，
默认 workspace/materials。

两条硬规则：
  1. **stdout 是协议通道**，任何 print 都会弄坏它 —— 日志只能走 stderr。
  2. 所有路径参数都限制在根目录之内。越界一律拒绝，不做"聪明"的修正。

工具的副作用等级由 Hub 那边的配置决定，server 自己不声明 —— 外部代码的
自我声明本来就不可信，这正是 M20 默认最严的理由。
"""

from __future__ import annotations

import mimetypes
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

# 预期内的失败（越界、不存在）要抛 ToolError：mcp 2.x 把其它异常一律包成
# "Error executing tool xxx" 并吞掉原文，客户端只会看到一句没有信息量的话。
from mcp.server.mcpserver.exceptions import ToolError

# 本包里唯一引的模块：拒绝表（只用标准库，不会把主进程那一串依赖带进来）
from ...domain.sensitive_paths import HARD_DENY, denied

KIND_EXT: dict[str, set[str]] = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"},
    "video": {".mp4", ".mov", ".mkv", ".webm", ".avi"},
    "audio": {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"},
    "doc": {".txt", ".md", ".srt", ".vtt", ".json", ".yaml", ".csv"},
}
TRASH = ".trash"
MAX_LIMIT = 100

server = MCPServer(
    "material_lib",
    instructions=(
        "本地素材库：按文件名关键词与类型（image/video/audio/doc）检索素材，"
        "返回相对路径、大小、时长等元数据，供剪辑与生成环节引用。"
    ),
)


# ---------------------------------------------------------------- 路径


def _workspace() -> Path:
    """和主进程同一个 workspace（AIGC_WORKSPACE），没设就用项目下的 workspace/。

    之前用 Path.cwd()/workspace —— 换个目录启动就在那里凭空建一个 workspace。
    """
    raw = os.environ.get("AIGC_WORKSPACE", "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[4] / "workspace"


# 导入源的拒绝规则：和主进程 fs_* 是同一份表、同一个匹配函数。
# 2026-09-29 审查 1.5：这里之前手抄了一份，漏了 rpa/profile 和 .config/gcloud ——
# 抖音、小红书的浏览器登录态能拷进素材库，主进程 fs_read 再从素材库读出来发给模型
_SENSITIVE = HARD_DENY


def _sensitive(p: Path) -> bool:
    try:
        resolved = p.resolve()
    except OSError:
        return True
    return denied(resolved, _SENSITIVE)


def root() -> Path:
    raw = os.environ.get("MATERIAL_LIB_ROOT", "")
    if not raw and len(sys.argv) > 1:
        raw = sys.argv[1]
    r = Path(raw) if raw else _workspace() / "materials"
    r.mkdir(parents=True, exist_ok=True)
    return r.resolve()


def safe(rel: str) -> Path:
    """把相对路径钉在根目录内。越界直接拒绝。"""
    base = root()
    target = (base / rel).resolve()
    if target != base and base not in target.parents:
        raise ToolError(f"路径越界：{rel!r} 不在素材库根目录内")
    return target


def kind_of(p: Path) -> str:
    ext = p.suffix.lower()
    for k, exts in KIND_EXT.items():
        if ext in exts:
            return k
    return "other"


def entry(p: Path) -> dict[str, Any]:
    st = p.stat()
    return {
        "path": p.relative_to(root()).as_posix(),
        "name": p.name,
        "kind": kind_of(p),
        "ext": p.suffix.lower(),
        "size": st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def iter_files() -> list[Path]:
    base = root()
    out: list[Path] = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if TRASH in p.relative_to(base).parts:
            continue
        out.append(p)
    return out


# ---------------------------------------------------------------- 工具


@server.tool()
def list_folders() -> dict[str, Any]:
    """列出素材库的顶层目录和每个目录里的文件数。"""
    base = root()
    folders = []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name != TRASH):
        n = sum(1 for f in d.rglob("*") if f.is_file())
        folders.append({"folder": d.name, "files": n})
    loose = sum(1 for f in base.iterdir() if f.is_file())
    return {"root": str(base), "folders": folders, "loose_files": loose}


@server.tool()
def search_materials(query: str = "", kind: str = "all", limit: int = 20) -> dict[str, Any]:
    """按文件名关键词检索素材。

    query 多个词用空格分开，全部命中才算；留空则列出全部。
    kind 可选 image / video / audio / doc / all。按修改时间倒序，最多 limit 条。
    """
    terms = [t.lower() for t in query.split() if t.strip()]
    want = kind if kind in KIND_EXT else "all"
    limit = max(1, min(int(limit or 20), MAX_LIMIT))

    hits = []
    for p in iter_files():
        if want != "all" and kind_of(p) != want:
            continue
        name = p.relative_to(root()).as_posix().lower()
        if all(t in name for t in terms):
            hits.append(p)
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "query": query,
        "kind": want,
        "total": len(hits),
        "items": [entry(p) for p in hits[:limit]],
    }


@server.tool()
def get_material(path: str) -> dict[str, Any]:
    """看一份素材的元数据。视频/音频会附上时长与分辨率（需要本机有 ffprobe）。"""
    p = safe(path)
    if not p.is_file():
        raise ToolError(f"没有这份素材：{path}")
    info = entry(p)
    info["mime"] = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    if info["kind"] in ("video", "audio"):
        info.update(probe(p))
    return info


@server.tool()
def import_material(source: str, folder: str = "") -> dict[str, Any]:
    """把一个外部文件复制进素材库（不移动原文件）。folder 留空放根目录。重名自动加序号。"""
    src = Path(source).expanduser()
    if not src.is_file():
        raise ToolError(f"源文件不存在：{source}")
    if _sensitive(src):
        # 2026-09-23 审查：源路径之前不受任何约束 —— 私钥/凭据复制进素材库后，
        # 主进程的 fs_read 就能从素材库这边读出来发给模型，绕过了 fs_* 的 deny
        raise ToolError(f"拒绝导入：{source} 是凭据/系统/私有目录下的文件")
    dest_dir = safe(folder) if folder else root()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    n = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{n}{src.suffix}"
        n += 1
    shutil.copy2(src, dest)
    return {"imported": entry(dest)}


@server.tool()
def remove_material(path: str) -> dict[str, Any]:
    """把素材移进 .trash（可恢复），不做物理删除。这是破坏性动作，Hub 侧保持 L-external。"""
    p = safe(path)
    if not p.is_file():
        raise ToolError(f"没有这份素材：{path}")
    trash = root() / TRASH
    trash.mkdir(exist_ok=True)
    dest = trash / f"{int(time.time())}_{p.name}"
    shutil.move(str(p), str(dest))
    return {"removed": path, "trash": dest.relative_to(root()).as_posix()}


# ---------------------------------------------------------------- 辅助


def probe(p: Path) -> dict[str, Any]:
    """ffprobe 拿时长/分辨率。没装就跳过，不报错 —— 元数据是加分项。"""
    if shutil.which("ffprobe") is None:
        return {}
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=width,height,codec_type",
                "-of", "default=noprint_wrappers=1", str(p),
            ],
            capture_output=True, text=True, timeout=15, encoding="utf-8", errors="replace",
        ).stdout
    except Exception:  # noqa: BLE001
        return {}
    info: dict[str, Any] = {}
    for line in out.splitlines():
        k, _, v = line.partition("=")
        if k == "duration":
            try:
                info["duration"] = round(float(v), 2)
            except ValueError:
                pass
        elif k in ("width", "height") and v.isdigit() and k not in info:
            info[k] = int(v)
    return info


def main() -> None:
    # 子进程继承的可能是 GBK 控制台；日志走 stderr，先保证它能编中文
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    root()  # 先建目录，起不来就在这里报，而不是第一次调用时
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
