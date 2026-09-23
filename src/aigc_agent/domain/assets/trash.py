"""回收站（workspace/trash）的清点与清理。

fs_delete、覆盖前备份、面容审查、存量迁移清下来的东西都按时间戳分文件夹放在这里，
从不自动删。2026-09-23 审查：之前也从不清理 —— 现在由人显式执行
`agent assets trash --older-than N --apply`。
迁移备份（migrate-*）不清：撤销迁移要用它们的 manifest。
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_STAMP = re.compile(r"(\d{8})-(\d{6})")
_KEEP_PREFIX = ("migrate-",)


@dataclass
class TrashEntry:
    name: str
    path: Path
    age_days: float
    files: int
    size: int

    @property
    def purgeable(self) -> bool:
        return not self.name.startswith(_KEEP_PREFIX)

    @property
    def size_text(self) -> str:
        n = float(self.size)
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
            n /= 1024
        return f"{n:.1f}GB"


def _created(folder: Path) -> float:
    """文件夹名里的时间戳（20260921-194918、cleanup-20260923-015025），没有就用 mtime。"""
    m = _STAMP.search(folder.name)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            pass
    try:
        return folder.stat().st_mtime
    except OSError:
        return time.time()


def trash_folders(workspace: Path) -> list[TrashEntry]:
    """回收站里的各个文件夹，最老的在前。"""
    root = Path(workspace) / "trash"
    if not root.is_dir():
        return []
    now = time.time()
    out: list[TrashEntry] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        files = [p for p in d.rglob("*") if p.is_file()]
        size = 0
        for p in files:
            try:
                size += p.stat().st_size
            except OSError:
                pass
        age = max(0.0, (now - _created(d)) / 86400)
        out.append(TrashEntry(d.name, d, age, len(files), size))
    out.sort(key=lambda e: -e.age_days)
    return out


def purge_trash(workspace: Path, older_than_days: float) -> list[Path]:
    """永久删除 older_than_days 天以前的文件夹（迁移备份除外）。返回删掉的。"""
    gone: list[Path] = []
    for e in trash_folders(workspace):
        if e.age_days < older_than_days or not e.purgeable:
            continue
        shutil.rmtree(e.path, ignore_errors=True)
        if not e.path.exists():
            gone.append(e.path)
    return gone
