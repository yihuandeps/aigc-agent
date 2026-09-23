"""项目维度（2026-09-23 审查缺口 A）：一部剧 / 一个选题 = 一个项目。

之前资产、记忆、预算、流水线全按「全库最新」或「会话」取：
  · 新剧显示「全剧已有剧本 79 集」、1–5 集「已有跳过」—— 跳过的是测试假剧本
  · /auto 拿旧剧的资产库和参考图包渲新剧，人物是上一部剧的脸
  · 台账上的「项目」其实是会话名 default —— 一个永不清零的终身上限
  · 换个文件夹启动，装回的是 default 会话的轮次、模型锁和产物目录

项目键从**产物目录**派生（用户定的：生成的东西落在他打开的那个文件夹，一个文件夹一部剧）：
目录名 + 规范化全路径的短哈希。同名不同盘的两个文件夹不会撞；大小写、斜杠写法不影响。
资产库、记忆、台账、会话快照共用这一个键；/out 换目录就是换项目。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .drama.card import read_state
from .output import slug


def normalize_root(root: Path | str) -> str:
    """同一个文件夹的不同写法（大小写、结尾斜杠、相对路径）归一成同一个串。"""
    p = os.path.normpath(os.path.abspath(str(root)))
    return os.path.normcase(p).rstrip("\\/") or p


def project_key(root: Path | str) -> str:
    """产物目录 → 项目键，形如 `西游记-3f2a9c1b`。文件名安全，可以直接当会话名。"""
    norm = normalize_root(root)
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
    name = Path(norm).name
    return f"{slug(name, limit=24) if name else 'root'}-{digest}"


def project_title(root: Path | str | None) -> str:
    """给人看的项目名：`.drama-state.json` 里的剧名优先，否则文件夹名。"""
    if root is None:
        return ""
    title = str(read_state(Path(root)).get("dramaTitle") or "").strip()
    return title or Path(os.path.abspath(str(root))).name or str(root)
