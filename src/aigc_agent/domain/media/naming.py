"""产物文件命名 —— 给后期剪辑用的、带顺序号的可读文件名（2026-09-18）。

之前产物目录里的图片/视频一律叫 `as_7f79f5c08d.mp4`：id 是给机器看的，人在剪辑软件里
对着几十个这样的文件根本排不出顺序。这里统一命名规则，生成时就按它落盘，
已经生成的用 `agent assets rename`（或 chat 里 `/rename`）按同一套规则补改。

规则（数字都补零，文件管理器和剪辑软件按名字排就是正确顺序）：

    分镜视频   第01集-03_2场_镜9-18.mp4      集-集内序号_场次_镜头范围
    参考图     参考图-角色-01_陆离.png         类别-类别内序号_名字
               参考图-服装-02_陆离-深色风衣-[8-1].png
    配方短视频 AI芯片_第03镜.mp4               主题_镜序号
    整集成片   第01集.mp4

改名只动**文件名**，资产 id、血缘、远端 url 都不变；资产上的 gen_params["local"] 跟着更新，
所以 compose / 回退 / 检索照常。只改仍叫 `as_…` 的文件 —— 用户自己改过的名字不碰，
重复跑不会来回改。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..assets.store import Asset, AssetStore, AssetType

_ILLEGAL = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')
_WS = re.compile(r"\s+")
_SCENE = re.compile(r"第\s*(\d+)\s*集\s*[-－—–]\s*(.+)")
_NUMBERED = re.compile(r"第\s*(\d+)\s*(集|场|镜)")

_EXT = {AssetType.VIDEO: ".mp4", AssetType.IMAGE: ".png", AssetType.AUDIO: ".mp3"}


def safe_name(text: str, limit: int = 80) -> str:
    """去掉 Windows 非法字符与控制字符，空白压成 -，截到 limit。"""
    s = _ILLEGAL.sub("-", (text or "").strip())
    s = _WS.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip(" .-")
    return s[:limit].rstrip(" .-") or "untitled"


def pad_numbers(text: str) -> str:
    """「第1集」「第3镜」补成两位，名字排序才对。"""
    return _NUMBERED.sub(lambda m: f"第{int(m.group(1)):02d}{m.group(2)}", text or "")


def parse_scene(scene_index: str) -> tuple[int, str]:
    """'[第1集-1场]' → (1, '1场')；解析不出集号 → (0, 原文)。"""
    raw = (scene_index or "").strip().strip("[]").strip()
    m = _SCENE.match(raw)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return 0, raw


def clip_name(scene_index: str, video_name: str, seq: int) -> str:
    """分镜视频：第01集-03_2场_镜9-18。seq 是这一集内的顺序号。"""
    ep, scene = parse_scene(scene_index)
    head = f"第{ep:02d}集-{seq:02d}" if ep else f"片段-{seq:02d}"
    shots = f"镜{video_name}" if video_name else ""
    return safe_name("_".join(p for p in (head, scene, shots) if p))


def reference_name(kind: str, seq: int, name: str) -> str:
    """参考图：参考图-角色-01_陆离。"""
    return safe_name(f"参考图-{kind}-{seq:02d}_{name}")


def recipe_shot_name(topic: str, i: int) -> str:
    """配方短视频的素材段：主题_第03镜。"""
    return safe_name(f"{topic}_第{i:02d}镜")


def episode_export_name(episode: int) -> str:
    return f"第{episode:02d}集.mp4"


def unique_path(directory: Path, name: str) -> Path:
    """同名文件已存在就加 -v2/-v3：重渲同一镜头时旧版不被覆盖。"""
    p = directory / name
    if not p.exists():
        return p
    stem, suffix = Path(name).stem, Path(name).suffix
    for k in range(2, 1000):
        q = directory / f"{stem}-v{k}{suffix}"
        if not q.exists():
            return q
    return p  # pragma: no cover


# ---------------------------------------------------------------- 已生成文件的补改


@dataclass
class Rename:
    asset_id: str
    old: Path
    new: Path
    why: str


def _local_file(asset: Asset, root: Path) -> Path | None:
    """资产的本地副本：优先 gen_params["local"]，其次产物目录里按 id 命名的默认文件。"""
    from ..assets.store import local_copy

    p = local_copy(asset)
    if p is not None:
        return p
    ext = _EXT.get(asset.type)
    if ext is None:
        return None
    sub = "videos" if asset.type is AssetType.VIDEO else "images"
    p = root / sub / f"{asset.id}{ext}"
    return p if p.exists() else None


def _default_named(path: Path, asset: Asset) -> bool:
    return path.name.startswith(asset.id)


def _shot_sequences(store: AssetStore, shots_id: str) -> dict[tuple[str, str], int]:
    """(scene_index, video_name) → 集内顺序号。按提示词资产里的完整顺序算，
    和渲染时的规则一致（limit / episode 过滤不影响编号）。"""
    from ..drama.parse import parse_shots  # 领域内引用，避免模块环

    try:
        shots, err = parse_shots(store.content(shots_id))
    except KeyError:
        return {}
    if err:
        return {}
    counters: dict[int, int] = {}
    out: dict[tuple[str, str], int] = {}
    for s in shots:
        ep, _ = parse_scene(s.scene_index)
        counters[ep] = counters.get(ep, 0) + 1
        out.setdefault((s.scene_index, s.video_name), counters[ep])
    return out


def library_reference_names(lib: Any) -> dict[str, str]:
    """资产库里每个名字 → 参考图文件名（不含扩展名）。类别内按资产库原序编号。
    渲染时和事后补改共用这一份，编号才一致。"""
    names: dict[str, str] = {}
    for i, c in enumerate(lib.characters, 1):
        names[c.name] = reference_name("角色", i, c.name)
    k = 0
    for c in lib.characters:
        for cos in c.costumes:
            k += 1
            names[cos.name] = reference_name("服装", k, cos.name)
    for i, n in enumerate(lib.scenes, 1):
        names[n.name] = reference_name("场景", i, n.name)
    for i, n in enumerate(lib.props, 1):
        names[n.name] = reference_name("道具", i, n.name)
    return names


def _reference_names(store: AssetStore, pack: Asset, images: dict[str, Any]) -> dict[str, str]:
    """参考图包里每个名字 → 目标文件名（不含扩展名）。类别与序号按资产库原序。"""
    from ..drama.parse import parse_assets

    names: dict[str, str] = {}
    if pack.parent_ids:
        try:
            lib, err = parse_assets(store.content(pack.parent_ids[0]))
        except KeyError:
            lib, err = None, "missing"
        if lib is not None and not err:
            names = library_reference_names(lib)
    # 包里有、资产库里对不上的（老包或库改过）：按包里的顺序和 kind 兜底
    for i, (name, entry) in enumerate(images.items(), 1):
        if name not in names:
            kind = str((entry or {}).get("kind") or "参考图")
            names[name] = reference_name(kind, i, name)
    return names


def plan_renames(store: AssetStore, root: Path) -> list[Rename]:
    """算出该改哪些文件。只碰仍叫 `as_…` 的默认文件名，重复跑是幂等的。"""
    plans: dict[str, Rename] = {}

    def propose(asset: Asset, stem: str, why: str) -> None:
        if asset.id in plans:
            return
        old = _local_file(asset, root)
        if old is None or not _default_named(old, asset):
            return
        new = old.with_name(safe_name(stem) + old.suffix)
        if new == old:
            return
        plans[asset.id] = Rename(asset.id, old, new, why)

    # ① 分镜视频：按渲染包里的顺序，序号按提示词资产的集内顺序
    for pack in store.find(creator="tool:drama_render_shots", newest_first=False):
        try:
            clips = json.loads(store.content(pack.id))
        except (json.JSONDecodeError, KeyError):
            continue
        seqs = _shot_sequences(store, pack.parent_ids[0]) if pack.parent_ids else {}
        fallback: dict[int, int] = {}
        for clip in clips if isinstance(clips, list) else []:
            if not isinstance(clip, dict):
                continue
            scene, name, aid = clip.get("scene", ""), clip.get("name", ""), clip.get("asset", "")
            seq = seqs.get((scene, name))
            if seq is None:
                ep, _ = parse_scene(scene)
                fallback[ep] = fallback.get(ep, 0) + 1
                seq = fallback[ep]
            try:
                asset = store.get(aid)
            except KeyError:
                continue
            propose(asset, clip_name(scene, name, seq), "分镜视频")

    # ② 参考图：按资产库的类别与原序
    for pack in store.find(creator="tool:drama_render_assets", newest_first=False):
        try:
            images = json.loads(store.content(pack.id))
        except (json.JSONDecodeError, KeyError):
            continue
        if not isinstance(images, dict):
            continue
        names = _reference_names(store, pack, images)
        for name, entry in images.items():
            aid = (entry or {}).get("asset") if isinstance(entry, dict) else None
            if not aid:
                continue
            try:
                asset = store.get(aid)
            except KeyError:
                continue
            propose(asset, names[name], "参考图")

    # ③ 其余带明确摘要的图/视频（配方短视频的素材段、模型直接生成的）：按摘要命名，序号补零
    for asset in store.find(newest_first=False):
        if asset.type not in (AssetType.VIDEO, AssetType.IMAGE) or asset.id in plans:
            continue
        summary = (asset.summary or "").strip()
        if not summary or summary.endswith("…") or summary.startswith("成片"):
            continue
        if not (asset.creator or "").startswith("model:"):
            continue
        propose(asset, pad_numbers(summary), "按摘要")

    return sorted(plans.values(), key=lambda r: (str(r.old.parent), r.new.name))


@dataclass
class RenameReport:
    done: list[Rename]
    failed: list[tuple[Rename, str]]  # (计划, 原因)

    def render_failed(self) -> str:
        if not self.failed:
            return ""
        lines = [f"  ✗ {r.old.name}：{why}" for r, why in self.failed]
        hint = "关掉占用它的播放器/剪辑软件后再跑一次即可，已改好的不会重复改。"
        return "\n".join(lines) + "\n" + hint


def apply_renames(store: AssetStore, plans: list[Rename]) -> RenameReport:
    """执行改名并更新资产上的本地路径。目标已存在就加 -v2。

    逐个文件容错：正在被播放器/剪辑软件占用的文件 Windows 不让改名（WinError 32），
    跳过它继续改其余的，最后一起报出来 —— 实测第一次上线就撞上了，之前一个异常
    让整批中断，图改了一半视频一个没动。
    """
    done: list[Rename] = []
    failed: list[tuple[Rename, str]] = []
    for r in plans:
        if not r.old.exists():
            failed.append((r, "文件已不在"))
            continue
        target = unique_path(r.new.parent, r.new.name)
        try:
            os.replace(r.old, target)
        except PermissionError:
            failed.append((r, "文件被其他程序占用，跳过"))
            continue
        except OSError as e:
            failed.append((r, f"{type(e).__name__}: {e}"))
            continue
        asset = store.get(r.asset_id)
        asset.gen_params["local"] = str(target)
        store.put(asset)
        done.append(Rename(r.asset_id, r.old, target, r.why))
    return RenameReport(done=done, failed=failed)


def write_manifest(store: AssetStore, root: Path) -> Path | None:
    """产物清单：文件名 ↔ 资产 id ↔ 说明。剪辑时对着看。"""
    rows: list[tuple[str, str, str, str]] = []
    for asset in store.find(newest_first=False):
        if asset.type not in (AssetType.VIDEO, AssetType.IMAGE):
            continue
        p = _local_file(asset, root)
        if p is None:
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        rows.append((str(rel).replace("\\", "/"), asset.id, asset.type.value, asset.summary))
    if not rows:
        return None
    rows.sort()
    lines = ["# 产物清单", "", "| 文件 | 资产 id | 类型 | 说明 |", "|---|---|---|---|"]
    lines += [f"| {f} | {aid} | {t} | {s} |" for f, aid, t, s in rows]
    path = root / "产物清单.md"
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        return None
    return path
