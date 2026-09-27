"""项目卡 —— 短剧长任务的持久事实，每轮 pin 进上下文（2026-09-17）。

关键词记忆对这类任务不管用：35 条记忆召回命中 0 次，因为用户输入几乎全是「继续」。
而模型真正需要的事实其实是结构化的：剧名、总集数、写到第几集、各产物的最新
资产 id。这些从资产库和 .drama-state.json 直接算出来，约 500 token，每轮更新，
放在 pre_input 位 —— 不占历史轮的缓存前缀。

它还顺带解决了"找不到自己的资产"：有了卡片，模型不用 list_assets 全吐、
不用凭记忆写 id。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..assets.store import AssetStore, AssetType
from ..media.naming import parse_scene
from .format import (
    EpisodeFormat,
    copied_opening,
    dialogue_rushed,
    has_flash,
    lost_lines,
    prompt_problems,
    split_script,
)
from .models import Episode, episode_ranges
from .parse import parse_assets, parse_episodes, parse_shots
from .refpack import pick_pack, same_library

STATE_FILE = ".drama-state.json"
CARD_PIN = "project_card"


def _ranges(nums: list[int]) -> str:
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


def read_state(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {}
    try:
        data = json.loads((root / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# 视频提示词资产 → (覆盖哪几集, 其中漏镜头 / 被压缩的集)。资产不可变，按 id 只算一次
_PROMPT_CHECK: dict[str, tuple[set[int], set[int]]] = {}


def _prompt_check(assets: AssetStore, a: Any) -> tuple[set[int], set[int]]:
    hit = _PROMPT_CHECK.get(a.id)
    if hit is not None:
        return hit
    covered: set[int] = set()
    broken: set[int] = set()
    try:
        shots, err = parse_shots(assets.content(a.id))
        if not err:
            covered = {parse_scene(s.scene_index)[0] for s in shots} - {0}
            if a.parent_ids:
                eps, err2 = parse_episodes(assets.content(a.parent_ids[0]))
                if not err2:
                    for e in eps:
                        if e.index in covered and prompt_problems(shots, [e]):
                            broken.add(e.index)
    except KeyError:
        pass
    _PROMPT_CHECK[a.id] = (covered, broken)
    return covered, broken


def _broken_prompts(assets: AssetStore) -> list[int]:
    """每一集最新的那份视频提示词里，漏了分镜镜头或被压缩的集。"""
    seen: set[int] = set()
    out: set[int] = set()
    for a in assets.find(creator="tool:drama_shots"):
        covered, broken = _prompt_check(assets, a)
        fresh = covered - seen
        seen |= covered
        out |= fresh & broken
    return sorted(out)


# 分镜资产 → 各集分镜（资产不可变，按 id 只解析一次）
_BOARD_EPS: dict[str, list[Episode]] = {}
# (分镜 id, 集号, 剧本 id) → (台词压快了?, 比剧本少了几句台词)
_BOARD_CHECK: dict[tuple[str, int, str, bool], tuple[bool, int, bool, bool]] = {}


def _board_eps(assets: AssetStore, a: Any) -> list[Episode]:
    hit = _BOARD_EPS.get(a.id)
    if hit is None:
        try:
            eps, err = parse_episodes(assets.content(a.id))
        except KeyError:
            eps, err = [], "gone"
        hit = [] if err else eps
        _BOARD_EPS[a.id] = hit
    return hit


def _board_problems(
    assets: AssetStore, total: int, fmt: EpisodeFormat | None = None
) -> tuple[list[int], dict[int, int], list[int], list[int]]:
    """每一集最新的分镜：把台词压快了的集；比剧本少了台词的集 → 少了几句；开场加了闪前的集；
    开场把后面的台词复制了一遍的集（只在跟剧本走的项目里查，fmt.follow_script）。

    total（全剧集数）已知时只看这部剧的集号 —— 没迁移的旧剧分镜在这里也看得见。"""
    seen: set[int] = set()
    rushed: list[int] = []
    lost: dict[int, int] = {}
    flashed: list[int] = []
    copied: list[int] = []
    follow = bool(fmt is not None and fmt.follow_script)
    for a in assets.find(creator="tool:drama_storyboard"):
        for e in _board_eps(assets, a):
            n = e.index
            if n <= 0 or n in seen or (total and n > total):
                continue
            seen.add(n)
            # 核对分镜用它当初拆的那份剧本（分镜的第一个父资产）；没记才退回这一集最新的剧本
            # （2026-09-26：之前一律拿最新剧本比，剧本改过一版，旧分镜就被判「少了台词」）
            script = _source_script(assets, a) or assets.script_of(n)
            key = (a.id, n, script.id if script is not None else "", follow)
            hit = _BOARD_CHECK.get(key)
            if hit is None:
                text = ""
                if script is not None:
                    try:
                        text = assets.content(script.id)
                    except KeyError:
                        text = ""
                src = split_script(text).get(n) or text
                hit = (
                    dialogue_rushed(e), lost_lines(src, e.desc) if src else 0, has_flash(e.desc),
                    bool(follow and src and fmt and copied_opening(e.title, src, e.desc, fmt)),
                )
                _BOARD_CHECK[key] = hit
            if hit[0]:
                rushed.append(n)
            if hit[1]:
                lost[n] = hit[1]
            if hit[2]:
                flashed.append(n)
            if hit[3]:
                copied.append(n)
    return sorted(rushed), lost, sorted(flashed), sorted(copied)


def _source_script(assets: AssetStore, board: Any) -> Any:
    """分镜是从哪份剧本拆的（drama_storyboard 落库时 parents=[剧本]）。不是剧本就返回 None。"""
    for pid in (board.parent_ids or [])[:1]:
        try:
            src = assets.get(pid)
        except KeyError:
            return None
        return src if src.type is AssetType.SCRIPT else None
    return None


def latest_storyboards(assets: AssetStore) -> dict[int, str]:
    """每一集最新的分镜资产 id（一份分镜可能覆盖好几集）。"""
    out: dict[int, str] = {}
    for a in assets.find(creator="tool:drama_storyboard"):
        for e in _board_eps(assets, a):
            if e.index > 0:
                out.setdefault(e.index, a.id)
    return out


def _outdated_prompts(assets: AssetStore, total: int, lib_id: str) -> list[int]:
    """每一集最新的视频提示词里，是按旧分镜或旧资产库出的集（落库时 parents = [分镜, 资产库]）。

    2026-09-26：20 集分镜全部重拆之后，第 11、15 集的旧提示词是「完整」的（和它自己的旧分镜
    对得上），漏镜头的检查拦不住，照着渲就是被压快的旧版。"""
    boards = latest_storyboards(assets)
    seen: set[int] = set()
    out: set[int] = set()
    for a in assets.find(creator="tool:drama_shots"):
        covered, _ = _prompt_check(assets, a)
        fresh = covered - seen
        seen |= covered
        ps = list(a.parent_ids or [])
        for n in fresh:
            if total and n > total:
                continue
            old_board = bool(ps) and n in boards and ps[0] != boards[n]
            # 同一条增量链上的资产库算同一套（定稿后只增量补新集，已有条目没动，2026-09-26）
            old_lib = len(ps) > 1 and bool(lib_id) and not same_library(assets, ps[1], lib_id)
            if old_board or old_lib:
                out.add(n)
    return sorted(out)


# 资产库 → (覆盖哪几集, 是不是按服装集数推出来的)。资产不可变，只算一次
_LIB_COVERS: dict[str, tuple[frozenset[int], bool]] = {}


def _lib_covers(assets: AssetStore, a: Any) -> tuple[frozenset[int], bool]:
    """资产库覆盖哪几集：2026-09-26 起记在 gen_params.covers（生成时用了哪几集的剧本）；
    老的库没记，按服装标的集数范围推。推不出来（有「全集」、有服装没标集数）返回空。"""
    hit = _LIB_COVERS.get(a.id)
    if hit is not None:
        return hit
    rec = a.gen_params.get("covers") or []
    hit = (frozenset(int(x) for x in rec if isinstance(x, int) or str(x).isdigit()), False)
    if not rec:
        hit = (frozenset(), True)
        try:
            lib, err = parse_assets(assets.content(a.id))
        except KeyError:
            lib, err = None, "gone"
        spans: list[list[tuple[int, int]]] = []
        if lib is not None and not err:
            spans = [episode_ranges(cos.episodes) for c in lib.characters for cos in c.costumes]
        if spans and all(s and all(hi < 9999 for _, hi in s) for s in spans):
            hit = (frozenset(n for s in spans for lo, hi in s for n in range(lo, hi + 1)), True)
    _LIB_COVERS[a.id] = hit
    return hit


def _own_script_episodes(assets: AssetStore) -> set[int]:
    """这个项目自己的剧本有哪几集（没迁移的旧剧没有项目键，不算）。"""
    scope = getattr(assets, "project", "") or ""
    return {
        n
        for a in assets.find(type_=AssetType.SCRIPT)
        if (not scope or a.project == scope) and (n := assets.episode_of(a)) > 0
    }


def build_project_card(
    assets: AssetStore, output_root: Path | None = None, fmt: EpisodeFormat | None = None
) -> str:
    """没有短剧痕迹（状态文件、方案/目录资产、分集剧本都没有）就返回空串，不 pin。
    fmt：这个项目的一集规格（跟剧本走的项目要多查「开场复制了后面的台词」）。"""
    state = read_state(output_root)
    # 真版本（有创建者、不是占位符、够长优先），不是「最新的一份」—— 之前最新的三份全是测试桩
    plan = assets.best_doc(AssetType.OUTLINE, "创作方案")
    chars = assets.best_doc(AssetType.OUTLINE, "角色档案")
    outline = assets.best_doc(AssetType.OUTLINE, "分集目录")
    done = assets.episodes_done()
    if not (state or plan or outline or done):
        return ""

    total = int(state.get("totalEpisodes") or 0)
    lines = ["## 项目卡（自动生成，每轮更新；资产 id 以这里为准，不要凭记忆写）"]

    head: list[str] = []
    if state.get("dramaTitle"):
        head.append(f"剧名《{state['dramaTitle']}》")
    if state.get("genre"):
        g = state["genre"]
        head.append("题材 " + ("/".join(map(str, g)) if isinstance(g, list) else str(g)))
    if total:
        head.append(f"共 {total} 集")
    if state.get("ethnicity"):
        head.append(f"面孔 {state['ethnicity']}")
    if state.get("language"):
        head.append(f"台词 {state['language']}")
    if state.get("currentStep"):
        head.append(f"阶段 {state['currentStep']}")
    if head:
        lines.append(" · ".join(head))

    refs: list[str] = []
    if plan:
        refs.append(f"创作方案 {plan.id}")
    if chars:
        refs.append(f"角色档案 {chars.id}")
    if outline:
        refs.append(f"分集目录 {outline.id}")
    if refs:
        lines.append(" · ".join(refs))

    if done:
        progress = f"剧本已完成：第 {_ranges(done)} 集"
        if total:
            progress += f"（{len(done)}/{total}）"
            missing = [n for n in range(1, total + 1) if n not in set(done)]
            if missing:
                progress += f" · 缺：{_ranges(missing)}"
        recent = []
        for n in done[-3:]:
            a = assets.episode_assets(n).get("剧本")
            if a is not None:
                recent.append(f"第{n}集 {a.id}")
        if recent:
            progress += "\n最近三集：" + "，".join(recent)
        lines.append(progress)
    elif total:
        lines.append(f"剧本：一集都还没写（0/{total}）")

    eng: list[str] = []
    n_sb = len(assets.find(creator="tool:drama_storyboard"))
    # 没标集号的（整季一次出的旧提示词、手动存的）不算「一集」（2026-09-26）
    n_shots = len(
        {a.gen_params.get("episode") for a in assets.find(creator="tool:drama_shots")} - {None, 0}
    )
    rendered = assets.find(creator="tool:drama_render_shots")
    n_rend = len({a.gen_params.get("episode") for a in rendered} - {None, 0})
    lib = assets.find(creator="tool:drama_assets")
    # 只认这套资产库的参考图包（没迁移的旧剧包在这里也看得见，同名角色会被当成这部剧的）
    refs_img = pick_pack(
        assets, assets.find(creator="tool:drama_render_assets"), lib[0].id if lib else ""
    )
    if n_sb:
        eng.append(f"分镜脚本 {n_sb} 份")
    if n_shots:
        eng.append(f"视频提示词 {n_shots} 集")
    if n_rend:
        eng.append(f"视频片段 {n_rend} 集")
    if lib:
        covers, _ = _lib_covers(assets, lib[0])
        span = f"（第 {_ranges(sorted(covers))} 集）" if covers else ""
        eng.append(f"资产库 {lib[0].id}{span}")
    if refs_img:
        eng.append(f"参考图 {refs_img}")
    if eng:
        lines.append("工程链：" + " · ".join(eng))
    # 2026-09-25：旧版 drama_shots 按「一集 4 分钟」压总时长，20 集里 18 集的视频提示词丢了镜头，
    # 模型对着它们报「干净、已完成」。每一集最新的提示词和分镜对不上的，在这里标出来
    broken = _broken_prompts(assets)
    if broken:
        lines.append(
            f"⚠ 第 {_ranges(broken)} 集的视频提示词漏了分镜镜头或被压缩：不能算完成，"
            "渲染时会被拦下；要按集重跑 drama_shots（现在的会逐镜核对、时长跟着分镜走）"
        )
    old_prompts = [
        n for n in _outdated_prompts(assets, total, lib[0].id if lib else "") if n not in broken
    ]
    if old_prompts:
        lines.append(
            f"⚠ 第 {_ranges(old_prompts)} 集的视频提示词是按旧的分镜或资产库出的：不能拿去渲，"
            "要按集重跑 drama_shots"
        )
    # 分镜本身也要核：旧规格下有几集把台词压快了（光念台词就比整集长）、有一集删了台词
    rushed, lost, flashed, copied = _board_problems(assets, total, fmt)
    if rushed:
        lines.append(
            f"⚠ 第 {_ranges(rushed)} 集的分镜把台词压快了（光念台词就占满了镜头总时长，渲出来"
            "念不完）：不能算完成。先请用户用 /length 放宽集长（或 /length auto 跟剧本走），"
            "再重拆这几集的分镜、重出视频提示词"
        )
    if lost:
        shown = "、".join(f"第{n}集少 {k} 句" for n, k in sorted(lost.items()))
        lines.append(f"⚠ 分镜比剧本少了台词（{shown}）：不能算完成，要重拆这几集的分镜")
    if copied:
        lines.append(
            f"⚠ 第 {_ranges(copied)} 集的分镜开场把后面的台词复制了一遍（跟剧本走的剧不许，"
            "2026-09-27 用户定的）：重拆这几集的分镜 —— 开场用剧本自己的，要加闪前只许画面"
        )
    if flashed:
        # 跟剧本走的剧开场是铺垫时加的无台词闪前（2026-09-27）：列出来，用户不要哪集就去掉
        lines.append(
            f"第 {_ranges(flashed)} 集的开场加了无台词画面闪前（剧本开场是铺垫才加的）："
            "用户不要哪集，就重拆那一集的分镜、note 写「开场不加闪前」"
        )
    # 资产库只覆盖半部剧（2026-09-26：20 集一次生成失败后，主模型拆成两段各生成了一份）
    if lib:
        covers, guessed = _lib_covers(assets, lib[0])
        want = set(range(1, total + 1)) if total else _own_script_episodes(assets)
        gap = sorted(want - covers) if covers else []
        if gap:
            what = "的服装只排到" if guessed else "只用了"
            lines.append(
                f"⚠ 最新的资产库 {lib[0].id} {what}第 {_ranges(sorted(covers))} 集，"
                f"第 {_ranges(gap)} 集不在里面：一部剧要一份覆盖全剧的资产库。把全剧各集的剧本 id "
                "一次传给 drama_assets 重新生成；不要分段生成、不要手工合并（同一个角色会被描述成"
                "两张脸，戏服前后对不上）"
            )

    lines.append(
        "取某一集用 find_episode(N)（连产物目录里这一集的本地文件一起列）；"
        "列资产用 list_assets(type=…, episode=N)；找本地素材用 find_materials；"
        "写剧本用 drama_write_episodes，不要在对话里写正文。"
    )
    return "\n".join(lines)
