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


def build_project_card(assets: AssetStore, output_root: Path | None = None) -> str:
    """没有短剧痕迹（状态文件、方案/目录资产、分集剧本都没有）就返回空串，不 pin。"""
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
    n_shots = len({a.gen_params.get("episode") for a in assets.find(creator="tool:drama_shots")})
    rendered = assets.find(creator="tool:drama_render_shots")
    n_rend = len({a.gen_params.get("episode") for a in rendered})
    lib = assets.find(creator="tool:drama_assets")
    refs_img = assets.find(creator="tool:drama_render_assets")
    if n_sb:
        eng.append(f"分镜脚本 {n_sb} 份")
    if n_shots:
        eng.append(f"视频提示词 {n_shots} 集")
    if n_rend:
        eng.append(f"视频片段 {n_rend} 集")
    if lib:
        eng.append(f"资产库 {lib[0].id}")
    if refs_img:
        eng.append(f"参考图 {refs_img[0].id}")
    if eng:
        lines.append("工程链：" + " · ".join(eng))

    lines.append(
        "取某一集用 find_episode(N)（连产物目录里这一集的本地文件一起列）；"
        "列资产用 list_assets(type=…, episode=N)；找本地素材用 find_materials；"
        "写剧本用 drama_write_episodes，不要在对话里写正文。"
    )
    return "\n".join(lines)
