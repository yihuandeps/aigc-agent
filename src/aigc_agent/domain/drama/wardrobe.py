"""服装按场景绑定（2026-09-18）—— 第③步之后的确定性校正。

用户发现的问题：不同场景下角色的服装会乱变。两个原因叠在一起：
  · 第②步的资产库一个角色只有一套服装，几十个场景一套穿到底，分镜里没有可引用的换装 ID
  · 第③步靠模型自觉选服装 ID，同一场景的几个镜头也可能各用各的

现在第②步按场景分配服装（每套服装带 scenes / episodes），这里再做一遍**不靠模型**的绑定：
镜头所在场景有绑定服装的，把镜头里该角色的服装引用换成那一套的 ID；裸角色名也换成服装 ID
（否则渲视频时对不上参考图）。同一场景内穿搭因此必然一致，换场景才换装。
第②步没给某个「角色 × 场景」配服装的，报出来交给人，不猜。
"""

from __future__ import annotations

from collections import Counter

from ..media.naming import parse_scene
from .models import AssetLibrary, Costume, ShotPrompt


def scene_of(shot: ShotPrompt, lib: AssetLibrary) -> str:
    """镜头所在场景：描述里第一个能对上资产库场景名的引用（通常在「场景设定: (…)」）。"""
    names = lib.scene_names()
    return next((r for r in shot.refs() if r in names), "")


def _wearable(cos: Costume, episode: int) -> bool:
    """这一集能不能穿这套：没标集数的、或集数范围包含这一集的。"""
    return not episode or not (cos.episodes or "").strip() or cos.covers_episode(episode)


def bind_costumes(
    shots: list[ShotPrompt], lib: AssetLibrary
) -> tuple[list[str], list[str], list[str]]:
    """校正每个镜头里的角色/服装引用（原地改 description）。返回 (改动, 缺口警告, 按剧情保留的)。

    规则（2026-09-25 改）：
      · 模型点名的服装、这一集能穿的 → 按剧情保留；同一场次里同一角色统一成点名最多的那套
        （回忆戏里的婚服、同一地点临时换装，之前被场景分配表一律改回默认服装，note 也拦不住）
      · 裸角色名、或点了这一集不穿的服装 → 换成分配表里这个场景 / 这一集的那套
      · 按剧情保留、但和分配表不一致的 → 报出来给人看
    """
    changes: list[str] = []
    warnings: list[str] = []
    kept: list[str] = []
    warned: set[tuple[str, str]] = set()
    # ① 每个场次里，每个角色被显式点名、而且这一集能穿的服装 → 取点名最多的那套
    tally: dict[tuple[str, str], Counter[str]] = {}
    scene_name: dict[str, str] = {}
    for s in shots:
        episode, _ = parse_scene(s.scene_index)
        scene_name.setdefault(s.scene_index, scene_of(s, lib))
        for ref in s.refs():
            ch = lib.character_of(ref)
            if ch is None or ref == ch.name:
                continue
            cos = next((c for c in ch.costumes if c.name == ref), None)
            if cos is not None and _wearable(cos, episode):
                tally.setdefault((s.scene_index, ch.name), Counter())[ref] += 1
    picked = {key: cnt.most_common(1)[0][0] for key, cnt in tally.items()}
    # ② 逐镜校正
    for s in shots:
        episode, _ = parse_scene(s.scene_index)
        scene = scene_name.get(s.scene_index) or scene_of(s, lib)
        for ref in s.refs():
            ch = lib.character_of(ref)
            if ch is None:
                continue
            name = picked.get((s.scene_index, ch.name))
            if name is not None:
                if name != ref:
                    s.description = s.description.replace(f"({ref})", f"({name})")
                    changes.append(f"{s.scene_index}：{ref} → {name}（同一场统一）")
                continue
            want = ch.costume_for(scene, episode)
            if want is None:
                if ref == ch.name and ch.costumes:
                    # 裸角色名：至少换成一套服装 ID，渲视频时才对得上参考图。先挑这一集能穿的；
                    # 这一集一套能穿的都没有是缺口，要报出来（2026-09-26：之前默认第一套、一声
                    # 不吭 —— 第一套可能是第 11–20 集的戏服）
                    want = next((c for c in ch.costumes if _wearable(c, episode)), ch.costumes[0])
                    key = (ch.name, scene)
                    if not _wearable(want, episode) and key not in warned:
                        warned.add(key)
                        where = scene or "未识别场景"
                        warnings.append(
                            f"{ch.name} 在「{where}」（{s.scene_index}）这一集没有能穿的服装，"
                            f"先用 {want.name}（它标的是第 {want.episodes} 集）"
                        )
                else:
                    key = (ch.name, scene)
                    if key not in warned:
                        warned.add(key)
                        where = scene or "未识别场景"
                        warnings.append(
                            f"{ch.name} 在「{where}」（{s.scene_index}）没有绑定的服装，沿用 {ref}"
                        )
                    continue
            if want.name != ref:
                s.description = s.description.replace(f"({ref})", f"({want.name})")
                changes.append(f"{s.scene_index}：{ref} → {want.name}")
    # ③ 按剧情保留、但和分配表不一致的：报出来
    for (scene_index, ch_name), name in picked.items():
        ch = lib.character_of(name)
        episode, _ = parse_scene(scene_index)
        planned = ch.costume_for(scene_name.get(scene_index, ""), episode) if ch else None
        if planned is not None and planned.name != name:
            kept.append(f"{scene_index}：{ch_name} 穿 {name}（分配表是 {planned.name}）")
    return changes, warnings, kept


def unbound_costumes(lib: AssetLibrary) -> list[str]:
    """既没标场景也没标集数的服装 —— 第③步只能拿它当兜底，换装时机说不清。"""
    return [
        cos.name
        for c in lib.characters
        for cos in c.costumes
        if not cos.scenes and not cos.episodes
    ]
