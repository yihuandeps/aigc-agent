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

from ..media.naming import parse_scene
from .models import AssetLibrary, ShotPrompt


def scene_of(shot: ShotPrompt, lib: AssetLibrary) -> str:
    """镜头所在场景：描述里第一个能对上资产库场景名的引用（通常在「场景设定: (…)」）。"""
    names = lib.scene_names()
    return next((r for r in shot.refs() if r in names), "")


def bind_costumes(shots: list[ShotPrompt], lib: AssetLibrary) -> tuple[list[str], list[str]]:
    """把每个镜头里的角色/服装引用换成该场景绑定的服装 ID（原地改 description）。

    返回 (改动说明, 缺口警告)。规则：
      场景显式绑定的服装 > 集数覆盖的服装 > 裸角色名换成第一套 > 不动并警告
    """
    changes: list[str] = []
    warnings: list[str] = []
    warned: set[tuple[str, str]] = set()
    for s in shots:
        episode, _ = parse_scene(s.scene_index)
        scene = scene_of(s, lib)
        for ref in s.refs():
            ch = lib.character_of(ref)
            if ch is None:
                continue
            want = ch.costume_for(scene, episode)
            if want is None:
                if ref == ch.name and ch.costumes:
                    # 裸角色名：至少换成一套服装 ID，渲视频时才对得上参考图
                    want = ch.costumes[0]
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
    return changes, warnings


def unbound_costumes(lib: AssetLibrary) -> list[str]:
    """既没标场景也没标集数的服装 —— 第③步只能拿它当兜底，换装时机说不清。"""
    return [
        cos.name
        for c in lib.characters
        for cos in c.costumes
        if not cos.scenes and not cos.episodes
    ]
