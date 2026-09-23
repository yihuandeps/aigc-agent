"""从 awesome-seedance 模板派生的两份配方（2026-09-23，用户要的）。

  ugc-vlog    ← handheld-ugc-vlog（案例库最富的一类，92 条）：用相机缺陷换真实感
  product-ad  ← product-commercial-shotlist（27 条）：先写死广告美学词，微距指名拍什么

两份是两个极端，案例库的结论是**混着写 = 塑料感**，所以这里钉住的都是"各自的味道
只出现在各自的 prompt 里"，以及配方骨架能被流水线正常消费。
"""

from __future__ import annotations

from pathlib import Path

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.short_video import ShortVideoFunctions
from aigc_agent.domain.pipeline.recipe import list_recipes, load_recipe

RECIPES = Path(__file__).resolve().parents[1] / "config" / "recipes"


# ---------------------------------------------------------------- 手持 vlog


def test_手持vlog_风格骨架():
    r = load_recipe("ugc-vlog", RECIPES)
    assert r.style_label == "手持vlog" and r.duration_bounds == (15, 45)
    assert r.style.get("footage") == "mixed", "用户实拍优先，缺的镜头按同一质感生成"
    assert r.cut_max == 3.0 and r.cut_min == 1.5
    assert r.grounded, "vlog 选题要接热点里的'一件小事'"
    assert r.total_seconds == 30 and r.script_chars > 0


def test_手持vlog_靠相机缺陷换真实感():
    """案例库：真实感是用'毛病'买来的。指名器材 + 缺陷清单 + 手动关掉电影感。"""
    p = load_recipe("ugc-vlog", RECIPES).shot_prompt("她把咖啡倒进杯子", "subtle")
    assert "手抖" in p and "对焦" in p and "曝光" in p, "相机缺陷清单要在每镜 prompt 里"
    assert "iPhone" in p, "要指名具体器材，不要笼统写'真实'"
    assert "no cinematic emulation" in p and "稳定器" in p and "磨皮" in p
    assert "胸口以上" in p, "不推大特写（会暴露 AI 脸）"
    assert "颗粒" in p, "realism: auto → 全局真实感档位挂上"
    assert "luxury" not in p and "体积光" not in p, "vlog 里不能混进广告光"


def test_手持vlog_台词跟着镜头走():
    r = load_recipe("ugc-vlog", RECIPES)
    hint = str(r.shots.get("prompt_hint"))
    assert "8 个词" in hint and "焊在动作里" in hint
    assert "单向递进" in hint, "身体状态只能单向递进，给模型不可逆的时间线"
    assert "塑料感" in hint, "要说清为什么不能混电影感"
    script = str(r.voiceover.get("script_hint")).format(chars=r.script_chars)
    assert "第一人称" in script and str(r.script_chars) in script


# ---------------------------------------------------------------- 产品广告


def test_产品广告_风格骨架():
    r = load_recipe("product-ad", RECIPES)
    assert r.style_label == "产品广告" and r.duration_bounds == (8, 20)
    assert r.style.get("footage") == "generate", "产品图是身份锁，不当实拍素材剪"
    assert not r.grounded, "产品广告选题来自产品本身，不抓热榜"
    assert r.cut_max == 3.0 and r.cut_min == 1.0, "单镜低于 1 秒模型解析不出动作"
    assert r.models.get("video_tier") == "quality", "投放级成片"
    assert not r.voiceover.get("enabled") and not r.subtitle.get("enabled"), "文字后期加"


def test_产品广告_先写死广告美学词_不挂皮肤真实感():
    r = load_recipe("product-ad", RECIPES)
    p = r.shot_prompt("金属反光扫过瓶身", "subtle")
    assert "luxury advertising aesthetic" in p and "anamorphic" in p and "体积光" in p
    assert "颗粒" not in p and "毛孔" not in p, "产品片没人，皮肤真实感段不该挂"
    assert "手抖" not in p and "iPhone" not in p, "广告里不能混进 UGC 质感"


def test_产品广告_微距要指名拍什么_收尾留干净英雄帧():
    hint = str(load_recipe("product-ad", RECIPES).shots.get("prompt_hint"))
    assert "指名拍什么" in hint and "泡沫" in hint and "液体" in hint
    assert "一拍只放一种物理效果" in hint
    assert "英雄帧" in hint and "后期上字" in hint
    assert "logo" in hint and "印刷标签" in hint, "品牌文字几乎必错，要避开"
    assert "@图片1" in hint, "产品图当身份锁要点名"


# ---------------------------------------------------------------- 接进流水线


def test_两份配方都在风格列表里():
    keys = [r.key for r in list_recipes(RECIPES)]
    assert "ugc-vlog" in keys and "product-ad" in keys
    assert len(keys) == 5, f"配方数变了：{keys}"


async def test_风格列表能看到新配方():
    fns = ShortVideoFunctions(None, AssetStore(), registry=None, catalog=None, recipes_dir=RECIPES)
    r = await fns.invoke("list_video_styles", {})
    assert r.ok, r.error
    assert "ugc-vlog · 手持vlog" in r.content and "15–45s" in r.content
    assert "product-ad · 产品广告" in r.content and "8–20s" in r.content
    assert "实拍与生成混用" in r.content and "画面靠 AI 生成" in r.content
