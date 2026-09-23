"""引用门（2026-09-20，用户定的规则）：没有引用成功的镜头不许生成，要在生成前拦住。

真实事故（20260921-000907 会话，第 11 集）：20 段里 4 段渲染失败 —— 两段是 5 张参考图按两个字段
双送 = 10 个，撞上 seedance "at most 9 reference images" 的 400；一段内容策略拦截；一段超时。
模型随后把这 4 段的提示词改写成英文、用 gen_videos **不带任何参考图**补生成，再拼成"完整版"，
用户看到的是后半段人物全变脸。

三道拦截：
  1. drama_render_shots 花钱之前逐段核对：引用都在包里、描述里的角色都带参考图、链接没过期且
     可访问、参考数不超上限；任一不过整批不发起
  2. 媒体层：参考数超模型上限提交前就拒；双送会超时只送实测有效的那份；短剧镜头不带参考图的
     gen_video / gen_videos 直接拦下（reference_guard）
  3. 某几段失败**不成片**，报清楚缺哪几段、该怎么补
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.drama import parse_assets
from aigc_agent.domain.functions.drama import (
    DramaFunctions,
    _Ref,
    _ref_gate_error,
    preflight_refs,
)
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.media import FakeMediaProvider, MediaGateway, MediaKind
from aigc_agent.harness.tools.provider import ToolResult

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"

LIB = {
    "characters": [
        {"baseRoleName": "陆离", "roleTotalDesc": "男 | 28岁",
         "roleCostumeList": [{"costumeName": "陆离-战术冲锋衣-[11]", "costumeDesc": "冲锋衣"}]},
        {"baseRoleName": "小满", "roleTotalDesc": "女 | 16岁",
         "roleCostumeList": [{"costumeName": "小满-破旧连帽衫-[11]", "costumeDesc": "连帽衫"}]},
    ],
    "scenes": [{"name": "隧道", "description": "x"}],
    "props": [{"name": "声波驱散器", "description": "y"}, {"name": "对讲机", "description": "z"}],
}


def _lib() -> Any:
    lib, err = parse_assets(json.dumps(LIB, ensure_ascii=False))
    assert not err, err
    return lib


def _shot(name: str, desc: str, scene: str = "[第11集-2场]") -> Any:
    from aigc_agent.domain.drama.models import ShotPrompt

    return ShotPrompt(scene_index=scene, video_name=name, duration="12s", description=desc)


# ---------------------------------------------------------------- 纯逻辑


def test_引用门_缺引用_角色没带图_人物超位():
    lib = _lib()
    resolved = {
        "陆离-战术冲锋衣-[11]": _Ref("陆离-战术冲锋衣-[11]", "https://i/luli", True),
        "隧道": _Ref("隧道", "https://i/tunnel", False),
        "声波驱散器": None,
        "小满-破旧连帽衫-[11]": _Ref("小满", "https://i/xiaoman", True),
    }
    shots = [
        _shot("14", "(隧道) (陆离-战术冲锋衣-[11]) 拿起 (声波驱散器)"),  # 道具缺图
        _shot("15", "(隧道) 小满跌坐在地，陆离低声说：“别动。”"),  # 提到两人都没引用
        _shot("16", "(隧道) (陆离-战术冲锋衣-[11]) 对着空气说：“小满在哪？”"),  # 台词里提到不算
        _shot("17", "(隧道) (陆离-战术冲锋衣-[11]) (小满-破旧连帽衫-[11]) 并肩"),
    ]
    probs = preflight_refs(shots, [0, 1, 2, 3], resolved, lib, max_people=0)
    assert len(probs) == 2
    assert "[第11集-2场] 14：引用了参考图包里没有的「声波驱散器」" in probs[0]
    assert "[第11集-2场] 15：提到了 陆离、小满 却没有引用其角色/服装参考图" in probs[1]
    assert preflight_refs(shots, [2, 3], resolved, lib) == [], "台词里提到的角色不算出镜"
    assert preflight_refs(shots, [3], resolved, lib, max_people=1)[0].endswith("拆镜或减少同框人物")
    assert preflight_refs(shots, [], resolved, lib) == [], "复用的段不查"
    err = _ref_gate_error(["x"], "as_lib")
    assert "没有发起任何生成" in err
    assert 'drama_render_assets(assets_id="as_lib", reuse=true)' in err
    assert "不要用 gen_video" in err


# ---------------------------------------------------------------- 渲染前整批拦


class Registry:
    def __init__(self, store: AssetStore, fail: set[str] | None = None) -> None:
        self.store = store
        self.fail = fail or set()
        self.calls: list[tuple[str, dict]] = []

    def videos(self) -> list[dict]:
        return [a for n, a in self.calls if n == "gen_video"]

    async def invoke(self, name: str, args: dict) -> ToolResult:
        self.calls.append((name, dict(args)))
        if name == "compose_video":
            a = self.store.create("mp4", summary="成片", creator="fake")
            a.uri = "https://fake/composed.mp4"
            self.store.put(a)
            return ToolResult(ok=True, content="拼好了", asset_ref=a.id)
        if args.get("summary") in self.fail:
            return ToolResult(
                ok=False, error="HTTP 400：This model accepts at most 9 reference images"
            )
        a = self.store.create("", type_=AssetType.VIDEO, summary=args["summary"], creator="model:x")
        a.uri = f"https://fake/{a.id}.mp4"
        self.store.put(a)
        return ToolResult(ok=True, content="ok", asset_ref=a.id)


def _seed(
    store: AssetStore, shots: list[dict], pack: dict[str, dict] | None
) -> tuple[str, str, str]:
    lib_id = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                          creator="tool:drama_assets", type_=AssetType.STORYBOARD).id
    shots_id = store.create(json.dumps(shots, ensure_ascii=False), summary="提示词",
                            creator="tool:drama_shots", parents=["sb", lib_id],
                            type_=AssetType.STORYBOARD).id
    pack_id = ""
    if pack is not None:
        pack_id = store.create(json.dumps(pack, ensure_ascii=False), summary="参考图包",
                               creator="tool:drama_render_assets", parents=[lib_id],
                               type_=AssetType.STORYBOARD).id
    return lib_id, shots_id, pack_id


PACK = {
    "陆离": {"asset": "a1", "url": "https://i/luli", "kind": "角色"},
    "小满": {"asset": "a2", "url": "https://i/xiaoman", "kind": "角色"},
    "隧道": {"asset": "a3", "url": "https://i/tunnel", "kind": "场景"},
    "声波驱散器": {"asset": "a4", "url": "https://i/device", "kind": "道具"},
    "对讲机": {"asset": "a5", "url": "https://i/radio", "kind": "道具"},
}


def _fns(store: AssetStore, reg: Registry, **drama_cfg: str) -> DramaFunctions:
    fns = DramaFunctions(None, store, registry=reg, catalog=None)
    fns.catalog = SimpleNamespace(
        drama={"voice_anchor": "false", "subtitle_gate": "false", "identity_gate": "false",
               "cut_gate": "false", **drama_cfg},
        max_concurrency=lambda k: 0,
    )
    return fns


async def test_描述里的角色没带参考图_整批不发起():
    store = AssetStore()
    shots = [
        {"scene_index": "[第11集-2场]", "video_name": "14", "video_duration": "12s",
         "description": "(隧道) (陆离-战术冲锋衣-[11]) 举起 (声波驱散器)"},
        {"scene_index": "[第11集-2场]", "video_name": "15", "video_duration": "12s",
         "description": "(隧道) 小满抓住陆离的手腕往上跑"},
    ]
    lib_id, shots_id, pack_id = _seed(store, shots, PACK)
    reg = Registry(store)
    r = await _fns(store, reg)._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert not r.ok and reg.videos() == [], "一段都不能发起，包括本来没问题的第 14 段"
    assert "[第11集-2场] 15：提到了 陆离、小满" in r.error
    assert f'drama_render_assets(assets_id="{lib_id}"' in r.error and "drama_shots" in r.error


async def test_有资产库没渲参考图_不许无参考生成():
    store = AssetStore()
    shots = [{"scene_index": "[第11集-1场]", "video_name": "1", "video_duration": "12s",
              "description": "(隧道) (陆离-战术冲锋衣-[11]) 回头"}]
    lib_id, shots_id, _ = _seed(store, shots, None)
    reg = Registry(store)
    r = await _fns(store, reg)._fn_drama_render_shots(shots_id)
    assert not r.ok and reg.videos() == []
    assert "没有参考图包" in r.error and f'drama_render_assets(assets_id="{lib_id}"' in r.error

    # 没有资产库（不是短剧项目、只是几段提示词）照旧渲，只警告
    store2 = AssetStore()
    loose = store2.create(json.dumps(shots, ensure_ascii=False), summary="提示词", creator="t").id
    reg2 = Registry(store2)
    r2 = await _fns(store2, reg2)._fn_drama_render_shots(loose)
    assert r2.ok and len(reg2.videos()) == 1 and "没有参考图包" in r2.content


async def test_链接失效_探测到就拦_网络不通只提示():
    store = AssetStore()
    shots = [{"scene_index": "[第11集-1场]", "video_name": "1", "video_duration": "12s",
              "description": "(隧道) (陆离-战术冲锋衣-[11]) 回头"}]
    _, shots_id, pack_id = _seed(store, shots, PACK)
    reg = Registry(store)
    fns = _fns(store, reg, ref_probe="true")
    probed: list[list[str]] = []

    async def dead(urls: list[str]) -> dict[str, str]:
        probed.append(list(urls))
        return {u: ("HTTP 404" if u.endswith("luli") else "") for u in urls}

    fns._probe_urls = dead  # type: ignore[method-assign]
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert not r.ok and reg.videos() == []
    assert "链接已失效" in r.error and "陆离" in r.error and "drama_refresh_refs" in r.error
    assert sorted(probed[0]) == ["https://i/luli", "https://i/tunnel"], "只探测这次要用的图"

    async def offline(urls: list[str]) -> dict[str, str]:
        return {u: "?ConnectError" for u in urls}

    fns._probe_urls = offline  # type: ignore[method-assign]
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r2.ok and len(reg.videos()) == 1 and "没验证上" in r2.content

    # 默认不探测（catalog 里没开）
    fns3 = _fns(store, Registry(store))
    assert fns3._ref_probe_on() is False


async def test_参考位不够_先省道具再省场景_人物必保():
    store = AssetStore()
    shots = [{"scene_index": "[第11集-3场]", "video_name": "24", "video_duration": "15s",
              "description": "(隧道) (陆离-战术冲锋衣-[11]) (小满-破旧连帽衫-[11]) "
              "(对讲机) (声波驱散器)"}]
    _, shots_id, pack_id = _seed(store, shots, PACK)
    reg = Registry(store)
    fns = _fns(store, reg, max_video_refs="0")  # 不留参考视频位：4 个位子全给图
    fns.catalog.get = lambda kind, model: SimpleNamespace(max_refs=4)  # type: ignore[attr-defined]
    assert fns._max_refs() == 4
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    sent = reg.videos()[0]["image"]
    assert sent == ["https://i/luli", "https://i/xiaoman", "https://i/tunnel", "https://i/radio"]
    assert "参考位不够（上限 4）" in r.content and "道具「声波驱散器」" in r.content

    # 人物本身就超位：花钱之前拦（上限 4，参考视频位留 3 → 人物最多 1）
    fns.catalog.drama["max_video_refs"] = "3"
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id, reuse=False)
    assert not r2.ok and "人物参考图 2 张超过模型参考位（最多 1）" in r2.error


async def test_某段失败_不成片_说清缺哪段():
    store = AssetStore()
    shots = [
        {"scene_index": "[第11集-1场]", "video_name": "1", "video_duration": "12s",
         "description": "(隧道) (陆离-战术冲锋衣-[11]) 回头"},
        {"scene_index": "[第11集-1场]", "video_name": "2", "video_duration": "12s",
         "description": "(隧道) (小满-破旧连帽衫-[11]) 跑"},
    ]
    _, shots_id, pack_id = _seed(store, shots, PACK)
    reg = Registry(store, fail={"[第11集-1场] 2"})
    fns = _fns(store, reg)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    assert "未成片：1 段没生成出来" in r.content and "不要用 gen_video" in r.content
    assert "[第11集-1场] 2" in r.content.split("失败：")[1]
    assert not any(n == "compose_video" for n, _ in reg.calls), "缺段不拼成片"

    reg.fail.clear()
    r2 = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r2.ok and "复用 1，新生成 1" in r2.content
    assert any(n == "compose_video" for n, _ in reg.calls), "补齐后才成片"


# ---------------------------------------------------------------- 媒体层


async def _media(tmp_path: Path) -> tuple[MediaFunctions, FakeMediaProvider]:
    provider = FakeMediaProvider(urls=["https://example.com/out.mp4"])
    gw = MediaGateway({"apimart": provider}, EventBus(), poll_interval=0.01, max_poll_interval=0.02)

    async def dl(url: str, dest: Path) -> bool:
        return False

    fns = MediaFunctions(gw, MediaCatalog.load(CATALOG_PATH), AssetStore(tmp_path / "a"),
                         prefs=OutputPrefs(tmp_path / "out", downloader=dl))
    return fns, provider


async def test_参考数超上限提交前就拒_双送会超时只送一份(tmp_path: Path):
    fns, provider = await _media(tmp_path)
    assert fns.catalog.get(MediaKind.VIDEO, "seedance-2.0").max_refs == 9

    imgs4 = [f"https://i/{k}" for k in range(4)]
    r = await fns._fn_gen_video("四张参考", model="seedance-2.0", image=imgs4)
    assert r.ok, r.error
    sent = provider.submitted[-1]
    assert sent["image"] == imgs4 and sent["image_urls"] == imgs4, "4 张双送 = 8 ≤ 9，两份都送"

    imgs5 = [f"https://i/{k}" for k in range(5)]
    r = await fns._fn_gen_video("五张参考", model="seedance-2.0", image=imgs5)
    assert r.ok, r.error
    sent = provider.submitted[-1]
    assert sent["image"] == imgs5, "5 张双送 = 10 > 9：只送实测有效的 image"
    assert "image_urls" not in sent

    n = len(provider.submitted)
    ten = [f"https://i/{k}" for k in range(10)]
    r = await fns._fn_gen_video("十张参考", model="seedance-2.0", image=ten)
    assert not r.ok and "超过 seedance-2.0 的上限 9" in r.error and "图 10" in r.error
    assert len(provider.submitted) == n, "没有提交"
    r = await fns._fn_gen_video("图加视频", model="seedance-2.0", image=imgs5,
                                video_urls=[f"https://v/{k}" for k in range(5)])
    assert not r.ok and "图 5 + 视频 5" in r.error


async def test_短剧镜头不带参考图_gen_video与gen_videos都拦(tmp_path: Path):
    fns, provider = await _media(tmp_path)
    fns.ref_guard = lambda prompt, summary: (
        "拦：短剧镜头" if "第11集" in summary or "陆离" in prompt else ""
    )

    r = await fns._fn_gen_video("a man in tactical jacket", model="seedance-2.0",
                                summary="第11集镜15-改写版")
    assert not r.ok and "拦：短剧镜头" in r.error and not provider.submitted
    r = await fns._fn_gen_video("陆离回头", model="seedance-2.0")
    assert not r.ok
    r = await fns._fn_gen_video("海边日落", model="seedance-2.0")
    assert r.ok, "不是短剧镜头照常"
    r = await fns._fn_gen_video("陆离回头", model="seedance-2.0", image=["https://i/luli"])
    assert r.ok, "带了参考图就放行"
    r = await fns._fn_gen_video("陆离的空镜", model="seedance-2.0", allow_no_refs=True)
    assert r.ok, "显式声明不需要参考图才放行"

    r = await fns._fn_gen_videos(
        [{"prompt": "a", "summary": "第11集镜16-17"}, {"prompt": "b", "summary": "第11集镜23"}],
        model="seedance-2.0",
    )
    assert not r.ok and r.error.count("拦：短剧镜头") == 2
    r = await fns._fn_gen_videos([{"prompt": "空镜"}], model="seedance-2.0", allow_no_refs=True)
    assert r.ok


def test_reference_guard_认集号镜号和资产库里的名字():
    store = AssetStore()
    store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库", creator="tool:drama_assets")
    fns = DramaFunctions(None, store, registry=None, catalog=None)
    why = fns.reference_guard("陆离走进隧道", "")
    assert "陆离" in why and "drama_render_shots" in why and "allow_no_refs" in why
    rewritten = fns.reference_guard("a man runs", "第11集镜15-声波驱散器爆亮（改写版）")
    assert "带集号或镜号" in rewritten
    assert "带集号或镜号" in fns.reference_guard("镜 3 的画面", "")
    assert fns.reference_guard("海边日落，无人", "") == ""
    assert fns.reference_guard("Cinematic close-up", "B-roll 3") == ""
    assert "小满-破旧连帽衫-[11]" in fns._known_names()
