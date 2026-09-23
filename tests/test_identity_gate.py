"""人物一致性门（2026-09-18 用户反馈：参考图传了也漂，人物漂移严重）。

三道：① 提示词里 @图片N 逐张点名参考图（只传图不点名模型常常不用）；② 生成后把参考图和
抽帧一起给视觉模型逐人判断是不是同一个人，不合格把差异写进提示词重生成，都没过留分最高的
并标出来；③ 参考图链接过期（24h）前渲染会提示先 drama_refresh_refs 用本地副本刷新。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.drama.identity import (
    IDENTITY_RETRY,
    REFERENCE_HEAD,
    identity_check_messages,
    identity_retry_prompt,
    parse_identity_verdict,
    reference_block,
)
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.domain.output import OutputPrefs
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.media import FakeMediaProvider, MediaGateway
from aigc_agent.harness.tools.provider import ToolResult

# ---------------------------------------------------------------- 纯逻辑


def test_参考锁定段_按顺序点名():
    text = reference_block([("角色", "陆离"), ("服装", "陆离-风衣-[全集]"), ("场景", "地铁车厢")])
    assert text.startswith(REFERENCE_HEAD)
    assert "@图片1 = 角色「陆离」本人" in text and "@图片2 = 「陆离-风衣-[全集]」这套服装" in text
    assert "@图片3 = 场景「地铁车厢」" in text
    assert reference_block([]) == ""


def test_参考锁定段带不继承声明():
    """awesome-seedance 案例库的结论：能不能锁住人，差别就在"不继承"那一句 ——
    我们的参考图是白底证件照，不说清楚，白背景和站姿会一起被搬进镜头。"""
    from aigc_agent.domain.drama.identity import NOT_INHERIT

    text = reference_block([("角色", "陆离")])
    assert NOT_INHERIT in text
    assert "纯白背景" in text and "站姿" in text and "同一张脸" in text
    assert text.index("@图片1") < text.index("纯白背景"), "先点名、再说不继承什么"


def test_重生成提示把差异写进去_幂等():
    p = identity_retry_prompt("正文", ["发型变短", "衣服换了"])
    assert p.startswith("【重新生成】") and "发型变短；衣服换了" in p and p.endswith("正文")
    assert identity_retry_prompt(p, ["x"]) == p
    assert "长相或服装变了" in identity_retry_prompt("正文", [])
    assert "{issues}" not in IDENTITY_RETRY.format(issues="x")


def test_校验消息_参考图在前_结果在后():
    refs = [("陆离", "data:a"), ("小满", "data:b")]
    msgs = identity_check_messages(refs, ["data:f1", "data:f2"], True)
    parts = msgs[0]["content"]
    assert parts[0]["type"] == "text" and "同一个人" in parts[0]["text"]
    labels = [p["text"] for p in parts if p["type"] == "text"]
    assert "参考图：陆离" in labels and any("视频抽帧" in t for t in labels)
    images = [p for p in parts if p["type"] == "image_url"]
    assert [p["image_url"]["url"] for p in images] == ["data:a", "data:b", "data:f1", "data:f2"]
    assert images[0]["image_url"]["detail"] == "high" and images[-1]["image_url"]["detail"] == "low"


def test_判定解析_按通过分():
    v = parse_identity_verdict(
        '{"pass": true, "score": 6, "issues": ["发型"], "characters": {"陆离": 6}}'
    )
    assert v.passed is False and v.score == 6 and v.issues == ["发型"]
    mixed = parse_identity_verdict('{"score": 8, "characters": {"陆离": 8, "小满": 5}}')
    assert mixed.passed is False, "任一角色低于通过分就不算过"
    assert parse_identity_verdict('{"score": 9, "characters": {"陆离": 9}}', pass_score=7).passed
    bad = parse_identity_verdict("看不清")
    assert bad.passed and bad.note and bad.score == -1


# ---------------------------------------------------------------- 渲染链路


class Registry:
    def __init__(self, store: AssetStore, tmp: Path) -> None:
        self.store = store
        self.tmp = tmp
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
        type_ = AssetType.VIDEO if name == "gen_video" else AssetType.IMAGE
        a = self.store.create("", type_=type_, summary=args["summary"], creator="model:x")
        ext = ".mp4" if name == "gen_video" else ".png"
        local = self.tmp / f"{a.id}{ext}"
        local.write_bytes(b"x")
        a.uri = f"https://fake/{a.id}{ext}"
        a.gen_params["local"] = str(local)
        self.store.put(a)
        return ToolResult(ok=True, content="ok", asset_ref=a.id)


LIB = {
    "characters": [
        {"baseRoleName": "陆离", "roleTotalDesc": "男 | 35岁",
         "roleCostumeList": [{"costumeName": "陆离-风衣-[全集]", "costumeDesc": "风衣"}]},
    ],
    "scenes": [{"name": "地铁车厢", "description": "x"}],
    "props": [],
}
SHOTS = [
    {"scene_index": "[第1集-1场]", "video_name": "1-4", "video_duration": "10s",
     "description": "(地铁车厢) (陆离-风衣-[全集]) 说：“别回头。”"},
]


def _seed(store: AssetStore, tmp: Path, portrait_age_h: float = 0) -> tuple[str, str]:
    lib_id = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库", creator="t").id
    shots_id = store.create(
        json.dumps(SHOTS, ensure_ascii=False), summary="提示词", creator="tool:drama_shots",
        parents=["sb", lib_id],
    ).id
    portrait = store.create("", type_=AssetType.IMAGE, summary="角色·陆离", creator="model:x")
    local = tmp / "luli.png"
    local.write_bytes(b"png")
    portrait.uri = "https://img/luli.png"
    portrait.gen_params["local"] = str(local)
    portrait.created_at = time.time() - portrait_age_h * 3600
    store.put(portrait)
    scene = store.create("", type_=AssetType.IMAGE, summary="场景·地铁车厢", creator="model:x")
    scene.uri = "https://img/car.png"
    store.put(scene)
    pack = {
        "陆离": {"asset": portrait.id, "url": "https://img/luli.png", "kind": "角色"},
        "地铁车厢": {"asset": scene.id, "url": "https://img/car.png", "kind": "场景"},
    }
    pack_id = store.create(
        json.dumps(pack, ensure_ascii=False), summary="参考图包",
        creator="tool:drama_render_assets", parents=[lib_id],
    ).id
    return shots_id, pack_id


def _fns(store: AssetStore, reg: Registry, verdicts: list[dict[str, Any]] | None = None) -> Any:
    calls: list[list[dict[str, Any]]] = []
    seq = list(verdicts or [])

    async def chat(role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        calls.append(messages)
        return SimpleNamespace(text=json.dumps(seq.pop(0) if seq else {"score": 9}))

    fns = DramaFunctions(SimpleNamespace(chat=chat), store, registry=reg, catalog=None)
    fns.catalog = SimpleNamespace(
        drama={"identity_gate": "true", "identity_retries": "1", "subtitle_gate": "false",
               "voice_anchor": "false"},
        max_concurrency=lambda k: 0,
    )
    fns._chat_calls = calls  # type: ignore[attr-defined]

    async def frames(video: Path, count: int = 4) -> list[bytes]:
        return [b"f1", b"f2"]

    fns._frames_of = frames  # type: ignore[method-assign]
    return fns


async def test_提示词里逐张点名参考图(tmp_path: Path):
    store = AssetStore()
    shots_id, pack_id = _seed(store, tmp_path)
    reg = Registry(store, tmp_path)
    fns = _fns(store, reg, [{"score": 9, "characters": {"陆离": 9}}])
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    v = reg.videos()[0]
    assert v["image"] == ["https://img/luli.png", "https://img/car.png"]
    assert "@图片1 = 角色「陆离」本人" in v["prompt"] and "@图片2 = 场景「地铁车厢」" in v["prompt"]
    assert v["prompt"].index(REFERENCE_HEAD) < v["prompt"].index("别回头")
    assert "人物一致 9/10" in r.content
    # 校验消息里参考图用的是本地副本（data URL），不是会过期的链接
    check = fns._chat_calls[0][0]["content"]  # type: ignore[attr-defined]
    ref_img = next(p for p in check if p["type"] == "image_url")
    assert ref_img["image_url"]["url"].startswith("data:image/png;base64,")


async def test_不像同一个人_写进差异重生成_仍不过留分高的并标出(tmp_path: Path):
    store = AssetStore()
    shots_id, pack_id = _seed(store, tmp_path)
    reg = Registry(store, tmp_path)
    fns = _fns(store, reg, [
        {"score": 4, "issues": ["陆离的发型从长发变短发"], "characters": {"陆离": 4}},
        {"score": 9, "characters": {"陆离": 9}},
    ])
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    vids = reg.videos()
    assert len(vids) == 2 and "长发变短发" in vids[1]["prompt"]
    assert vids[1]["prompt"].index("【重新生成】") < 40
    assert "已重生成" in r.content and "人物一致 9/10（重生成后）" in r.content

    # 两次都不过：留分高的那版，结果里标「仍与参考图不符」
    store2 = AssetStore()
    shots_id, pack_id = _seed(store2, tmp_path)
    reg2 = Registry(store2, tmp_path)
    verdicts = [{"score": 3, "issues": ["换人了"]}, {"score": 5, "issues": ["还是不像"]}]
    fns2 = _fns(store2, reg2, verdicts)
    r2 = await fns2._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r2.ok and "仍与参考图不符" in r2.content and "人物漂移" in r2.content
    record = next(a for a in store2.all() if a.creator == "tool:drama_render_shots")
    kept = json.loads(store2.content(record.id))[0]["asset"]
    second = [a for a in store2.all() if a.summary == "[第1集-1场] 1-4"]
    assert len(second) == 2 and kept == second[1].id, "5 分的第二版比 3 分的第一版高，保留第二版"


async def test_参考图链接过期_渲染前提示刷新(tmp_path: Path):
    store = AssetStore()
    shots_id, pack_id = _seed(store, tmp_path, portrait_age_h=30)
    reg = Registry(store, tmp_path)
    fns = _fns(store, reg)
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    # 2026-09-20 用户定的规则：链接过期 = 引用不会成功 → 花钱之前拦下，不是渲完再提示
    assert not r.ok and reg.videos() == []
    assert "链接超过 20 小时" in r.error and "drama_refresh_refs" in r.error
    assert "没有发起任何生成" in r.error


async def test_刷新参考图链接_用本地副本复刻(tmp_path: Path):
    store = AssetStore()
    _, pack_id = _seed(store, tmp_path, portrait_age_h=30)
    reg = Registry(store, tmp_path)
    fns = _fns(store, reg)
    r = await fns._fn_drama_refresh_refs()
    assert r.ok, r.error
    gen = [a for n, a in reg.calls if n == "gen_image"]
    assert len(gen) == 1 and gen[0]["image"][0].startswith("data:image/png;base64,")
    assert gen[0]["aspect_ratio"] == "3:4" and "复刻" in gen[0]["prompt"]
    new_pack = json.loads(store.content(r.asset_ref))
    assert new_pack["陆离"]["asset"] != json.loads(store.content(pack_id))["陆离"]["asset"]
    assert new_pack["地铁车厢"]["url"] == "https://img/car.png", "没过期的不动"
    assert store.get(r.asset_ref).parent_ids[0] == pack_id
    r2 = await fns._fn_drama_refresh_refs(rendered_id=r.asset_ref)
    assert not r2.ok and "没有需要刷新" in r2.error


async def test_服装图与主形象不像_重生成一次(tmp_path: Path):
    store = AssetStore()
    reg = Registry(store, tmp_path)
    # 顺序：第一版真实感 → 第一版一致性（不像）→ 重生成 → 第二版一致性（像）
    # 一致性门自己循环，不再重过真实感门
    fns = _fns(store, reg, [
        {"pass": True, "score": 8, "issue_type": "ok", "issues": []},
        {"score": 4, "issues": ["脸换了"], "characters": {"陆离": 4}},
        {"score": 9, "characters": {"陆离": 9}},
    ])
    fns.catalog.drama.update({"realism_gate": "true", "realism_retries": "1"})
    portrait = tmp_path / "p.png"
    portrait.write_bytes(b"p")
    pa = store.create("", type_=AssetType.IMAGE, summary="角色·陆离", creator="model:x")
    pa.uri = "https://img/luli.png"
    pa.gen_params["local"] = str(portrait)
    store.put(pa)
    report: list[dict[str, Any]] = []
    aid, _, err = await fns._gen_image(
        "陆离 风衣 全身", "16:9", "服装·陆离-风衣", ref=["https://img/luli.png"], person=True,
        report=report,
    )
    assert err == ""
    gens = [a for n, a in reg.calls if n == "gen_image"]
    assert len(gens) == 2 and "脸换了" in gens[1]["prompt"]
    assert gens[1]["prompt"].startswith("【重新生成】上一版画面里的人物与参考图不是同一个人")
    assert report[0]["identity"]["pass"] is True and report[0]["identity"]["attempts"] == 2
    made = [a for a in store.all() if a.summary == "服装·陆离-风衣"]
    assert len(made) == 2 and aid == made[1].id, "留的是通过一致性的第二版"


CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"


async def test_gen_video参考图两个字段都送(tmp_path: Path):
    catalog = MediaCatalog.load(CATALOG_PATH)
    assert catalog.image_ref_alias == "image_urls"
    provider = FakeMediaProvider(urls=["https://example.com/out.mp4"])
    gw = MediaGateway({"apimart": provider}, EventBus(), poll_interval=0.01, max_poll_interval=0.02)

    async def dl(url: str, dest: Path) -> bool:
        return False

    fns = MediaFunctions(gw, catalog, AssetStore(tmp_path / "a"),
                         prefs=OutputPrefs(tmp_path / "out", downloader=dl))
    r = await fns._fn_gen_video("x", model="seedance-2.0", image=["https://i/1"])
    assert r.ok
    sent = provider.submitted[-1]
    assert sent["image"] == ["https://i/1"] and sent["image_urls"] == ["https://i/1"]
    catalog.image_ref_alias = ""
    await fns._fn_gen_video("x", model="seedance-2.0", image=["https://i/2"])
    assert "image_urls" not in provider.submitted[-1]
    await asyncio.sleep(0)
