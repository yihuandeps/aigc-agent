"""缺口 A（2026-09-23 审查）：项目维度。

真实发生过的事：
  · 新剧开写，工具说「全剧已有剧本 79 集」，第 1 集「已有跳过」—— 跳过的是测试假剧本
  · /auto 拿旧剧的资产库和参考图包渲新剧
  · 台账上的「项目」其实是会话名 default，¥400 成了永不清零的终身上限
  · 换个文件夹启动，装回的是 default 会话的产物目录（E:\\西游记）
  · 资产库里「最新」的创作方案 / 角色档案 / 分集目录全是测试桩
  · 两个终端同时开，seq 各算各的，库里撞出 61 个重复 seq；写一半崩溃的 JSON 被静默跳过
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aigc_agent.domain.assets.migrate import (
    apply_migration,
    is_test_stub,
    plan_migration,
    restore_migration,
)
from aigc_agent.domain.assets.store import LEGACY_PROJECT, AssetStatus, AssetStore, AssetType
from aigc_agent.domain.drama.card import build_project_card
from aigc_agent.domain.project import normalize_root, project_key, project_title

_BODY = "正文。" * 120  # 够长，算真剧本


def _script(store: AssetStore, n: int, body: str = _BODY):
    return store.create(body, type_=AssetType.SCRIPT, summary=f"第{n}集", creator="model",
                        gen_params={"episode": n})


# ---------------------------------------------------------------- 项目键


def test_项目键稳定且不撞(tmp_path: Path):
    a = tmp_path / "剧A"
    a.mkdir()
    k = project_key(a)
    # 目录名进键（Windows 上按规范化后的小写，同一个文件夹怎么写都是同一个键）
    assert k.lower().startswith("剧a-") and len(k.split("-")[-1]) == 8
    assert project_key(str(a) + os.sep) == k, "结尾斜杠不影响"
    if os.name == "nt":
        assert project_key(str(a).upper()) == k, "Windows 上大小写不影响"
    other = tmp_path / "别处" / "剧A"
    other.mkdir(parents=True)
    assert project_key(other) != k, "同名不同路径不撞"
    assert not any(c in k for c in '\\/:*?"<>|'), "能直接当文件名"


def test_项目名优先剧名(tmp_path: Path):
    assert project_title(tmp_path) == tmp_path.name
    (tmp_path / ".drama-state.json").write_text(
        json.dumps({"dramaTitle": "山海守门人"}, ensure_ascii=False), encoding="utf-8"
    )
    assert project_title(tmp_path) == "山海守门人"


# ---------------------------------------------------------------- 按项目查


def test_查询默认只看当前项目():
    store = AssetStore()
    store.project = "剧A-1"
    a1 = _script(store, 1)
    store.project = "剧B-2"
    b1 = _script(store, 1)
    b2 = _script(store, 2)
    assert a1.project == "剧A-1" and b1.project == "剧B-2"

    assert store.episodes_done() == [1, 2]
    assert store.script_of(1).id == b1.id, "拿的是本项目的第 1 集，不是全库最新"
    assert {a.id for a in store.find()} == {b1.id, b2.id}
    assert {a.id for a in store.find(project="*")} == {a1.id, b1.id, b2.id}
    assert store.get(a1.id).id == a1.id, "按 id 仍取得到别的项目的"

    store.project = "剧A-1"
    assert store.episodes_done() == [1]
    assert store.latest(AssetType.SCRIPT).id == a1.id


def test_legacy看不见_没分项目的都看得见():
    store = AssetStore()
    old = _script(store, 1)  # 没开项目维度时存的：project == ""
    legacy = _script(store, 2)
    legacy.project = LEGACY_PROJECT
    store.put(legacy)
    store.project = "剧A-1"
    mine = _script(store, 3)
    ids = {a.id for a in store.find()}
    assert old.id in ids, "迁移前的存量先照老样子看得见，不会一升级就「全没了」"
    assert legacy.id not in ids and mine.id in ids


def test_作废和被替代的默认不取():
    store = AssetStore()
    v1 = _script(store, 1)
    v2 = _script(store, 1)
    store.set_status(v2.id, AssetStatus.VOID, note="/rollback")
    assert store.script_of(1).id == v1.id
    assert store.latest(AssetType.SCRIPT).id == v1.id
    assert v2.id in {a.id for a in store.find(include_inactive=True)}
    assert store.get(v2.id).status_note == "/rollback"


def test_改稿归当前项目():
    store = AssetStore()
    base = _script(store, 1)
    store.project = "剧A-1"
    rev = store.revise(base.id, _BODY + "改", summary="第1集·改")
    assert rev.project == "剧A-1" and rev.gen_params["episode"] == 1


# ---------------------------------------------------------------- 文档取真版本


def test_文档取真版本不取测试桩():
    store = AssetStore()
    real = store.create("# 创作方案\n" + "三幕结构。" * 80, type_=AssetType.OUTLINE,
                        summary="创作方案", creator="model")
    store.create("三幕结构，7 个付费卡点", type_=AssetType.OUTLINE, summary="创作方案")  # 桩，更新
    assert store.best_doc(AssetType.OUTLINE, "创作方案").id == real.id
    card = build_project_card(store)
    assert real.id in card

    store2 = AssetStore()
    short = store2.create("短方案", type_=AssetType.OUTLINE, summary="创作方案", creator="model")
    longer = store2.create("稍长一点的方案", type_=AssetType.OUTLINE, summary="创作方案",
                           creator="model")
    store2.create("更新但没有创建者", type_=AssetType.OUTLINE, summary="创作方案")
    assert store2.best_doc(AssetType.OUTLINE, "创作方案").id == longer.id, "都偏短就退到最长的"
    assert short.id != longer.id


# ---------------------------------------------------------------- 多进程


def test_两个进程同时写_seq不撞_互相看得见(tmp_path: Path):
    root = tmp_path / "assets"
    a = AssetStore(root)
    b = AssetStore(root)
    a.refresh_interval = b.refresh_interval = 0
    made = []
    for i in range(10):
        made.append((a if i % 2 else b).create(f"x{i}", summary=f"第{i}份"))
    seqs = [m.seq for m in made]
    assert len(set(seqs)) == 10 and seqs == sorted(seqs), "交替写也全局单调"
    assert {m.id for m in made} <= {x.id for x in a.find()}
    assert {m.id for m in made} <= {x.id for x in b.find()}, "对方写的也看得见"
    assert not list(root.glob(".*.tmp")), "临时文件都换成正式文件了"

    # 另一个进程改了某份资产（打了状态）→ 这边读到新版本
    target = made[0]
    b.set_status(target.id, AssetStatus.REJECTED)
    assert target.id not in {x.id for x in a.find()}
    assert a.get(target.id).status is AssetStatus.REJECTED
    # 另一个进程把文件挪走了（迁移进回收站）→ 这边也移出去
    (root / f"{made[1].id}.json").unlink()
    assert made[1].id not in {x.id for x in a.find(include_inactive=True)}


def test_坏文件告警不静默(tmp_path: Path):
    root = tmp_path / "assets"
    good = AssetStore(root).create("好的", summary="好")
    (root / "as_brokenbroke.json").write_text("{半截", encoding="utf-8")
    store = AssetStore(root)
    assert store.has(good.id)
    assert any("as_brokenbroke.json" in e for e in store.load_errors)


# ---------------------------------------------------------------- 装配：一个文件夹一个项目


async def test_换文件夹就是换项目(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from aigc_agent.app import Agent

    ws = tmp_path / "ws"
    drama_a = tmp_path / "剧A"
    drama_b = tmp_path / "剧B"
    drama_a.mkdir()
    drama_b.mkdir()
    monkeypatch.chdir(drama_a)
    agent = Agent.create(workspace=ws)
    try:
        assert agent.bus.session_id == project_key(drama_a), "不带 --session：一个文件夹一个会话"
        assert agent.project == project_key(drama_a)
        assert agent.guard.project_id == agent.project, "单项目预算按项目累计，不再是 default"
        assert agent.mem_agent.project_id == agent.project
        s1 = _script(agent.assets, 1)
        assert s1.project == agent.project

        key_b = agent.switch_project(drama_b)
        assert key_b == project_key(drama_b) and agent.output_prefs.root == drama_b
        assert agent.assets.project == key_b and agent.guard.project_id == key_b
        assert agent.recorder.project_id == key_b and agent.memory_source.project_id == key_b
        assert agent.assets.episodes_done() == [], "新项目看不到剧 A 的剧本"
        assert agent.session_store.output_dir == str(drama_b)
    finally:
        await agent.aclose()


async def test_流水线有活在跑时不许换项目(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from aigc_agent.app import Agent

    monkeypatch.chdir(tmp_path)
    agent = Agent.create(workspace=tmp_path / "ws")
    try:
        agent.pipeline._inflight.add("render:3")  # noqa: SLF001
        with pytest.raises(RuntimeError, match="流水线"):
            agent.switch_project(tmp_path / "别的剧")
        agent.pipeline._inflight.clear()  # noqa: SLF001
        agent.pipeline._failed.add("render:3")  # noqa: SLF001
        agent.switch_project(tmp_path / "别的剧")
        assert not agent.pipeline._failed, "上一部剧的失败标记不能挡住新剧的同号集"  # noqa: SLF001
    finally:
        await agent.aclose()


# ---------------------------------------------------------------- 存量迁移


def _raw(ws: Path, **fields) -> str:
    """直接写一份「老版本」资产文件（没有 project 字段）。"""
    d = {
        "id": fields.pop("id"), "type": "text", "summary": "", "creator": "model",
        "parent_ids": [], "gen_params": {}, "seq": fields.pop("seq", 1), "created_at": 1.0,
        "inline": "内容",
    }
    d.update(fields)
    (ws / "assets").mkdir(parents=True, exist_ok=True)
    (ws / "assets" / f"{d['id']}.json").write_text(json.dumps(d, ensure_ascii=False),
                                                   encoding="utf-8")
    return d["id"]


def test_测试桩按原文认():
    assert is_test_stub({"type": "outline", "creator": "", "inline": "三幕结构，7 个付费卡点"})
    assert is_test_stub({"type": "text", "creator": "stub", "inline": "【大纲 · 第 1 版】"})
    assert is_test_stub({"type": "script", "creator": "model",
                         "inline": "# 第1集：标题\n\n正文…\n\n> 🎣 本集钩子：第1集的悬念"})
    assert not is_test_stub({"type": "outline", "creator": "", "inline": "用户自己写的短方案"})
    assert not is_test_stub({"type": "outline", "creator": "model",
                             "inline": "三幕结构，7 个付费卡点"}), "有创建者的不算"


def test_迁移分项目_可撤销(tmp_path: Path):
    ws = tmp_path / "ws"
    drama = tmp_path / "山海"
    (drama / "images").mkdir(parents=True)
    (drama / "texts").mkdir()
    img = drama / "images" / "a.png"
    img.write_bytes(b"x")
    script = _raw(ws, id="as_00000000a1", type="script", summary="第1集", seq=1)
    (drama / "texts" / f"第1集-{script}.md").write_text("正文", encoding="utf-8")
    pic = _raw(ws, id="as_00000000a2", type="image", gen_params={"local": str(img)}, seq=2)
    shots = _raw(ws, id="as_00000000a3", parent_ids=[script], seq=3)  # 血缘跟着剧本走
    orphan = _raw(ws, id="as_00000000a4", seq=4)
    stub = _raw(ws, id="as_00000000a5", type="outline", creator="", summary="创作方案",
                inline="三幕结构，7 个付费卡点", seq=9)
    (ws / "memory" / "sessions").mkdir(parents=True)
    (ws / "memory" / "sessions" / "default.json").write_text(
        json.dumps({"output_dir": str(drama), "turns": []}), encoding="utf-8"
    )
    (ws / "memory" / "mem_0000000001.json").write_text(
        json.dumps({"id": "mem_0000000001", "content": f"第1集剧本是 {script}", "project_id": "",
                    "layer": "project"}, ensure_ascii=False),
        encoding="utf-8",
    )

    plan = plan_migration(ws)
    key = project_key(drama)
    assert plan.assign[script] == key and plan.how[script] in ("local", "mirror")
    assert plan.assign[pic] == key and plan.how[pic] == "local"
    assert plan.assign[shots] == key and plan.how[shots] == "lineage"
    assert plan.assign[orphan] == LEGACY_PROJECT
    assert plan.stubs == [stub]
    assert plan.memories == {"mem_0000000001": key}
    assert plan.snapshots == [("default.json", f"{key}.json")]
    assert "山海" in plan.render()

    manifest = apply_migration(plan, ws)
    store = AssetStore(ws / "assets")
    store.project = key
    assert {a.id for a in store.find()} == {script, pic, shots}, "legacy 和测试桩都不在本项目里"
    assert not store.has(stub) and (manifest.parent / "assets" / f"{stub}.json").exists()
    assert (ws / "memory" / "sessions" / f"{key}.json").exists()
    assert int((ws / "assets" / ".seq").read_text()) >= 9

    done = restore_migration(manifest)
    assert done["assets"] == 4 and done["stubs"] == 1 and done["memories"] == 1
    back = json.loads((ws / "assets" / f"{script}.json").read_text(encoding="utf-8"))
    assert "project" not in back and (ws / "assets" / f"{stub}.json").exists()
    mem = json.loads((ws / "memory" / "mem_0000000001.json").read_text(encoding="utf-8"))
    assert mem["project_id"] == ""
    assert not (ws / "memory" / "sessions" / f"{key}.json").exists()


def test_工作区内部目录不算产物目录(tmp_path: Path):
    ws = tmp_path / "ws"
    dub = ws / "drama_dub" / "d1" / "clips"
    dub.mkdir(parents=True)
    (dub / "x.wav").write_bytes(b"x")
    aid = _raw(ws, id="as_00000000b1", type="audio", gen_params={"local": str(dub / "x.wav")})
    out = ws / "output" / "default" / "images"
    out.mkdir(parents=True)
    (out / "y.png").write_bytes(b"y")
    bid = _raw(ws, id="as_00000000b2", type="image", gen_params={"local": str(out / "y.png")})
    plan = plan_migration(ws)
    assert plan.assign[aid] == LEGACY_PROJECT, "配音临时目录不是用户的产物目录"
    assert plan.assign[bid] == project_key(ws / "output" / "default")
    assert normalize_root(plan.roots[plan.assign[bid]]) == normalize_root(ws / "output" / "default")
