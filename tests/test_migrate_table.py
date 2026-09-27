"""存量迁移：按整条链判 + 先出映射表、人改完再执行（2026-09-26 用户定的，待拍板 一·8）。

真实事故（dry-run）：《姐姐们抢着给我当妈》的剧本 1–12 集、资产库、参考图包被判给了「内容测试」
—— 剧本的镜像 .md 在 E:\\内容测试\\texts，图片的本地文件也落在那儿（当时产物目录指着它）；
可 E:\\西游记\\.drama-state.json 里清清楚楚列着这些 id。分镜因为没记父资产，进了 legacy。
现在：血缘连成一条链整体判；.drama-state 的结构化 id 最硬，镜像最后参考；最硬的证据指向两个
项目就标「待你定」，不自动判。执行前先出一张表，人改过再 --apply。
"""

from __future__ import annotations

import json
from pathlib import Path

from aigc_agent.domain.assets.migrate import (
    PENDING,
    apply_migration,
    latest_plan_table,
    plan_from_table,
    plan_migration,
    read_plan_table,
    write_plan_table,
)
from aigc_agent.domain.assets.store import LEGACY_PROJECT
from aigc_agent.domain.project import project_key


def _raw(ws: Path, aid: str, **fields) -> str:
    d = {
        "id": aid, "type": "text", "summary": "", "creator": "model", "parent_ids": [],
        "gen_params": {}, "seq": int(aid[-2:], 16), "created_at": 1.0, "inline": "内容",
    }
    d.update(fields)
    (ws / "assets").mkdir(parents=True, exist_ok=True)
    (ws / "assets" / f"{aid}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return aid


def _session(ws: Path, *roots: Path) -> None:
    """会话快照记着这些产物目录（迁移据此认识它们）。"""
    d = ws / "memory" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    for n, r in enumerate(roots):
        (d / f"s{n}.json").write_text(json.dumps({"output_dir": str(r)}), encoding="utf-8")


def _project(ws: Path, aid: str) -> str | None:
    return json.loads((ws / "assets" / f"{aid}.json").read_text(encoding="utf-8")).get("project")


def _chain(ws: Path, root: Path) -> dict[str, str]:
    """一整条短剧链：剧本 → 分镜 / 资产库 → 参考图包（列着一张图）→ 提示词 → 片段。
    只有那张图有本地文件（落在 root/images），别的都没有自己的证据。"""
    img = root / "images" / "角色-01_阿蛛.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"png")
    ids = {
        "script": _raw(ws, "as_00000000a1", type="script", summary="第1集"),
        "board": _raw(ws, "as_00000000a2", creator="tool:drama_storyboard",
                      parent_ids=["as_00000000a1"]),
        "lib": _raw(ws, "as_00000000a3", creator="tool:drama_assets",
                    parent_ids=["as_00000000a1"]),
        "image": _raw(ws, "as_00000000a4", type="image", creator="model:gpt-image-2",
                      gen_params={"local": str(img)}),
        "pack": _raw(ws, "as_00000000a5", creator="tool:drama_render_assets",
                     parent_ids=["as_00000000a3"],
                     inline=json.dumps({"阿蛛": {"asset": "as_00000000a4", "url": "https://x"}})),
        "prompt": _raw(ws, "as_00000000a6", creator="tool:drama_shots",
                       parent_ids=["as_00000000a2", "as_00000000a3"]),
        "clip": _raw(ws, "as_00000000a7", type="video", creator="model:seedance-2.0",
                     gen_params={"tags": {"shots_id": "as_00000000a6", "pack": "as_00000000a5"}}),
    }
    return ids


def test_血缘连成一条链_整条归同一个项目(tmp_path: Path):
    ws, root = tmp_path / "ws", tmp_path / "西游记"
    ids = _chain(ws, root)
    _raw(ws, "as_00000000b1")  # 和链不相干的孤儿
    plan = plan_migration(ws)
    key = project_key(root)
    assert {plan.assign[i] for i in ids.values()} == {key}, "整条链归同一个项目"
    assert plan.how[ids["image"]] == "local" and plan.how[ids["script"]] == "lineage"
    assert plan.chain[ids["script"]] == plan.chain[ids["clip"]], "片段靠标签里的 shots_id 连上链"
    assert plan.assign["as_00000000b1"] == LEGACY_PROJECT
    assert "本地文件" in plan.why[ids["script"]] and "跟着链走" in plan.why[ids["script"]]


def test_drama_state的id压过镜像和本地文件(tmp_path: Path):
    ws, xyj, test = tmp_path / "ws", tmp_path / "西游记", tmp_path / "内容测试"
    ids = _chain(ws, test)  # 图片本地文件、剧本镜像都在「内容测试」（当时产物目录指着它）
    (test / "texts").mkdir(parents=True)
    (test / "texts" / f"第1集-{ids['script']}.md").write_text("正文", encoding="utf-8")
    xyj.mkdir()
    (xyj / ".drama-state.json").write_text(json.dumps({
        "dramaTitle": "姐姐们抢着给我当妈",
        "assetsId": ids["lib"],
        "scriptIds": {"1": ids["script"]},
        "note": "作废资产勿用：as_00000000c1（别的剧分镜）",  # 备注里的 id 不认
    }, ensure_ascii=False), encoding="utf-8")
    other = _raw(ws, "as_00000000c1", creator="tool:drama_storyboard")
    _session(ws, xyj)
    plan = plan_migration(ws)
    key = project_key(xyj)
    assert {plan.assign[i] for i in ids.values()} == {key}
    assert plan.how[ids["script"]] == "state" and plan.how[ids["image"]] == "lineage"
    assert "scriptIds" in plan.why[ids["script"]] and "文本镜像" in plan.why[ids["script"]]
    assert plan.label(key) == "姐姐们抢着给我当妈"
    assert plan.assign[other] == LEGACY_PROJECT, "state 备注里提到的 id 不算这部剧的"
    assert not plan.pending


def test_链上最硬的证据指向两个项目_标待你定(tmp_path: Path):
    ws, a_root, b_root = tmp_path / "ws", tmp_path / "A", tmp_path / "B"
    img_a, img_b = a_root / "images" / "x.png", b_root / "images" / "y.png"
    for p in (img_a, img_b):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"png")
    _raw(ws, "as_00000000d1", type="image", gen_params={"local": str(img_a)})
    _raw(ws, "as_00000000d2", type="image", gen_params={"local": str(img_b)})
    listed = {"甲": {"asset": "as_00000000d1"}, "乙": {"asset": "as_00000000d2"}}
    pack = _raw(ws, "as_00000000d3", creator="tool:drama_render_assets",
                inline=json.dumps(listed))
    plan = plan_migration(ws)
    for aid in ("as_00000000d1", "as_00000000d2", pack):
        assert aid in plan.pending and aid not in plan.assign
        assert set(plan.pending[aid]) == {project_key(a_root), project_key(b_root)}
    assert "不止一个项目" in plan.why[pack]


def test_文件放在一个项目_内容讲的是另一部剧_也标待你定(tmp_path: Path):
    ws, xyj, test = tmp_path / "ws", tmp_path / "西游记", tmp_path / "内容测试"
    xyj.mkdir()
    lib = _raw(ws, "as_00000000e1", creator="tool:drama_assets",
               inline=json.dumps({"characters": [{"baseRoleName": "阿蛛"}]}, ensure_ascii=False))
    (xyj / ".drama-state.json").write_text(json.dumps({"assetsId": lib}), encoding="utf-8")
    img = test / "images" / "a.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"png")
    stray = _raw(ws, "as_00000000e2", type="image", summary="阿蛛写实主形象",
                 gen_params={"local": str(img)})
    _session(ws, xyj)
    plan = plan_migration(ws)
    assert plan.assign[lib] == project_key(xyj)
    assert set(plan.pending[stray]) == {project_key(xyj), project_key(test)}
    assert "本地文件" in plan.why[stray] and "剧名 / 角色名" in plan.why[stray]


def test_映射表往返_按人改过的执行_待你定没改的不动(tmp_path: Path):
    ws, a_root, b_root = tmp_path / "ws", tmp_path / "A", tmp_path / "B"
    img_a, img_b = a_root / "images" / "x.png", b_root / "images" / "y.png"
    for p in (img_a, img_b):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"png")
    a1 = _raw(ws, "as_00000000f1", type="image", gen_params={"local": str(img_a)})
    a2 = _raw(ws, "as_00000000f2", type="image", gen_params={"local": str(img_b)})
    pack = _raw(ws, "as_00000000f3", creator="tool:drama_render_assets",
                inline=json.dumps({"甲": {"asset": a1}, "乙": {"asset": a2}}))
    solo = _raw(ws, "as_00000000f4", type="image", gen_params={"local": str(img_a)})
    orphan = _raw(ws, "as_00000000f5")

    plan = plan_migration(ws)
    table = write_plan_table(plan, ws)
    assert table.parent == ws and table.name.startswith("migrate-plan-")
    assert latest_plan_table(ws) == table
    raw = table.read_text(encoding="utf-8-sig").splitlines()
    assert raw[0].startswith("资产id,类型,摘要,链,项目（改这一列）")
    assert raw[1].split(",")[0] in (a1, a2, pack), "待你定排在最前面"
    cells = read_plan_table(table)
    assert cells[a1].startswith(PENDING) and cells[solo] == str(a_root)
    assert cells[orphan] == "legacy"

    # 人改表：pack 定给 B（写剧名 / 路径都行），solo 改成 legacy；a1、a2 不改（待你定）
    text = table.read_text(encoding="utf-8-sig")
    lines = []
    for ln in text.splitlines():
        cols = ln.split(",")
        if cols[0] == pack:
            cols[4] = str(b_root)
        elif cols[0] == solo:
            cols[4] = "legacy"
        lines.append(",".join(cols))
    table.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    final = plan_from_table(ws, table)
    assert final.assign[pack] == project_key(b_root) and final.how[pack] == "table"
    assert final.assign[solo] == LEGACY_PROJECT and final.how[solo] == "table"
    assert set(final.pending) == {a1, a2}
    apply_migration(final, ws)
    assert _project(ws, pack) == project_key(b_root) and _project(ws, solo) == LEGACY_PROJECT
    assert _project(ws, a1) is None and _project(ws, a2) is None, "待你定没改的：不写项目键"
    assert _project(ws, orphan) == LEGACY_PROJECT

    again = plan_migration(ws)
    assert again.assign[a1] == again.assign[a2] == project_key(b_root), (
        "没定的下次迁移重新进表；链上已经有人定过的项目键，就跟着它建议"
    )
    assert again.how[a1] == "lineage" and "已有项目键" in again.why[a1]


def test_Excel另存成GBK_照样读得出(tmp_path: Path):
    ws, root = tmp_path / "ws", tmp_path / "西游记"
    ids = _chain(ws, root)
    table = write_plan_table(plan_migration(ws), ws, tmp_path / "表.csv")
    gbk = tmp_path / "表-gbk.csv"
    gbk.write_bytes(table.read_text(encoding="utf-8-sig").encode("gbk"))
    assert read_plan_table(gbk) == read_plan_table(table)
    assert read_plan_table(gbk)[ids["clip"]] == str(root)


def test_填得认不出的不猜_不动(tmp_path: Path):
    ws, root = tmp_path / "ws", tmp_path / "西游记"
    ids = _chain(ws, root)
    table = write_plan_table(plan_migration(ws), ws)
    text = table.read_text(encoding="utf-8-sig").replace(
        f"{ids['clip']},", f"{ids['clip']},", 1
    )
    lines = []
    for ln in text.splitlines():
        cols = ln.split(",")
        if cols[0] == ids["clip"]:
            cols[4] = "随便写写"
        lines.append(",".join(cols))
    table.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    final = plan_from_table(ws, table)
    assert ids["clip"] in final.pending and final.how[ids["clip"]] == "unknown"
    assert "认不出" in final.why[ids["clip"]]
    assert final.assign[ids["script"]] == project_key(root)


def test_命令行两步走_先出表_没表不许apply(tmp_path: Path, monkeypatch):
    from typer.testing import CliRunner

    from aigc_agent.interfaces.cli import assets_cmd

    ws, root = tmp_path / "ws", tmp_path / "西游记"
    ids = _chain(ws, root)
    monkeypatch.setattr(assets_cmd, "WORKSPACE", ws)
    monkeypatch.setattr(assets_cmd, "agents_running", lambda _ws: [])
    runner = CliRunner()

    r = runner.invoke(assets_cmd.app, ["migrate", "--apply"])
    assert r.exit_code == 1 and "没有映射表" in r.output
    assert _project(ws, ids["script"]) is None

    r = runner.invoke(assets_cmd.app, ["migrate"])
    assert r.exit_code == 0 and "映射表" in r.output and latest_plan_table(ws) is not None
    assert _project(ws, ids["script"]) is None, "出表这一步什么都不改"

    r = runner.invoke(assets_cmd.app, ["migrate", "--apply"])
    assert r.exit_code == 0 and "迁移完成" in r.output, r.output
    assert _project(ws, ids["script"]) == project_key(root)
