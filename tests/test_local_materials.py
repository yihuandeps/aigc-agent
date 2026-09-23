"""本地素材可见（2026-09-20）—— 用户目录里摆着 6–37 集剧本、分段视频、成片，
模型却说「只写到第 5 集」。

真实日志（20260920-181003 / 194303）里的两层根因：
  · 资产库按 type=script + gen_params.episode 认集；模型自己 save_draft(kind="outline") 存的整集、
    合规修订版（revise 还会把 gen_params 丢掉）全都不算数，项目卡和 find_episode 都说 6 集以后没有
  · 产物目录里的文件从没被看过：find_episode 只查资产库，模型不知道磁盘上有什么，
    用户说「结合已有的素材」它只会 list_assets
另加：用户的剧本 / 人物小传常是 .docx，fs_read 之前只认纯文本。
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

from aigc_agent.domain.assets.store import Asset, AssetStore, AssetType
from aigc_agent.domain.documents import extract_text
from aigc_agent.domain.functions.content import ContentFunctions
from aigc_agent.domain.functions.files import FileFunctions, FsPolicy
from aigc_agent.domain.local_materials import (
    LOCAL_PIN,
    LocalMaterials,
    episode_in_name,
    kind_of,
)
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.tools.registry import ToolRegistry

SCRIPT = "# 第6集：昆仑的捐赠仪式\n\n" + "△ 陆离走进会展中心，人群骚动。\n陆离：别动。\n" * 20


def _seed_dir(root: Path) -> Path:
    files = {
        "texts/第6集·合规修订版-as_2908badc53.md": SCRIPT,
        "texts/第7集·合规修订版-as_0c4284cb0e.md": SCRIPT,
        "texts/分集目录·合规修订版v2-as_cf06c99f28.md": "第1集…",
        "texts/60集合规审核报告-as_b5cf2bcd94.md": "报告",
        "videos/ep6_r2_scene1.mp4": "v",
        "videos/ep6_r2_scene2.mp4": "v",
        "videos/ep10_scene1.mp4": "v",
        "exports/山海守门人_第6集_重制锁脸版.mp4": "v",
        "images/char_陆离.png": "i",
        "images/参考图-角色-01_陆离.png": "i",
        "notes/step1.txt": "不是第几集",
        "产物清单.md": "| 文件 |",
        ".drama-state.json": "{}",
    }
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return root


def _docx(path: Path, paragraphs: list[str]) -> Path:
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(
        f"<w:p><w:r><w:t>{t}</w:t></w:r></w:p>" if "\t" not in t
        else "<w:p>" + "".join(
            f"<w:r><w:t>{part}</w:t></w:r>" + ("<w:r><w:tab/></w:r>" if i == 0 else "")
            for i, part in enumerate(t.split("\t"))
        ) + "</w:p>"
        for t in paragraphs
    )
    xml = (
        f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{w}">'
        f'<w:body>{body}</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", xml)
    return path


# ---------------------------------------------------------------- 文件名 → 集号 / 类型


def test_文件名认集号与类型():
    assert episode_in_name("第6集·合规修订版-as_2908badc53.md") == 6
    assert episode_in_name("ep10_scene1.mp4") == 10
    assert episode_in_name("EP06_scene2.mp4") == 6
    assert episode_in_name("山海守门人_S1E07.mp4") == 7
    assert episode_in_name("第01集-03_2场_镜9-18.mp4") == 1
    assert episode_in_name("Episode 12 final.mov") == 12
    assert episode_in_name("step1.txt") == 0, "step 里的 ep 不是集号"
    assert episode_in_name("sleep3.mp4") == 0
    assert episode_in_name("60集合规审核报告.md") == 0, "没有「第」不算"
    assert kind_of(Path("a.docx")) == "doc"
    assert kind_of(Path("a.MP4")) == "video"
    assert kind_of(Path("a.srt")) == "subtitle"
    assert kind_of(Path("a.xyz")) == "other"


# ---------------------------------------------------------------- 索引


def test_索引按集归组与摘要(tmp_path: Path):
    root = _seed_dir(tmp_path / "out")
    lm = LocalMaterials(root)
    idx = lm.index()
    assert len(idx) == 12, "隐藏文件不索引"
    assert idx.output_root == root.resolve()
    six = idx.by_episode(6)
    assert [f.rel for f in six] == [
        "exports/山海守门人_第6集_重制锁脸版.mp4",
        "texts/第6集·合规修订版-as_2908badc53.md",
        "videos/ep6_r2_scene1.mp4",
        "videos/ep6_r2_scene2.mp4",
    ]
    assert idx.episodes() == [6, 7, 10]
    assert idx.episodes(kind="video") == [6, 10]
    text6 = next(f for f in six if f.kind == "text")
    assert text6.asset_id == "as_2908badc53" and text6.folder == "texts"

    s = idx.summary()
    assert "产物目录" in s and "共 12 个文件" in s
    assert "texts 4（文本：第 6–7 集）" in s
    assert "videos 3（视频：第 6, 10 集）" in s
    assert "exports 1（视频：第 6 集）" in s
    assert "根目录 1" in s
    assert "find_episode" in s and "find_materials" in s and "view_video" in s

    ep = idx.render_episode(6)
    assert "第 6 集的本地文件（4 个" in ep
    assert "texts/第6集·合规修订版-as_2908badc53.md" in ep and "资产 as_2908badc53" in ep
    assert "exports/山海守门人_第6集_重制锁脸版.mp4" in ep
    assert idx.render_episode(99) == ""

    assert len(idx.filter(episode=6, kind="video")) == 3, "exports 里的成片也是视频"
    assert len(idx.filter(episode=6, kind="video", folder="videos")) == 2
    assert len(idx.filter(query="陆离")) == 2
    only = [f.rel for f in idx.filter(folder="exports")]
    assert only == ["exports/山海守门人_第6集_重制锁脸版.mp4"]
    assert idx.filter(kind="all", episode=10)[0].rel == "videos/ep10_scene1.mp4"


def test_索引缓存_目录变了就重扫_没变就复用(tmp_path: Path):
    root = _seed_dir(tmp_path / "out")
    lm = LocalMaterials(root, ttl=100)
    time.sleep(0.02)  # NTFS 对刚写完的文件延迟更新目录 mtime，等它落定
    first = lm.index()
    assert lm.index() is first, "目录没变、没过期：复用"
    (root / "texts" / "第8集·修表铺的老人-as_58769915c1.md").write_text(SCRIPT, encoding="utf-8")
    again = lm.index(force=True)
    assert again is not first and 8 in again.episodes()
    (root / "texts" / "第9集·蛊雕再现.md").write_text(SCRIPT, encoding="utf-8")
    time.sleep(0.05)
    assert 9 in lm.index().episodes(), "子目录 mtime 变了要自动重扫"
    stale = LocalMaterials(root, ttl=0)
    assert stale.index() is not stale.index(), "ttl=0 每次都重扫"


def test_空目录_不存在的目录_素材目录(tmp_path: Path):
    assert LocalMaterials(tmp_path / "nope").summary() == ""
    assert LocalMaterials(None).summary() == ""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert LocalMaterials(empty).summary() == ""

    extra = _seed_dir(tmp_path / "素材")
    lm = LocalMaterials(empty, extra_dirs=[extra])
    idx = lm.index()
    assert idx.output_root == empty.resolve() and len(idx) == 12
    s = idx.summary()
    assert "素材目录" in s and "产物目录" not in s.split("\n")[1]
    # 不在产物目录下的文件给绝对路径，模型不用拼
    line = idx.render_episode(6).split("\n")[1]
    assert str(extra.resolve()).replace("\\", "/") in line


def test_文件太多只取前面并标出来(tmp_path: Path):
    root = _seed_dir(tmp_path / "out")
    idx = LocalMaterials(root, max_files=3).index()
    assert len(idx) == 3 and idx.truncated
    assert "只索引了前面一部分" in idx.summary()


# ---------------------------------------------------------------- 资产库按摘要认集号


def test_资产库按摘要认集号_修订版保留集号():
    store = AssetStore()
    loose = store.create(
        SCRIPT, type_=AssetType.OUTLINE, summary="第6集·合规修订版", creator="model"
    )
    assert store.episode_of(loose) == 6
    assert store.find(episode=6) == [loose]
    assert store.episodes_done() == [6]
    assert store.script_of(6) is loose and store.episode_assets(6)["剧本"] is loose

    store.create("待写", type_=AssetType.TEXT, summary="第7集待写")  # 太短，不是剧本
    store.create(SCRIPT, type_=AssetType.OUTLINE, summary="第8集大纲")  # 大纲不是正文
    store.create("第1集…第60集", type_=AssetType.OUTLINE, summary="分集目录·合规修订版v2")
    store.create(SCRIPT, type_=AssetType.REPORT, summary="第9集审核")  # 类型不对
    store.put(Asset(type=AssetType.IMAGE, summary="第6集海报"))
    assert store.episodes_done() == [6]
    assert store.find(episode=7)[0].summary == "第7集待写", "find 按集过滤仍然能找到它"

    # type=script 的永远优先，哪怕 outline 那份更新
    typed = store.create(SCRIPT, type_=AssetType.SCRIPT, summary="第6集·旧哨站",
                         gen_params={"episode": 6}, creator="tool:drama_write_episode")
    store.create(SCRIPT + "改", type_=AssetType.OUTLINE, summary="第6集·合规修订版v2")
    assert store.script_of(6) is typed

    # 修订版跟着带集号：之前 revise 丢 gen_params，第 6 集改一稿就从 find_episode 里消失
    rev = store.revise(typed.id, SCRIPT + "修", summary="第6集·修订", creator="model")
    assert rev.gen_params["episode"] == 6 and rev.type is AssetType.SCRIPT
    assert store.episode_assets(6)["剧本"] is rev


# ---------------------------------------------------------------- find_episode / find_materials


async def _content(store: AssetStore, local: LocalMaterials | None):
    reg = ToolRegistry(EventBus())
    reg.register(ContentFunctions(store, local=local))
    await reg.refresh()
    return reg


async def test_find_episode连产物目录里的本地文件一起列(tmp_path: Path):
    root = _seed_dir(tmp_path / "out")
    store = AssetStore()
    reg = await _content(store, LocalMaterials(root))

    r = await reg.invoke("find_episode", {"episode": 6})
    assert r.ok and "资产库里没有第 6 集的记录" in r.content
    assert "texts/第6集·合规修订版-as_2908badc53.md" in r.content
    assert "videos/ep6_r2_scene2.mp4" in r.content and r.asset_ref is None

    r99 = await reg.invoke("find_episode", {"episode": 99})
    assert "资产库和产物目录都没有" in r99.content

    a = store.create(SCRIPT, type_=AssetType.OUTLINE, summary="第6集·合规修订版", creator="model")
    r2 = await reg.invoke("find_episode", {"episode": 6})
    assert f"剧本：[{a.id}|outline|v1]" in r2.content and r2.asset_ref == a.id
    assert "第 6 集的本地文件" in r2.content

    listed = await reg.invoke("list_assets", {"episode": 6})
    assert "匹配 1 份" in listed.content and " 第6集" in listed.content

    # 没接本地索引：行为与原来一致
    reg0 = await _content(AssetStore(), None)
    r0 = await reg0.invoke("find_episode", {"episode": 6})
    assert "还没有任何产物" in r0.content


def _files(tmp_path: Path, root: Path, local: LocalMaterials | None, **policy):
    store = AssetStore(tmp_path / "assets")
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    pol = FsPolicy(roots=[], **policy)
    return FileFunctions(store, ws, root, pol, project_root=tmp_path / "proj", local=local), store


async def test_find_materials工具(tmp_path: Path):
    root = _seed_dir(tmp_path / "out")
    fns, _ = _files(tmp_path, root, LocalMaterials(root))
    assert "find_materials" in {m.name for m in await fns.list_tools()}

    r = await fns.invoke("find_materials", {})
    assert r.ok and "共 12 个文件" in r.content and "命中 12 个文件" in r.content

    r6 = await fns.invoke("find_materials", {"episode": 6, "kind": "video"})
    assert "命中 3 个文件" in r6.content, "videos/ 两段 + exports/ 一部成片"
    assert "第6集 · videos/ep6_r2_scene1.mp4" in r6.content and "texts/" not in r6.content

    r7 = await fns.invoke("find_materials", {"query": "不存在的名字"})
    assert "没有匹配的文件" in r7.content and "有文件的集：6–7, 10" in r7.content

    rf = await fns.invoke("find_materials", {"folder": "exports"})
    assert "命中 1 个文件" in rf.content

    page = await fns.invoke("find_materials", {"limit": 5})
    assert "显示第 1–5 个" in page.content and "offset=5" in page.content

    assert not (await fns.invoke("find_materials", {"kind": "胶片"})).ok

    fns0, _ = _files(tmp_path, root, None)
    assert not (await fns0.invoke("find_materials", {})).ok


async def test_素材目录自动进白名单(tmp_path: Path):
    extra = _seed_dir(tmp_path / "素材")
    out = tmp_path / "out"
    out.mkdir()
    fns, _ = _files(tmp_path, out, LocalMaterials(out, extra_dirs=[extra]), material_dirs=[extra])
    roots = (await fns.invoke("fs_roots", {})).content
    assert "material_dirs" in roots and "素材" in roots
    target = extra / "texts" / "第6集·合规修订版-as_2908badc53.md"
    r = await fns.invoke("fs_read", {"path": str(target)})
    assert r.ok and "昆仑的捐赠仪式" in r.content
    found = await fns.invoke("find_materials", {"episode": 7})
    assert "命中 1 个文件" in found.content
    assert str(extra.resolve()).replace("\\", "/") in found.content


# ---------------------------------------------------------------- docx / pptx / xlsx


async def test_fs_read读docx_搜索与导入也认(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    doc = _docx(out / "人物小传.docx", ["陆离，28 岁，守门人。", "苏晏\t记者", "第二段"])
    fns, store = _files(tmp_path, out, None)

    r = await fns.invoke("fs_read", {"path": "人物小传.docx"})
    assert r.ok and "docx 抽出的文字" in r.content
    assert "陆离，28 岁，守门人。\n苏晏\t记者\n第二段" in r.content

    text, how = extract_text(doc)
    assert how == "docx" and text.startswith("陆离")

    s = await fns.invoke("fs_search", {"path": str(out), "query": "记者"})
    assert "人物小传.docx:2:" in s.content

    imp = await fns.invoke("fs_import", {"path": str(doc), "kind": "script"})
    assert imp.ok and imp.asset_ref
    assert "苏晏" in store.content(imp.asset_ref)
    auto = await fns.invoke("fs_import", {"path": str(doc)})
    assert auto.ok and store.get(auto.asset_ref).type is AssetType.TEXT

    (out / "旧剧本.doc").write_bytes(b"\xd0\xcf\x11\xe0 old word")
    old = await fns.invoke("fs_read", {"path": "旧剧本.doc"})
    assert not old.ok and "另存为 .docx" in old.error
    (out / "简介.pdf").write_bytes(b"%PDF-1.4 ...")
    pdf = await fns.invoke("fs_read", {"path": "简介.pdf"})
    assert not pdf.ok and "PDF" in pdf.error

    (out / "坏的.docx").write_bytes(b"not a zip")
    bad = await fns.invoke("fs_read", {"path": "坏的.docx"})
    assert not bad.ok and "打不开" in bad.error


def test_pptx与xlsx抽文字(tmp_path: Path):
    a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    with zipfile.ZipFile(tmp_path / "deck.pptx", "w") as z:
        for no, txt in ((2, "第二页标题"), (1, "第一页标题")):
            z.writestr(
                f"ppt/slides/slide{no}.xml",
                f'<p:sld xmlns:a="{a}" xmlns:p="x"><a:p><a:r><a:t>{txt}</a:t></a:r></a:p></p:sld>',
            )
    text, how = extract_text(tmp_path / "deck.pptx")
    assert how == "pptx" and text.index("第一页标题") < text.index("第二页标题")
    assert "## 第 1 页" in text

    s = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(tmp_path / "表.xlsx", "w") as z:
        z.writestr(
            "xl/sharedStrings.xml",
            f'<sst xmlns="{s}"><si><t>集数</t></si><si><t>标题</t></si></sst>',
        )
        z.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{s}"><sheetData>'
            '<row><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row>'
            '<row><c><v>6</v></c><c t="inlineStr"><is><t>捐赠仪式</t></is></c></row>'
            "</sheetData></worksheet>",
        )
    text, how = extract_text(tmp_path / "表.xlsx")
    assert how == "xlsx" and "集数\t标题" in text and "6\t捐赠仪式" in text


# ---------------------------------------------------------------- Agent 装配层（不连网）


async def test_Agent每轮pin本地素材摘要(tmp_path: Path, monkeypatch):
    from aigc_agent.app import Agent

    monkeypatch.setenv("KIMI_API_KEY", "sk-test")
    monkeypatch.setenv("APIMART_API_KEY", "sk-test")
    agent = Agent.create(session_id="p6-test")
    try:
        agent.assets._items.clear()  # noqa: SLF001 — 隔离：不看磁盘上已有的资产
        agent.assets.root = None  # 也不往真实资产库写（见 test_p6_drama_tools 同名处）
        agent.assets.mirror = None
        await agent.registry.refresh()  # 工具目录在 setup() 里刷；这里不连 MCP，手动刷一次
        names = {m.name for m in agent.registry.catalog()}
        assert "find_materials" in names and "find_episode" in names
        assert agent.local_materials is not None

        agent.output_prefs.root = _seed_dir(tmp_path / "out")  # 相当于用户 /out 改了产物目录
        await agent.prepare_turn("现在我需要结合已有的素材生成第11集的内容")
        pin = agent.memory.pins[LOCAL_PIN]
        assert pin.position == "pre_input" and "texts 4（文本：第 6–7 集）" in pin.content

        r = await agent.registry.invoke("find_episode", {"episode": 6})
        assert "videos/ep6_r2_scene1.mp4" in r.content

        empty = tmp_path / "empty"
        empty.mkdir()
        agent.output_prefs.root = empty
        await agent.prepare_turn("继续")
        assert LOCAL_PIN not in agent.memory.pins, "目录空了就不 pin"
    finally:
        await agent.aclose()
