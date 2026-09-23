"""本地文件系统工具（2026-09-18）。

边界：只在白名单目录内；凭据/虚拟环境/系统目录永远不碰；写操作可回滚
（覆盖前备份、删除 = 移到回收目录）；文本自动识别 UTF-8 / GBK；媒体文件登记成资产不复制。
"""

from __future__ import annotations

from pathlib import Path

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.files import FileFunctions, FsPolicy, decode_text


def _fns(tmp_path: Path, **policy) -> tuple[FileFunctions, AssetStore, Path]:
    store = AssetStore(tmp_path / "assets")
    ws = tmp_path / "ws"
    out = tmp_path / "out"
    ext = tmp_path / "ext"
    for d in (ws, out, ext):
        d.mkdir(parents=True, exist_ok=True)
    pol = FsPolicy(roots=[ext], **policy)
    fns = FileFunctions(store, ws, out, pol, project_root=tmp_path / "proj")
    return fns, store, out


# ---------------------------------------------------------------- 边界


async def test_根目录白名单_外面的拒绝_相对路径按产物目录(tmp_path: Path):
    fns, _, out = _fns(tmp_path)
    (tmp_path / "secret.txt").write_text("x", encoding="utf-8")
    r = await fns.invoke("fs_read", {"path": str(tmp_path / "secret.txt")})
    assert not r.ok and "不在允许访问的目录内" in r.error and "filesystem.yaml" in r.error
    (out / "a.txt").write_text("hello", encoding="utf-8")
    r = await fns.invoke("fs_read", {"path": "a.txt"})
    assert r.ok and "hello" in r.content
    roots = (await fns.invoke("fs_roots", {})).content
    assert "ext" in roots and "out" in roots and ".env" in roots


async def test_凭据文件永远不碰(tmp_path: Path):
    fns, _, out = _fns(tmp_path)
    (out / ".env").write_text("KEY=1", encoding="utf-8")
    (out / "id_rsa").write_text("---", encoding="utf-8")
    for name in (".env", "id_rsa"):
        r = await fns.invoke("fs_read", {"path": str(out / name)})
        assert not r.ok and "禁止访问" in r.error, name
    r = await fns.invoke("fs_write", {"path": str(out / ".env.local"), "content": "x"})
    assert not r.ok
    # 列目录时也不露出来
    listed = (await fns.invoke("fs_list", {"path": str(out)})).content
    assert ".env" not in listed and "id_rsa" not in listed


async def test_权限分级(tmp_path: Path):
    fns, _, _ = _fns(tmp_path)
    metas = {m.name: m.permission.value for m in await fns.list_tools()}
    for name in ("fs_roots", "fs_list", "fs_info", "fs_read", "fs_search", "find_materials"):
        assert metas[name] == "L-read", name
    writes = ("fs_write", "fs_mkdir", "fs_move", "fs_copy", "fs_delete", "fs_import", "fs_export")
    for name in writes:
        assert metas[name] == "L-write", name
    assert len(metas) == 13


# ---------------------------------------------------------------- 只读


async def test_列目录_过滤_递归_翻页(tmp_path: Path):
    fns, _, out = _fns(tmp_path)
    (out / "sub").mkdir()
    (out / "a.txt").write_text("a", encoding="utf-8")
    (out / "b.mp4").write_bytes(b"\x00" * 2048)
    (out / "sub" / "c.txt").write_text("c", encoding="utf-8")
    r = await fns.invoke("fs_list", {"path": str(out)})
    assert r.ok and "[目录] sub/" in r.content and "a.txt" in r.content and "2.0KB" in r.content
    assert "共 3 项" in r.content
    r = await fns.invoke("fs_list", {"path": str(out), "pattern": "*.txt"})
    assert "a.txt" in r.content and "b.mp4" not in r.content
    r = await fns.invoke("fs_list", {"path": str(out), "recursive": True, "pattern": "*.txt"})
    assert "sub/c.txt" in r.content
    r = await fns.invoke("fs_list", {"path": str(out), "limit": 1, "offset": 1})
    assert "还有 1 项" in r.content and "offset=2" in r.content
    r = await fns.invoke("fs_list", {"path": str(out / "a.txt")})
    assert not r.ok and "不是目录" in r.error


async def test_信息(tmp_path: Path):
    fns, _, out = _fns(tmp_path)
    (out / "v.mp4").write_bytes(b"0" * 10)
    r = await fns.invoke("fs_info", {"path": str(out / "v.mp4")})
    assert r.ok and "video/mp4" in r.content and "资产类型 video" in r.content
    assert "目录" in (await fns.invoke("fs_info", {"path": str(out)})).content
    assert "不存在" in (await fns.invoke("fs_info", {"path": str(out / "nope")})).content


def test_编码识别():
    assert decode_text("你好".encode("utf-8-sig")) == ("你好", "utf-8-sig")
    assert decode_text("你好".encode("gbk")) == ("你好", "gbk")
    assert decode_text(b"\x00\x01binary") is None


async def test_读文本_分段_二进制_超大(tmp_path: Path):
    fns, _, out = _fns(tmp_path, max_read_bytes=100)
    (out / "g.txt").write_bytes("第一行\n第二行".encode("gbk"))
    r = await fns.invoke("fs_read", {"path": str(out / "g.txt")})
    assert r.ok and "gbk" in r.content and "第二行" in r.content
    r = await fns.invoke("fs_read", {"path": str(out / "g.txt"), "offset": 4, "max_chars": 2})
    assert "第二" in r.content and "第一行" not in r.content and "offset=6" in r.content
    (out / "b.bin").write_bytes(b"\x00\x01\x02")
    r = await fns.invoke("fs_read", {"path": str(out / "b.bin")})
    assert r.ok and "二进制文件" in r.content and "fs_import" in r.content
    (out / "big.txt").write_bytes(b"x" * 200)
    r = await fns.invoke("fs_read", {"path": str(out / "big.txt")})
    assert not r.ok and "超过单次读取上限" in r.error


async def test_搜索(tmp_path: Path):
    fns, _, out = _fns(tmp_path)
    (out / "sub").mkdir()
    (out / "sub" / "ep1.txt").write_text("陆离：别回头。\n小满：为什么？", encoding="utf-8")
    (out / "notes.md").write_text("todo 陆离 换装", encoding="utf-8")
    (out / "clip.mp4").write_bytes(b"\x00" + "陆离".encode())
    r = await fns.invoke("fs_search", {"path": str(out), "query": "陆离"})
    assert r.ok and "sub/ep1.txt:1: 陆离：别回头。" in r.content and "notes.md:1:" in r.content
    assert "clip.mp4" not in r.content
    r = await fns.invoke("fs_search", {"path": str(out), "query": "^小满", "regex": True})
    assert "ep1.txt:2:" in r.content and "notes.md" not in r.content
    r = await fns.invoke("fs_search", {"path": str(out), "query": "陆离", "glob": "*.md"})
    assert "ep1.txt" not in r.content and "notes.md" in r.content
    r = await fns.invoke("fs_search", {"path": str(out), "query": "(", "regex": True})
    assert not r.ok and "正则" in r.error


# ---------------------------------------------------------------- 写（可回滚）


async def test_写文件_新建拒绝覆盖_覆盖有备份_追加(tmp_path: Path):
    fns, _, out = _fns(tmp_path)
    target = out / "deep" / "s.txt"
    r = await fns.invoke("fs_write", {"path": str(target), "content": "v1"})
    assert r.ok and target.read_text(encoding="utf-8") == "v1"
    r = await fns.invoke("fs_write", {"path": str(target), "content": "v2"})
    assert not r.ok and "已存在" in r.error
    r = await fns.invoke("fs_write", {"path": str(target), "content": "v2", "mode": "overwrite"})
    assert r.ok and target.read_text(encoding="utf-8") == "v2" and "备份到" in r.content
    backups = list((tmp_path / "ws" / "trash").rglob("s.txt"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == "v1"
    r = await fns.invoke("fs_write", {"path": str(target), "content": "+3", "mode": "append"})
    assert r.ok and target.read_text(encoding="utf-8") == "v2+3"
    r = await fns.invoke("fs_write", {"path": str(target), "content": "x", "mode": "clobber"})
    assert not r.ok


async def test_建目录_移动_复制_删除到回收(tmp_path: Path):
    fns, _, out = _fns(tmp_path)
    assert (await fns.invoke("fs_mkdir", {"path": str(out / "x" / "y")})).ok
    assert (out / "x" / "y").is_dir()
    src = out / "a.txt"
    src.write_text("a", encoding="utf-8")
    r = await fns.invoke("fs_move", {"src": str(src), "dst": str(out / "x" / "b.txt")})
    assert r.ok and not src.exists() and (out / "x" / "b.txt").exists()
    r = await fns.invoke("fs_copy", {"src": str(out / "x" / "b.txt"), "dst": str(out / "x" / "y")})
    assert r.ok and (out / "x" / "y" / "b.txt").exists(), "目标是目录就复制进去"
    dup = {"src": str(out / "x" / "b.txt"), "dst": str(out / "x" / "y" / "b.txt")}
    r = await fns.invoke("fs_copy", dup)
    assert not r.ok and "已存在" in r.error
    r = await fns.invoke("fs_delete", {"path": str(out / "x" / "y" / "b.txt")})
    assert r.ok and not (out / "x" / "y" / "b.txt").exists() and "回收目录" in r.content
    assert list((tmp_path / "ws" / "trash").rglob("b.txt"))
    r = await fns.invoke("fs_delete", {"path": str(out)})
    assert not r.ok and "根目录" in r.error


# ---------------------------------------------------------------- 与资产互通


async def test_登记本地文件为资产(tmp_path: Path):
    fns, store, out = _fns(tmp_path)
    (out / "剧本.txt").write_text("第1集\n陆离：别回头。", encoding="utf-8")
    (out / "素材.mp4").write_bytes(b"\x00" * 10)
    r = await fns.invoke("fs_import", {"path": str(out / "剧本.txt"), "kind": "script"})
    assert r.ok and r.asset_ref
    a = store.get(r.asset_ref)
    assert a.type is AssetType.SCRIPT and "别回头" in store.content(a.id)
    r = await fns.invoke("fs_import", {"path": str(out / "素材.mp4"), "summary": "用户素材"})
    assert r.ok
    v = store.get(r.asset_ref)
    assert v.type is AssetType.VIDEO and v.uri == str((out / "素材.mp4").resolve())
    assert v.gen_params["local"] == v.uri and v.summary == "用户素材" and "原文件不动" in r.content
    r = await fns.invoke("fs_import", {"path": str(out / "素材.xyz")})
    assert not r.ok


async def test_导出资产(tmp_path: Path):
    fns, store, out = _fns(tmp_path)
    script = store.create("第1集剧本正文", type_=AssetType.SCRIPT, summary="第1集 剧本")
    # 给的是文件路径 → 按这个名字写
    r = await fns.invoke("fs_export", {"asset_id": script.id, "path": str(out / "导出.md")})
    assert r.ok and (out / "导出.md").read_text(encoding="utf-8") == "第1集剧本正文"
    r = await fns.invoke("fs_export", {"asset_id": script.id, "path": str(out / "导出.md")})
    assert not r.ok and "已存在" in r.error
    # 给的是目录 → 用资产摘要起名
    (out / "导出目录").mkdir()
    r = await fns.invoke("fs_export", {"asset_id": script.id, "path": str(out / "导出目录")})
    assert r.ok
    files = list((out / "导出目录").glob("*.md"))
    assert len(files) == 1 and "剧本正文" in files[0].read_text(encoding="utf-8")
    clip = store.create("", type_=AssetType.VIDEO, summary="片段")
    (out / "src.mp4").write_bytes(b"mp4")
    clip.gen_params["local"] = str(out / "src.mp4")
    store.put(clip)
    target = out / "导出目录" / "c.mp4"
    r = await fns.invoke("fs_export", {"asset_id": clip.id, "path": str(target)})
    assert r.ok and target.read_bytes() == b"mp4"
    r = await fns.invoke("fs_export", {"asset_id": "as_nope", "path": str(out)})
    assert not r.ok and "没有资产" in r.error


async def test_错误参数不炸(tmp_path: Path):
    fns, _, _ = _fns(tmp_path)
    r = await fns.invoke("fs_read", {"path": "", "bogus": 1})
    assert not r.ok and "TypeError" in r.error
