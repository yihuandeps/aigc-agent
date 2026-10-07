"""渲参考图边渲边记进度，超时按张数放宽（2026-09-29 审查 1.4）。

之前参考图包要到整批渲完才落库，工具超时写死 1 小时；《不渡》174 张按实测速度要 50–90 分钟，
一超时已付费的图不进任何包，重跑全部重付。现在：
  · 每出好一张就记进进度（按资产库 id，不进资产库），下次渲染认领、只补没出的；
  · 整批落包后清掉进度；
  · 注册表按这次要生成的张数放宽超时。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.tools.registry import ToolRegistry
from tests.test_drama_render import FakeRegistry, _lib_asset

COSTUME = "服装·安妮-制服-[全集]"
FIRST_LAYER = {"安妮", "老王", "酒店走廊", "行李箱"}


async def _interrupted(store: AssetStore) -> tuple[str, DramaFunctions, FakeRegistry]:
    """第一层（2 角色 + 场景 + 道具）很快出完，服装卡住 → 外面的超时把整个调用取消。"""
    lib = _lib_asset(store)
    slow = FakeRegistry(store, delays={COSTUME: 30})
    fns = DramaFunctions(None, store, registry=slow)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(fns._fn_drama_render_assets(lib), timeout=1.0)
    assert not store.find(creator="tool:drama_render_assets"), "半成品不进资产库"
    return lib, fns, slow


async def test_渲到一半断了_出好的接着用_只补没出的():
    store = AssetStore()
    lib, fns, slow = await _interrupted(store)
    assert len(slow.calls) == 5  # 第一层 4 张 + 卡住的那张服装

    fast = FakeRegistry(store)
    fns.registry = fast  # 同一个会话里重跑
    r = await fns._fn_drama_render_assets(lib)
    assert r.ok, r.error
    assert list(fast.calls) == [COSTUME], "只补服装，第一层 4 张不重付"
    assert "其中 4 张是上次渲到一半留下的" in r.content
    pack = json.loads(store.content(r.asset_ref))
    assert set(pack) == FIRST_LAYER | {"安妮-制服-[全集]"}
    assert fns._scratch_load("refpack", lib) is None, "整批落包后进度清掉"


async def test_换了会话也接得上_进度存在资产库目录里(tmp_path: Path):
    lib, _, _ = await _interrupted(AssetStore(tmp_path / "assets"))
    files = list((tmp_path / "assets" / "progress" / "refpack").glob("*.json"))
    assert len(files) == 1
    assert set(json.loads(files[0].read_text(encoding="utf-8"))) == FIRST_LAYER

    store = AssetStore(tmp_path / "assets")  # 新会话：重新读盘
    fast = FakeRegistry(store)
    r = await DramaFunctions(None, store, registry=fast)._fn_drama_render_assets(lib)
    assert r.ok, r.error
    assert list(fast.calls) == [COSTUME]
    assert not list((tmp_path / "assets" / "progress" / "refpack").glob("*.json"))


async def test_没断过的照常_不留进度(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    lib = _lib_asset(store)
    fake = FakeRegistry(store)
    r = await DramaFunctions(None, store, registry=fake)._fn_drama_render_assets(lib)
    assert r.ok and len(fake.calls) == 5 and "渲到一半" not in r.content
    assert not list((tmp_path / "assets" / "progress").rglob("*.json"))


# ---------------------------------------------------------------- 超时按张数放宽


async def test_超时按要生成的张数放宽_不少于一小时():
    store = AssetStore()
    chars = [
        {"baseRoleName": f"角色{i}", "roleTotalDesc": "x",
         "roleCostumeList": [{"costumeName": f"角色{i}-服装{j}-[全集]", "costumeDesc": "x"}
                             for j in range(3)]}
        for i in range(30)
    ]
    lib = store.create(json.dumps({
        "characters": chars,
        "scenes": [{"name": f"场景{i}", "description": "x"} for i in range(40)],
        "props": [{"name": f"道具{i}", "description": "x"} for i in range(14)],
    }, ensure_ascii=False), summary="资产库", creator="t").id
    fns = DramaFunctions(None, store)
    assert fns.timeout_for("drama_render_assets", {"assets_id": lib}) == 60.0 * 174
    assert fns.timeout_for("drama_render_assets", {"assets_id": lib, "only": "scenes"}) == 3600
    assert fns.timeout_for("drama_shots", {"assets_id": lib}) is None

    reg = ToolRegistry(EventBus())
    reg.register(fns)
    await reg.refresh()
    assert reg.meta_for_call("drama_render_assets", {"assets_id": lib}).timeout == 60.0 * 174
    small = _lib_asset(store)
    assert reg.meta_for_call("drama_render_assets", {"assets_id": small}).timeout == 3600
