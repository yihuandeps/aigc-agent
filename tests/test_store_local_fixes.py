"""2026-09-23 审查的资产底座 / 本地素材问题。

  · 资产库和本地素材都不认「第十二集」这种中文数字
  · 回收站不记原路径，恢复只能靠猜
  · fs_import 在线程里建资产，ASSET_CREATED 发不出去，按集流水漏接
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.local_materials import episode_in_name
from aigc_agent.domain.numerals import cn_to_int, episode_from
from aigc_agent.domain.system_prompt import MINOR_STAGE
from aigc_agent.harness.events.bus import EventBus, EventType


def test_中文数字集号():
    assert [cn_to_int(x) for x in ("十二", "二十", "一百零五", "两", "九十九", "12")] == \
        [12, 20, 105, 2, 99, 12]
    assert cn_to_int("十二a") == 0
    assert episode_from("第十二集·合规修订版") == 12 and episode_from("第 7 集") == 7
    assert episode_in_name("第三十集-02_镜1.mp4") == 30
    assert MINOR_STAGE.search("剧本第十二集")


def test_资产库认中文集号():
    store = AssetStore()
    store.create("正文。" * 120, type_=AssetType.OUTLINE, summary="第十二集·合规修订版",
                 creator="model")
    assert store.episodes_done() == [12]
    assert store.script_of(12) is not None


async def test_线程里建的资产也发落库事件():
    bus = EventBus()
    got: list[str] = []
    bus.subscribe(lambda e: got.append(e.data["id"]) if e.type is EventType.ASSET_CREATED else None)
    store = AssetStore()
    store.bus = bus
    store.bind_loop()
    a = await asyncio.to_thread(lambda: store.create("x", summary="线程里建的"))
    for _ in range(20):
        if a.id in got:
            break
        await asyncio.sleep(0.01)
    assert a.id in got, "fs_import 走线程，事件之前直接丢了"


async def test_回收站记原路径(tmp_path: Path):
    from aigc_agent.domain.functions.files import FileFunctions, FsPolicy
    from aigc_agent.domain.output import OutputPrefs

    ws = tmp_path / "ws"
    out = tmp_path / "out"
    out.mkdir()
    doomed = out / "旧稿.md"
    doomed.write_text("旧", encoding="utf-8")
    fns = FileFunctions(AssetStore(), ws, OutputPrefs(out), FsPolicy(), project_root=tmp_path / "p")
    r = await fns.invoke("fs_delete", {"path": str(doomed)})
    assert r.ok, r.error
    def check() -> None:
        manifests = list((ws / "trash").rglob("manifest.jsonl"))
        assert manifests, "回收站要记下原路径"
        row = json.loads(manifests[0].read_text(encoding="utf-8").splitlines()[0])
        assert row["from"] == str(doomed) and Path(row["to"]).exists()

    await asyncio.to_thread(check)
