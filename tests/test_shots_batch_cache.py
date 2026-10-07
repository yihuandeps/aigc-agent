"""分批出提示词：一批失败，出好的批先存着，再跑只补失败的（2026-09-29 审查 2.4）。

之前任一批抛异常就整个往上抛、任一批报错就 return []，已经成功的批全部丢掉，重跑整集重付
（日志里因此白付了 21 次调用）。现在按「这一批发给模型的完整消息」算指纹存结果：
同样的输入再跑直接沿用；整份落库之后清掉，之后再跑还是重新出。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.drama.format import EpisodeFormat
from aigc_agent.domain.functions.drama import DramaFunctions
from tests.test_episode_format import LIB
from tests.test_review_0924 import _ChunkGateway, _long_storyboard, _shots_setup


class _FlakyGateway(_ChunkGateway):
    """写第 51 镜的那一批，第一次调用抛错（网关 502），之后正常。"""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def chat(self, role: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
        if "〔51〕" in messages[1]["content"] and not self.failed:
            self.failed = True
            self.calls.append(messages)
            raise RuntimeError("网关 502")
        return await super().chat(role, messages, **kw)


async def test_一批失败_出好的批存着_再跑只补失败的那批():
    gw = _FlakyGateway()
    fns, store, sb, lib = _shots_setup(gw)
    r = await fns.invoke("drama_shots", {"storyboard_id": sb, "assets_id": lib})
    assert not r.ok and "RuntimeError: 网关 502" in r.error
    assert "已经出好的 2 批提示词先存着" in r.error
    assert len(gw.calls) == 3

    r = await fns.invoke("drama_shots", {"storyboard_id": sb, "assets_id": lib})
    assert r.ok, r.error
    assert len(gw.calls) == 4, "只补失败的那一批，出好的两批不重付"
    assert "〔51〕" in gw.calls[-1][1]["content"]
    saved = json.loads(store.content(r.asset_ref))
    assert len(saved) == 30, "三批拼起来覆盖全部 150 镜"


async def test_整份落库后清掉_再跑是重新出():
    gw = _ChunkGateway()
    fns, _, sb, lib = _shots_setup(gw)
    assert (await fns.invoke("drama_shots", {"storyboard_id": sb, "assets_id": lib})).ok
    assert (await fns.invoke("drama_shots", {"storyboard_id": sb, "assets_id": lib})).ok
    assert len(gw.calls) == 6, "成功落库的那次不留缓存：用户要重出就该真的重出"


async def test_换了会话也接得上_存在资产库目录里(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    sb = store.create(
        json.dumps([{"episodeIndex": 1, "episodeTitle": "第1集",
                     "episodeDesc": _long_storyboard()}], ensure_ascii=False),
        summary="分镜", creator="tool:drama_storyboard",
    )
    lib = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库",
                       creator="tool:drama_assets")
    args = {"storyboard_id": sb.id, "assets_id": lib.id}
    first = _FlakyGateway()
    fns = DramaFunctions(first, store, registry=None, catalog=None, fmt=EpisodeFormat())
    assert not (await fns.invoke("drama_shots", args)).ok
    assert list((tmp_path / "assets" / "progress" / "shots").glob("*.json"))

    again = _ChunkGateway()  # 新会话：新的实例，只剩磁盘上的那份
    fns2 = DramaFunctions(again, store, registry=None, catalog=None, fmt=EpisodeFormat())
    r = await fns2.invoke("drama_shots", args)
    assert r.ok, r.error
    assert len(again.calls) == 1 and "〔51〕" in again.calls[0][1]["content"]
    assert not list((tmp_path / "assets" / "progress" / "shots").glob("*.json")), "落库后清掉"
