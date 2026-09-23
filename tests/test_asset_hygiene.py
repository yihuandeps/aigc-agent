"""资产卫生：空壳不许落库、取剧本不许取到空壳和假货（2026-09-22）。

真实事故链：
  9-21 模型照着自己被折叠的历史写 save_draft，12 集分镜存成 12 份 11 字的
       「<4939 字已折叠>」—— 工具报成功，直到 drama_shots 读它才炸。
  9-18 起测试往真实 workspace 写「第1集·标题」的 30 字假剧本，攒了几十份。
两批垃圾都带 type=script、都比真剧本新，`script_of` 取最新就永远取到它们，
下游拿着 30 个字去拆分镜 —— 用户看到的就是"读资产时报占位符错误"。

所以三道：参数层拦（见 test_p6_context_control）、**落库层拦**、**取的时候筛**。
"""

from __future__ import annotations

import pytest

from aigc_agent.domain.assets.store import AssetStore, AssetType


def _script(store: AssetStore, text: str, episode: int, summary: str) -> str:
    """绕开 create 的守卫直接塞一份（模拟存量脏数据）。"""
    from aigc_agent.domain.assets.store import Asset

    a = Asset(type=AssetType.SCRIPT, inline=text, summary=summary,
              gen_params={"episode": episode}, creator="model")
    return store.put(a).id


# ---------------------------------------------------------------- 落库层


def test_整段是占位符的内容不许落库():
    store = AssetStore()
    with pytest.raises(ValueError, match="占位符"):
        store.create("<4939 字已折叠>", type_=AssetType.SCRIPT, summary="分镜·第1集")
    with pytest.raises(ValueError, match="占位符"):
        store.create("  <29000字剧本全文>  ", type_=AssetType.TEXT)


def test_报错要说清楚该怎么办():
    store = AssetStore()
    with pytest.raises(ValueError) as e:
        store.create("<812 字已折叠>")
    msg = str(e.value)
    assert "重新写出来" in msg and "fs_write" in msg, "光说不存没用，得给一条能走的路"


def test_正常内容照存():
    store = AssetStore()
    a = store.create("正常的一整集剧本正文。" * 40, type_=AssetType.SCRIPT)
    assert a.id and len(store.content(a.id)) > 300
    # 正文里**提到**折叠这回事，不算占位符
    b = store.create("上下文里 12 字已折叠的部分要 read_asset 取回。" * 20)
    assert b.id


# ---------------------------------------------------------------- 取剧本


def test_取剧本跳过折叠空壳():
    store = AssetStore()
    real = _script(store, "真剧本正文。" * 100, 1, "第1集·天上掉下个蛛妹妹")
    _script(store, "<4939 字已折叠>", 1, "分镜·第1集（手写JSON正式版）")  # 比真剧本新
    assert store.script_of(1).id == real


def test_取剧本跳过太短的假货():
    """30 字的「第1集·标题」带着 type=script，比真剧本新 —— 不按长度筛就永远挡在前面。"""
    store = AssetStore()
    real = _script(store, "真剧本正文。" * 100, 1, "第1集·天上掉下个蛛妹妹")
    for _ in range(5):
        _script(store, "第1集·标题", 1, "第1集·标题")
    assert store.script_of(1).id == real


def test_全都偏短时退回最长的一份():
    """别因为筛得狠就装作没有 —— 也可能真是一集很短的剧本。"""
    store = AssetStore()
    _script(store, "短的", 2, "第2集·甲")
    longer = _script(store, "稍微长一点的那份", 2, "第2集·乙")
    got = store.script_of(2)
    assert got is not None and got.id == longer


def test_全是空壳就当没有():
    store = AssetStore()
    _script(store, "<4939 字已折叠>", 3, "分镜·第3集")
    _script(store, "<812 字已折叠>", 3, "分镜·第3集 v2")
    assert store.script_of(3) is None, "空壳一份都不该被当成剧本"


def test_没标type也能认出剧本_但空壳不算():
    """模型 save_draft(kind="outline") 存的整集也算剧本；但摘要带「大纲/目录/报告」
    这类词的是别的东西，本来就排除在外（_NOT_SCRIPT_WORDS）。"""
    store = AssetStore()
    from aigc_agent.domain.assets.store import Asset

    store.put(Asset(type=AssetType.OUTLINE, inline="整集正文。" * 100,
                    summary="第4集·盘丝洞的梦（合规修订版）", gen_params={"episode": 4}))
    store.put(Asset(type=AssetType.OUTLINE, inline="<4939 字已折叠>",
                    summary="第4集·盘丝洞的梦（更新）", gen_params={"episode": 4}))
    got = store.script_of(4)
    assert got is not None and "合规修订版" in got.summary, "空壳更新，但取的该是真那份"
