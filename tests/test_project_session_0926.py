"""项目设置按项目走、/out 换会话、第二个窗口另起会话、产物目录以外的写要确认
（2026-09-26 用户定的）。

- /out 到别的项目：对话窗口、挂起的人审、集长 / 画幅 / 模型锁都换成那个项目自己的；
  原来的存回原项目，/out 回去接着聊（之前 /out 只换资产和记忆，西游记按《不渡》的 20 分钟跑）
- 换锁写回当前项目的快照，不写到上一个项目
- 记忆提取按入队时的项目键记：/out 之后排着没提取完的旧项目对话，不能记到新项目名下
- 同一个文件夹的第二个窗口：另起「名字~2」会话、设置抄主窗口一份；主窗口关了再开就是主窗口
- 文件工具写 / 移 / 删落在产物目录以外要人确认；复制进产物目录不用
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aigc_agent.capabilities.memory.agent import MemoryAgent
from aigc_agent.capabilities.memory.session import SessionLock, SessionSnapshot, open_session
from aigc_agent.capabilities.memory.store import MemoryStore
from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.files import FileFunctions, FsPolicy
from aigc_agent.domain.project import project_key
from aigc_agent.harness.context.window import ShortTermMemory
from aigc_agent.harness.tools.provider import PermissionLevel


def _say(agent: Any, text: str) -> Any:
    t = agent.memory.new_turn()
    t.messages = [{"role": "user", "content": text}, {"role": "assistant", "content": "好"}]
    return t


# ---------------------------------------------------------------- /out 换会话


async def test_out换项目连对话和项目设置一起换_回来接着聊(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from aigc_agent.app import Agent

    ws, a_dir, b_dir = tmp_path / "ws", tmp_path / "不渡", tmp_path / "西游记"
    a_dir.mkdir()
    b_dir.mkdir()
    monkeypatch.chdir(a_dir)
    agent = Agent.create(workspace=ws)
    try:
        base = agent.base_episode_fmt.minutes
        agent.set_episode_minutes(20)
        agent.set_aspect_ratio("16:9")
        agent.media_fns.set_video_lock("veo3.1-quality")
        t = _say(agent, "不渡第 3 集改一下")
        agent.loop.pending_review = {"turn_index": t.index, "stage": "剧本", "question": "?"}

        key_b = agent.switch_project(b_dir, session=True)
        assert agent.session_store.name == key_b
        assert agent.memory.turns == [], "西游记还没有对话，不带《不渡》的"
        assert agent.loop.pending_review is None, "《不渡》挂着的人审不带过来"
        assert agent.episode_fmt.minutes == base, "西游记没设过集长：用默认，不是《不渡》的 20 分钟"
        assert not agent.aspect_custom
        assert agent.media_fns.video_lock != "veo3.1-quality", "模型锁也是按项目的"

        agent.set_episode_minutes(None, auto=True)
        agent.media_fns.set_image_lock("nano-banana")
        _say(agent, "西游记开个头")

        saved_a = SessionSnapshot(ws / "memory" / "sessions", project_key(a_dir))
        assert saved_a.episode_minutes == 20 and saved_a.video_model == "veo3.1-quality"
        assert saved_a.image_model == "", "西游记换的生图锁写到西游记，不写到《不渡》"

        agent.switch_project(a_dir, session=True)
        assert agent.session_store.name == project_key(a_dir)
        assert [m["content"] for m in agent.memory.turns[0].messages][0] == "不渡第 3 集改一下"
        assert agent.loop.pending_review is not None, "回来之后挂着的人审还在"
        assert agent.episode_fmt.minutes == 20 and agent.aspect_ratio == "16:9"
        assert agent.media_fns.video_lock == "veo3.1-quality"

        saved_b = SessionSnapshot(ws / "memory" / "sessions", key_b)
        assert saved_b.episode_auto is True and saved_b.image_model == "nano-banana"
        assert saved_b.load_into(ShortTermMemory()) == 1, "西游记那一轮存回了西游记"
    finally:
        await agent.aclose()


async def test_开工时切到记住的目录不换会话(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """带 --session 的会话带着它记住的目录走：switch_project 默认不换会话。"""
    from aigc_agent.app import Agent

    monkeypatch.chdir(tmp_path)
    agent = Agent.create(workspace=tmp_path / "ws", session_id="我的会话")
    try:
        _say(agent, "你好")
        agent.switch_project(tmp_path / "别的剧")
        assert agent.session_store.name == "我的会话"
        assert len(agent.memory.turns) == 1
    finally:
        await agent.aclose()


# ---------------------------------------------------------------- 记忆按入队时的项目记


async def test_换项目之后排着的旧对话仍记在旧项目名下():
    class _Gw:
        async def chat(self, role: str, messages: list[dict[str, Any]], **_: Any) -> Any:
            await asyncio.sleep(0)
            return SimpleNamespace(
                text='[{"content": "唐僧的台词别太文绉绉", "terms": ["唐僧"], '
                '"polarity": "negative", "category": "constraint"}]'
            )

    store = MemoryStore()
    mem = MemoryAgent(_Gw(), store, project_id="不渡")
    assert mem.submit("用户：唐僧的台词别太文绉绉")
    mem.project_id = "西游记"  # /out 换了项目，提取还没跑
    await mem.close()
    assert [m.project_id for m in store.all()] == ["不渡"]


# ---------------------------------------------------------------- 第二个窗口


def test_会话锁_同一份只给一个窗口(tmp_path: Path):
    first = SessionLock(tmp_path / "a.lock")
    assert first.acquire()
    second = SessionLock(tmp_path / "a.lock")
    assert not second.acquire(), "别的窗口占着"
    first.release()
    assert second.acquire(), "主窗口关了就能占"
    second.release()


def test_第二个窗口另起会话_设置抄主窗口一份(tmp_path: Path):
    root = tmp_path / "sessions"
    main, lock, second = open_session(root, "proj")
    assert not second and main.name == "proj"
    main.set_episode_minutes(20)
    main.set_video_model("seedance-2.0")
    other, lock2, second2 = open_session(root, "proj")
    try:
        assert second2 and other.name == "proj~2"
        assert other.episode_minutes == 20 and other.video_model == "seedance-2.0"
        assert other.pending_review is None
        third, lock3, second3 = open_session(root, "proj")
        assert second3 and third.name == "proj~3"
        lock3.release()
    finally:
        lock2.release()
        lock.release()
    again, lock4, second4 = open_session(root, "proj")
    assert not second4 and again.name == "proj"
    lock4.release()


async def test_同一个文件夹开第二个Agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from aigc_agent.app import Agent

    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "ws"
    first = Agent.create(workspace=ws)
    first.set_episode_minutes(12)
    second = Agent.create(workspace=ws)
    try:
        assert not first.second_window
        assert second.second_window
        assert second.session_store.name == f"{first.session_store.name}~2"
        assert second.episode_fmt.minutes == 12, "项目设置抄主窗口的"
        assert second.project == first.project, "同一个项目：资产、记忆照样共用"
    finally:
        await second.aclose()
        await first.aclose()
    third = Agent.create(workspace=ws)
    try:
        assert not third.second_window, "主窗口关了，再开就是主窗口"
    finally:
        await third.aclose()


# ---------------------------------------------------------------- 产物目录以外的写要确认


def _files(tmp_path: Path) -> FileFunctions:
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "别处").mkdir()
    (tmp_path / "别处" / "a.txt").write_text("x", encoding="utf-8")
    ws = tmp_path / "ws"
    ws.mkdir()
    return FileFunctions(
        AssetStore(), ws, output_root=out, policy=FsPolicy(roots=[tmp_path]),
        project_root=tmp_path / "proj",
    )


def test_产物目录以外的写移删要确认(tmp_path: Path):
    fns = _files(tmp_path)
    out, other = tmp_path / "out", tmp_path / "别处"
    assert fns.permission_for("fs_write", {"path": str(out / "笔记.md"), "content": "x"}) is None
    assert fns.permission_for("fs_write", {"path": "相对路径.md", "content": "x"}) is None
    for tool, args in (
        ("fs_write", {"path": str(other / "b.txt"), "content": "x"}),
        ("fs_delete", {"path": str(other / "a.txt")}),
        ("fs_move", {"src": str(other / "a.txt"), "dst": str(out / "a.txt")}),
        ("fs_mkdir", {"path": str(other / "新目录")}),
        ("fs_export", {"asset_id": "as_x", "path": str(other)}),
    ):
        got = fns.permission_for(tool, args)
        assert got is not None, tool
        level, why = got
        assert level is PermissionLevel.EXTERNAL
        assert "产物目录" in why and "别处" in why
    # 复制进产物目录：源不动，不用问
    assert fns.permission_for(
        "fs_copy", {"src": str(other / "a.txt"), "dst": str(out / "a.txt")}
    ) is None
