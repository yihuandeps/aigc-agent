"""TTS / ASR 验收。

和图像视频的关键区别：**调用形状不同**。
  TTS 同步返回二进制音频（没有 URL 可轮询）
  ASR 走 multipart 上传文件

重点验：
  1. 二进制产物正确落盘成资产，上下文里只出现路径不出现字节
  2. srt/vtt 直接是字幕资产 —— 短视频链路的字幕由此而来
  3. 输入校验（4096 字上限、25MB 上限、格式白名单）在**发请求之前**拦住
  4. 配音 → 转写(srt) 串起来时血缘不断
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.audio import AudioFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.audio import AudioGateway, FakeAudioProvider
from aigc_agent.harness.tools.registry import ToolRegistry

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"


async def _funcs(tmp_path: Path, provider: FakeAudioProvider | None = None):
    p = provider or FakeAudioProvider()
    bus = EventBus()
    store = AssetStore(tmp_path)
    catalog = MediaCatalog.load(CATALOG_PATH)
    # TTS 和 ASR 现在可以走不同 provider（中文口播要换中文原生的 TTS），
    # 所以假 provider 两个槽都要占上，否则真配置一改这里就全红。
    providers = {catalog.provider: p, catalog.speech_provider: p}
    fns = AudioFunctions(AudioGateway(providers, bus), catalog, store)
    registry = ToolRegistry(bus)
    registry.register(fns)
    await registry.refresh()
    return registry, store, p


# ---------------------------------------------------------------- 目录


def test_音色与模型目录从yaml加载():
    """音色表和模型表都按 provider 分区。

    换 TTS provider 之后音色名整套都变了 —— 拿 nova 去请求 MiniMax
    它**不会报错**，而是静默换成自己的默认音色，比报错更难发现。
    """
    c = MediaCatalog.load(CATALOG_PATH)
    assert c.speech_voices, "当前 TTS provider 没有可用音色"
    assert c.has_voice(c.speech_default_voice), "默认音色不在自己的音色表里"
    # 当前 provider 的模型才该出现在候选里
    for m in c.audio_models("tts"):
        assert (m.provider or c.provider) == c.speech_provider
    assert [m.id for m in c.audio_models("asr")] == ["whisper-1"]


def test_跨provider的音色和模型不会混进候选():
    """选到别家的模型就是 'not have model'，这个错该在目录层拦住。"""
    c = MediaCatalog.load(CATALOG_PATH)
    ids = {m.id for m in c.audio_models("tts")}
    if c.speech_provider == "minimax":
        assert "gpt-4o-mini-tts" not in ids
        assert not c.has_voice("nova")
    else:
        assert not any(i.startswith("speech-") for i in ids)


def test_tts与asr分别选型互不干扰():
    c = MediaCatalog.load(CATALOG_PATH)
    # 默认 TTS 模型跟着 tts_provider 走 —— 换成中文原生 provider 之后还拿
    # gpt-4o-mini-tts 去请求会直接报 "not have model"，所以两者绑死。
    default_tts = c.choose_audio(kind="tts")[0]
    assert default_tts in {m.id for m in c.audio_models("tts")}
    if c.speech_provider == "minimax":
        assert default_tts.startswith("speech-"), "切到 minimax 了默认模型却没跟着换"
    else:
        assert default_tts == "gpt-4o-mini-tts"
    assert c.choose_audio(kind="asr")[0] == "whisper-1"
    # 拿 asr 模型当 tts 用会被挡下
    picked, why = c.choose_audio("whisper-1", kind="tts")
    assert picked == "" and "未知 tts 模型" in why


def test_目录渲染包含音色说明():
    c = MediaCatalog.load(CATALOG_PATH)
    text = c.render_audio()
    assert "## TTS 模型" in text and "## 音色" in text and "## 转写模型" in text
    # 音色只列当前 TTS provider 那一套（2026-09-29：这里原来断言 onyx / 默认 nova ——
    # 把 bug 锁死了：TTS 走 MiniMax，tts 只认 speech_voices，模型照着目录传 nova 被拒）
    assert all(v.name in text for v in c.speech_voices)
    assert f"默认：{c.speech_default_voice}" in text


async def test_三个function注册且权限正确(tmp_path: Path):
    registry, _, _ = await _funcs(tmp_path)
    perms = {m.name: m.permission.value for m in registry.catalog()}
    assert set(perms) == {"list_voices", "tts", "transcribe"}
    assert perms["list_voices"] == "L-read"
    assert perms["tts"] == "L-compute"  # 花钱
    assert perms["transcribe"] == "L-compute"


# ---------------------------------------------------------------- TTS


async def test_合成音频落盘成资产(tmp_path: Path):
    registry, store, p = await _funcs(tmp_path)
    script = store.create("大家好，今天聊露营装备", summary="口播稿")
    voice = MediaCatalog.load(CATALOG_PATH).speech_default_voice

    r = await registry.invoke(
        "tts", {"text": "大家好，今天聊露营装备", "voice": voice, "parent_id": script.id}
    )
    assert r.ok, r.error

    a = store.get(r.asset_ref)
    assert a.type is AssetType.AUDIO
    assert a.parent_ids == [script.id]  # 血缘挂在稿子下面
    expected = MediaCatalog.load(CATALOG_PATH).choose_audio(kind="tts")[0]
    assert a.creator == f"model:{expected}"
    assert a.gen_params["voice"] == MediaCatalog.load(CATALOG_PATH).speech_default_voice

    # 文件真落盘了，且上下文里只出现路径不出现字节
    blob = Path(a.uri)
    assert blob.exists()  # noqa: ASYNC240 — 测试断言，非运行时热路径
    assert blob.read_bytes() == p.audio  # noqa: ASYNC240
    assert "ID3" not in r.content
    assert a.id in r.content


async def test_未指定音色时用默认(tmp_path: Path):
    registry, store, p = await _funcs(tmp_path)
    await registry.invoke("tts", {"text": "测试"})
    assert p.spoken[0]["voice"] == MediaCatalog.load(CATALOG_PATH).speech_default_voice


async def test_未知音色被挡下且不发请求(tmp_path: Path):
    registry, _, p = await _funcs(tmp_path)
    r = await registry.invoke("tts", {"text": "测试", "voice": "小明"})
    assert not r.ok and "未知音色" in r.error
    # 报错里要列出**当前 provider** 的候选，列别家的等于没帮上忙
    assert MediaCatalog.load(CATALOG_PATH).speech_default_voice in r.error
    assert p.spoken == []


async def test_超长文本在发请求前就被拦住(tmp_path: Path):
    """4096 是硬上限，白跑一趟才报错既慢又浪费。"""
    registry, _, p = await _funcs(tmp_path)
    r = await registry.invoke("tts", {"text": "字" * 5000})
    assert not r.ok
    assert "4096" in r.error and "分段" in r.error
    assert p.spoken == []


async def test_空文本被拦(tmp_path: Path):
    registry, _, p = await _funcs(tmp_path)
    r = await registry.invoke("tts", {"text": "   "})
    assert not r.ok and p.spoken == []


async def test_情绪与语速参数透传(tmp_path: Path):
    """字段名必须是 instructions，不是 instruct。

    这条测试原来断言的是 instruct —— 把 bug 锁死了。接口只认 instructions，
    传 instruct 会被当成未知字段丢掉，所以语气指令一直没生效过。

    （实测经 APIMart 转发的 gpt-4o-mini-tts 连 instructions 也不响应，
    但那是服务端的事；我们这边发对字段是底线，换 provider 就能用上。）
    """
    registry, _, p = await _funcs(tmp_path)
    await registry.invoke(
        "tts", {"text": "你好", "instruct": "轻松亲切", "speed": 1.2, "language": "Chinese"}
    )
    sent = p.spoken[0]
    assert "instruct" not in sent, "又发成 instruct 了，接口不认这个字段"
    assert sent["instructions"] == "轻松亲切"
    assert sent["speed"] == 1.2
    assert sent["language"] == "Chinese"


async def test_合成失败不留垃圾资产(tmp_path: Path):
    registry, store, _ = await _funcs(tmp_path, FakeAudioProvider(fail="speak"))
    r = await registry.invoke("tts", {"text": "你好"})
    assert not r.ok and "合成失败" in r.error
    assert len(store) == 0


async def test_返回json错误体不会被当成音频存下来(tmp_path: Path):
    """有些网关出错时仍返回 200 + JSON，存下来就是个坏文件。"""
    registry, store, _ = await _funcs(tmp_path, FakeAudioProvider(return_json_error=True))
    r = await registry.invoke("tts", {"text": "你好"})
    assert not r.ok
    assert len(store) == 0


# ---------------------------------------------------------------- ASR


async def test_转写出纯文本(tmp_path: Path):
    registry, store, _ = await _funcs(tmp_path)
    audio = store.create_blob(b"fake-audio", ".mp3", type_=AssetType.AUDIO)

    r = await registry.invoke("transcribe", {"asset_id": audio.id})
    assert r.ok, r.error
    a = store.get(r.asset_ref)
    assert a.type is AssetType.TEXT
    assert a.parent_ids == [audio.id]
    assert "转写出来的文本" in store.content(a.id)


async def test_srt直接成为字幕资产(tmp_path: Path):
    """短视频链路的字幕由此而来 —— 不用自己按字数估时间轴。"""
    registry, store, _ = await _funcs(tmp_path)
    audio = store.create_blob(b"fake-audio", ".mp3", type_=AssetType.AUDIO)

    r = await registry.invoke("transcribe", {"asset_id": audio.id, "format": "srt"})
    assert r.ok
    a = store.get(r.asset_ref)
    assert a.type is AssetType.SUBTITLE
    assert "-->" in store.content(a.id)  # 时间轴


async def test_配音再转写形成完整血缘链(tmp_path: Path):
    """脚本 --tts--> 配音 --transcribe--> 字幕，三级血缘不断。"""
    registry, store, _ = await _funcs(tmp_path)
    script = store.create("口播稿正文", summary="脚本")

    r1 = await registry.invoke("tts", {"text": "口播稿正文", "parent_id": script.id})
    r2 = await registry.invoke("transcribe", {"asset_id": r1.asset_ref, "format": "srt"})

    chain = store.lineage(r2.asset_ref)
    assert [a.type for a in chain] == [
        AssetType.TEXT,
        AssetType.AUDIO,
        AssetType.SUBTITLE,
    ]


async def test_不支持的格式在上传前被拦(tmp_path: Path):
    registry, store, p = await _funcs(tmp_path)
    bad = store.create_blob(b"data", ".flac", type_=AssetType.AUDIO)
    r = await registry.invoke("transcribe", {"asset_id": bad.id})
    assert not r.ok and "不支持的格式" in r.error
    assert p.transcribed == []


async def test_超过25MB在上传前被拦(tmp_path: Path):
    registry, store, p = await _funcs(tmp_path)
    big = store.create_blob(b"x" * (26 * 1024 * 1024), ".mp3", type_=AssetType.AUDIO)
    r = await registry.invoke("transcribe", {"asset_id": big.id})
    assert not r.ok and "25MB" in r.error and "切段" in r.error
    assert p.transcribed == []


async def test_不存在的资产给出可读错误(tmp_path: Path):
    registry, _, _ = await _funcs(tmp_path)
    r = await registry.invoke("transcribe", {"asset_id": "as_0000000000"})
    assert not r.ok and "资产不存在" in r.error


async def test_对纯文本资产转写会被拦(tmp_path: Path):
    registry, store, _ = await _funcs(tmp_path)
    text = store.create("这是文本不是音频")
    r = await registry.invoke("transcribe", {"asset_id": text.id})
    assert not r.ok and "没有文件路径" in r.error


async def test_非法输出格式被拦(tmp_path: Path):
    registry, store, p = await _funcs(tmp_path)
    audio = store.create_blob(b"a", ".mp3", type_=AssetType.AUDIO)
    r = await registry.invoke("transcribe", {"asset_id": audio.id, "format": "docx"})
    assert not r.ok and "不支持的输出格式" in r.error
    assert p.transcribed == []


async def test_转写失败不留垃圾资产(tmp_path: Path):
    registry, store, _ = await _funcs(tmp_path, FakeAudioProvider(fail="transcribe"))
    audio = store.create_blob(b"a", ".mp3", type_=AssetType.AUDIO)
    before = len(store)
    r = await registry.invoke("transcribe", {"asset_id": audio.id})
    assert not r.ok and len(store) == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
