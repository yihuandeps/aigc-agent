"""list_voices 只列当前 TTS provider 的音色（2026-09-29 优化审查 1.5）。

之前 render_audio 列的是 OpenAI 那套音色、默认 nova，而 TTS 实际走 MiniMax，tts 校验音色
查的也是 speech_voices。tts 的参数说明写「见 list_voices」，9-23 模型照着传了 nova，被拒。
"""

from __future__ import annotations

from pathlib import Path

from aigc_agent.domain.assets.store import AssetStore
from aigc_agent.domain.functions.audio import AudioFunctions
from aigc_agent.domain.generators.catalog import AudioModel, MediaCatalog, Voice
from aigc_agent.harness.events.bus import EventBus
from aigc_agent.harness.model.audio import AudioGateway, FakeAudioProvider

CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "media_models.yaml"
ANCHOR = "Chinese (Mandarin)_News_Anchor"


def _catalog(tts_provider: str = "minimax") -> MediaCatalog:
    """主 provider 是 apimart（OpenAI 那套音色），TTS 单独走 minimax —— 和线上配置同形。"""
    return MediaCatalog(
        provider="apimart",
        tts_provider=tts_provider,
        audio=[
            AudioModel(id="gpt-4o-mini-tts", kind="tts", strengths="OpenAI 口播"),
            AudioModel(id="speech-2.8-hd", kind="tts", provider="minimax", strengths="中文原生"),
            AudioModel(id="whisper-1", kind="asr", strengths="出字幕"),
        ],
        voices=[
            Voice(name="nova", lang="multi", note="女声，明亮自然"),
            Voice(name="onyx", lang="multi", note="男声，低沉厚重"),
        ],
        default_voice="nova",
        provider_voices={
            "minimax": [
                Voice(name=ANCHOR, note="新闻女声，资讯播报"),
                Voice(name="Chinese (Mandarin)_Radio_Host", note="电台男主播，叙事"),
            ]
        },
        provider_default_voice={"minimax": ANCHOR},
    )


def _fns(tmp_path: Path, catalog: MediaCatalog) -> tuple[AudioFunctions, FakeAudioProvider]:
    p = FakeAudioProvider()
    providers = {catalog.provider: p, catalog.speech_provider: p}
    return AudioFunctions(AudioGateway(providers, EventBus()), catalog, AssetStore(tmp_path)), p


def _listed_voices(text: str) -> list[str]:
    """从 list_voices 的「## 音色」那一节里抠出音色 id。"""
    block = text.split("## 音色", 1)[1].split("\n## ", 1)[0]
    names = []
    for line in block.splitlines():
        if line.startswith("- "):
            names.append(line[2:].split("：", 1)[0].split("（", 1)[0].strip())
    return names


async def test_TTS走MiniMax时list_voices只列MiniMax音色并标出provider(tmp_path: Path):
    fns, _ = _fns(tmp_path, _catalog())
    r = await fns.invoke("list_voices", {})
    assert r.ok, r.error
    assert _listed_voices(r.content) == [ANCHOR, "Chinese (Mandarin)_Radio_Host"]
    assert "nova" not in r.content and "onyx" not in r.content
    assert f"默认：{ANCHOR}" in r.content
    assert "minimax" in r.content.split("## 音色", 1)[1].splitlines()[0], "音色那节要标出是哪家的"
    # 别家的 TTS 模型也不该出现：选了就是 "not have model"
    assert "speech-2.8-hd" in r.content and "gpt-4o-mini-tts" not in r.content


async def test_list_voices列出来的音色tts都认(tmp_path: Path):
    """目录和校验必须是同一张表：照着 list_voices 填，tts 不能拒。"""
    cat = _catalog()
    fns, p = _fns(tmp_path, cat)
    listed = _listed_voices((await fns.invoke("list_voices", {})).content)
    assert listed
    for name in listed:
        r = await fns.invoke("tts", {"text": "你好", "voice": name})
        assert r.ok, (name, r.error)
    assert [s["voice"] for s in p.spoken] == listed


async def test_传了别家的音色_报错说清TTS走的是哪家(tmp_path: Path):
    fns, p = _fns(tmp_path, _catalog())
    r = await fns.invoke("tts", {"text": "你好", "voice": "nova"})
    assert not r.ok and "未知音色" in r.error
    assert "minimax" in r.error and ANCHOR in r.error
    assert p.spoken == []


async def test_TTS跟主provider走时照样列OpenAI那套(tmp_path: Path):
    cat = _catalog(tts_provider="")
    fns, _ = _fns(tmp_path, cat)
    text = (await fns.invoke("list_voices", {})).content
    assert _listed_voices(text) == ["nova", "onyx"]
    assert "默认：nova" in text and "apimart" in text
    assert "speech-2.8-hd" not in text


async def test_健康检查数的是当前TTS那家的音色(tmp_path: Path):
    fns, _ = _fns(tmp_path, _catalog())
    detail = (await fns.health()).detail
    assert "2 个音色" in detail and "minimax" in detail


def test_真实配置_list_voices列的就是tts认的那套():
    c = MediaCatalog.load(CATALOG_PATH)
    text = c.render_audio()
    listed = _listed_voices(text)
    assert listed == [v.name for v in c.speech_voices]
    assert all(c.has_voice(n) for n in listed)
    assert f"默认：{c.speech_default_voice}" in text
    assert c.has_voice(c.speech_default_voice)
