"""MiniMax TTS 接入验收。

接它的原因：APIMart 那 364 个模型里没有一个中文原生 TTS，只有 OpenAI
的英文音色库，中文口播听着假是先天的，调参数改文案都只能缓解。

这里盯的是三个照搬 OpenAI 形状就会踩的坑：
  · GroupId 走 query 参数，不是请求头
  · 音频是 **hex 字符串**，不是 base64（解错了会得到能播放的白噪音文件，
    极难发现 —— 文件大小正常、时长正常、就是全是沙沙声）
  · 失败时 HTTP 也是 200，错误藏在 base_resp 里
"""

from __future__ import annotations

from typing import Any

import pytest

from aigc_agent.harness.model.audio import MiniMaxSpeechProvider

REAL_AUDIO = b"\xff\xfb\x90\x00fake-mp3-bytes"


class _Resp:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status
        self.content = b"{}"

    def json(self) -> Any:
        return self._payload


class _Client:
    """记下请求，返回预设响应。"""

    def __init__(self, resp: _Resp) -> None:
        self.resp = resp
        self.calls: list[dict[str, Any]] = []

    async def post(self, path: str, **kw: Any) -> _Resp:
        self.calls.append({"path": path, **kw})
        return self.resp

    async def aclose(self) -> None: ...


def _provider(resp: _Resp) -> tuple[MiniMaxSpeechProvider, _Client]:
    p = MiniMaxSpeechProvider("https://api.minimaxi.com", "k", "g")
    c = _Client(resp)
    p._client = c  # type: ignore[assignment]
    return p, c


OK_PAYLOAD = {
    "data": {"audio": REAL_AUDIO.hex(), "status": 2},
    "base_resp": {"status_code": 0, "status_msg": "success"},
}


# ---------- 成功路径 ----------


async def test_音频按hex解码而不是base64():
    """解成 base64 会得到一堆噪音 —— 文件能生成能播放，全是沙沙声。"""
    p, _ = _provider(_Resp(OK_PAYLOAD))
    r = await p.speak("speech-2.5-hd-preview", "你好")
    assert r.ok
    assert r.audio == REAL_AUDIO, "音频没按 hex 解出来"


async def test_groupid走query参数():
    p, c = _provider(_Resp(OK_PAYLOAD))
    await p.speak("speech-2.5-hd-preview", "你好")
    assert c.calls[0]["params"] == {"GroupId": "g"}, "GroupId 必须在 query 里，不是请求头"


async def test_参数是嵌套的不是平铺():
    p, c = _provider(_Resp(OK_PAYLOAD))
    await p.speak("speech-2.5-hd-preview", "你好", voice="presenter_female", speed=1.1)
    body = c.calls[0]["json"]
    assert body["voice_setting"]["voice_id"] == "presenter_female"
    assert body["voice_setting"]["speed"] == 1.1
    assert "voice" not in body, "平铺的 voice 字段会被忽略"


async def test_voice_setting是必填的():
    """不传会被拒（2013 invalid params, empty field）。

    这点和 OpenAI 相反 —— 那边省略 voice 是合法的，照搬会 100% 失败。
    """
    p, c = _provider(_Resp(OK_PAYLOAD))
    await p.speak("speech-2.5-hd-preview", "你好")
    vs = c.calls[0]["json"].get("voice_setting")
    assert vs and vs.get("voice_id"), "没指定音色时也必须带一个默认 voice_id"


async def test_没有group_id时不传该参数():
    """新的 sk-api- key 自带归属，传 GroupId 反而 1004 token not match group。"""
    p = MiniMaxSpeechProvider("https://api.minimaxi.com", "k", "")
    c = _Client(_Resp(OK_PAYLOAD))
    p._client = c  # type: ignore[assignment]
    await p.speak("speech-2.5-hd-preview", "你好")
    assert c.calls[0]["params"] is None, "没有 group_id 就不该带这个 query 参数"


async def test_只有api_key也能用():
    """GroupId 不再是必需项。"""
    p = MiniMaxSpeechProvider("https://api.minimaxi.com", "k", "")
    c = _Client(_Resp(OK_PAYLOAD))
    p._client = c  # type: ignore[assignment]
    r = await p.speak("speech-2.5-hd-preview", "你好")
    assert r.ok


async def test_情绪只在显式传入时才带():
    """emotion 是枚举不是自由文本，配方里那种整句描述传进去会被拒。"""
    p, c = _provider(_Resp(OK_PAYLOAD))
    await p.speak("speech-2.5-hd-preview", "你好", instruct="冷静清晰的科技口播语气")
    body = c.calls[0]["json"]
    assert "emotion" not in body.get("voice_setting", {})
    assert "instruct" not in body and "instructions" not in body


# ---------- 失败路径 ----------


async def test_业务错误藏在base_resp里():
    """MiniMax 失败时 HTTP 也是 200，不查 base_resp 会把错误响应当音频存下来。"""
    p, _ = _provider(
        _Resp({"base_resp": {"status_code": 1004, "status_msg": "invalid api key"}})
    )
    r = await p.speak("speech-2.5-hd-preview", "你好")
    assert not r.ok
    assert "1004" in (r.error or "") and "invalid api key" in (r.error or "")


async def test_没有音频字段算失败():
    p, _ = _provider(_Resp({"data": {"status": 2}, "base_resp": {"status_code": 0}}))
    r = await p.speak("speech-2.5-hd-preview", "你好")
    assert not r.ok


async def test_非法hex算失败而不是存成坏文件():
    p, _ = _provider(
        _Resp({"data": {"audio": "这不是hex"}, "base_resp": {"status_code": 0}})
    )
    r = await p.speak("speech-2.5-hd-preview", "你好")
    assert not r.ok and "hex" in (r.error or "")


async def test_http错误被归一():
    p, _ = _provider(_Resp({"message": "boom"}, status=500))
    r = await p.speak("speech-2.5-hd-preview", "你好")
    assert not r.ok and "500" in (r.error or "")


async def test_没配key时给出怎么配的提示():
    """静默退回到英文音色比报错更糟 —— 用户会以为已经换成中文了。"""
    p = MiniMaxSpeechProvider("https://api.minimaxi.com", "", "")
    r = await p.speak("speech-2.5-hd-preview", "你好")
    assert not r.ok
    assert "MINIMAX_API_KEY" in (r.error or "")


async def test_余额不足要说清楚():
    """1008 是最容易撞上的一个 —— key 有效、格式正确，就是没钱。

    报错必须让人一眼看出是充值问题，而不是去查代码。
    """
    p, _ = _provider(
        _Resp({"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}})
    )
    r = await p.speak("speech-2.5-hd-preview", "你好")
    assert not r.ok
    assert "1008" in (r.error or "") and "insufficient balance" in (r.error or "")


async def test_转写不走这条通路():
    """MiniMax 这边只接了 TTS，转写留在 whisper。"""
    p = MiniMaxSpeechProvider("https://api.minimaxi.com", "k", "g")
    with pytest.raises(NotImplementedError, match="whisper"):
        await p.transcribe("x", b"", "a.mp3")


# ---------- 配置 ----------


def test_catalog_默认tts跟随主provider():
    from aigc_agent.domain.generators.catalog import MediaCatalog

    assert MediaCatalog(provider="apimart").speech_provider == "apimart"
    assert MediaCatalog(provider="apimart", tts_provider="minimax").speech_provider == "minimax"


# ---------- 国内站 vs 国际站 ----------


def test_域名指向国内站():
    """platform.minimax.cn 和 platform.minimaxi.com 是**两条产品线**，
    账号和余额都不通。和 Kimi Code vs platform.moonshot.cn 同一类坑。

    这条钉住配置，防止后续迭代里被"顺手统一成 minimaxi.com"。
    """
    import yaml

    from aigc_agent.app import PROJECT_ROOT

    raw = yaml.safe_load((PROJECT_ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    mm = (raw["image"]["providers"] or {})["minimax"]
    assert "minimax.cn" in mm["base_url"], "指到国际站了，国内站的 key 在那边没余额"


def test_配置里没有编造的模型名():
    """国内站**没有** speech-2.5-*，那是早先猜的。

    真实的是 speech-2.8/2.6/02/01 系列（platform.minimax.cn 文档 2026-09-12 核对）。
    余额检查排在模型校验之前，没钱时任何名字都返回 1008，所以这层只能靠文档守。
    """
    from aigc_agent.app import PROJECT_ROOT
    from aigc_agent.domain.generators.catalog import MediaCatalog

    c = MediaCatalog.load(PROJECT_ROOT / "config" / "media_models.yaml")
    real = {
        "speech-2.8-hd", "speech-2.8-turbo", "speech-2.6-hd",
        "speech-2.6-turbo", "speech-02-hd", "speech-02-turbo",
        "speech-01-hd", "speech-01-turbo",
    }
    for m in c.audio_models("tts"):
        if m.id.startswith("speech-"):
            assert m.id in real, f"{m.id} 不在国内站的模型列表里"


def test_默认音色是实拉验证过的():
    """303 个系统音色是从 /v1/get_voice 拉的，不是猜的。"""
    from aigc_agent.harness.model.audio import DEFAULT_MINIMAX_VOICE

    assert DEFAULT_MINIMAX_VOICE == "Chinese (Mandarin)_News_Anchor"


async def test_粤语要带language_boost():
    """不带的话粤语音色会按普通话念。"""
    p, c = _provider(_Resp(OK_PAYLOAD))
    await p.speak(
        "speech-2.8-hd", "你好", voice="Cantonese_ProfessionalHost（F)",
        language_boost="Chinese,Yue",
    )
    assert c.calls[0]["json"]["language_boost"] == "Chinese,Yue"
