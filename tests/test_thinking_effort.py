"""thinking_effort 取值校验（2026-09-29 审查）。

抖音线简报角色 short_video_planner 配了 `thinking_effort: medium`，而合法值只有
low / high / max —— 服务端 6 次调用 6 次 400，这条线从没跑通过，配置加载时又不校验。
现在加载配置时就拦住，启动即报错，不再等到调用时才失败。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aigc_agent.harness.model.config import THINKING_EFFORTS, ModelsConfig, TextConfig

ROOT = Path(__file__).resolve().parents[1]


def test_仓库里的models_yaml每个角色的取值都合法():
    cfg = ModelsConfig.load(ROOT / "config" / "models.yaml")
    bad = {
        role: p.get("thinking_effort")
        for role, p in cfg.text.role_params.items()
        if p.get("thinking_effort") and p["thinking_effort"] not in THINKING_EFFORTS
    }
    assert not bad, bad
    assert cfg.text.role_params["short_video_planner"]["thinking_effort"] == "low"


def test_写错取值_加载配置时就报错并点名角色():
    with pytest.raises(ValueError, match="short_video_planner.thinking_effort"):
        TextConfig(
            providers={}, roles={},
            role_params={"short_video_planner": {"thinking_effort": "medium"}},
        )


def test_没写或合法取值都放行():
    cfg = TextConfig(
        providers={}, roles={},
        role_params={"a": {}, "b": {"thinking_effort": "high"}, "c": {"temperature": 0.1}},
    )
    assert cfg.role_params["b"]["thinking_effort"] == "high"
