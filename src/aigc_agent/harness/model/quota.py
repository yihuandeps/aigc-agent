"""余额 / 额度用完的识别（2026-09-29 审查 1.1）。

APIMart 余额不足时先回 500（insufficient quota: balance=85356, required=102911），再回 402；
Kimi 订阅额度窗口用完回 403 / 429（usage limit / quota）。这些重试没用、换模型也没用 ——
同一个账号的钱或窗口没了。之前文本侧把 500 当普通错误重试，9-27 两次 drama_shots 共重试
12 次、原始的 402 直接甩给主模型；9-23 连着 20 次 402，主模型当成模型的问题换了两次模型。
媒体侧一批 8 段并发，全部 402。
"""

from __future__ import annotations

import re

_QUOTA_TEXT = re.compile(
    r"usage limit|quota|insufficient|额度|余额|balance|欠费|arrears|payment required", re.I
)
# 500 只认明确说钱 / 额度的：「insufficient resources」这类服务端过载不算
_QUOTA_500 = re.compile(r"quota|余额|额度|balance|欠费|arrears", re.I)


def is_quota_error(status: int | None, text: str) -> bool:
    """这个错误是不是余额 / 额度用完了（重试、换模型都没用）。"""
    text = text or ""
    if status == 402:
        return True
    if status in (403, 429):
        return bool(_QUOTA_TEXT.search(text))
    if status == 500:
        return bool(_QUOTA_500.search(text))
    return False


def quota_advice(provider: str, status: int | None) -> str:
    """给主模型（再转给用户）的一句话：怎么办、别怎么办。"""
    if status in (403, 429):
        what = f"「{provider}」的额度窗口用完了"
        todo = "告诉用户等额度窗口恢复后再继续"
    else:
        what = f"「{provider}」的余额不足"
        todo = "告诉用户去充值，充好后再继续"
    return (
        f"{what}。重试、换模型、拆小批、换工具绕开都没用（同一个账号的钱或额度没了）：{todo}；"
        "已经做好的部分都留着，恢复后原样再跑一次就行"
    )


class QuotaExhaustedError(Exception):
    """某个 provider 的余额 / 额度用完了。消息本身写清楚该怎么办 —— 工具层照例只把
    「异常类名: 消息」报给主模型，这段话会原样到它眼前。"""

    def __init__(self, provider: str, status: int | None, detail: str, role: str = "") -> None:
        self.provider, self.status, self.detail, self.role = provider, status, detail, role
        who = f"角色「{role}」用的 " if role else ""
        super().__init__(
            f"{who}provider「{provider}」调用被拒（HTTP {status}，服务端原话：{detail[:200]}）。"
            + quota_advice(provider, status)
        )
