"""M3 轮内压缩 —— 把"引用不放内容"的原则落到消息历史上。

滑窗按轮驱逐解决不了**一轮之内**的膨胀：模型在一轮里连调几十次工具，每次
save_draft 的参数就是一整集剧本、每次 read_asset 的返回就是一整篇正文。
实测（2026-09-17）一轮 12 次迭代把上下文从 19 万推到 24 万 token，最后撞上
模型上限报 401，整轮作废；而这些正文**早就落在资产库里**，留在上下文里的
只是重复品。

做法是折叠而不是删除：
  · 工具调用的参数：长字符串换成「<N 字已折叠>」，短字段（kind / summary /
    episode / parent_id …）原样保留 —— 模型仍能看出"那次调用干了什么"
  · 工具结果：留开头一小段 + 产物资产 id，正文靠 read_asset 按需取回
  · 助手正文：超长的截断，留开头
  · 用户输入永远不折叠

折叠只作用于**装配时的视图**，Turn 里的原始消息一字不动 —— 快照、回放、
记忆提取看到的还是全文。折叠是确定性的：同一段消息折出来永远一样，这样
历史轮的前缀稳定，prompt cache 不会被折叠本身打穿。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_ASSET_RE = re.compile(r"\bas_[0-9a-f]{10}\b")
_FAIL_PREFIX = "[工具执行失败]"


@dataclass
class CompactionPolicy:
    """阈值都是估算 token 或字符数；默认值按 256K 窗口、软目标 80K 定。"""

    # 当前轮里最近几次迭代原样保留（模型正在处理的东西不能折）
    keep_recent_iterations: int = 2
    # 当前轮估算超过这么多 token 才开始折旧迭代（小轮不折，省得前缀乱变）
    current_turn_tokens: int = 40_000
    # 历史轮估算超过这么多 token 才折叠（小轮原样保留，保住缓存前缀）
    history_turn_tokens: int = 6_000
    # 折叠阈值（字符）
    fold_args_over: int = 600
    fold_result_over: int = 800
    fold_assistant_over: int = 3_000
    # 折叠后保留的开头长度（字符）
    result_head: int = 200
    assistant_head: int = 1_000
    # 参数里的字符串字段超过这个长度就换成占位
    arg_scalar_max: int = 80


# 撞上模型上限后的应急档：能折的都折，只留最后一次迭代
HARD = CompactionPolicy(
    keep_recent_iterations=1,
    current_turn_tokens=0,
    history_turn_tokens=0,
    fold_args_over=200,
    fold_result_over=200,
    fold_assistant_over=800,
    result_head=120,
    assistant_head=300,
    arg_scalar_max=60,
)


def fold_arguments(raw: str, scalar_max: int = 80) -> str:
    """把工具调用参数里的长字段换成占位，短字段原样保留。永远返回合法 JSON。"""
    try:
        data = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"_folded": f"参数 {len(raw or '')} 字已折叠"}, ensure_ascii=False)
    if not isinstance(data, dict):
        return json.dumps({"_folded": f"参数 {len(raw)} 字已折叠"}, ensure_ascii=False)
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str):
            out[k] = v if len(v) <= scalar_max else f"<{len(v)} 字已折叠>"
        elif isinstance(v, (list, dict)):
            s = json.dumps(v, ensure_ascii=False)
            out[k] = v if len(s) <= scalar_max * 2 else f"<{len(s)} 字已折叠>"
        else:
            out[k] = v
    return json.dumps(out, ensure_ascii=False)


# 本文件产出的全部折叠标记。数字必须是真数字 —— 源码里的 f-string 模板不会误中
_FOLD_MARK_RE = re.compile(
    # 整个值就是「<数字字…>」：模型会自编变体，如 9-21 drama_assets 收到的
    # 「<29000字剧本全文>」。只在整串是它时才算，免得误伤正文里的方括号描述
    r"^\s*<\d+\s*字[^<>\n]{0,20}>\s*$"
    # 本文件产出的原样标记 —— **夹在正文中间也算**（2026-09-22 补）。
    # 模型拼接长正文时会把半截真内容和占位符缝在一起，那种最阴：
    # 整串检查放行，结果存进去的是缺了一大块的稿子，下游解析还不一定报错
    r"|<\d+\s*字已折叠>"
    r"|…（(?:工具结果 |助手正文 )?\d+ 字已折叠"  # 结果/正文/失败信息截断后的尾巴
    r"|参数 \d+ 字已折叠"  # 整个参数 JSON 坏掉时的兜底
)

# 整段内容就是一个占位符 —— 这永远不可能是真内容，零误判，可以直接拒绝落库
_ONLY_FOLD_RE = re.compile(r"^\s*<\d+\s*字[^<>\n]{0,20}>\s*$")


def is_only_fold_mark(text: Any) -> bool:
    """整段内容只是一个折叠占位符吗。

    2026-09-21 的事故：模型照着自己被折叠的历史写 save_draft，12 集分镜存成了
    12 个 11 字的占位符，工具还报成功；之后 drama_shots 读这些资产才炸出来。
    参数层已经拦了一道，这个判定给**落库**再兜一道 —— 任何路径写进来都挡得住。
    """
    return isinstance(text, str) and bool(_ONLY_FOLD_RE.match(text))


def find_fold_marks(args: Any, path: str = "") -> list[str]:
    """找出工具参数里原样抄回来的折叠标记，返回字段路径。

    折叠只作用于给模型看的视图，但模型会照着历史里的样子写参数：
    2026-09-21 主模型连发 12 次 save_draft，content 全是「<7xxx 字已折叠>」，
    12 集分镜存成了 12 个 11 字的占位符，工具照单全收还报成功。
    """
    if isinstance(args, str):
        return [path or "(参数)"] if _FOLD_MARK_RE.search(args) else []
    if isinstance(args, dict):
        hits: list[str] = []
        for k, v in args.items():
            hits += find_fold_marks(v, f"{path}.{k}" if path else str(k))
        return hits
    if isinstance(args, list):
        return [h for i, v in enumerate(args) for h in find_fold_marks(v, f"{path}[{i}]")]
    return []


def _fold_tool_result(content: str, policy: CompactionPolicy) -> str:
    head = content[: policy.result_head].rstrip()
    ids = list(dict.fromkeys(_ASSET_RE.findall(content)))[:6]
    tail = f"…（工具结果 {len(content)} 字已折叠"
    if ids:
        tail += f"；涉及资产 {', '.join(ids)}，需要全文用 read_asset 取"
    return head + tail + "）"


def fold_message(m: dict[str, Any], policy: CompactionPolicy) -> tuple[dict[str, Any], int]:
    """折叠一条消息。返回 (新消息, 改动了几处)。不改原对象。"""
    role = m.get("role")
    if role == "assistant":
        changed = 0
        new = dict(m)
        calls = m.get("tool_calls")
        if calls:
            folded_calls = []
            for tc in calls:
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments") or ""
                if len(args) > policy.fold_args_over:
                    fn["arguments"] = fold_arguments(args, policy.arg_scalar_max)
                    changed += 1
                folded_calls.append({**tc, "function": fn})
            new["tool_calls"] = folded_calls
        content = m.get("content")
        if isinstance(content, str) and len(content) > policy.fold_assistant_over:
            new["content"] = (
                content[: policy.assistant_head].rstrip()
                + f"…（助手正文 {len(content)} 字已折叠）"
            )
            changed += 1
        return new, changed
    if role == "tool":
        content = m.get("content")
        if isinstance(content, str) and len(content) > policy.fold_result_over:
            # 失败信息比成功结果值钱，多留一点
            if content.startswith(_FAIL_PREFIX):
                keep = max(policy.result_head, 300)
                if len(content) <= keep:
                    return m, 0
                return {**m, "content": content[:keep] + f"…（{len(content)} 字已折叠）"}, 1
            return {**m, "content": _fold_tool_result(content, policy)}, 1
        return m, 0
    return m, 0


def fold_turn_messages(
    messages: list[dict[str, Any]],
    *,
    keep_tail_iterations: int,
    policy: CompactionPolicy,
) -> tuple[list[dict[str, Any]], int]:
    """折叠一轮的消息。最近 keep_tail_iterations 次迭代（以助手消息为界）原样保留，
    keep_tail_iterations=0 表示整轮都折（历史轮）。用户消息永远不动。
    返回 (新消息列表, 折叠处数)。"""
    if not messages:
        return [], 0
    assistant_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if keep_tail_iterations > 0 and assistant_idx:
        k = min(keep_tail_iterations, len(assistant_idx))
        boundary = assistant_idx[-k]
    else:
        boundary = len(messages)

    out: list[dict[str, Any]] = []
    n = 0
    for i, m in enumerate(messages):
        if i >= boundary or m.get("role") == "user":
            out.append(m)
            continue
        fm, changed = fold_message(m, policy)
        n += changed
        out.append(fm)
    return out, n
