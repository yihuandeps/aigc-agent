"""参考图包（drama_render_assets 的产物）的读取与「内容身份」。

包 = {名字: {asset, url, kind?, portrait?}}。渲视频时片段带着「用的是哪个包」的标签，下次
只复用同一套参考图渲出的段 —— 换脸、补服装之后旧片段不该再复用。

但「同一套参考图」不能按包 id 判：链接过期后 `_rehost_pack` / `drama_refresh_refs` 会新建一个
包资产（图没变，只是换了链接）。2026-09-24 审查发现按 id 比的后果是托管一次就整集重渲、重付，
按集流水还会把它当成「输入变了」清掉失败标记、未经人确认自动重渲。所以这里按**每个键用的是
哪份图片资产**算签名：托管 / 刷新不换资产，签名不变；重生成了脸或服装（新资产）签名才变。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def load_pack(content: str) -> dict[str, dict[str, str]]:
    """参考图包 = {名字: {asset, url, kind?}}。不是这个形状就返回空。"""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {
        k: v for k, v in data.items() if isinstance(v, dict) and isinstance(v.get("url"), str)
    }
    return out if out else {}


def pack_identity(images: dict[str, dict[str, str]]) -> str:
    """包的内容签名：键 → 图片资产 id（没记资产的老包退回 url）。空包返回空串。"""
    if not images:
        return ""
    items = sorted((k, str(v.get("asset") or v.get("url") or "")) for k, v in images.items())
    raw = "\n".join(f"{k}={v}" for k, v in items)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- 包属于哪套资产库（2026-09-25）
# 资产迁移没做之前，旧剧的参考图包没有项目归属、在新剧里也看得见。两部西游题材的剧角色同名
# （唐僧、悟空…），按名字匹配能对上 —— 会拿旧剧的脸渲新剧。所以挑包、用包都要核对血缘。

_LIB_OF: dict[str, str] = {}  # 包 id → 资产库 id（血缘落库后不变，算一次）


def pack_library(store: Any, pack_id: str, depth: int = 12) -> str:
    """参考图包属于哪套资产库：沿 parents 往上找第一个资产库（drama_assets 的产物）。

    重新托管 / 刷新 / 换脸都会新建包，血缘是「新包 → 旧包 → … → 资产库」，要一路往上追。
    找不到（没记血缘的老包、测试数据）返回空串 = 不知道。
    """
    if pack_id in _LIB_OF:
        return _LIB_OF[pack_id]
    seen: set[str] = set()
    frontier = [pack_id]
    found = ""
    for _ in range(depth):
        nxt: list[str] = []
        for aid in frontier:
            if not aid or aid in seen:
                continue
            seen.add(aid)
            try:
                a = store.get(aid)
            except KeyError:
                continue
            if aid != pack_id and a.creator == "tool:drama_assets":
                found = a.id
                break
            nxt += list(a.parent_ids or [])
        if found or not nxt:
            break
        frontier = nxt
    if found:  # 查不到的不缓存：父资产可能是别的进程稍后才写进来的
        _LIB_OF[pack_id] = found
    return found


def library_of_shots(store: Any, shots_id: str) -> str:
    """视频提示词用的是哪套资产库（drama_shots 落库时 parents = [分镜, 资产库]）。"""
    try:
        parents = store.get(shots_id).parent_ids[1:]
    except KeyError:
        return ""
    for aid in parents:
        try:
            if store.get(aid).creator == "tool:drama_assets":
                return aid
        except KeyError:
            continue
    return ""


def library_chain(store: Any, lib_id: str, depth: int = 30) -> list[str]:
    """资产库的增量链：[这一版, 它增量补充的上一版, …]（2026-09-26 用户定的：资产库定稿后冻结，
    只增量补新集）。增量版原样保留上一版的全部条目，所以链上各版对已有条目来说是同一套库；
    整份重做（rebuild）的不算 —— 条目可能改了，链到那里就断。"""
    out: list[str] = []
    cur = lib_id
    for _ in range(depth):
        if not cur or cur in out:
            break
        out.append(cur)
        try:
            gp = store.get(cur).gen_params or {}
        except KeyError:
            break
        if gp.get("mode") != "incremental":
            break
        cur = str(gp.get("base") or "")
    return out


def same_library(store: Any, a: str, b: str) -> bool:
    """两个资产库 id 是不是同一套库（相同，或在同一条增量链上）。"""
    if not a or not b:
        return False
    return a == b or a in library_chain(store, b) or b in library_chain(store, a)


def pick_pack(store: Any, packs: list[Any], library: str) -> str:
    """一堆参考图包（新到旧）里挑属于 library 的最新一个；没有就挑血缘不明的最新一个（老数据 /
    测试）；属于别的资产库的一律不要。"""
    unknown = ""
    for a in packs:
        lib = pack_library(store, a.id)
        if library and lib == library:
            return a.id
        if not lib and not unknown:
            unknown = a.id
    return unknown
