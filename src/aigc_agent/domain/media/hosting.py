"""本地素材托管（2026-09-19）—— 本地文件 → 公网链接，生成接口才能拿它当参考。

用户的素材一定在本地；生成接口只收公网链接（seedance 的报错原话：Only http/https URL
or asset:// private asset URL），本地路径、base64 都不行；生成结果自带的链接约 24 小时失效。
所以需要一层"把文件放到用户自己的存储上"的能力，配置在 config/hosting.yaml：

  imghost   传到图床拿链接（2026-09-22 加）—— **没有对象存储时用这个**，
            preset 选站：imgbb / smms 免费注册拿 key，catbox / litterbox 免注册
  command   跑用户自己的上传命令，取 stdout 最后一行 http 链接
  s3        S3 兼容对象存储（OSS / COS / R2 / MinIO / AWS），SigV4 签名 PUT，不依赖 boto3
  http_put  自建 nginx / WebDAV：PUT 到 base_url/文件名

`Hosting.ensure_asset` 是统一入口：资产已有可用公网链接就直接用，否则拿本地副本上传，
把链接写回资产（gen_params["hosted"] 记录来源，托管链接默认不过期）。
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import mimetypes
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx2 as httpx
import yaml

from ...harness.model.media import default_proxy

HttpPut = Callable[[str, dict[str, str], bytes], Awaitable[tuple[int, str]]]
# (url, headers, 表单字段, 文件字段名, 文件内容, 文件名) -> (状态码, 响应文本)
HttpPost = Callable[
    [str, dict[str, str], dict[str, str], str, bytes, str], Awaitable[tuple[int, str]]
]


async def _default_put(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
    async with httpx.AsyncClient(timeout=300.0, proxy=default_proxy()) as c:
        resp = await c.put(url, headers=headers, content=body)
    return resp.status_code, resp.text[:300]


async def _default_post(
    url: str,
    headers: dict[str, str],
    form: dict[str, str],
    field: str,
    body: bytes,
    filename: str,
) -> tuple[int, str]:
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    # 跟随重定向：图床换域名是常事（sm.ms 的 sm.ms → smms.app 就是 308），
    # 不跟的话拿到的是一页 Cloudflare 的重定向 HTML，报错还看不出原因
    async with httpx.AsyncClient(
        timeout=300.0, proxy=default_proxy(), follow_redirects=True
    ) as c:
        resp = await c.post(
            url, headers=headers, data=form, files={field: (filename, body, mime)}
        )
    return resp.status_code, resp.text[:4000]


@dataclass
class HostingConfig:
    type: str = "none"  # none | imghost | command | s3 | http_put
    command: str = ""
    s3: dict[str, Any] = field(default_factory=dict)
    http_put: dict[str, Any] = field(default_factory=dict)
    imghost: dict[str, Any] = field(default_factory=dict)
    ttl_hours: float = 0.0

    @classmethod
    def load(cls, path: str | Path) -> HostingConfig:
        p = Path(path)
        if not p.exists():
            return cls()
        raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        imghost = dict(raw.get("imghost") or {})
        ttl = float(raw.get("ttl_hours") or 0)
        # 临时图床（litterbox 72h）到点就删，ttl 留 0 会被当成"永不过期"，
        # 渲染时拿着死链去生图 —— 没填就按 preset 的默认值，留一点余量。
        if not ttl and str(raw.get("type") or "").strip().lower() == "imghost":
            preset = _IMGHOST_PRESETS.get(str(imghost.get("preset") or "").strip().lower(), {})
            ttl = float(preset.get("default_ttl") or 0)
        return cls(
            type=str(raw.get("type") or "none").strip().lower(),
            command=str(raw.get("command") or ""),
            s3=dict(raw.get("s3") or {}),
            http_put=dict(raw.get("http_put") or {}),
            imghost=imghost,
            ttl_hours=ttl,
        )


# ---------------------------------------------------------------- 上传器


class CommandUploader:
    """跑用户自己的命令。命令要把公网链接打印到 stdout（取最后一行以 http 开头的）。"""

    def __init__(self, template: str) -> None:
        self.template = template

    async def upload(self, path: Path, name: str) -> tuple[str, str]:
        if not self.template.strip():
            return "", "hosting.yaml 的 command 是空的"
        cmd = self.template.format(
            file=str(path), name=name, stem=Path(name).stem, ext=Path(name).suffix
        )
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=300)
        except TimeoutError:
            return "", "上传命令超过 300s 没结束"
        except Exception as e:  # noqa: BLE001
            return "", f"上传命令跑不起来：{type(e).__name__}: {e}"
        text = out.decode("utf-8", errors="replace")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        urls = [ln for ln in lines if ln.startswith(("http://", "https://"))]
        if proc.returncode != 0 or not urls:
            tail = err.decode("utf-8", errors="replace").strip()[-300:]
            return "", f"上传命令失败（退出码 {proc.returncode}，stdout 里没有 http 链接）{tail}"
        return urls[-1], ""


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _uri_encode(path: str) -> str:
    return "/".join(quote(seg, safe="-_.~") for seg in path.split("/"))


class S3Uploader:
    """S3 兼容对象存储的 SigV4 PUT（阿里云 OSS / 腾讯云 COS / Cloudflare R2 / MinIO / AWS）。"""

    def __init__(
        self,
        cfg: dict[str, Any],
        env: dict[str, str] | None = None,
        http_put: HttpPut | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.cfg = cfg
        self.env = env if env is not None else os.environ  # type: ignore[assignment]
        self._put = http_put or _default_put
        self._now = now or time.time

    def _creds(self) -> tuple[str, str, str]:
        ak = str(self.env.get(str(self.cfg.get("access_key_env") or "S3_ACCESS_KEY")) or "")
        sk = str(self.env.get(str(self.cfg.get("secret_key_env") or "S3_SECRET_KEY")) or "")
        missing = []
        if not str(self.cfg.get("endpoint") or ""):
            missing.append("hosting.yaml s3.endpoint")
        if not str(self.cfg.get("bucket") or ""):
            missing.append("hosting.yaml s3.bucket")
        if not ak or not sk:
            missing.append(".env 里的 S3_ACCESS_KEY / S3_SECRET_KEY")
        return ak, sk, "、".join(missing)

    def _target(self, key: str) -> tuple[str, str, str]:
        """返回 (PUT url, host, 公网 url)。"""
        endpoint = str(self.cfg.get("endpoint") or "").rstrip("/")
        bucket = str(self.cfg.get("bucket") or "")
        parts = urlsplit(endpoint)
        scheme = parts.scheme or "https"
        host = parts.netloc or parts.path
        enc_key = _uri_encode(key)
        if self.cfg.get("path_style"):
            put_url = f"{scheme}://{host}/{bucket}/{enc_key}"
            put_host = host
        else:
            put_host = f"{bucket}.{host}"
            put_url = f"{scheme}://{put_host}/{enc_key}"
        public_base = str(self.cfg.get("public_base") or "").rstrip("/")
        public = f"{public_base}/{enc_key}" if public_base else put_url
        return put_url, put_host, public

    async def upload(self, path: Path, name: str) -> tuple[str, str]:
        ak, sk, missing = self._creds()
        if missing:
            return "", f"S3 托管缺配置：{missing}"
        body = await asyncio.to_thread(path.read_bytes)
        prefix = str(self.cfg.get("prefix") or "")
        key = f"{prefix}{name}"
        put_url, host, public = self._target(key)
        region = str(self.cfg.get("region") or "auto")
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        t = time.gmtime(self._now())
        amz_date = time.strftime("%Y%m%dT%H%M%SZ", t)
        date = time.strftime("%Y%m%d", t)
        payload_hash = hashlib.sha256(body).hexdigest()
        headers = {
            "host": host,
            "content-type": content_type,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        }
        acl = str(self.cfg.get("acl") or "")
        if acl:
            headers["x-amz-acl"] = acl
        signed = ";".join(sorted(headers))
        canonical_headers = "".join(f"{k}:{headers[k].strip()}\n" for k in sorted(headers))
        canonical_uri = urlsplit(put_url).path or "/"
        canonical = "\n".join(["PUT", canonical_uri, "", canonical_headers, signed, payload_hash])
        scope = f"{date}/{region}/s3/aws4_request"
        string_to_sign = "\n".join(
            ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()]
        )
        k = _sign(("AWS4" + sk).encode("utf-8"), date)
        k = _sign(k, region)
        k = _sign(k, "s3")
        k = _sign(k, "aws4_request")
        signature = hmac.new(k, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        auth = (
            f"AWS4-HMAC-SHA256 Credential={ak}/{scope}, "
            f"SignedHeaders={signed}, Signature={signature}"
        )
        send = {k2: v for k2, v in headers.items() if k2 != "host"}
        send["Authorization"] = auth
        try:
            status, text = await self._put(put_url, send, body)
        except Exception as e:  # noqa: BLE001
            return "", f"上传失败：{type(e).__name__}: {e}"
        if status >= 300:
            return "", f"上传失败 HTTP {status}：{text[:200]}"
        return public, ""


class HttpPutUploader:
    def __init__(self, cfg: dict[str, Any], http_put: HttpPut | None = None) -> None:
        self.cfg = cfg
        self._put = http_put or _default_put

    async def upload(self, path: Path, name: str) -> tuple[str, str]:
        base = str(self.cfg.get("base_url") or "").rstrip("/")
        public_base = str(self.cfg.get("public_base") or "").rstrip("/")
        if not base or not public_base:
            return "", "http_put 托管缺配置：hosting.yaml http_put.base_url / public_base"
        body = await asyncio.to_thread(path.read_bytes)
        headers = {"Content-Type": mimetypes.guess_type(name)[0] or "application/octet-stream"}
        auth = str(self.cfg.get("auth_header") or "")
        if auth:
            headers["Authorization"] = auth
        enc = quote(name, safe="-_.~")
        try:
            status, text = await self._put(f"{base}/{enc}", headers, body)
        except Exception as e:  # noqa: BLE001
            return "", f"上传失败：{type(e).__name__}: {e}"
        if status >= 300:
            return "", f"上传失败 HTTP {status}：{text[:200]}"
        return f"{public_base}/{enc}", ""


# ---------------------------------------------------------------- 图床

# 2026-09-22 用户：「我没有储存」。没有对象存储就没法把本地图变成参考链接，
# 整条"用自己的图当参考"的路就断了。图床是唯一不用自建存储的办法：
#   imgbb / smms   免费注册拿一个 key，图归你的账号，能自己删
#   catbox/litterbox 连注册都不用，但**谁拿到链接谁就能看**，litterbox 72h 自动删
# ⚠️ 图床都是第三方公网服务。只传当参考用的角色图 / 场景图，别传不该外流的东西。
# 站点参数全做成配置，换站不改代码（preset 选一个，或 custom 自己写）。
_IMGHOST_PRESETS: dict[str, dict[str, Any]] = {
    "imgbb": {
        "endpoint": "https://api.imgbb.com/1/upload",
        "field": "image",
        "url_path": "data.url",
        "token_env": "IMGBB_API_KEY",
        "token_in": "query:key",
        "note": "imgbb.com 免费注册拿 API key；图片长期保存、公开可访问",
    },
    # 2026-09-22 实测：sm.ms 已整体迁到 S.EE（原班团队），老域名的上传接口对所有人只读，
    # 带正确 token 也回 401；新接口是 /api/v1/file/upload，链接落在 i.see.you。
    # 国内可直连（同日实测 sm.ms/s.ee 都通，catbox 系被墙）。
    "see": {
        "endpoint": "https://s.ee/api/v1/file/upload",
        "field": "file",
        "url_path": "data.url",
        "token_env": "SEE_API_TOKEN",
        "token_in": "header:Authorization",
        "note": "S.EE（原 sm.ms 团队）国内可直连；注册拿 API key，图归你账号能自己删",
    },
    "catbox": {
        "endpoint": "https://catbox.moe/user/api.php",
        "field": "fileToUpload",
        "form": {"reqtype": "fileupload"},
        "note": "不用注册；图片永久公开，拿到链接的人都能看",
    },
    "litterbox": {
        "endpoint": "https://litterbox.catbox.moe/resources/internals/api.php",
        "field": "fileToUpload",
        "form": {"reqtype": "fileupload", "time": "72h"},
        "default_ttl": 70.0,
        "note": "不用注册；72 小时后自动删除，够渲染期间用（过期渲染前会自动重传）",
    },
}


def _dig(data: Any, path: str) -> str:
    """按 "data.url" 这种点路径取字符串，取不到返回空。"""
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key)
    return cur if isinstance(cur, str) else ""


def _pick_url(text: str, url_path: str) -> tuple[str, str]:
    """图床返回里挑链接：配了 url_path 走 JSON，否则按纯文本取第一条 http 链接。"""
    if url_path:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return "", "返回不是 JSON"
        url = _dig(data, url_path) or _dig(data, "images")  # sm.ms 重复上传走 images
        if url.startswith(("http://", "https://")):
            return url, ""
        return "", f"{url_path} 里没有链接"
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith(("http://", "https://")):
            return s, ""
    return "", "返回里没有 http 链接"


class ImageHostUploader:
    """传到图床拿公网链接。站点参数走配置，preset 认不得时说清楚有哪些。"""

    def __init__(self, cfg: dict[str, Any], post: HttpPost | None = None) -> None:
        self.cfg = cfg
        self._post = post or _default_post

    def settings(self) -> tuple[dict[str, Any], str]:
        """preset 与用户自定义合并后的站点参数。返回 (参数, 错误)。"""
        name = str(self.cfg.get("preset") or "").strip().lower()
        if name and name != "custom" and name not in _IMGHOST_PRESETS:
            known = " / ".join(_IMGHOST_PRESETS)
            return {}, f"hosting.yaml imghost.preset 认不得「{name}」，可选：{known} / custom"
        s: dict[str, Any] = dict(_IMGHOST_PRESETS.get(name) or {})
        # 选了 preset 就**完全以 preset 为准**，不吃配置文件里那几个自定义字段 ——
        # 它们在模板里带着占位值（token_in: "query:key"），非空，会把 preset 的
        # header:Authorization 盖掉，token 就跑进了 query，服务端只回一句 401 很难查。
        # 要自己调参数就写 preset: custom。
        if name in ("", "custom"):
            for k in ("endpoint", "field", "url_path", "token_env", "token_in"):
                if self.cfg.get(k):
                    s[k] = self.cfg[k]
            s["form"] = {**(s.get("form") or {}), **(self.cfg.get("form") or {})}
            s["headers"] = {**(s.get("headers") or {}), **(self.cfg.get("headers") or {})}
        s.setdefault("form", {})
        s.setdefault("headers", {})
        if not s.get("endpoint"):
            return {}, (
                "imghost 托管缺配置：hosting.yaml 的 imghost.preset 填一个站"
                f"（{' / '.join(_IMGHOST_PRESETS)}），或 preset: custom 自己写 endpoint"
            )
        s.setdefault("field", "file")
        return s, ""

    async def upload(self, path: Path, name: str) -> tuple[str, str]:
        s, err = self.settings()
        if err:
            return "", err
        endpoint = str(s["endpoint"])
        headers = {str(k): str(v) for k, v in (s.get("headers") or {}).items()}
        form = {str(k): str(v) for k, v in (s.get("form") or {}).items()}
        # key 放 .env，配置里只写变量名；放哪儿（query / header / form）按站定
        token_env = str(s.get("token_env") or "")
        token = os.environ.get(token_env, "").strip() if token_env else ""
        if token_env and not token:
            return "", f"{token_env} 没设：把图床的 key 写进 .env 再重启"
        if token:
            where, _, key = str(s.get("token_in") or "query:key").partition(":")
            if where == "header":
                headers[key or "Authorization"] = token
            elif where == "form":
                form[key or "key"] = token
            else:
                sep = "&" if "?" in endpoint else "?"
                endpoint = f"{endpoint}{sep}{quote(key or 'key')}={quote(token)}"
        body = await asyncio.to_thread(path.read_bytes)
        try:
            status, text = await self._post(endpoint, headers, form, str(s["field"]), body, name)
        except Exception as e:  # noqa: BLE001
            return "", f"上传失败：{type(e).__name__}: {e}"
        if status >= 300:
            return "", f"上传失败 HTTP {status}：{text[:200]}"
        url, why = _pick_url(text, str(s.get("url_path") or ""))
        if not url:
            return "", f"上传成功但没拿到链接（{why}）：{text[:200]}"
        return url, ""


# ---------------------------------------------------------------- 统一入口


class Hosting:
    def __init__(self, config: HostingConfig, uploader: Any = None) -> None:
        self.config = config
        if uploader is not None:
            self.uploader = uploader
        elif config.type == "imghost":
            self.uploader = ImageHostUploader(config.imghost)
        elif config.type == "command":
            self.uploader = CommandUploader(config.command)
        elif config.type == "s3":
            self.uploader = S3Uploader(config.s3)
        elif config.type == "http_put":
            self.uploader = HttpPutUploader(config.http_put)
        else:
            self.uploader = None

    @property
    def enabled(self) -> bool:
        return self.uploader is not None

    def brief(self) -> str:
        if not self.enabled:
            return (
                "未配置（config/hosting.yaml type: none）—— 本地素材只能拼接，不能当参考。"
                "没有对象存储就把 type 改成 imghost 并选 preset："
                f"{' / '.join(_IMGHOST_PRESETS)}（imgbb / smms 要一个免费 key，"
                "catbox / litterbox 免注册但图片公开可访问）"
            )
        if self.config.type == "imghost":
            s, err = self.uploader.settings() if hasattr(self.uploader, "settings") else ({}, "")
            if err:
                return f"imghost 配置有问题：{err}"
            preset = str(self.config.imghost.get("preset") or "custom")
            note = str(s.get("note") or "")
            return f"imghost:{preset}（config/hosting.yaml）{('—— ' + note) if note else ''}"
        return f"{self.config.type}（config/hosting.yaml）"

    async def host_file(self, path: Path, name: str = "") -> tuple[str, str]:
        """本地文件 → 公网链接。返回 (url, 错误)。"""
        if not self.enabled:
            return "", (
                "没有配置素材托管：生成接口只收公网链接，本地文件当不了参考。"
                "在 config/hosting.yaml 里配 imghost / command / s3 / http_put 之一后重启"
                "（没有对象存储就用 imghost）。"
            )
        if not await asyncio.to_thread(path.exists):
            return "", f"{path} 不存在"
        return await self.uploader.upload(path, name or path.name)

    def is_fresh(self, asset: Any, ttl_hours: float) -> bool:
        """资产当前的链接还能不能用：托管过的按托管 ttl（默认不过期），否则按生成链接 ttl。"""
        uri = asset.uri or ""
        if not uri.startswith(("http://", "https://")):
            return False
        hosted = asset.gen_params.get("hosted") if isinstance(asset.gen_params, dict) else None
        if isinstance(hosted, dict) and hosted.get("url") == uri:
            ttl = self.config.ttl_hours
            return ttl <= 0 or time.time() - float(hosted.get("at") or 0) <= ttl * 3600
        return ttl_hours <= 0 or time.time() - asset.created_at <= ttl_hours * 3600

    @staticmethod
    def local_file(asset: Any) -> Path | None:
        gp = asset.gen_params if isinstance(asset.gen_params, dict) else {}
        local = str(gp.get("local") or "")
        for cand in (local, asset.uri or ""):
            if cand and not cand.startswith(("http://", "https://")) and Path(cand).exists():
                return Path(cand)
        return None

    def _hosted_ok(self, asset: Any) -> bool:
        """已经托管到自家存储、且托管链接没过期。"""
        gp = asset.gen_params if isinstance(asset.gen_params, dict) else {}
        hosted = gp.get("hosted")
        if not isinstance(hosted, dict) or hosted.get("url") != (asset.uri or ""):
            return False
        return self.is_fresh(asset, 0)

    async def ensure_asset(
        self, store: Any, asset: Any, ttl_hours: float, force: bool = False
    ) -> tuple[str, str]:
        """资产 → 可用的公网链接：链接还新鲜就直接用，否则拿本地副本上传并写回资产。

        force=True：不管生成链接新不新鲜都重新上传（已托管到自家存储的不重复传）。
        """
        if self._hosted_ok(asset):
            return asset.uri, ""
        if not force and self.is_fresh(asset, ttl_hours):
            return asset.uri, ""
        local = await asyncio.to_thread(self.local_file, asset)
        if local is None:
            if (asset.uri or "").startswith(("http://", "https://")):
                return asset.uri, "链接可能已过期且没有本地副本，只能原样用"
            return "", "没有本地副本也没有链接"
        url, err = await self.host_file(local, f"{asset.id}{local.suffix.lower()}")
        if err:
            return (asset.uri or ""), err
        asset.uri = url
        asset.gen_params["hosted"] = {"url": url, "at": time.time(), "via": self.config.type}
        asset.gen_params.setdefault("local", str(local))
        store.put(asset)
        return url, ""
