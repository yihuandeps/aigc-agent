"""本地素材托管（2026-09-19）：用户的素材在本地，生成接口只收公网链接。

三种上传方式（command / s3 / http_put）用假的执行器验签名、URL 拼法与错误提示；
Hosting.ensure_asset 把链接写回资产；短剧链里：本地图进参考图包、过期参考图渲染前自动重新
托管、drama_refresh_refs 优先重新上传而不是复刻。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.functions.drama import DramaFunctions
from aigc_agent.domain.functions.files import FileFunctions, FsPolicy
from aigc_agent.domain.functions.hosting import HostingFunctions
from aigc_agent.domain.functions.media import MediaFunctions
from aigc_agent.domain.generators.catalog import MediaCatalog
from aigc_agent.domain.media.hosting import (
    CommandUploader,
    Hosting,
    HostingConfig,
    HttpPutUploader,
    ImageHostUploader,
    S3Uploader,
)
from aigc_agent.harness.tools.provider import ToolResult

CONFIG = Path(__file__).resolve().parents[1] / "config" / "hosting.yaml"


def _write(path: Path, data: bytes = b"png") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------- 上传器


def test_真实配置文件能用():
    """config/hosting.yaml 是用户按自己的机器改的，不钉死具体是哪种；
    钉住的是：认得出来、构造得出上传器、brief() 说得清现在什么状态。"""
    cfg = HostingConfig.load(CONFIG)
    assert cfg.type in ("none", "imghost", "command", "s3", "http_put")
    h = Hosting(cfg)
    assert "hosting.yaml" in h.brief()
    if cfg.type == "none":
        assert not h.enabled
    else:
        assert h.enabled, f"配了 {cfg.type} 却造不出上传器：{h.brief()}"
        if cfg.type == "imghost":
            _, err = h.uploader.settings()
            assert not err, err


async def test_command上传_取stdout最后一行链接(tmp_path: Path):
    f = await asyncio.to_thread(_write, tmp_path / "a.png")
    py = sys.executable.replace("\\", "/")
    up = CommandUploader(f'"{py}" -c "print(\'uploading\'); print(\'https://cdn.example.com/{{name}}\')"')
    url, err = await up.upload(f, "as_1.png")
    assert err == "" and url == "https://cdn.example.com/as_1.png"
    bad = CommandUploader(f'"{py}" -c "import sys; print(\'no url\'); sys.exit(2)"')
    url, err = await bad.upload(f, "x.png")
    assert url == "" and "退出码 2" in err
    assert (await CommandUploader("").upload(f, "x.png"))[1]


async def test_s3上传_SigV4签名与地址拼法(tmp_path: Path):
    f = await asyncio.to_thread(_write, tmp_path / "陆离.png", b"data")
    calls: list[tuple[str, dict[str, str], bytes]] = []

    async def put(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
        calls.append((url, headers, body))
        return 200, ""

    cfg = {
        "endpoint": "https://oss-cn-hangzhou.aliyuncs.com", "region": "oss-cn-hangzhou",
        "bucket": "my-bucket", "prefix": "aigc-refs/", "acl": "public-read",
    }
    env = {"S3_ACCESS_KEY": "AKID", "S3_SECRET_KEY": "SECRET"}
    up = S3Uploader(cfg, env=env, http_put=put, now=lambda: 1_700_000_000.0)
    url, err = await up.upload(f, "as_1.png")
    assert err == "" and url == "https://my-bucket.oss-cn-hangzhou.aliyuncs.com/aigc-refs/as_1.png"
    put_url, headers, body = calls[0]
    assert put_url == url and body == b"data"
    auth = headers["Authorization"]
    cred = "AWS4-HMAC-SHA256 Credential=AKID/20231114/oss-cn-hangzhou/s3/aws4_request"
    assert auth.startswith(cred)
    assert "SignedHeaders=content-type;host;x-amz-acl;x-amz-content-sha256;x-amz-date" in auth
    assert headers["x-amz-date"] == "20231114T221320Z" and headers["x-amz-acl"] == "public-read"
    assert headers["content-type"] == "image/png" and "host" not in headers
    assert len(auth.rsplit("Signature=", 1)[1]) == 64
    # path-style + 自定义公网前缀 + 中文对象名要 URL 编码
    cfg2 = {**cfg, "path_style": True, "public_base": "https://cdn.example.com", "prefix": "参考/"}
    url2, _ = await S3Uploader(cfg2, env=env, http_put=put).upload(f, "陆离.png")
    assert url2 == "https://cdn.example.com/%E5%8F%82%E8%80%83/%E9%99%86%E7%A6%BB.png"
    assert calls[1][0].startswith("https://oss-cn-hangzhou.aliyuncs.com/my-bucket/%E5%8F%82%E8%80%83/")
    # 缺配置要说清楚缺什么
    url3, err3 = await S3Uploader({"endpoint": "x"}, env={}, http_put=put).upload(f, "a.png")
    assert url3 == "" and "s3.bucket" in err3 and "S3_ACCESS_KEY" in err3


async def test_http_put上传(tmp_path: Path):
    f = await asyncio.to_thread(_write, tmp_path / "a.png")
    calls: list[tuple[str, dict[str, str]]] = []

    async def put(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
        calls.append((url, headers))
        return 201, ""

    cfg = {"base_url": "https://files.example.com/upload/", "public_base": "https://files.example.com",
           "auth_header": "Bearer t"}
    url, err = await HttpPutUploader(cfg, http_put=put).upload(f, "a.png")
    assert err == "" and url == "https://files.example.com/a.png"
    assert calls[0][0] == "https://files.example.com/upload/a.png"
    assert calls[0][1]["Authorization"] == "Bearer t" and calls[0][1]["Content-Type"] == "image/png"


# ---------------------------------------------------------------- 资产层


class FakeUploader:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[Path, str]] = []
        self.fail = fail

    async def upload(self, path: Path, name: str) -> tuple[str, str]:
        self.calls.append((path, name))
        if self.fail:
            return "", "模拟上传失败"
        return f"https://cdn.example.com/{name}", ""


def _hosting(fail: bool = False) -> tuple[Hosting, FakeUploader]:
    up = FakeUploader(fail)
    return Hosting(HostingConfig(type="command", command="x"), uploader=up), up


async def test_ensure_asset_新鲜链接直接用_过期或本地就上传写回(tmp_path: Path):
    store = AssetStore()
    h, up = _hosting()
    fresh = store.create("", type_=AssetType.IMAGE, summary="新")
    fresh.uri = "https://gen/fresh.png"
    store.put(fresh)
    assert await h.ensure_asset(store, fresh, 20) == ("https://gen/fresh.png", "")
    assert not up.calls

    local = await asyncio.to_thread(_write, tmp_path / "old.png")
    old = store.create("", type_=AssetType.IMAGE, summary="旧", gen_params={"local": str(local)})
    old.uri = "https://gen/old.png"
    old.created_at = time.time() - 30 * 3600
    store.put(old)
    url, err = await h.ensure_asset(store, old, 20)
    assert err == "" and url == f"https://cdn.example.com/{old.id}.png"
    kept = store.get(old.id)
    assert kept.uri == url and kept.gen_params["hosted"]["via"] == "command"
    assert h.is_fresh(store.get(old.id), 20), "托管过的不按 24h 过期"
    # 第二次不再上传
    assert await h.ensure_asset(store, store.get(old.id), 20) == (url, "")
    assert len(up.calls) == 1

    only_local = store.create(
        "", type_=AssetType.IMAGE, summary="本地", gen_params={"local": str(local)}
    )
    only_local.uri = str(local)
    store.put(only_local)
    url, err = await h.ensure_asset(store, only_local, 20)
    assert url.startswith("https://cdn.example.com/") and err == ""

    gone = store.create("", type_=AssetType.IMAGE, summary="没副本")
    gone.uri = "https://gen/gone.png"
    gone.created_at = time.time() - 30 * 3600
    store.put(gone)
    url, err = await h.ensure_asset(store, gone, 20)
    assert url == "https://gen/gone.png" and "没有本地副本" in err


async def test_托管工具_host_file登记并上传(tmp_path: Path):
    store = AssetStore()
    out = tmp_path / "out"
    out.mkdir()
    img = await asyncio.to_thread(_write, out / "产品.png")
    files = FileFunctions(store, tmp_path / "ws", out, FsPolicy(), project_root=tmp_path / "p")
    h, up = _hosting()
    fns = HostingFunctions(store, h, files)
    r = await fns.invoke("host_file", {"path": str(img), "summary": "产品图"})
    assert r.ok, r.error
    a = store.get(r.asset_ref)
    assert a.uri == f"https://cdn.example.com/{a.id}.png" and "参考链接" in r.content
    status = (await fns.invoke("hosting_status", {})).content
    assert status.endswith("command（config/hosting.yaml）")
    off = HostingFunctions(store, Hosting(HostingConfig()), files)
    r2 = await off.invoke("host_file", {"path": str(img)})
    assert not r2.ok and "hosting.yaml" in r2.error


# ---------------------------------------------------------------- 短剧链


LIB = {
    "characters": [
        {"baseRoleName": "陆离", "roleTotalDesc": "男 | 35岁",
         "roleCostumeList": [{"costumeName": "陆离-风衣-[全集]", "costumeDesc": "风衣"}]},
    ],
    "scenes": [{"name": "地铁车厢", "description": "x"}],
    "props": [],
}


class Registry:
    def __init__(self, store: AssetStore) -> None:
        self.store = store
        self.calls: list[tuple[str, dict]] = []

    async def invoke(self, name: str, args: dict) -> ToolResult:
        self.calls.append((name, dict(args)))
        if name == "compose_video":
            a = self.store.create("mp4", summary="成片", creator="fake")
            a.uri = "https://fake/c.mp4"
            self.store.put(a)
            return ToolResult(ok=True, content="拼好了", asset_ref=a.id)
        a = self.store.create("", type_=AssetType.VIDEO if name == "gen_video" else AssetType.IMAGE,
                              summary=args["summary"], creator="model:x")
        a.uri = f"https://fake/{a.id}"
        self.store.put(a)
        return ToolResult(ok=True, content="ok", asset_ref=a.id)


def _drama(store: AssetStore, tmp_path: Path, hosting: Hosting) -> DramaFunctions:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    files = FileFunctions(store, tmp_path / "ws", out, FsPolicy(), project_root=tmp_path / "p")
    fns = DramaFunctions(None, store, registry=Registry(store), catalog=None, hosting=hosting)
    fns.files = files
    return fns


async def test_本地图放进参考图包_之后渲染沿用(tmp_path: Path):
    store = AssetStore()
    h, up = _hosting()
    fns = _drama(store, tmp_path, h)
    lib = store.create(
        json.dumps(LIB, ensure_ascii=False), summary="资产库", creator="tool:drama_assets"
    )
    img = await asyncio.to_thread(_write, tmp_path / "out" / "陆离定妆.png")
    r = await fns.invoke("drama_use_local_ref", {"name": "陆离", "path": str(img)})
    assert r.ok, r.error
    pack = json.loads(store.content(r.asset_ref))
    assert pack["陆离"]["kind"] == "角色" and pack["陆离"]["source"] == "user"
    assert pack["陆离"]["url"].startswith("https://cdn.example.com/")
    assert store.get(r.asset_ref).parent_ids[0] == lib.id
    # 名字不在库里要报出可用名字；类型自动判断服装/场景
    bad = await fns.invoke("drama_use_local_ref", {"name": "路人甲", "path": str(img)})
    assert not bad.ok and "陆离-风衣-[全集]" in bad.error
    r2 = await fns.invoke("drama_use_local_ref", {"name": "地铁车厢", "path": str(img)})
    assert r2.ok and json.loads(store.content(r2.asset_ref))["地铁车厢"]["kind"] == "场景"
    assert "陆离" in json.loads(store.content(r2.asset_ref)), "新包在旧包基础上加"
    # 渲参考图时 reuse：用户给的主形象不再生成
    rr = await fns._fn_drama_render_assets(lib.id, reuse=True)
    assert rr.ok, rr.error
    gens = [a for n, a in fns.registry.calls if n == "gen_image"]  # type: ignore[attr-defined]
    assert all("角色·陆离" != a["summary"] for a in gens) and "复用" in rr.content
    # 没配托管时说清楚
    off = _drama(AssetStore(), tmp_path, Hosting(HostingConfig()))
    r3 = await off.invoke("drama_use_local_ref", {"name": "陆离", "path": str(img)})
    assert not r3.ok and "hosting.yaml" in r3.error


async def test_渲染前过期参考图自动重新托管(tmp_path: Path):
    store = AssetStore()
    h, up = _hosting()
    fns = _drama(store, tmp_path, h)
    lib_id = store.create(json.dumps(LIB, ensure_ascii=False), summary="资产库", creator="t").id
    shots_id = store.create(
        json.dumps([{"scene_index": "[第1集-1场]", "video_name": "1-2", "video_duration": "10s",
                     "description": "(地铁车厢) (陆离) 开场"}], ensure_ascii=False),
        summary="提示词", creator="tool:drama_shots", parents=["sb", lib_id],
    ).id
    local = await asyncio.to_thread(_write, tmp_path / "luli.png")
    portrait = store.create("", type_=AssetType.IMAGE, summary="角色·陆离", creator="model:x",
                            gen_params={"local": str(local)})
    portrait.uri = "https://gen/old.png"
    portrait.created_at = time.time() - 30 * 3600
    store.put(portrait)
    scene = store.create("", type_=AssetType.IMAGE, summary="场景·地铁车厢", creator="model:x")
    scene.uri = "https://cdn.example.com/car.png"
    store.put(scene)
    pack_id = store.create(
        json.dumps({
            "陆离": {"asset": portrait.id, "url": "https://gen/old.png", "kind": "角色"},
            "地铁车厢": {
                "asset": scene.id, "url": "https://cdn.example.com/car.png", "kind": "场景"
            },
        }),
        summary="参考图包", creator="tool:drama_render_assets", parents=[lib_id],
    ).id
    r = await fns._fn_drama_render_shots(shots_id, rendered_id=pack_id)
    assert r.ok, r.error
    v = [a for n, a in fns.registry.calls if n == "gen_video"][0]  # type: ignore[attr-defined]
    assert v["image"][0] == f"https://cdn.example.com/{portrait.id}.png", "传给模型的是新链接"
    assert "已用本地副本重新托管" in r.content and "链接超过" not in r.content
    assert store.get(portrait.id).uri.startswith("https://cdn.example.com/")


def _stale_pack(store: AssetStore, local: Path) -> tuple[str, str]:
    portrait = store.create("", type_=AssetType.IMAGE, summary="角色·陆离", creator="model:x",
                            gen_params={"local": str(local)})
    portrait.uri = "https://gen/old.png"
    portrait.created_at = time.time() - 30 * 3600
    store.put(portrait)
    lib_id = store.create("{}", summary="资产库", creator="tool:drama_assets").id
    pack_id = store.create(
        json.dumps({"陆离": {"asset": portrait.id, "url": "https://gen/old.png", "kind": "角色"}}),
        summary="参考图包", creator="tool:drama_render_assets", parents=[lib_id],
    ).id
    return portrait.id, pack_id


async def test_刷新参考图_优先重新上传不复刻(tmp_path: Path):
    store = AssetStore()
    h, up = _hosting()
    fns = _drama(store, tmp_path, h)
    local = await asyncio.to_thread(_write, tmp_path / "luli.png")
    portrait_id, pack_id = _stale_pack(store, local)
    r = await fns._fn_drama_refresh_refs(rendered_id=pack_id)
    assert r.ok, r.error
    gens = [a for n, a in fns.registry.calls if n == "gen_image"]  # type: ignore[attr-defined]
    assert not gens, "没有复刻"
    new_pack = json.loads(store.content(r.asset_ref))
    assert new_pack["陆离"]["asset"] == portrait_id and "重新托管" in r.content
    assert new_pack["陆离"]["url"] == f"https://cdn.example.com/{portrait_id}.png"

    # 上传挂了 → 退回复刻（另起一个库，别让上面已托管的资产干扰）
    store2 = AssetStore()
    h2, _ = _hosting(fail=True)
    fns2 = _drama(store2, tmp_path, h2)
    _, pack2 = _stale_pack(store2, local)
    r2 = await fns2._fn_drama_refresh_refs(rendered_id=pack2)
    assert r2.ok and "改用复刻" in r2.content
    assert [a for n, a in fns2.registry.calls if n == "gen_image"]  # type: ignore[attr-defined]


def test_钩子_属性():
    assert isinstance(SimpleNamespace(), object)


# ---------------------------------------------------------------- 图床（2026-09-22）
#
# 用户没有对象存储，"用自己的图当参考"这条路本来是断的。图床是唯一不用自建存储的办法，
# 站点参数全做成配置 —— 这里钉住 preset 合并、key 放哪、链接怎么从返回里挑出来。


class _FakePost:
    """记下请求、按构造时给的内容返回。"""

    def __init__(self, status: int = 200, text: str = "") -> None:
        self.status = status
        self.text = text
        self.calls: list[dict] = []

    async def __call__(self, url, headers, form, field, body, filename):
        self.calls.append(
            {"url": url, "headers": headers, "form": form,
             "field": field, "body": body, "name": filename}
        )
        return self.status, self.text


def test_免注册图床_纯文本返回里取链接(tmp_path: Path):
    """catbox 直接回一行链接，没有 JSON 可解。"""
    post = _FakePost(text="https://files.catbox.moe/ab12cd.png\n")
    up = ImageHostUploader({"preset": "catbox"}, post=post)
    url, err = asyncio.run(up.upload(_write(tmp_path / "a.png"), "陆离.png"))
    assert err == "" and url == "https://files.catbox.moe/ab12cd.png"
    c = post.calls[0]
    assert c["url"] == "https://catbox.moe/user/api.php"
    assert c["field"] == "fileToUpload" and c["form"]["reqtype"] == "fileupload"


def test_图床key进query_返回按JSON路径取(tmp_path: Path, monkeypatch):
    """imgbb 的 key 走 query、链接在 data.url。key 放 .env，配置里只写变量名。"""
    monkeypatch.setenv("IMGBB_API_KEY", "k123")
    post = _FakePost(text=json.dumps({"data": {"url": "https://i.ibb.co/x/luli.png"}}))
    up = ImageHostUploader({"preset": "imgbb"}, post=post)
    url, err = asyncio.run(up.upload(_write(tmp_path / "a.png"), "luli.png"))
    assert err == "" and url == "https://i.ibb.co/x/luli.png"
    assert post.calls[0]["url"].endswith("?key=k123"), post.calls[0]["url"]


def test_图床key进请求头(tmp_path: Path, monkeypatch):
    """S.EE：key 走 Authorization 头、文件字段叫 file、链接在 data.url。
    2026-09-22 对着真接口核过：sm.ms 已整体迁到 S.EE，老域名上传一律 401。"""
    monkeypatch.setenv("SEE_API_TOKEN", "t456")
    post = _FakePost(text=json.dumps({"data": {"url": "https://i.see.you/x/a.png"}}))
    up = ImageHostUploader({"preset": "see"}, post=post)
    url, err = asyncio.run(up.upload(_write(tmp_path / "a.png"), "a.png"))
    assert err == "" and url == "https://i.see.you/x/a.png"
    c = post.calls[0]
    assert c["headers"]["Authorization"] == "t456" and "key=" not in c["url"]
    assert c["url"] == "https://s.ee/api/v1/file/upload" and c["field"] == "file"


def test_选了preset就不吃配置里的自定义字段(tmp_path: Path, monkeypatch):
    """hosting.yaml 的模板里 custom 那几项带着占位值（token_in: query:key），
    非空。要是让它们覆盖 preset，key 就从请求头跑进 query，服务端只回一句 401 ——
    真踩过这个坑，查了一轮才找到。选了 preset 就完全以 preset 为准。"""
    monkeypatch.setenv("SEE_API_TOKEN", "t456")
    post = _FakePost(text=json.dumps({"data": {"url": "https://i.see.you/x/a.png"}}))
    up = ImageHostUploader(
        {"preset": "see", "endpoint": "", "field": "file", "form": {}, "headers": {},
         "url_path": "", "token_env": "", "token_in": "query:key"},
        post=post,
    )
    url, err = asyncio.run(up.upload(_write(tmp_path / "a.png"), "a.png"))
    assert err == "" and url == "https://i.see.you/x/a.png"
    assert post.calls[0]["headers"]["Authorization"] == "t456", "占位的 token_in 把 key 挪走了"


def test_custom能自己指定链接在返回里的位置(tmp_path: Path):
    """preset: custom 才吃配置；链接不在 data.url 的站也接得上。"""
    post = _FakePost(text=json.dumps({"code": "image_repeated", "images": "https://x/a.png"}))
    up = ImageHostUploader(
        {"preset": "custom", "endpoint": "https://x/up", "field": "f", "url_path": "data.url"},
        post=post,
    )
    url, err = asyncio.run(up.upload(_write(tmp_path / "a.png"), "a.png"))
    assert err == "" and url == "https://x/a.png", "data.url 取不到时该退回 images"


def test_缺key时说清楚要设哪个变量(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("IMGBB_API_KEY", raising=False)
    up = ImageHostUploader({"preset": "imgbb"}, post=_FakePost())
    url, err = asyncio.run(up.upload(_write(tmp_path / "a.png"), "a.png"))
    assert url == "" and "IMGBB_API_KEY" in err and ".env" in err


def test_preset认不得时列出可选的(tmp_path: Path):
    up = ImageHostUploader({"preset": "随便写的"}, post=_FakePost())
    url, err = asyncio.run(up.upload(_write(tmp_path / "a.png"), "a.png"))
    assert url == "" and "see" in err and "imgbb" in err and "litterbox" in err


def test_没填preset也没填endpoint时不瞎传(tmp_path: Path):
    up = ImageHostUploader({}, post=_FakePost())
    url, err = asyncio.run(up.upload(_write(tmp_path / "a.png"), "a.png"))
    assert url == "" and "preset" in err


def test_上传失败带上返回内容(tmp_path: Path):
    up = ImageHostUploader({"preset": "catbox"}, post=_FakePost(status=413, text="too big"))
    url, err = asyncio.run(up.upload(_write(tmp_path / "a.png"), "a.png"))
    assert url == "" and "413" in err and "too big" in err


def test_临时图床按preset给默认过期时间(tmp_path: Path):
    """litterbox 72h 就删。ttl 留 0 会被当成永不过期，渲染时拿死链去生图。"""
    cfg = tmp_path / "h.yaml"
    cfg.write_text("type: imghost\nimghost:\n  preset: litterbox\n", encoding="utf-8")
    assert HostingConfig.load(cfg).ttl_hours == 70.0
    cfg.write_text("type: imghost\nimghost:\n  preset: catbox\n", encoding="utf-8")
    assert HostingConfig.load(cfg).ttl_hours == 0.0, "永久图床不该被当成会过期"


def test_没配托管时告诉用户可以用图床():
    h = Hosting(HostingConfig(type="none"))
    assert not h.enabled
    assert "imghost" in h.brief() and "catbox" in h.brief()


def test_配了图床_Hosting走图床上传器():
    h = Hosting(HostingConfig(type="imghost", imghost={"preset": "catbox"}))
    assert h.enabled and isinstance(h.uploader, ImageHostUploader)
    assert "catbox" in h.brief()


# ---------------------------------------------------------------- 提交前拦本地路径
#
# 生成接口只收 http/https。本地路径提交上去只会换回一句英文报错，一段视频的
# 等待时间全白搭 —— 在这儿拦住，并告诉模型该走 host_file。


def _media(tmp_path: Path) -> MediaFunctions:
    catalog = MediaCatalog.load(Path(__file__).resolve().parents[1] / "config/media_models.yaml")
    return MediaFunctions(None, catalog, AssetStore(tmp_path / "assets"))


async def test_本地路径当参考图会被拦在提交前(tmp_path: Path):
    fns = _media(tmp_path)
    r = await fns.invoke(
        "gen_video",
        {"prompt": "陆离走进实验室", "image": ["E:/照片/陆离.png"], "model": "seedance-2.0"},
    )
    assert not r.ok
    assert "不是公网链接" in r.error and "host_file" in r.error
    assert "drama_use_local_ref" in r.error


async def test_生图的参考图同样拦(tmp_path: Path):
    fns = _media(tmp_path)
    r = await fns.invoke("gen_image", {"prompt": "白底全身", "image": ["C:\\a\\b.png"]})
    assert not r.ok and "不是公网链接" in r.error


async def test_公网链接照常放行(tmp_path: Path):
    """拦的是本地路径，不能把正常链接一起拦了。"""
    from aigc_agent.domain.functions.media import _not_public

    assert _not_public(["https://a/b.png", "http://c/d.png", "asset://x"]) == []
    assert _not_public(["E:/a.png", "data:image/png;base64,xx"]) == [
        "E:/a.png", "data:image/png;base64,xx"
    ]
