"""2026-09-23 审查的合规 / 分发问题。

整集剧本被当广告文案查；待发布包漏本地成片、隐式标识只在清单里。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from aigc_agent.domain.assets.store import AssetStore, AssetType
from aigc_agent.domain.compliance import ComplianceChecker, ComplianceRules
from aigc_agent.domain.distribution import Packager, PlatformCatalog
from aigc_agent.domain.functions.compliance import run_check
from aigc_agent.domain.media import ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def test_整集剧本按剧本审_不查广告法极限词():
    store = AssetStore()
    body = "# 第6集：逆袭\n\n陆离：这是最强的一击！\n" + "正文台词。" * 80
    ep = store.create(body, type_=AssetType.OUTLINE, summary="第6集·合规修订版", creator="model")
    ad = store.create("这是最强的产品，绝对第一", type_=AssetType.TEXT, summary="广告文案",
                      creator="model")
    checker = ComplianceChecker(ComplianceRules.load(ROOT / "config" / "compliance.yaml"))
    ep_report, _ = run_check(checker, store, ep.id)
    ad_report, _ = run_check(checker, store, ad.id)
    assert not any("最强" in f.message + f.excerpt for f in ep_report.findings), \
        "剧本台词不按广告法查"
    assert any("最强" in f.message + f.excerpt for f in ad_report.findings)


@pytest.mark.skipif(not ffmpeg.have_ffmpeg(), reason="需要 ffmpeg")
async def test_待发布包拷本地成片_隐式标识写进文件(tmp_path: Path):
    store = AssetStore()
    clip = tmp_path / "ep1.mp4"
    code, err = await ffmpeg.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=160x284:d=1:r=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
    ])
    assert code == 0, err
    video = store.create("", type_=AssetType.VIDEO, summary="成片", creator="model:seedance",
                         gen_params={"local": str(clip)})
    video.uri = "https://cdn.example.com/expired.mp4"  # 远端链接早过期了
    store.put(video)
    text = store.create("第一集上线啦", summary="文案", creator="model:kimi")
    packager = Packager(tmp_path / "releases",
                        PlatformCatalog.load(ROOT / "config" / "platforms.yaml"), store)
    asset, manifest = packager.build(text.id, "douyin", media_asset_ids=[video.id])
    copied = Path(asset.uri or "") / manifest.files[video.id]
    assert copied.exists(), "本地成片要进包（之前只看 uri，远端链接不拷）"
    out = await asyncio.to_thread(
        lambda: subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "json",
             str(copied)],
            capture_output=True, text=True, encoding="utf-8", check=False,
        ).stdout
    )
    tags = json.loads(out).get("format", {}).get("tags", {})
    assert '"AIGC": true' in tags.get("comment", ""), "隐式标识写进了上传的那个文件"
