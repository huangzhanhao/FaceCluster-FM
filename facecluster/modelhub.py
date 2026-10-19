# -*- coding: utf-8 -*-
"""模型下载：HF 单文件 / InsightFace 官方 release 模型包，流式下载 + 手动放置提示。

两个来源分工：
- SigLIP2、预处理配置等**单文件**走 HF（`repo/resolve/main/file`）；endpoint 取
  `HF_ENDPOINT`，未设置时先试官方源、失败自动落 hf-mirror.com（国内常态）。
- 人脸识别**模型包**走 InsightFace 官方 release `model-zoo` 的 `{包名}.zip`。
  HF 的 public-data/insightface 只放了 buffalo_l，其它包在那里必然 404，仅作兜底。
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

MIRROR = "https://hf-mirror.com"
OFFICIAL = "https://huggingface.co"

# 官方 release 的包名白名单 = 可下载清单；未列入的包名直接报错，
# 不再让用户拿到一句无从下手的 404。
INSIGHTFACE_RELEASE = "https://github.com/deepinsight/insightface/releases/download/model-zoo"
INSIGHTFACE_PACKS: tuple = (
    "antelopev2", "buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc", "cheetah_l", "cheetah_s",
)


def _endpoints() -> list[str]:
    env = os.environ.get("HF_ENDPOINT")
    if env:
        return [env.rstrip("/")]
    return [OFFICIAL, MIRROR]


def _stream(url: str, part: Path, log) -> None:
    """单个 URL 流式写入 .part（每累计 64MB 打一行进度）。"""
    log(f"下载 {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "FaceCluster-FM/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        next_mark = 0
        with open(part, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done >= next_mark:
                    pct = f" {done * 100 // total}%" if total else ""
                    log(f"  {done / 1e6:.0f} MB{pct}")
                    next_mark += 64 << 20


def _fetch(urls: list[str], dest: Path, log, min_bytes: int, manual: str) -> Path:
    """按候选 URL 顺序流式下载 → dest；已存在且达标即跳过；全失败抛 ConnectionError。

    字节数小于 min_bytes 视为源 404 / 半截响应（不落盘），换下一个候选。
    """
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    for url in urls:
        try:
            log(f"  → {dest}")
            _stream(url, part, log)
            if part.stat().st_size < min_bytes:
                raise IOError(f"文件过小 {part.stat().st_size}B（源可能 404 或不完整）")
            part.replace(dest)
            log(f"✓ {dest.name} {dest.stat().st_size / 1e6:.0f} MB")
            return dest
        except (urllib.error.URLError, OSError, IOError) as e:
            last_err = e
            log(f"✗ {url} 失败：{e}")
            part.unlink(missing_ok=True)
    raise ConnectionError(
        f"下载失败（{last_err}）。可手动下载后放到：\n"
        f"  {dest}\n"
        f"  源：{manual}"
    )


def download(repo: str, file: str, dest: Path, log=print, min_bytes: int = 1 << 20) -> Path:
    """HF 单文件：`repo/resolve/main/file`（官方源 → hf-mirror 镜像）。"""
    urls = [f"{b}/{repo}/resolve/main/{file}" for b in _endpoints()]
    return _fetch(urls, dest, log, min_bytes,
                  manual=f"{OFFICIAL}/{repo} （国内可换 {MIRROR}/{repo}）")


def _unpack(zf: Path, unpack_dir: Path, log=print) -> Path:
    """解压 zip → unpack_dir 后删 zip；非法路径（路径穿越）直接拒绝。"""
    import zipfile

    unpack_dir.mkdir(parents=True, exist_ok=True)
    root = unpack_dir.resolve()
    with zipfile.ZipFile(zf) as z:
        for m in z.namelist():
            # is_relative_to 按路径段判定：startswith 字符串比较会被 buffalo_l_evil
            # 这类同名前缀兄弟目录绕过（绝对路径/../ 同理拦在 root 之外）
            if not (unpack_dir / m).resolve().is_relative_to(root):
                raise ValueError(f"zip 内非法路径（疑似路径穿越）: {m}")
        z.extractall(unpack_dir)
    zf.unlink()
    log(f"✓ 已解压 → {unpack_dir}")
    return unpack_dir


def download_pack(name: str, pack_dir: Path, log=print, min_bytes: int = 10_000_000) -> Path:
    """下载 InsightFace 模型包并解压到 pack_dir（官方 release 为主，HF 兜底）。

    HF 兜底只对 buffalo_l 有效（那里没放其它包），其余包会在两个候选各失败一次后
    给出官方手动下载地址——不会静默变成一句 404。
    """
    if name not in INSIGHTFACE_PACKS:
        raise ValueError(f"未知模型包 {name}（可下载：{', '.join(INSIGHTFACE_PACKS)}）")
    urls = [f"{INSIGHTFACE_RELEASE}/{name}.zip"]
    urls += [f"{b}/public-data/insightface/resolve/main/models/{name}.zip" for b in _endpoints()]
    zf = _fetch(urls, pack_dir.parent / f"{name}.zip", log, min_bytes,
                manual=f"{INSIGHTFACE_RELEASE}/{name}.zip")
    return _unpack(zf, pack_dir, log)
