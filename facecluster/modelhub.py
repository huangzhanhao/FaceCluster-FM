# -*- coding: utf-8 -*-
"""模型下载：HF resolve 流式下载 + 手动放置提示。

endpoint 取 HF_ENDPOINT 环境变量；未设置时先试官方源，连接失败自动落到
hf-mirror.com 镜像（国内网络常态）。失败时打印手动放置路径。
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

MIRROR = "https://hf-mirror.com"
OFFICIAL = "https://huggingface.co"


def _endpoints() -> list[str]:
    env = os.environ.get("HF_ENDPOINT")
    if env:
        return [env.rstrip("/")]
    return [OFFICIAL, MIRROR]


def download(repo: str, file: str, dest: Path, log=print, min_bytes: int = 1 << 20) -> Path:
    """下载 repo/file → dest。已存在则跳过；.part 临时文件防半截文件。"""
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    for base in _endpoints():
        url = f"{base}/{repo}/resolve/main/{file}"
        try:
            log(f"下载 {url}")
            log(f"  → {dest}")
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
            if part.stat().st_size < min_bytes:
                raise IOError(f"文件过小 {part.stat().st_size}B（源可能 404 或不完整）")
            part.replace(dest)
            log(f"✓ {dest.name} {dest.stat().st_size / 1e6:.0f} MB")
            return dest
        except (urllib.error.URLError, OSError, IOError) as e:
            last_err = e
            log(f"✗ {base} 失败：{e}")
            part.unlink(missing_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    raise ConnectionError(
        f"下载失败（{last_err}）。可手动下载后放到：\n"
        f"  {dest}\n"
        f"  源：{OFFICIAL}/{repo} （国内可换 {MIRROR}/{repo}）\n"
        f"  或设环境变量 HF_ENDPOINT=https://hf-mirror.com 后重试"
    )
