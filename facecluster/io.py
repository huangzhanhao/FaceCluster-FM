# -*- coding: utf-8 -*-
"""输入层：扫描 / 解码 / EXIF / 内容哈希（L0）。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scan_images(root: Path, exts=IMAGE_EXTS) -> list[Path]:
    """递归扫描目录下的图片文件，按文件名排序。"""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def load_bgr(path: Path, max_side: int = 1600) -> np.ndarray:
    """解码 + EXIF 旋转 + 长边缩放，返回 BGR 数组。"""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
    arr = np.asarray(im.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def content_hash(path: Path) -> str:
    """blake2b 内容哈希 —— 图片身份标识（不用文件名：改名后缓存仍命中）。"""
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
