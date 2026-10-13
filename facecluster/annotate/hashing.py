# -*- coding: utf-8 -*-
"""pHash：整图 + 人脸区（同源变体的并列裁决与精确重复标记）。"""
from __future__ import annotations

import cv2
import imagehash
import numpy as np
from PIL import Image


def _phash(bgr: np.ndarray) -> str:
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(im))


def whole_phash(bgr: np.ndarray) -> str:
    return _phash(bgr)


def face_region_phash(bgr: np.ndarray, bbox: list[float], expand: float = 1.4) -> str:
    """人脸框外扩 1.4 倍裁剪后 pHash（原图/裁剪变体的人脸区哈希更稳）。"""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_x, half_y = (x2 - x1) / 2.0 * expand, (y2 - y1) / 2.0 * expand
    a, b = max(0, int(cx - half_x)), min(w, int(cx + half_x))
    c, d = max(0, int(cy - half_y)), min(h, int(cy + half_y))
    return _phash(bgr[c:d, a:b])
