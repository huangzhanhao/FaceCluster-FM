# -*- coding: utf-8 -*-
"""三属性判定：发色 / 肤色(0-19级) / 发长（需求定稿 §4.2 + 校准日志）。

共用原则：掩码腐蚀后采样；统计用中位数；输出 = 离散桶 + 连续值。
None = 无效（像素不足 / 无掩码），调用方按"弃权"处理。

实现要点（校准踩坑记录）：
- OpenCV 浮点色空间要求输入 0-1（L*∈[0,100]，a*/b*∈[-127,127]）；uint8 输入的
  Lab 是 0-255 且 a/b 偏移 128——两种约定混用是 v0.3 肤色失真的根源。
- 灰世界白平衡只对"皮肤像素集"做会把皮肤色度中和成灰（a*/b*→0），禁止使用。
- 近黑像素的 HSV 色相是噪声，暗发判棕/黑走"色相一致率"而不是色相均值本身。
- 绝对亮度 L 与"场景位次"都无法在混合光照下区分 阴影浅肤 vs 亮光深肤
  （African95 vs 2748 面部 Lab 几乎相同）——0-19 级用 L 为主、色度有界加分的
  单调评分，残余混淆记为已知边界（M3 零样本交叉验证）。
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import Config
from ..types import Attributes


def _erode_bool(mask: np.ndarray, ksize: int = 5, iters: int = 2) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(m, k, iterations=iters).astype(bool)


def _head_roi_mask(hair_mask: np.ndarray, bbox: list[float]) -> np.ndarray:
    """发区限制在头部 ROI（脸框上 0.9H~下 0.4H、左右 0.3W），防衣服/背景误吸。"""
    H_img, W_img = hair_mask.shape[:2]
    x1, y1, x2, y2 = bbox
    H, W = max(1.0, y2 - y1), max(1.0, x2 - x1)
    ry1, ry2 = max(0, int(y1 - 0.9 * H)), min(H_img, int(y2 + 0.4 * H))
    rx1, rx2 = max(0, int(x1 - 0.3 * W)), min(W_img, int(x2 + 0.3 * W))
    out = np.zeros_like(hair_mask)
    out[ry1:ry2, rx1:rx2] = hair_mask[ry1:ry2, rx1:rx2]
    return out


def _scaled_min_px(cfg: Config, area: int, ref_area: int = 350 * 350) -> int:
    return max(cfg.min_attr_px, int(cfg.min_attr_px * area / ref_area * 0.5))


def _hue_stats(hue_deg: np.ndarray) -> tuple[float, float]:
    """圆形均值色相 + 一致率（±15° 内像素占比）。"""
    ang = np.deg2rad(hue_deg)
    mh = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))) % 360.0
    d = np.abs((hue_deg - mh + 180.0) % 360.0 - 180.0)
    return mh, float((d <= 15.0).mean())


# ---------------- 发色 ----------------

def _hue_bucket(h: float, L: float) -> str:
    if 345.0 <= h or h < 18.0:
        return "红棕"
    if 300.0 <= h < 345.0:
        return "粉"
    if 262.0 <= h < 300.0:
        return "紫"
    if 180.0 <= h < 262.0:
        return "蓝"
    if 70.0 <= h < 180.0:
        return "绿"
    return "金黄" if L >= 55.0 else "棕"


def hair_color(
    bgr: np.ndarray,
    hair_mask: np.ndarray | None,
    cfg: Config,
    face_bbox: list[float] | None = None,
) -> tuple[str | None, float]:
    if hair_mask is None:
        return None, 0.0
    m = _head_roi_mask(hair_mask, face_bbox) if face_bbox is not None else hair_mask
    er = _erode_bool(m)
    if int(er.sum()) < _scaled_min_px(cfg, hair_mask.size):
        return None, 0.0
    img01 = bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img01, cv2.COLOR_BGR2Lab)
    L = lab[..., 0][er]          # 0..100
    medL = float(np.median(L))
    if medL < 8.0:
        return "黑", float((L <= cfg.hair_black_L + 10.0).mean())
    if medL < cfg.hair_dark_L:
        # 暗发：色相一致率高的暖色相 → 棕（用户 5 例黑→棕的修复）；否则黑
        hsv = cv2.cvtColor(img01, cv2.COLOR_BGR2HSV)
        medV = float(np.median(hsv[..., 2][er]))
        if medV < cfg.dark_v_floor:
            return "黑", float((L <= cfg.hair_black_L + 10.0).mean())
        mh, conf = _hue_stats(hsv[..., 0][er] * 2.0)
        if conf >= cfg.hair_dark_conf and cfg.hair_warm_hue[0] <= mh <= cfg.hair_warm_hue[1]:
            return "棕", conf
        return "黑", conf
    hsv = cv2.cvtColor(img01, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1][er]          # 0..1
    medS = float(np.median(S))
    if medS < cfg.hair_achroma_sat:  # 无彩色：灰/白（黑已在暗发分支接住）
        bucket = "灰" if medL < cfg.hair_white_L else "白"
        return bucket, float((np.abs(L - medL) <= 10.0).mean())
    mh, conf = _hue_stats(hsv[..., 0][er] * 2.0)
    bucket = _hue_bucket(mh, medL)
    if conf < cfg.min_color_conf:
        return None, conf
    return bucket, conf


# ---------------- 肤色（0-19 级） ----------------

def skin_level(
    bgr: np.ndarray,
    face_skin_mask: np.ndarray | None,
    cfg: Config,
) -> tuple[int | None, str | None, float | None]:
    """返回 (0-19 级, 描述词, eff 连续值)。None = 无效。"""
    if face_skin_mask is None:
        return None, None, None
    er = _erode_bool(face_skin_mask, ksize=7, iters=2)
    if int(er.sum()) < _scaled_min_px(cfg, face_skin_mask.size):
        return None, None, None
    img01 = bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img01, cv2.COLOR_BGR2Lab)
    L = lab[..., 0][er]
    a = lab[..., 1][er]
    b = lab[..., 2][er]
    medL = float(np.median(L))
    medC = float(np.median(np.hypot(a, b)))
    # 单调评分：亮度为主（越暗越深），色度有界加分（黑色素信号）
    eff = (100.0 - medL) + min(medC, cfg.skin_chroma_cap) * cfg.skin_chroma_credit
    level = int(round((eff - cfg.skin_level_floor) / cfg.skin_level_span * 19.0))
    level = max(0, min(19, level))
    lo, hi = cfg.skin_word_ranges
    word = "白皮肤" if level <= lo else ("黄皮肤" if level <= hi else "黑皮肤")
    return level, word, eff


# ---------------- 发长 ----------------

def hair_length(
    face_bbox: list[float],
    kps: list[list[float]] | None,
    img_shape: tuple[int, int],
    hair_mask: np.ndarray | None,
    cfg: Config,
) -> tuple[str | None, float | None, float | None, bool, bool]:
    """返回 (桶, d延伸, 上头皮厚, censored, updo_suspect)。五桶：光头/寸头/短发/中发/长发。"""
    if hair_mask is None or face_bbox is None:
        return None, None, None, False, False
    H_img, W_img = img_shape[:2]
    x1, y1, x2, y2 = face_bbox
    H = max(1.0, y2 - y1)
    Wf = max(1.0, x2 - x1)
    chin = int(round(y2))
    top = int(round(y1))
    xi1, xi2 = max(0, int(x1)), min(W_img, int(x2))
    chin_c = min(H_img, chin)

    # 上头皮区 = 脸框上 0.55H 到鼻尖（关键点定位）——把胡须区排除在外
    # （MediaPipe 头发类含胡须，大胡子会把"头皮"统计污染：Central European219 案例）
    nose_y = kps[2][1] if kps and len(kps) > 2 else y1 + 0.75 * H
    ry1 = max(0, int(y1 - 0.55 * H))
    ry2 = int(min(y2, nose_y))
    upper = hair_mask[ry1:ry2, xi1:xi2]
    upper_area = max(1.0, (ry2 - ry1) * Wf)
    upper_ratio = float(upper.sum() / upper_area)
    urows = upper.sum(axis=1).astype(np.float32)
    upper_thick = float(urows.mean() / Wf) if urows.size else 0.0

    head_area = max(1.0, 0.6 * H * Wf)
    hair_pts = np.argwhere(hair_mask)
    vol = float(hair_mask.sum() / head_area)
    wide = 0.0
    if hair_pts.size:
        band = hair_mask[max(0, top - int(0.5 * H)):chin_c, :]
        if band.size:
            widths = band.sum(axis=1)
            wide = float(widths.max() / Wf) if widths.size else 0.0

    # 越下巴延伸：脸框±0.5W 竖带与全宽取大
    bx1, bx2 = max(0, int(x1 - 0.5 * Wf)), min(W_img, int(x2 + 0.5 * Wf))
    max_y = -1
    for region in (hair_mask[chin_c:, bx1:bx2], hair_mask[chin_c:, :]):
        ys = np.where(region.any(axis=1))[0]
        if ys.size:
            max_y = max(max_y, int(ys.max()) + chin_c)
    d = (max_y - chin) / H if max_y >= 0 else 0.0
    censored = bool(max_y >= H_img - 3)

    updo = False
    if hair_pts.size:
        hy1, _ = hair_pts.min(axis=0)
        hy2, _ = hair_pts.max(axis=0)
        bb_h = (float(hy2) - float(hy1)) / H
        updo = bool(bb_h < 0.9 and vol > 0.35)

    if upper_ratio < cfg.bald_upper_ratio and upper_thick < cfg.bald_upper_thick:
        return "光头", 0.0, upper_thick, False, updo
    if upper_thick < cfg.len_buzz_upper:
        return "寸头", 0.0, upper_thick, censored, updo
    if vol >= cfg.len_vol_long and wide >= cfg.len_wide_long:
        # 体积型长发：卷发/爆炸头，下垂延伸小但发量巨大（校准轮①两例）
        return "长发", d, upper_thick, censored, updo
    if d < cfg.len_short_d:
        return "短发", 0.0, upper_thick, censored, updo
    if d <= cfg.len_medium_d:
        return "中发", d, upper_thick, censored, updo
    return "长发", d, upper_thick, censored, updo


# ---------------- 汇总 ----------------

def compute_attributes(
    bgr: np.ndarray,
    masks: dict[str, np.ndarray] | None,
    face_bbox: list[float] | None,
    cfg: Config,
    kps: list[list[float]] | None = None,
) -> Attributes:
    attrs = Attributes()
    if masks is None:
        return attrs
    hc, hcc = hair_color(bgr, masks.get("hair"), cfg, face_bbox)
    sl, sw, _eff = skin_level(bgr, masks.get("face_skin"), cfg)
    hl, d, th, cen, updo = hair_length(face_bbox, kps, bgr.shape, masks.get("hair"), cfg)
    attrs.hair_color, attrs.hair_color_conf = hc, hcc
    attrs.skin_level, attrs.skin_word = sl, sw
    attrs.hair_length, attrs.hair_len_d, attrs.hair_thickness = hl, d, th
    attrs.censored = cen
    attrs.updo_suspect = updo
    return attrs
