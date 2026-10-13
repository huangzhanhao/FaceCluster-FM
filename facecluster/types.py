# -*- coding: utf-8 -*-
"""数据类型：标注阶段的全部产物契约（M1 聚簇 / M2 排序消费这些类型）。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class FaceInfo:
    """单张图的主要人脸（多人合影取最大脸）。"""
    bbox: list[float]                  # x1, y1, x2, y2（原图坐标）
    kps: list[list[float]]             # 5 关键点
    det_score: float
    embedding: Optional[list[float]] = None  # 512 维，L2 归一化后存 npz，JSON 不带


@dataclass
class Attributes:
    """离散属性标签 + 连续值双份（需求定稿 §4.2）。None = 该属性无效（弃权）。"""
    hair_color: Optional[str] = None       # 黑/灰/白/红棕/棕/金黄/蓝/绿/紫/粉
    hair_color_conf: float = 0.0
    skin_level: Optional[int] = None       # 0-19，越大越深（用户需求）
    skin_word: Optional[str] = None        # 白皮肤/黄皮肤/黑皮肤（级别区间的描述词）
    hair_length: Optional[str] = None      # 光头/寸头/短发/中发/长发（五桶，校准轮②定）
    hair_len_d: Optional[float] = None     # 越下巴延伸（脸高单位）
    hair_thickness: Optional[float] = None # 头皮区平均发厚（脸高单位）
    censored: bool = False                 # 发梢被画幅底截断 → 长度"至少"当前桶
    updo_suspect: bool = False             # 盘束疑似（M3 零样本交叉验证）
    hairstyle: Optional[str] = None        # 发型组/类目（M3 SigLIP2 三模式写入）


@dataclass
class ImageRecord:
    """一张图的完整标注结果。"""
    path: str
    content_hash: str
    width: int
    height: int
    face: Optional[FaceInfo]
    attrs: Attributes = field(default_factory=Attributes)
    whole_phash: Optional[str] = None      # 16 位 hex
    face_phash: Optional[str] = None
    unread: bool = False                   # 读取/推理失败
    error: Optional[str] = None
    timings_ms: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
