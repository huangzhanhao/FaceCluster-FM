# -*- coding: utf-8 -*-
"""全局配置：路径、模型、阈值。所有可调参数集中此处（需求定稿 §3/§6）。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _home(*parts: str) -> Path:
    return Path.home().joinpath(*parts)


@dataclass
class Config:
    # ---- 路径 ----
    sample_dir: Path = PROJECT_ROOT / "样本"
    out_dir: Path = PROJECT_ROOT / "out"
    cache_root: Path = _home(".facecluster", "cache")
    models_dir: Path = _home(".facecluster", "models")
    insightface_root: Path = _home(".insightface")

    # ---- 人脸（检测+识别）----
    face_pack: str = "buffalo_s"      # 本地已有；精度不足换 buffalo_l（需下载）
    det_size: int = 0                 # 0 = 从检测模型输入 shape 自动读取
    use_gpu: bool = True              # DirectML；不可用时显式告警（不静默）

    # ---- 掩码 ----
    mask_provider: str = "mediapipe"  # mediapipe | off
    mask_norm: str = "auto"           # auto | unit(0-1) | byte(0-255)

    # ---- 属性阈值（初始值，M0 校准后更新）----
    min_attr_px: int = 48             # 区域最少像素（掩码腐蚀后）
    hair_achroma_sat: float = 0.16    # HSV 饱和度低于此 → 无彩色（黑/灰/白）
    hair_black_L: float = 30.0        # L* 低于此 = 黑
    hair_white_L: float = 62.0        # L* 高于此 = 白
    hue_tol: float = 15.0             # 色相置信度的容差（度）
    skin_top_light_frac: float = 0.5  # 诊断保留：L_top 统计用
    # 肤色 0-19 级（用户需求：数字越大越深；描述词 白皮肤/黄皮肤/黑皮肤 映射区间）
    skin_level_floor: float = 44.0    # eff 分下限（eff=(100-L)+min(C,cap)*credit）
    skin_level_span: float = 48.0     # eff 跨度 [floor, floor+span] → 0..19
    skin_chroma_credit: float = 1.0   # 色度对"深"的加分系数（黑色素信号；校准 round2 由 0.6 提至 1.0）
    skin_chroma_cap: float = 22.0     # 色度加分上限（防高饱和浅肤反超真深肤）
    skin_word_ranges: tuple = (5, 12)  # 白皮肤 ≤5；黄皮肤 6-12；黑皮肤 ≥13（校准轮②用户定）
    min_color_conf: float = 0.45      # 色相置信度低于此 → 发色弃权（暗发色相噪声防护）
    dark_v_floor: float = 0.12        # HSV 明度低于此 → 直接判黑（暗像素色相无意义）
    hair_dark_L: float = 30.0         # L* 低于此走"暗发"分支（色相一致性判棕/黑）
    hair_dark_conf: float = 0.40      # 暗发暖色相一致率 ≥ 此值 → 棕（用户 5 例黑→棕的修复）
    hair_warm_hue: tuple = (12.0, 58.0)  # 暗发判棕的色相窗口（度）
    # 发长五桶：光头/寸头/短发/中发/长发（校准轮②用户定；原"超长"并入长发）
    len_buzz_upper: float = 0.255     # 上头皮平均发厚（脸高单位）< 此值 = 寸头（界于 Asian23 0.253 与 355 0.240 之间）
    len_short_d: float = 0.10         # 越下巴延伸 < 0.10H = 短发
    len_medium_d: float = 0.50        # ≤ 0.50H = 中发
    len_vol_long: float = 1.75        # 发量体积比 ≥ 此值（卷发/爆炸头，d≈0 也算长发；校准②由 1.30 提至 1.75）
    len_wide_long: float = 1.55       # 发区最大行宽/脸宽 ≥ 此值（与体积双条件）
    bald_upper_ratio: float = 0.10    # 上头皮区发像素占比 < 此值 → 光头候选
    bald_upper_thick: float = 0.11    # 且上头皮平均发厚 < 此值 → 光头
    bald_scalp_ratio: float = 0.05    # 头皮区发像素占比 < 此值 = 光头

    # ---- 扫描 ----
    max_side: int = 1600              # 长边上限，超过先缩放
    image_exts: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    # ---- M1 身份聚簇 ----
    face_det_min: float = 0.50        # 检测分低于此 = 坏脸（不可靠嵌入）
    face_min_area_frac: float = 0.02  # 脸框面积占比低于此 = 小脸（kps/嵌入不可靠）
    face_min_side_px: int = 48        # 脸框短边下限
    identity_thr: float = 0.55        # 同人余弦阈值（起始值，含变体包容余量）
    medoid_sample_cap: int = 500      # 簇成员超过此数时采样算 medoid

    # ---- M2 分区/排序/重命名 ----
    key_order: list = field(default_factory=lambda: ["肤色", "发长", "发色"])
    rename_mode: str = "copy"         # copy | inplace
    natural_order: bool = False       # True = 纯自然序号 1.jpg..N.jpg（R8）
    sorted_dir: Path | None = None    # None → 样本目录同级 {name}_sorted

    # ---- M3 发型（SigLIP2 三模式）----
    hairstyle_mode: str = "B"         # A 固定清单 | B 嵌入聚类 | C 示例标注
    hairstyle_k: int | None = None    # 模式 B 组数；None = 轮廓系数自动（[2,12]）
    hair_mask_thr: float = 0.5        # 发区掩码二值化阈值（缓存掩码为 0/1，取 1 的像素）
    siglip_repo: str = "onnx-community/siglip2-base-patch16-224-ONNX"


# ---- 属性键注册表：键名 → (从 Attributes 取值的函数, 桶次序) ----
# 肤色桶次序 = 数值升序（0-19，越大越深）；发型键由 hairstyle.py 运行时注册。
HAIR_COLOR_ORDER: tuple = ("黑", "灰", "白", "红棕", "棕", "金黄", "蓝", "绿", "紫", "粉")
HAIR_LEN_ORDER: tuple = ("光头", "寸头", "短发", "中发", "长发")

KEY_REGISTRY: dict[str, tuple] = {
    "肤色": (lambda a: a.skin_level, None),
    "发长": (lambda a: a.hair_length, HAIR_LEN_ORDER),
    "发色": (lambda a: a.hair_color, HAIR_COLOR_ORDER),
}


def register_hairstyle_key(group_names: list[str]) -> None:
    """发型组作为分区键注册（模式 B 组名 / 模式 A 类目 / 模式 C 自定组）。"""
    KEY_REGISTRY["发型"] = (lambda a: a.hairstyle, tuple(group_names))


def stable_hash(**kw) -> str:
    """参数指纹：工件与生成参数绑定，GUI/CLI 重放时检测参数漂移。"""
    h = hashlib.blake2b(digest_size=8)
    h.update(json.dumps(kw, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return h.hexdigest()
