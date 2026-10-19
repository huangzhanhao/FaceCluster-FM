# -*- coding: utf-8 -*-
"""人脸检测+识别引擎（InsightFace）。

- 多人合影取最大脸（需求定稿 §6）。
- det 输入尺寸从模型文件自动读取（本地 det_500m 已是固定 352）。
- provider 断言：要求 DML 时若实际挂在 CPU，打印显著告警，不静默。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Config
from ..types import FaceInfo


def _read_onnx_input_hw(onnx_path: Path) -> tuple[int, int] | None:
    try:
        import onnx

        m = onnx.load(str(onnx_path))
        dims = m.graph.input[0].type.tensor_type.shape.dim
        vals = [(d.dim_value or 0) for d in dims]
        if len(vals) == 4 and vals[2] > 0 and vals[3] > 0:
            return vals[2], vals[3]
    except Exception:
        pass
    return None


class FaceEngine:
    def __init__(self, cfg: Config, log=print):
        from insightface.app import FaceAnalysis

        self.cfg = cfg
        self.log = log
        providers = (
            ["DmlExecutionProvider", "CPUExecutionProvider"]
            if cfg.use_gpu
            else ["CPUExecutionProvider"]
        )
        pack_dir = cfg.insightface_root / "models" / cfg.face_pack
        if not pack_dir.is_dir():
            raise FileNotFoundError(f"未找到模型包目录: {pack_dir}")

        det_hw = None
        for p in sorted(pack_dir.glob("det_*.onnx")):
            det_hw = _read_onnx_input_hw(p)
            if det_hw:
                break
        # cfg.det_size 是方形边长（int），必须归一成 (h, w) 元组——动态 shape 模型也靠它
        # 显式指定输入尺寸，不必去改模型文件
        det_size = (cfg.det_size, cfg.det_size) if cfg.det_size else (det_hw or (640, 640))
        if det_hw is None:
            log(f"⚠ 检测模型输入为动态 shape，det_size 用默认 {det_size}；DML 可能不兼容")

        self.app = FaceAnalysis(
            root=str(cfg.insightface_root),
            name=cfg.face_pack,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_size=(int(det_size[0]), int(det_size[1])))

        det = self.app.models.get("detection")
        det_sess = getattr(det, "session", None) or getattr(det, "det_session", None)
        if det is not None and cfg.use_gpu and det_sess is not None:
            active = det_sess.get_providers()
            if active[0] != "DmlExecutionProvider":
                log(f"⚠ 人脸检测实际挂在 {active[0]}（要求 DML）—— 性能差 35~50 倍，请检查")
            else:
                log(f"✓ 人脸检测/识别：DML（det 输入 {det_size[0]}×{det_size[1]}）")

    def largest_face(self, bgr: np.ndarray) -> FaceInfo | None:
        """取最大脸。embedding 为 L2 归一化后向量。"""
        faces = self.app.get(bgr)
        if not faces:
            return None
        f = max(faces, key=lambda x: float(x.det_score) * float(x.bbox[2] - x.bbox[0]) * float(x.bbox[3] - x.bbox[1]))
        emb = getattr(f, "normed_embedding", None)
        return FaceInfo(
            bbox=[float(v) for v in f.bbox],
            kps=[[float(a), float(b)] for a, b in f.kps] if f.kps is not None else [],
            det_score=float(f.det_score),
            embedding=emb.tolist() if emb is not None else None,
        )
