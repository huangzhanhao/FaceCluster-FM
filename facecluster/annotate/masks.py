# -*- coding: utf-8 -*-
"""区域掩码引擎：MediaPipe multiclass selfie segmentation（需求定稿 §4.1 主档）。

6 类：0 背景 / 1 头发 / 2 身体皮肤 / 3 面部皮肤 / 4 衣服 / 5 其他。
输入 NHWC float32 [1,256,256,3]；归一化(0-1 或 0-255)按 mask_norm，auto 时
用第一张有人脸的图探测（面部皮肤类与人脸框的交叠比大者胜）。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import Config

MODEL_NAME = "selfie_multiclass_256x256.onnx"
_IN = "input_29"
_OUT = "Identity"


class MaskEngine:
    def __init__(self, cfg: Config, log=print):
        self.cfg = cfg
        self.log = log
        self.ok = False
        self.norm: str | None = None
        if cfg.mask_provider != "mediapipe":
            log("掩码引擎已关闭（mask_provider=off），属性将全部无效")
            return
        path = cfg.models_dir / MODEL_NAME
        if not path.is_file():
            log(f"⚠ 缺少掩码模型 {path} —— 属性将无效（可从 HF senty-au/selfie_multiclass_256x256-ONNX 下载）")
            return
        import onnxruntime as ort

        providers = (
            ["DmlExecutionProvider", "CPUExecutionProvider"]
            if cfg.use_gpu
            else ["CPUExecutionProvider"]
        )
        self.sess = ort.InferenceSession(str(path), providers=providers)
        if cfg.use_gpu and self.sess.get_providers()[0] != "DmlExecutionProvider":
            log(f"⚠ 掩码模型实际挂在 {self.sess.get_providers()[0]}（要求 DML）")
        else:
            log("✓ 区域掩码：DML（MediaPipe multiclass 6 类）")
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.ok = True
        self.norm = None if cfg.mask_norm == "auto" else cfg.mask_norm

    # ---- 归一化探测：auto 时只在第一张有人脸的图上跑一次 ----
    def autotune_norm(self, bgr: np.ndarray, bbox: list[float]) -> None:
        if self.norm is not None:
            return
        h, w = bgr.shape[:2]
        # 脸框是原图坐标 → 缩到 256×256 模型空间
        sx, sy = 256.0 / w, 256.0 / h
        x1, y1, x2, y2 = bbox
        bx1, by1 = max(0, int(x1 * sx)), max(0, int(y1 * sy))
        bx2, by2 = min(256, int(x2 * sx)), min(256, int(y2 * sy))
        best, best_score = None, -1.0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        base = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
        for cand in ("unit", "byte"):
            x = base / 255.0 if cand == "unit" else base
            logits = self.sess.run([self.out_name], {self.in_name: x[None]})[0][0]
            if float(np.abs(logits).max()) > 1e4:  # 荒谬 logits = 预处理错误，直接排除
                continue
            if bx2 <= bx1 or by2 <= by1:
                continue
            share = float((logits.argmax(-1)[by1:by2, bx1:bx2] == 3).mean())
            if share > best_score:
                best, best_score = cand, share
        self.norm = best or "unit"
        self.log(f"掩码归一化探测：{self.norm}（脸框内面部皮肤占比 {best_score:.2f}）")

    def _infer_classmap(self, bgr: np.ndarray, norm: str) -> np.ndarray | None:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
        if norm == "unit":
            x /= 255.0
        logits = self.sess.run([self.out_name], {self.in_name: x[None]})[0]  # [1,256,256,6]
        return logits[0].argmax(-1).astype(np.uint8)

    def infer(self, bgr: np.ndarray, face_bbox: list[float] | None) -> dict[str, np.ndarray] | None:
        """返回 hair / body_skin / face_skin（原图分辨率 bool）与 class_map(uint8)。"""
        if not self.ok:
            return None
        if self.norm is None:
            if face_bbox is None:
                return None
            self.autotune_norm(bgr, face_bbox)
        cls = self._infer_classmap(bgr, self.norm or "unit")
        if cls is None:
            return None
        h, w = bgr.shape[:2]
        full = cv2.resize(cls, (w, h), interpolation=cv2.INTER_NEAREST)
        return {
            "hair": full == 1,
            "body_skin": full == 2,
            "face_skin": full == 3,
            "class_map": full,
        }
