# -*- coding: utf-8 -*-
"""DirectML 可用性自检（M0 简版）：
1) ORT DML provider 枚举
2) 掩码模型在 DML 上建会话 + 跑一张
3) insightface 检测/识别在 DML 上建会话
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run(cfg) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import onnxruntime as ort

    results: list[tuple[str, bool, str]] = []

    dml = "DmlExecutionProvider" in ort.get_available_providers()
    results.append(("ORT 枚举 DmlExecutionProvider", dml, f"providers={ort.get_available_providers()}"))
    if not dml:
        for name, ok, note in results:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} {note}")
        print("修复：pip install onnxruntime-directml==1.24.4（新版本有 DML 枚举回归），"
              "并确保 onnxruntime 与 onnxruntime-directml 不并存")
        return 1

    # 2) 掩码模型
    try:
        mp = cfg.models_dir / "selfie_multiclass_256x256.onnx"
        s = ort.InferenceSession(str(mp), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        x = np.random.rand(1, 256, 256, 3).astype(np.float32)
        s.run([s.get_outputs()[0].name], {s.get_inputs()[0].name: x})
        ok = s.get_providers()[0] == "DmlExecutionProvider"
        results.append(("掩码模型 DML 推理", ok, f"active={s.get_providers()[0]}"))
    except Exception as e:
        results.append(("掩码模型 DML 推理", False, str(e)[:120]))

    # 3) insightface
    try:
        from facecluster.annotate.face import FaceEngine

        eng = FaceEngine(cfg, log=lambda *_: None)
        img = (np.random.rand(352, 352, 3) * 255).astype(np.uint8)
        eng.largest_face(img)
        det = eng.app.models["detection"]
        sess = getattr(det, "session", None) or getattr(det, "det_session", None)
        active = sess.get_providers()
        ok = active[0] == "DmlExecutionProvider"
        results.append(("人脸检测/识别 DML", ok, f"active={active}"))
    except Exception as e:
        results.append(("人脸检测/识别 DML", False, str(e)[:120]))

    for name, ok, note in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {note}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    from facecluster.config import Config

    raise SystemExit(run(Config()))
