# -*- coding: utf-8 -*-
"""标注管线：扫描 → 人脸 → 掩码 → 三属性 → pHash → 缓存 → annotations.jsonl。"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..cache_store import load as cache_load
from ..cache_store import save as cache_save
from ..config import Config
from ..io import content_hash, load_bgr, scan_images
from ..types import Attributes, FaceInfo, ImageRecord
from . import hashing
from .attributes import compute_attributes
from .face import FaceEngine
from .masks import MaskEngine


def _rec_from_json(d: dict) -> ImageRecord:
    face = d.get("face")
    if face is not None:
        face["embedding"] = None
        face = FaceInfo(**face)
    attrs = Attributes(**(d.get("attrs") or {}))
    return ImageRecord(
        path=d["path"],
        content_hash=d["content_hash"],
        width=d["width"],
        height=d["height"],
        face=face,
        attrs=attrs,
        whole_phash=d.get("whole_phash"),
        face_phash=d.get("face_phash"),
        unread=d.get("unread", False),
        error=d.get("error"),
        timings_ms=d.get("timings_ms", {}),
    )


def _annotate_one(
    cfg: Config, p: Path, chash: str, face_eng: FaceEngine, mask_eng: MaskEngine, log
) -> tuple[ImageRecord, np.ndarray | None, dict[str, np.ndarray] | None]:
    rec = ImageRecord(path=str(p), content_hash=chash, width=0, height=0, face=None)
    try:
        t0 = time.perf_counter()
        bgr = load_bgr(p, cfg.max_side)
        rec.width, rec.height = bgr.shape[1], bgr.shape[0]
        t_load = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        face = face_eng.largest_face(bgr)
        t_face = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        masks = mask_eng.infer(bgr, face.bbox if face else None)
        t_mask = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        rec.attrs = compute_attributes(
            bgr, masks, face.bbox if face else None, cfg, face.kps if face else None
        )
        t_attr = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        rec.whole_phash = hashing.whole_phash(bgr)
        if face is not None:
            rec.face_phash = hashing.face_region_phash(bgr, face.bbox)
        t_hash = (time.perf_counter() - t0) * 1000

        rec.face = face
        rec.timings_ms = {
            "load": round(t_load, 1),
            "face": round(t_face, 1),
            "mask": round(t_mask, 1),
            "attr": round(t_attr, 1),
            "hash": round(t_hash, 1),
        }

        emb = np.asarray(face.embedding, dtype=np.float32) if face and face.embedding else None
        mq = None
        if masks is not None:
            qh, qw = bgr.shape[0] // 4, bgr.shape[1] // 4
            mq = {
                "hair": cv2resize(masks["hair"], (qw, qh)),
                "face_skin": cv2resize(masks["face_skin"], (qw, qh)),
            }
        cache_save(cfg.cache_root, chash, rec.to_json_dict(), emb, mq)
        return rec, emb, mq
    except Exception as e:  # 单图失败不阻断批次
        rec.unread = True
        rec.error = f"{type(e).__name__}: {e}"
        log(f"✗ {p.name}: {rec.error}")
        return rec, None, None


def cv2resize(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)


def annotate(cfg: Config, limit: int | None = None, log=print) -> Path:
    t_start = time.perf_counter()
    paths = scan_images(cfg.sample_dir)
    if limit:
        paths = paths[:limit]
    log(f"待标注 {len(paths)} 张（目录 {cfg.sample_dir}）")

    face_eng = FaceEngine(cfg, log)
    mask_eng = MaskEngine(cfg, log)

    records: list[ImageRecord] = []
    n_cache = 0
    for i, p in enumerate(paths, 1):
        chash = content_hash(p)
        d, _emb = cache_load(cfg.cache_root, chash)
        if d is not None and not d.get("unread"):
            records.append(_rec_from_json(d))
            n_cache += 1
            continue
        rec, _emb, _mq = _annotate_one(cfg, p, chash, face_eng, mask_eng, log)
        records.append(rec)
        if i % 50 == 0 or i == len(paths):
            el = time.perf_counter() - t_start
            log(f"[{i}/{len(paths)}] {el:.1f}s（缓存命中 {n_cache}）")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = cfg.out_dir / "annotations.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_json_dict(), ensure_ascii=False) + "\n")

    _summary(records, n_cache, time.perf_counter() - t_start, log)
    return jsonl


def _summary(records: list[ImageRecord], n_cache: int, elapsed: float, log) -> None:
    n = len(records)
    if n == 0:
        return
    fresh = [r for r in records if not n_cache or r.timings_ms]
    log(f"—— 标注完成：{n} 张（缓存命中 {n_cache}，新增 {n - n_cache}），耗时 {elapsed:.1f}s ——")
    if fresh:
        for stage in ("load", "face", "mask", "attr", "hash"):
            vals = sorted(r.timings_ms.get(stage, 0) for r in fresh)
            log(f"  {stage:>5}: 中位 {vals[len(vals) // 2]:.1f} ms")
    noface = sum(1 for r in records if r.face is None and not r.unread)
    unread = sum(1 for r in records if r.unread)
    log(f"  有脸 {n - noface - unread} / 无人脸 {noface} / 失败 {unread}")
    for key, title in (("hair_color", "发色"), ("skin_tone", "肤色"), ("hair_length", "发长")):
        c = Counter(getattr(r.attrs, key) or "无效" for r in records if not r.unread)
        log(f"  {title}: {dict(c.most_common())}")
