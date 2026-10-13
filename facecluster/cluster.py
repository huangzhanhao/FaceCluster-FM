# -*- coding: utf-8 -*-
"""M1 身份聚簇：坏脸门控 → 代表点法聚簇 → medoid → 簇属性众数（需求定稿 §2②/R5/R6）。

簇 = 同一个人；属性不参与"谁跟谁一簇"，只决定簇的排列位置（R2）。
代表点（seed）法防相似度传递把不同人串成一簇；收尾 seed 互检合并消除建立顺序偏差。
产出 out/clusters.json，是 M2 分区与 M3 GUI 的唯一聚簇数据源。
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .cache_store import load as cache_load
from .config import KEY_REGISTRY, Config, stable_hash
from .types import ImageRecord

Rec = ImageRecord


def load_annotations(cfg: Config) -> list[Rec]:
    """annotations.jsonl → ImageRecord 列表（嵌入不含在内，按需从缓存取）。"""
    jsonl = cfg.out_dir / "annotations.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"先运行 annotate 生成 {jsonl}")
    from .annotate.pipeline import _rec_from_json

    out = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(_rec_from_json(json.loads(line)))
    return out


def load_embeddings(cfg: Config, chashes: list[str], log=print) -> dict[str, np.ndarray]:
    """按内容哈希批量取 512 维嵌入。"""
    out: dict[str, np.ndarray] = {}
    for c in chashes:
        _, emb = cache_load(cfg.cache_root, c)
        if emb is not None:
            out[c] = emb
    if len(out) < len(chashes):
        log(f"⚠ {len(chashes) - len(out)} 张嵌入缺失（缓存不完整，建议重跑 annotate）")
    return out


def face_quality_fail(rec: Rec, cfg: Config) -> str | None:
    """坏脸门控：返回失败原因，None = 质量可用（§2② 对齐质量门槛）。"""
    f = rec.face
    if f is None:
        return "无人脸"
    if f.det_score < cfg.face_det_min:
        return f"det_score {f.det_score:.2f} < {cfg.face_det_min}"
    x1, y1, x2, y2 = f.bbox
    w, h = x2 - x1, y2 - y1
    if w * h < cfg.face_min_area_frac * rec.width * rec.height:
        return f"脸框过小 {w * h / max(1, rec.width * rec.height):.3f}"
    if min(w, h) < cfg.face_min_side_px:
        return f"脸框短边 {min(w, h):.0f}px"
    # 嵌入缺失在 build() 按缓存判定（JSON 不携带嵌入，恒为 None）
    if len(f.kps) < 5:
        return "关键点缺失"
    # kps 塌缩：任两点过近 / 两眼距过小 / 越出脸框外扩 0.5 边长
    k = np.asarray(f.kps, dtype=np.float32)
    d = np.hypot(k[:, None, 0] - k[None, :, 0], k[:, None, 1] - k[None, :, 1])
    np.fill_diagonal(d, np.inf)  # 对角线是自身距离 0，不参与塌缩判定
    if d.min() < 2.0:
        return "关键点塌缩"
    eye = float(np.hypot(*(k[0] - k[1])))
    if eye < 0.05 * max(1.0, w):
        return f"两眼距过小 {eye:.1f}px"
    pad = 0.5 * max(w, h)
    if (k[:, 0] < x1 - pad).any() or (k[:, 0] > x2 + pad).any() \
            or (k[:, 1] < y1 - pad).any() or (k[:, 1] > y2 + pad).any():
        return "关键点越出脸框"
    return None


def cluster_identities(embs: np.ndarray, det_scores: np.ndarray, thr: float) -> np.ndarray:
    """代表点法：按检测分降序，与已有 seed 最大余弦 ≥ thr 归簇，否则新建 seed。

    收尾对 seed 做并查集合并（seed 间 cos≥thr 的小簇并入大簇）。
    返回 labels[i] = 簇号（0..K-1）。全程不构造 N² 矩阵。
    """
    n = len(embs)
    order = np.argsort(-det_scores)
    labels = np.full(n, -1, dtype=np.int64)
    seed_idx: list[int] = []
    for i in order:
        if seed_idx:
            sims = embs[seed_idx] @ embs[i]
            j = int(np.argmax(sims))
            if float(sims[j]) >= thr:
                labels[i] = j
                continue
        labels[i] = len(seed_idx)
        seed_idx.append(int(i))

    # seed 互检合并（并查集，按簇大小并入）
    parent = list(range(len(seed_idx)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    S = embs[seed_idx]
    sim = S @ S.T
    sizes = np.bincount(labels, minlength=len(seed_idx))
    for a in range(len(seed_idx)):
        for b in range(a + 1, len(seed_idx)):
            if sim[a, b] >= thr:
                ra, rb = find(a), find(b)
                if ra == rb:
                    continue
                big, small = (ra, rb) if sizes[ra] >= sizes[rb] else (rb, ra)
                parent[small] = big
                sizes[big] += sizes[small]
                sizes[small] = 0
    roots = sorted({find(i) for i in range(len(seed_idx))})
    remap = {r: k for k, r in enumerate(roots)}
    labels = np.asarray([remap[find(int(l))] if l >= 0 else -1 for l in labels], dtype=np.int64)
    return labels


def _medoid(E: np.ndarray, cap: int) -> int:
    """簇内 medoid：到其他成员平均余弦最大者（R5）。超过 cap 随机采样。"""
    if len(E) > cap:
        E = E[np.random.default_rng(0).choice(len(E), cap, replace=False)]
    sims = E @ E.T
    return int(np.argmax(sims.mean(axis=1)))


def _cluster_attr(members: list[Rec], medoid_rec: Rec, key: str):
    """簇属性 = 成员众数；并列取 medoid 的属性（R6/§6）；桶次序保证输出稳定。"""
    getter, order = KEY_REGISTRY[key]
    vals = [v for v in (getter(m.attrs) for m in members) if v is not None]
    if not vals:
        return None
    cnt = Counter(vals)
    top = max(cnt.values())
    cands = sorted({v for v, n in cnt.items() if n == top},
                   key=(order.index if order else (lambda x: x)))  # type: ignore
    med = getter(medoid_rec.attrs)
    if len(cands) > 1 and med in cands:
        return med
    return cands[0]


def build(cfg: Config, thr: float | None = None, log=print, progress_cb=None) -> Path:
    """全量聚簇 → out/clusters.json。返回工件路径。"""
    t0 = time.perf_counter()
    thr = cfg.identity_thr if thr is None else thr
    records = load_annotations(cfg)
    log(f"标注 {len(records)} 张，阈值 cos ≥ {thr}")

    unread, no_face, bad_face, have = [], [], [], []
    for r in records:
        if r.unread:
            unread.append(r)
            continue
        if r.face is None:
            no_face.append(r)
            continue
        reason = face_quality_fail(r, cfg)
        if reason:
            bad_face.append((r, reason))
        else:
            have.append(r)

    embs_map = load_embeddings(cfg, [r.content_hash for r in have], log)
    no_emb = [r for r in have if r.content_hash not in embs_map]
    bad_face.extend((r, "嵌入缺失（缓存不完整，重跑 annotate 可修复）") for r in no_emb)
    have = [r for r in have if r.content_hash in embs_map]
    if not have:
        log("⚠ 无可用人脸（检查 annotate 输出），输出空聚簇")
    embs = np.stack([embs_map[r.content_hash] for r in have]).astype(np.float32) \
        if have else np.zeros((0, 512), dtype=np.float32)
    scores = np.asarray([r.face.det_score for r in have], dtype=np.float32)
    labels = cluster_identities(embs, scores, thr) if have else np.zeros(0, dtype=np.int64)

    clusters = []
    for k in range(int(labels.max()) + 1):
        idx = np.where(labels == k)[0]
        members = [have[i] for i in idx]
        m_local = _medoid(embs[idx], cfg.medoid_sample_cap)
        medoid = members[m_local]
        clusters.append({
            "cid": k + 1,
            "members": [m.content_hash for m in members],
            "medoid": medoid.content_hash,
            "medoid_path": medoid.path,
            "attrs": {key: _cluster_attr(members, medoid, key) for key in KEY_REGISTRY},
        })
    clusters.sort(key=lambda c: -len(c["members"]))

    art = cfg.out_dir / "clusters.json"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps({
        "config_hash": stable_hash(thr=thr, face_det_min=cfg.face_det_min),
        "thr": thr,
        "n_images": len(records),
        "clusters": clusters,
        "tail": {
            "bad_face": [{"chash": r.content_hash, "path": r.path, "reason": reason}
                         for r, reason in bad_face],
            "no_face": [{"chash": r.content_hash, "path": r.path} for r in no_face],
            "unread": [{"chash": r.content_hash, "path": r.path, "error": r.error}
                       for r in unread],
        },
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    sizes = sorted((len(c["members"]) for c in clusters), reverse=True)
    log(f"—— 聚簇完成 {time.perf_counter() - t0:.1f}s：{len(clusters)} 簇 / 尾部 坏脸 {len(bad_face)}"
        f" 无人脸 {len(no_face)} 失败 {len(unread)} ——")
    log(f"  簇大小 top10: {sizes[:10]}")
    log(f"  单图簇 {sum(1 for s in sizes if s == 1)} 个")
    if bad_face:
        for r, reason in bad_face[:5]:
            log(f"  坏脸: {Path(r.path).name} —— {reason}")
    return art
