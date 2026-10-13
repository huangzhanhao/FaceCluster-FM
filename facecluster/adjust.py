# -*- coding: utf-8 -*-
"""人工簇调整：GUI 合并/拆分身份簇 → 重写 clusters.json。

人工微调是会话级修正（纠正识别错分），重跑 cluster 命令会覆盖。
合并/拆分不改变全体成员集合，因此 out/emb_cache.npz 仍然有效，重排秒级。
"""
from __future__ import annotations

import json

import numpy as np

from .cache_store import load as cache_load
from .cluster import _cluster_attr, _medoid, load_annotations
from .config import KEY_REGISTRY, Config
from .hairstyle import _kmeans


def _load_clusters(cfg: Config) -> dict:
    p = cfg.out_dir / "clusters.json"
    if not p.is_file():
        raise FileNotFoundError(f"先运行 cluster 生成 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _write_clusters(cfg: Config, clu: dict) -> None:
    (cfg.out_dir / "clusters.json").write_text(
        json.dumps(clu, ensure_ascii=False, indent=1), encoding="utf-8")


def _renumber(clusters: list[dict]) -> None:
    """簇号全局 1..N（R4）：按大小降序重排后连续编号。"""
    clusters.sort(key=lambda c: -len(c["members"]))
    for i, c in enumerate(clusters, 1):
        c["cid"] = i


def _make_cluster(cfg: Config, members: list[str]) -> dict:
    """由成员 chash 列表重建 medoid 与簇属性（众数，并列取 medoid，R5/R6）。"""
    records = {r.content_hash: r for r in load_annotations(cfg)}
    recs = [records[c] for c in members if c in records]
    embs = [cache_load(cfg.cache_root, c)[1] for c in members]
    pairs = [(r, e) for r, e in zip(recs, embs) if e is not None]
    if pairs:
        E = np.stack([e for _, e in pairs]).astype(np.float32)
        medoid = pairs[_medoid(E, cfg.medoid_sample_cap)][0]
    else:
        medoid = recs[0] if recs else None
    return {
        "members": members,
        "medoid": medoid.content_hash if medoid else None,
        "medoid_path": medoid.path if medoid else None,
        "attrs": {k: _cluster_attr(recs, medoid, k) if medoid else None
                  for k in KEY_REGISTRY if k != "发型"},
    }


def merge(cfg: Config, cids: list[int], log=print) -> None:
    """把 cids[1:] 并入 cids[0]（成员并集，medoid/属性重建），全表重编号。"""
    if len(cids) < 2:
        raise ValueError("合并至少需要两个簇")
    clu = _load_clusters(cfg)
    cls = clu["clusters"]
    keep = next(c for c in cls if c["cid"] == cids[0])
    gone = [c for c in cls if c["cid"] in cids[1:]]
    if len(gone) != len(cids) - 1:
        raise ValueError(f"簇号不存在: {cids}")
    members = sorted({m for c in [keep, *gone] for m in c["members"]})
    rest = [c for c in cls if c["cid"] not in cids]
    merged = _make_cluster(cfg, members)
    clu["clusters"] = rest + [merged]
    _renumber(clu["clusters"])
    _write_clusters(cfg, clu)
    log(f"✓ 合并 {cids} → {len(members)} 图一个簇；全表 {len(clu['clusters'])} 簇已重编号")


def split(cfg: Config, cid: int, k: int = 2, log=print) -> None:
    """把簇按人脸嵌入 k-means 分成 k 组（默认 2），全表重编号。"""
    clu = _load_clusters(cfg)
    cls = clu["clusters"]
    target = next(c for c in cls if c["cid"] == cid)
    if len(target["members"]) < k:
        raise ValueError(f"簇 {cid} 只有 {len(target['members'])} 图，无法拆成 {k} 组")
    embs_map: dict[str, np.ndarray] = {}
    for m in target["members"]:
        _, e = cache_load(cfg.cache_root, m)
        if e is not None:
            embs_map[m] = e
    chashes = sorted(embs_map)
    labels = _kmeans(np.stack([embs_map[c] for c in chashes]).astype(np.float32), k)
    new = []
    assigned = set()
    for g in range(k):
        members = [c for c, lb in zip(chashes, labels) if lb == g]
        assigned.update(members)
        if members:
            new.append(_make_cluster(cfg, members))
    leftovers = [m for m in target["members"] if m not in assigned]  # 无嵌入成员归末组
    if leftovers:
        new[-1]["members"] = sorted(set(new[-1]["members"]) | set(leftovers))
    rest = [c for c in cls if c["cid"] != cid]
    clu["clusters"] = rest + new
    _renumber(clu["clusters"])
    _write_clusters(cfg, clu)
    log(f"✓ 拆分簇 {cid}（{len(target['members'])} 图）→ {len(new)} 组；"
        f"全表 {len(clu['clusters'])} 簇已重编号")
