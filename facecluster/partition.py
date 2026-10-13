# -*- coding: utf-8 -*-
"""M2 硬分区+排序：键序层级分区 → 段内贪心最近邻链 → 簇内相似度排序 → out/plan.json。

- 硬分区（R4）：同属性值的所有簇编号连续；簇号全局 1..N，不补零（R12）。
- 簇内排序（R5）：到 medoid 余弦降序；并列 face_pHash 汉明距裁决（R7）；再并列原名序。
- 尾部（R9/§6）：序列末两段相连——有脸但属性无效的簇 → 无脸段（no_face+bad_face+unread
  合一组内不细分，按原扫描序）。
- F 值（R11）：每级段内相邻图余弦之和 + 段内最差相邻对 top-10。
改键序只重算本模块（纯内存 + 嵌入磁盘缓存），秒级重放。
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .config import KEY_REGISTRY, Config, register_hairstyle_key, stable_hash
from .io import IMAGE_EXTS

INVALID_SEG = "属性无效"
NOFACE_SEG = "无脸"


def natural_key(s: str):
    """Windows 资源管理器式自然序（数字段按数值比较）。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _bucket_index(key: str, value, order) -> int:
    """桶在固定次序表中的位置；无次序表的键（肤色）按数值升序。"""
    if order is None:
        return int(value)
    return order.index(value)


def _load_plan_inputs(cfg: Config, log=print):
    """clusters.json + annotations.jsonl + 全量嵌入（含 out/emb_cache.npz 磁盘缓存）。"""
    clu_path = cfg.out_dir / "clusters.json"
    if not clu_path.is_file():
        raise FileNotFoundError(f"先运行 cluster 生成 {clu_path}")
    clu = json.loads(clu_path.read_text(encoding="utf-8"))

    from .cluster import load_annotations

    records = load_annotations(cfg)
    by_chash = {r.content_hash: r for r in records}

    chashes = sorted(m for c in clu["clusters"] for m in c["members"])
    emb_cache = cfg.out_dir / "emb_cache.npz"
    embs: dict[str, np.ndarray] = {}
    if emb_cache.is_file():
        z = np.load(emb_cache, allow_pickle=False)
        if z["chashes"].tolist() == chashes:
            for c, e in zip(chashes, z["embs"]):
                embs[c] = e
            log(f"嵌入缓存命中 {len(chashes)} 条（{emb_cache.name}）")
    if not embs:
        from .cluster import load_embeddings

        embs = load_embeddings(cfg, chashes, log)
        if embs:
            np.savez_compressed(
                emb_cache,
                chashes=np.asarray(chashes),
                embs=np.stack([embs[c] for c in chashes]).astype(np.float32),
            )
            log(f"嵌入磁盘缓存已写 {emb_cache.name}（下次改键序免读逐图 npz）")
    return clu, by_chash, embs


def _chain(med_embs: dict[int, np.ndarray], start: int) -> list[int]:
    """贪心最近邻链：从 start 出发每次走向最近的未访问 medoid。"""
    remaining = set(med_embs)
    out: list[int] = []
    cur = start
    while remaining:
        out.append(cur)
        remaining.discard(cur)
        if not remaining:
            break
        v = med_embs[cur]
        cur = max(remaining, key=lambda c: float(med_embs[c] @ v))
    return out


def _hamming(a: str | None, b: str | None) -> int:
    if not a or not b:
        return 1 << 30
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _overlay_hairstyle(cfg: Config, clu: dict, log=print) -> None:
    """发型键叠加：hairstyle.json 的图级标签 → 簇级众数（并列取组序靠前者）。

    簇内无任何有效标签的簇 attrs["发型"]=None，自然落入"属性无效"段（R4 尾部规则）。
    必须在 KEY_REGISTRY 校验前调用——"发型"是运行时注册键。
    """
    art = cfg.out_dir / "hairstyle.json"
    if not art.is_file():
        raise FileNotFoundError(f"分区键含'发型'需先运行 hairstyle 生成 {art}")
    hs = json.loads(art.read_text(encoding="utf-8"))
    register_hairstyle_key(hs["group_order"])
    tag = {c: g for g, ms in hs["groups"].items() for c in ms}
    order = hs["group_order"]
    for c in clu["clusters"]:
        vals = [tag[m] for m in c["members"] if m in tag]
        if not vals:
            c["attrs"]["发型"] = None
            continue
        cnt = Counter(vals)
        top = max(cnt.values())
        cands = [g for g in order if cnt.get(g, 0) == top]
        c["attrs"]["发型"] = cands[0] if cands else min(cnt, key=cnt.get)
    log(f"发型键已叠加：{sum(1 for c in clu['clusters'] if c['attrs']['发型'])}"
        f"/{len(clu['clusters'])} 簇有效（来自 {art.name}）")


def build(cfg: Config, keys: list[str] | None = None, log=print) -> Path:
    """全量分区+排序 → out/plan.json。返回工件路径。"""
    t0 = time.perf_counter()
    keys = list(cfg.key_order if keys is None else keys)
    clu, by_chash, embs = _load_plan_inputs(cfg, log)
    if "发型" in keys:
        _overlay_hairstyle(cfg, clu, log)  # 注册"发型"键 + 簇属性覆盖，须在校验前
    for k in keys:
        if k not in KEY_REGISTRY:
            raise ValueError(f"未知分区键 {k}（可用：{list(KEY_REGISTRY)}；发型键需先跑 hairstyle）")

    # 簇 → 键值桶序（任一键无效 → 整簇进"属性无效"段，硬分区要求完整键路径）
    def kv(c: dict) -> list:
        return [c["attrs"].get(k) for k in keys]

    valid, invalid = [], []
    for c in clu["clusters"]:
        (valid if all(v is not None for v in kv(c)) else invalid).append(c)

    segs: list[dict] = []  # {seg:[键值...], cids:[cid], sort_key:tuple}
    if valid:
        orders = [KEY_REGISTRY[k][1] for k in keys]
        for c in valid:
            c["_sort"] = tuple(
                _bucket_index(k, v, o) for k, v, o in zip(keys, kv(c), orders)
            )
        valid.sort(key=lambda c: c["_sort"])
        cur_key, bucket = None, []
        for c in valid:
            key = tuple(kv(c))
            if key != cur_key and bucket:
                segs.append({"seg": list(cur_key), "clusters": bucket})
                bucket = []
            cur_key = key
            bucket.append(c)
        if bucket:
            segs.append({"seg": list(cur_key), "clusters": bucket})

    # medoid 嵌入 / 检测分（链起点用）
    med_emb = {c["cid"]: embs[c["medoid"]] for c in clu["clusters"] if c["medoid"] in embs}
    med_det = {
        c["cid"]: (by_chash[c["medoid"]].face.det_score if by_chash[c["medoid"]].face else 0.0)
        for c in clu["clusters"]
    }

    # 段排序：段内贪心链；起点 = 首段取检测分最高，其余衔接上一段末簇
    ordered_segs: list[dict] = []
    prev_last: int | None = None
    for s in segs:
        cids = [c["cid"] for c in s["clusters"]]
        me = {cid: med_emb[cid] for cid in cids if cid in med_emb}
        if not me:
            ordered_segs.append({**s, "order": cids, "chain": False})
            continue
        start = prev_last if (prev_last in me) else max(me, key=lambda c: med_det.get(c, 0.0))
        ordered_segs.append({**s, "order": _chain(me, start), "chain": True})
        prev_last = ordered_segs[-1]["order"][-1]

    # 属性无效段：簇间也走链，保持人相邻
    invalid_ordered: list[int] = []
    if invalid:
        me = {c["cid"]: med_emb[c["medoid"]] for c in invalid if c["medoid"] in med_emb}
        start = max(me, key=lambda c: med_det.get(c, 0.0)) if me else None
        invalid_ordered = _chain(me, start) if me else [c["cid"] for c in invalid]

    # ---- 展开成条目序列 ----
    clu_by_cid = {c["cid"]: c for c in clu["clusters"]}
    sequence: list[dict] = []

    def emit_cluster(cid: int, seg_display: list[str], cluster_no: int) -> None:
        c = clu_by_cid[cid]
        members = list(c["members"])
        mvec = embs.get(c["medoid"])
        med_rec = by_chash[c["medoid"]]

        def sort_key(chash: str):
            e = embs.get(chash)
            sim = float(e @ mvec) if e is not None and mvec is not None else -1.0
            rec = by_chash[chash]
            return (-sim, _hamming(rec.face_phash, med_rec.face_phash), natural_key(rec.path))

        members.sort(key=sort_key)
        for i, chash in enumerate(members, 1):
            rec = by_chash[chash]
            e = embs.get(chash)
            sequence.append({
                "chash": chash,
                "path": rec.path,
                "cid": cluster_no,
                "seg": seg_display,
                "sim": round(float(e @ mvec), 4) if e is not None and mvec is not None else None,
                "name_hint": f"{cluster_no}-{i}",
            })

    next_cid = 1
    for s in ordered_segs:
        for cid in s["order"]:
            emit_cluster(cid, [str(v) for v in s["seg"]], next_cid)
            next_cid += 1
    for cid in invalid_ordered:
        emit_cluster(cid, [INVALID_SEG], next_cid)
        next_cid += 1

    # 无脸尾段：no_face + bad_face + unread 合一，组内按原扫描序（R9）
    tail: list[dict] = []
    for group in ("no_face", "bad_face", "unread"):
        tail.extend(clu["tail"].get(group) or [])
    seen: set[str] = set()
    tail_items = []
    for t in tail:
        if t["chash"] in seen:
            continue
        seen.add(t["chash"])
        tail_items.append(t)
    tail_items.sort(key=lambda t: natural_key(t["path"]))
    noface_cid = next_cid
    for t in tail_items:
        sequence.append({
            "chash": t["chash"], "path": t["path"], "cid": noface_cid,
            "seg": [NOFACE_SEG], "sim": None, "name_hint": None,
        })

    # ---- 命名（R8/R12）----
    n_valid = sum(1 for it in sequence if it["seg"] != [NOFACE_SEG])
    for pos, it in enumerate(sequence, 1):
        ext = Path(it["path"]).suffix or ".jpg"
        if cfg.natural_order:
            it["new_name"] = f"{pos}{ext}"
        elif it["name_hint"]:
            it["new_name"] = f"{it['name_hint']}{ext}"
        else:
            it["new_name"] = f"{noface_cid}-{pos - n_valid}{ext}"
        p = Path(it["path"])
        try:
            st = p.stat()
            it["size"], it["mtime"] = st.st_size, st.st_mtime
        except OSError:
            it["size"], it["mtime"] = None, None
        it["pos"] = pos
        for drop in ("name_hint",):
            it.pop(drop, None)

    # ---- F 值（R11）：每级段内相邻 cos 之和 + 叶子级最差相邻对 top-10 ----
    def level_f(level: int) -> tuple[float, int]:
        groups: dict[tuple, list] = {}
        for it in sequence:
            if it["seg"][0] in (INVALID_SEG, NOFACE_SEG):
                continue
            groups.setdefault(tuple(it["seg"][:level]), []).append(it)
        total = 0.0
        for items in groups.values():
            for a, b in zip(items, items[1:]):
                ea, eb = embs.get(a["chash"]), embs.get(b["chash"])
                if ea is not None and eb is not None:
                    total += float(ea @ eb)
        return total, len(groups)

    per_level = [
        {"level": i + 1, "keys": keys[: i + 1], "f": round(f, 2), "n_seg": n}
        for i, (f, n) in enumerate(level_f(i + 1) for i in range(len(keys)))
    ]
    worst = []
    leaf_groups: dict[tuple, list] = {}
    for it in sequence:
        if it["seg"][0] in (INVALID_SEG, NOFACE_SEG):
            continue
        leaf_groups.setdefault(tuple(it["seg"]), []).append(it)
    for items in leaf_groups.values():
        for a, b in zip(items, items[1:]):
            ea, eb = embs.get(a["chash"]), embs.get(b["chash"])
            if ea is not None and eb is not None:
                worst.append({"a": a["path"], "b": b["path"], "sim": round(float(ea @ eb), 4)})
    worst.sort(key=lambda w: w["sim"])
    worst = worst[:10]
    overall_f = per_level[-1]["f"] if per_level else 0.0

    plan = {
        "config_hash": stable_hash(
            keys=keys, natural=cfg.natural_order, rename_mode=cfg.rename_mode,
            cluster_hash=clu["config_hash"],
        ),
        "key_order": keys,
        # 无脸尾段自身占一个簇号（命名 {cid}-i），计入 N 才满足 R4 簇号连续 1..N
        "n_clusters": noface_cid if tail_items else next_cid - 1,
        "n_items": len(sequence),
        "sequence": sequence,
        "fvalue": {"per_level": per_level, "overall_f": overall_f, "worst_pairs": worst},
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.out_dir / "plan.json"
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")

    log(f"—— 分区排序完成 {time.perf_counter() - t0:.1f}s：{plan['n_clusters']} 簇 "
        f"{len(sequence)} 条目，叶子段 {len(ordered_segs)} + 无效 {len(invalid_ordered)} + 无脸 {len(tail_items)} ——")
    for lv in per_level:
        log(f"  F(L{lv['level']} {'→'.join(lv['keys'])}): {lv['f']:.1f}（{lv['n_seg']} 段）")
    if worst:
        w = worst[0]
        log(f"  最差相邻对: {Path(w['a']).name} ↔ {Path(w['b']).name} cos={w['sim']}")
    return out
