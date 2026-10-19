# -*- coding: utf-8 -*-
"""M2 排序预览报告：plan.json → sort_report.html + sort_plan.csv（R11 可观测）。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import Config


def _esc(s) -> str:
    return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(cfg: Config) -> tuple[Path, Path]:
    plan_path = cfg.out_dir / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"先运行 order 生成 {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    seq = plan["sequence"]

    # 属性列回查（plan 不冗余存属性）；整图 pHash 供完全重复标记（§6：删否由用户手动）
    attrs_by_chash: dict[str, dict] = {}
    phash_by_chash: dict[str, str] = {}
    jsonl = cfg.out_dir / "annotations.jsonl"
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                attrs_by_chash[d["content_hash"]] = d.get("attrs") or {}
                if d.get("whole_phash"):
                    phash_by_chash[d["content_hash"]] = d["whole_phash"]

    # ---- CSV：全序列（pos, 簇, 段, 原名, 新名, 相似度） ----
    csv_path = cfg.out_dir / "sort_plan.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["位次", "簇号", "段", "原名", "新名", "到medoid相似度"])
        for it in seq:
            w.writerow([
                it["pos"], it["cid"], " → ".join(it["seg"]),
                Path(it["path"]).name, it["new_name"],
                "" if it["sim"] is None else it["sim"],
            ])

    # ---- HTML ----
    fv = plan["fvalue"]
    parts = [
        "<html><head><meta charset='utf-8'><title>FaceCluster-FM 排序预览</title>",
        "<style>body{font-family:'Microsoft YaHei',sans-serif;background:#111;color:#ddd;margin:16px}",
        "h2{border-bottom:1px solid #444;padding-bottom:4px;margin-top:28px}",
        ".g{display:flex;flex-wrap:wrap;gap:8px}",
        ".c{width:150px;text-align:center;font-size:11px;background:#1c1c1c;border-radius:6px;padding:6px}",
        ".c img{width:136px;height:136px;object-fit:cover;border-radius:4px}",
        ".c small{color:#8bd;display:block}",
        "table{border-collapse:collapse}td,th{border:1px solid #444;padding:4px 10px;font-size:13px}</style></head><body>",
        f"<h1>排序预览 · {plan['n_items']} 张 / {plan['n_clusters']} 簇 · 键序 {'→'.join(plan['key_order'])}</h1>",
        "<h2>F 值（段内相邻相似度之和，越大越好）</h2><table><tr><th>层级</th><th>键</th><th>段数</th><th>F</th></tr>",
    ]
    for lv in fv["per_level"]:
        parts.append(f"<tr><td>L{lv['level']}</td><td>{_esc(' → '.join(lv['keys']))}</td>"
                     f"<td>{lv['n_seg']}</td><td>{lv['f']}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>最差相邻对 top-10（叶子段内）</h2><table><tr><th>sim</th><th>图 A</th><th>图 B</th></tr>")
    for w in fv["worst_pairs"]:
        parts.append(f"<tr><td>{w['sim']}</td><td>{_esc(Path(w['a']).name)}</td>"
                     f"<td>{_esc(Path(w['b']).name)}</td></tr>")
    parts.append("</table>")

    # ---- 完全重复（整图 pHash 相同；§6：不做删除，标记后由用户手动处理） ----
    dup_groups: dict[str, list[str]] = {}
    for it in seq:
        ph = phash_by_chash.get(it["chash"])
        if ph:
            dup_groups.setdefault(ph, []).append(it["path"])
    dups = [ps for ps in dup_groups.values() if len(ps) > 1]
    parts.append(f"<h2>完全重复（整图 pHash 相同，{len(dups)} 组）</h2>")
    if dups:
        for ps in dups:
            parts.append("<div class='g'>")
            for p in ps:
                src = Path(p).resolve().as_posix()
                parts.append(f"<div class='c'><img src='file:///{src}' loading='lazy'>"
                             f"<small>{_esc(Path(p).name)}</small></div>")
            parts.append("</div>")
    else:
        parts.append("<p>无</p>")

    parts.append("<h2>输出序列（按段）</h2>")
    cur_seg = None
    for it in seq:
        seg_key = tuple(it["seg"])
        if seg_key != cur_seg:
            cur_seg = seg_key
            parts.append(f"<h3>{_esc(' → '.join(it['seg']))}</h3><div class='g'>")
        a = attrs_by_chash.get(it["chash"], {})
        sk = f"肤{a.get('skin_level')}" if a.get("skin_level") is not None else "肤—"
        src = Path(it["path"]).resolve().as_posix()
        parts.append(
            f"<div class='c'><img src='file:///{src}' loading='lazy'>"
            f"<small>{_esc(it['new_name'])}</small>"
            f"<small>{_esc(Path(it['path']).name)}</small>"
            f"<small>簇{it['cid']} · {_esc(a.get('hair_color') or '—')}·{_esc(a.get('hair_length') or '—')}·{sk}</small></div>"
        )
    parts.append("</div></body></html>")

    html_path = cfg.out_dir / "sort_report.html"
    html_path.write_text("".join(parts), encoding="utf-8")
    return html_path, csv_path
