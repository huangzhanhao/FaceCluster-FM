# -*- coding: utf-8 -*-
"""肤色混淆交叉核验表（需求 §4.2 已知边界：阴影浅肤 vs 亮光深肤可能同档）。

按 0-19 级分组的全量缩略图 + 词表分界带（5/6 白→黄、12/13 黄→黑）重点放大，
供人工核对分桶是否有系统性混淆。产出 out/skin_audit.html。
用法：python -m tools.skin_audit
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from facecluster.config import Config


def _esc(s) -> str:
    return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(cfg: Config) -> Path:
    jsonl = cfg.out_dir / "annotations.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"先运行 annotate 生成 {jsonl}")
    by_level: dict[int, list[dict]] = defaultdict(list)
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("unread"):
            continue
        a = d.get("attrs") or {}
        if a.get("skin_level") is None:
            continue
        by_level[int(a["skin_level"])].append(d)

    def card(d: dict) -> str:
        src = Path(d["path"]).resolve().as_posix()
        a = d["attrs"]
        return (f"<div class='c'><img src='file:///{src}' loading='lazy'>"
                f"<small>{_esc(Path(d['path']).name)}</small>"
                f"<small>级{a.get('skin_level')}·{_esc(a.get('skin_word') or '')}</small></div>")

    parts = [
        "<html><head><meta charset='utf-8'><title>肤色交叉核验</title>",
        "<style>body{font-family:'Microsoft YaHei',sans-serif;background:#111;color:#ddd;margin:16px}",
        "h2{color:#9cf;border-bottom:1px solid #444;padding-bottom:4px}",
        "h3{color:#fc9;margin-top:24px}",
        ".g{display:flex;flex-wrap:wrap;gap:8px}",
        ".c{width:110px;text-align:center;font-size:11px;background:#1c1c1c;border-radius:6px;padding:6px}",
        ".c img{width:98px;height:98px;object-fit:cover;border-radius:4px}</style></head><body>",
        f"<h1>肤色交叉核验 · {sum(len(v) for v in by_level.values())} 张有效</h1>",
        "<p>核对要点：①同级内是否混入明显不同深浅（阴影/亮光混淆）；"
        "②相邻级边界是否符合单调性；③词表分界带（级5/6 白→黄、级12/13 黄→黑）样本是否贴切。</p>",
    ]

    # 词表分界带优先放大
    lo, hi = cfg.skin_word_ranges
    for band, members in (
        (f"分界带：级{lo}（白皮肤末）与级{lo + 1}（黄皮肤首）",
         by_level.get(lo, []) + by_level.get(lo + 1, [])),
        (f"分界带：级{hi}（黄皮肤末）与级{hi + 1}（黑皮肤首）",
         by_level.get(hi, []) + by_level.get(hi + 1, [])),
    ):
        parts.append(f"<h2>{_esc(band)}（{len(members)} 张）</h2><div class='g'>")
        parts.extend(card(d) for d in members)
        parts.append("</div>")

    # 全量按级分组（级内按脸框亮度无——annotations 未存 eff，按文件名自然序）
    parts.append("<h2>全量 0-19 级分组</h2>")
    for lv in range(20):
        members = by_level.get(lv) or []
        if not members:
            continue
        word = members[0]["attrs"].get("skin_word") or ""
        parts.append(f"<h3>级 {lv} · {_esc(word)}（{len(members)} 张）</h3><div class='g'>")
        parts.extend(card(d) for d in members)
        parts.append("</div>")

    parts.append("</body></html>")
    out = cfg.out_dir / "skin_audit.html"
    out.write_text("".join(parts), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build(Config()))
