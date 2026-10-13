# -*- coding: utf-8 -*-
"""接触表 HTML：按 发色 → 发长 分组缩略图，供人工抽检校准（M0 出口条件）。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from facecluster.config import Config


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(cfg: Config) -> Path:
    jsonl = cfg.out_dir / "annotations.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"先运行 annotate 生成 {jsonl}")
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

    ok = [r for r in records if not r.get("unread")]
    unread = [r for r in records if r.get("unread")]

    parts = [
        "<html><head><meta charset='utf-8'><title>FaceCluster-FM 标注接触表</title>",
        "<style>body{font-family:'Microsoft YaHei',sans-serif;background:#111;color:#ddd;margin:16px}",
        "h2{border-bottom:1px solid #444;padding-bottom:4px;margin-top:28px}",
        "h3{color:#9cf;margin:18px 0 6px}",
        ".g{display:flex;flex-wrap:wrap;gap:8px}",
        ".c{width:150px;text-align:center;font-size:11px;background:#1c1c1c;border-radius:6px;padding:6px}",
        ".c img{width:136px;height:136px;object-fit:cover;border-radius:4px}",
        ".c small{color:#8bd;display:block}",
        "table{border-collapse:collapse}td,th{border:1px solid #444;padding:4px 10px;font-size:13px}</style></head><body>",
        f"<h1>标注接触表 · {len(records)} 张</h1>",
    ]

    # 统计
    parts.append("<h2>分布</h2><table><tr><th>发色</th><th>肤色级</th><th>发长</th><th>计数</th></tr>")
    combo = Counter(
        (
            r["attrs"].get("hair_color") or "无效",
            (f"{r['attrs'].get('skin_level')}{r['attrs'].get('skin_word') or ''}"
             if r["attrs"].get("skin_level") is not None else "无效"),
            r["attrs"].get("hair_length") or "无效",
        )
        for r in ok
    )
    for (hc, st, hl), n in combo.most_common():
        parts.append(f"<tr><td>{_esc(hc)}</td><td>{_esc(st)}</td><td>{_esc(hl)}</td><td>{n}</td></tr>")
    parts.append("</table>")

    def card(r) -> str:
        src = Path(r["path"]).resolve().as_posix()
        a = r["attrs"]
        cap = f"{_esc(Path(r['path']).name)}"
        sk = f"肤{a.get('skin_level')}{a.get('skin_word') or ''}" if a.get("skin_level") is not None else "肤无效"
        sub = (
            f"{_esc(a.get('hair_color') or '—')}·{a.get('hair_color_conf', 0):.2f} | "
            f"{_esc(a.get('hair_length') or '—')} d={a.get('hair_len_d') if a.get('hair_len_d') is not None else '—'} | "
            f"{sk}"
        )
        return f"<div class='c'><img src='file:///{src}' loading='lazy'><small>{cap}</small><small>{sub}</small></div>"

    # 肤色级 → 发长 → 发色 三级分组（用户默认键序：肤色→发长→发色）
    by_skin: dict[int, list] = {}
    for r in ok:
        by_skin.setdefault(r["attrs"].get("skin_level") if r["attrs"].get("skin_level") is not None else -1, []).append(r)
    parts.append("<h2>按 肤色级 → 发长 → 发色 分组</h2>")
    for sl in sorted(by_skin):
        rs = by_skin[sl]
        title = f"肤色级 {sl}" if sl >= 0 else "肤色无效"
        word = next((r["attrs"].get("skin_word") for r in rs if r["attrs"].get("skin_word")), "")
        parts.append(f"<h2>{_esc(title)} {_esc(word or '')}（{len(rs)}）</h2>")
        by_len: dict[str, list] = {}
        for r in rs:
            by_len.setdefault(r["attrs"].get("hair_length") or "无效", []).append(r)
        for hl, rs2 in sorted(by_len.items(), key=lambda kv: -len(kv[1])):
            parts.append(f"<h3>{_esc(hl)}（{len(rs2)}）</h3><div class='g'>")
            parts.extend(card(r) for r in rs2)
            parts.append("</div>")

    if unread:
        parts.append(f"<h2>失败（{len(unread)}）</h2><ul>")
        parts.extend(f"<li>{_esc(Path(r['path']).name)}: {_esc(r.get('error') or '')}</li>" for r in unread)
        parts.append("</ul>")

    parts.append("</body></html>")
    out = cfg.out_dir / "contact_sheet.html"
    out.write_text("".join(parts), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build(Config()))
