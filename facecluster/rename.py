# -*- coding: utf-8 -*-
"""M2 重命名：copy（默认，原包不动）/ inplace 两阶段改名 + 失败自动回滚（R8）。

plan.json 是唯一输入；journal 是 inplace 的回滚依据（先写 journal 再动文件）。
copy 冲突 = 目标名已存在且内容不同（视为他人文件，拒绝）；同名同大小视为上次输出，覆盖。
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

from .config import Config

JOURNAL = "rename_journal.json"


def load_plan(cfg: Config) -> dict:
    p = cfg.out_dir / "plan.json"
    if not p.is_file():
        raise FileNotFoundError(f"先运行 order 生成 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def target_dir(cfg: Config) -> Path:
    if cfg.sorted_dir:
        return Path(cfg.sorted_dir)
    return cfg.sample_dir.parent / f"{cfg.sample_dir.name}_sorted"


def apply_natural(plan: dict) -> None:
    """--natural：忽略 plan 内命名，按全局位次 1.jpg..N.jpg（R8 纯自然序模式）。"""
    for it in plan["sequence"]:
        it["new_name"] = f"{it['pos']}{Path(it['path']).suffix or '.jpg'}"


def _drifted(it: dict) -> str | None:
    """plan 与磁盘漂移快检（size/mtime 任一不符即跳过，不动文件）。"""
    if it.get("size") is None:
        return "plan 中无 stat 基线"
    try:
        st = Path(it["path"]).stat()
    except OSError as e:
        return f"源文件不可达 {e}"
    if st.st_size != it["size"] or st.st_mtime != it["mtime"]:
        return f"size/mtime 漂移（plan {it['size']}/{it['mtime']} vs 磁盘 {st.st_size}/{st.st_mtime}）"
    return None


def check_conflicts(cfg: Config, plan: dict) -> list[str]:
    """执行前置冲突检查，返回冲突描述列表（非空则拒绝执行）。"""
    seq = plan["sequence"]
    involved = {str(Path(it["path"]).resolve()) for it in seq}
    conflicts: list[str] = []

    if cfg.rename_mode == "copy":
        tdir = target_dir(cfg)
        if tdir.exists():
            for it in seq:
                dst = tdir / it["new_name"]
                if dst.is_file():
                    try:
                        same = dst.stat().st_size == Path(it["path"]).stat().st_size
                    except OSError:
                        same = False
                    if not same:
                        conflicts.append(f"copy 目标被占: {dst.name}（内容与源不同）")
    else:  # inplace：新名不得与包内未参与文件重名
        sample_files = {
            str(p.resolve()) for p in cfg.sample_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in cfg.image_exts
        }
        outsiders = {p for p in sample_files if p not in involved}
        by_name: dict[str, str] = {}
        for p in outsiders:
            by_name.setdefault(Path(p).name.lower(), p)
        for it in seq:
            hit = by_name.get(it["new_name"].lower())
            if hit and str(Path(it["path"]).resolve()) != hit:
                conflicts.append(f"inplace 新名撞未参与文件: {it['new_name']} ← {hit}")
    return conflicts


def run(cfg: Config, dry_run: bool = False, log=print) -> Path | None:
    """执行重命名。返回 journal 路径（inplace）或目标目录（copy）。"""
    plan = load_plan(cfg)
    seq = plan["sequence"]
    mode = cfg.rename_mode

    conflicts = check_conflicts(cfg, plan)
    if conflicts:
        log(f"✗ 发现 {len(conflicts)} 个冲突，拒绝执行：")
        for c in conflicts[:20]:
            log(f"  - {c}")
        raise SystemExit(1)

    drifted = [(it, d) for it in seq if (d := _drifted(it))]
    if drifted:
        log(f"⚠ {len(drifted)} 条与 plan 漂移，跳过：")
        for it, d in drifted[:5]:
            log(f"  - {Path(it['path']).name}: {d}")
    drift_ids = {id(it) for it, _ in drifted}
    todo = [it for it in seq if id(it) not in drift_ids]

    if dry_run:
        log(f"—— 预演 {len(todo)}/{len(seq)} 条（mode={mode}，目标 "
            f"{target_dir(cfg) if mode == 'copy' else cfg.sample_dir}）——")
        for it in todo[:50]:
            log(f"  {Path(it['path']).name} → {it['new_name']}")
        return None

    if mode == "copy":
        return _do_copy(cfg, todo, log)
    return _do_inplace(cfg, todo, log)


def _do_copy(cfg: Config, todo: list[dict], log=print) -> Path:
    t0 = time.perf_counter()
    tdir = target_dir(cfg)
    tdir.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(todo, 1):
        shutil.copy2(it["path"], tdir / it["new_name"])
        if i % 500 == 0:
            log(f"  copy {i}/{len(todo)}")
    log(f"—— copy 完成 {time.perf_counter() - t0:.1f}s：{len(todo)} 张 → {tdir} ——")
    return tdir


def _do_inplace(cfg: Config, todo: list[dict], log=print) -> Path:
    t0 = time.perf_counter()
    journal_path = cfg.out_dir / JOURNAL
    entries = [
        {"src": str(Path(it["path"]).resolve()), "dst": it["new_name"],
         "tmp": None, "status": "pending"}
        for it in todo
    ]
    journal_path.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode": "inplace", "status": "running", "entries": entries,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    def persist():
        journal_path.write_text(json.dumps(journal, ensure_ascii=False, indent=1),
                                encoding="utf-8")

    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    done_undo: list[dict] = []  # 已完成的实际操作，按序逆向回滚
    try:
        # 阶段 1：全部 src → tmp（防"新名 = 另一原文件名"互相覆盖）
        for e in journal["entries"]:
            src = Path(e["src"])
            tmp = src.with_name(src.name + f".fc_tmp-{uuid.uuid4().hex[:8]}")
            src.rename(tmp)
            e["tmp"], e["status"] = str(tmp), "tmp"
            done_undo.append({"type": "tmp", "src": src, "tmp": tmp})
        persist()
        # 阶段 2：tmp → dst
        for e in journal["entries"]:
            dst = Path(e["src"]).with_name(e["dst"])
            Path(e["tmp"]).rename(dst)
            e["status"] = "done"
            done_undo.append({"type": "dst", "tmp": Path(e["tmp"]), "dst": dst})
        journal["status"] = "done"
        persist()
        log(f"—— inplace 完成 {time.perf_counter() - t0:.1f}s：{len(todo)} 张 ——")
        return journal_path
    except Exception as e:
        log(f"✗ inplace 中途失败：{type(e).__name__}: {e} —— 自动回滚 {len(done_undo)} 步")
        for op in reversed(done_undo):
            if op["type"] == "dst" and op["dst"].is_file():
                op["dst"].rename(op["tmp"])  # 撤销阶段 2：dst → tmp
            elif op["type"] == "tmp" and op["tmp"].is_file():
                op["tmp"].rename(op["src"])  # 撤销阶段 1：tmp → src
        for e in journal["entries"]:
            e["status"], e["tmp"] = "pending", None
        journal["status"] = "rolled_back"
        persist()
        log("  已回滚到原状；journal 保留供检查")
        raise


def rollback(cfg: Config, log=print) -> int:
    """按 journal 逆向恢复 inplace 改名：done 态 dst→src，崩溃残留的 tmp 态 tmp→src。"""
    journal_path = cfg.out_dir / JOURNAL
    if not journal_path.is_file():
        raise FileNotFoundError(f"无 journal：{journal_path}")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if journal.get("status") == "rolled_back":
        log("journal 显示上次执行已回滚，无需操作")
        return 0
    n = 0
    for e in reversed(journal["entries"]):
        src = Path(e["src"])
        if e.get("status") == "tmp" and e.get("tmp") and Path(e["tmp"]).is_file():
            Path(e["tmp"]).rename(src)  # 崩溃时停在阶段 1/2 之间的残留
            n += 1
            continue
        dst = src.with_name(e["dst"])
        if dst.is_file() and not src.is_file():
            dst.rename(src)
            n += 1
        elif dst.is_file() and src.is_file():
            log(f"  ⚠ 源与目标并存，跳过: {dst.name}")
    journal["status"] = "rolled_back"
    journal_path.write_text(json.dumps(journal, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    log(f"—— 回滚完成：恢复 {n} 个文件 ——")
    return n
