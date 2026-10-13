# -*- coding: utf-8 -*-
"""特征缓存：按内容哈希逐图落盘（需求定稿：缓存身份 = 内容哈希，不用文件名）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _dir(root: Path) -> Path:
    p = root / "features"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load(root: Path, chash: str) -> tuple[dict | None, np.ndarray | None]:
    """返回 (record_json, embedding)。miss 返回 (None, None)。"""
    jf = _dir(root) / f"{chash}.json"
    if not jf.is_file():
        return None, None
    d = json.loads(jf.read_text(encoding="utf-8"))
    emb = None
    nf = _dir(root) / f"{chash}.npz"
    if nf.is_file():
        z = np.load(nf)
        if "emb" in z.files:
            emb = z["emb"]
    return d, emb


def save(
    root: Path,
    chash: str,
    record_dict: dict,
    emb: np.ndarray | None,
    masks_quarter: dict[str, np.ndarray] | None = None,
) -> None:
    _dir(root)
    (_dir(root) / f"{chash}.json").write_text(
        json.dumps(record_dict, ensure_ascii=False), encoding="utf-8"
    )
    arrays: dict[str, np.ndarray] = {}
    if emb is not None:
        arrays["emb"] = emb.astype(np.float32)
    if masks_quarter:
        for k, v in masks_quarter.items():
            arrays[f"mask_{k}"] = v.astype(np.uint8)
    if arrays:
        merge_arrays(root, chash, arrays)


def merge_arrays(root: Path, chash: str, arrays: dict[str, np.ndarray]) -> None:
    """向 {chash}.npz 合并写入（保留未涉及的既有数组）——发型嵌入追加复用此入口。"""
    nf = _dir(root) / f"{chash}.npz"
    old: dict[str, np.ndarray] = {}
    if nf.is_file():
        with np.load(nf) as z:
            old = {k: z[k] for k in z.files}
    old.update({k: v for k, v in arrays.items()})
    np.savez_compressed(nf, **old)


def load_array(root: Path, chash: str, key: str) -> np.ndarray | None:
    """读 npz 中的单个数组（如 hair_emb），缺文件或缺键返回 None。"""
    nf = _dir(root) / f"{chash}.npz"
    if not nf.is_file():
        return None
    with np.load(nf) as z:
        return z[key] if key in z.files else None
