# -*- coding: utf-8 -*-
"""M3 发型：SigLIP2 ViT-B/16 FP32 嵌入 + 三模式（需求定稿 §4.3）。

分工：光头~长发是"发长轴"（M0 几何量测已定）；本模块管"发型"（卷直/辫子/马尾…）。
- 模式 A 固定清单零样本：提示词嵌入一次性预计算，运行时不需 text_model（省 1.1GB 加载）。
- 模式 B 嵌入聚类（默认）：k-means++，K 自动（轮廓系数，上限 12），用户看图合并/拆组。
- 模式 C 示例标注：用户给示例图，其余按最近示例分配。
嵌入（hair_emb 768-d）算一次进缓存 npz，三模式共用，切换零重算。
发型标签落在 out/hairstyle.json；作分区键时由 partition.py 叠加进簇属性。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .cache_store import load_array, merge_arrays
from .config import Config, stable_hash
from .io import load_bgr

INPUT_SIZE = 224
MAX_TOKENS = 64
HAIR_MIN_FRAC = 0.005  # 发区像素占比低于此 = 掩码不可靠，标记无效

# 模式 A 内置清单（~20 类）。base 模型为英文词表训练，嵌入只用英文提示词；
# 中文仅作展示映射。同义词条平均后归一。
HAIRSTYLE_PROMPTS: list[tuple[str, list[str]]] = [
    ("直发", ["straight hair", "sleek straight hairstyle"]),
    ("卷发", ["curly hair", "tight curly hair"]),
    ("波浪发", ["wavy hair", "loose wavy hairstyle"]),
    ("编发", ["braided hair", "hair braids"]),
    ("脏辫", ["dreadlocks", "locs hairstyle"]),
    ("马尾", ["ponytail", "hair tied in a ponytail"]),
    ("高马尾", ["high ponytail"]),
    ("低马尾", ["low ponytail"]),
    ("丸子头", ["hair bun", "hair in a bun"]),
    ("双丸子头", ["space buns", "double bun hairstyle"]),
    ("双马尾", ["twintails", "pigtails hairstyle"]),
    ("半扎", ["half-up half-down hairstyle"]),
    ("刘海", ["bangs", "fringe hairstyle"]),
    ("空气刘海", ["wispy bangs", "see-through bangs"]),
    ("蘑菇头", ["bowl cut", "bowl haircut"]),
    ("莫霍克", ["mohawk hairstyle"]),
    ("油头", ["slicked back hair"]),
    ("中分", ["middle part hairstyle"]),
    ("爆炸头", ["afro hairstyle"]),
    ("寸头刺头", ["buzz cut", "crew cut"]),
]


def model_dir(cfg: Config) -> Path:
    return cfg.models_dir / "siglip2"


def prompts_file(cfg: Config) -> Path:
    """模式 A 外部清单（GUI 可增删改），存在则覆盖内置 HAIRSTYLE_PROMPTS。

    放项目根而非 out/：它是用户手工维护的配置，不应随生成物目录一起被清理，
    且可进版本库与其他环境共享。
    """
    from .config import PROJECT_ROOT

    return PROJECT_ROOT / "hairstyle_prompts.json"


def load_prompt_list(cfg: Config) -> list[tuple[str, list[str]]]:
    """模式 A 类目清单：优先读外部 JSON（[{label, prompts:[...]}]），否则内置。"""
    pf = prompts_file(cfg)
    if pf.is_file():
        items = json.loads(pf.read_text(encoding="utf-8"))
        return [(d["label"], list(d["prompts"])) for d in items if d.get("label") and d.get("prompts")]
    return HAIRSTYLE_PROMPTS


def _session(cfg: Config, onnx_path: Path, log=print):
    """DML 会话；FP16/int8 文件名直接拒绝（DML LayerNorm 崩溃先例 + int8 不可用）。"""
    name = onnx_path.name.lower()
    if "_fp16" in name or "_int8" in name or "_q4" in name or "_quantized" in name:
        raise ValueError(f"拒绝非 FP32 模型: {onnx_path.name}（DML 上 FP16/int8 不可用）")
    import onnxruntime as ort

    providers = (
        ["DmlExecutionProvider", "CPUExecutionProvider"]
        if cfg.use_gpu
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    if cfg.use_gpu and sess.get_providers()[0] != "DmlExecutionProvider":
        log(f"⚠ SigLIP2 实际挂在 {sess.get_providers()[0]}（要求 DML）")
    return sess


def _norm_params(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """preprocessor_config.json 的 mean/std；缺省 0.5。"""
    mean = std = np.full(3, 0.5, dtype=np.float32)
    pf = model_dir(cfg) / "preprocessor_config.json"
    if pf.is_file():
        d = json.loads(pf.read_text(encoding="utf-8"))
        if d.get("image_mean"):
            mean = np.asarray(d["image_mean"], dtype=np.float32)
        if d.get("image_std"):
            std = np.asarray(d["image_std"], dtype=np.float32)
    return mean, std


class SiglipEngine:
    """视觉塔：发区图 → 768-d 归一嵌入。文本塔仅模式 A 预计算时加载。"""

    def __init__(self, cfg: Config, log=print):
        self.cfg = cfg
        self.log = log
        base = model_dir(cfg)
        vp = base / "onnx" / "vision_model.onnx"
        if not vp.is_file():
            raise FileNotFoundError(
                f"缺少 SigLIP2 视觉塔 {vp}\n"
                f"  运行 python -m facecluster.cli fetch-model --siglip 下载（~355MB，需联网一次）"
            )
        self.sess = _session(cfg, vp, log)
        self.in_name = self.sess.get_inputs()[0].name
        ishape = self.sess.get_inputs()[0].shape  # [1,3,224,224] 或 [1,224,224,3]
        self.nchw = not (len(ishape) == 4 and isinstance(ishape[1], int) and ishape[1] in (224, 256))
        self.out_name = self.sess.get_outputs()[0].name
        self.mean, self.std = _norm_params(cfg)
        log("✓ SigLIP2 视觉塔就绪（FP32）")

    def image_embed(self, bgr: np.ndarray, hair_mask_q: np.ndarray) -> tuple[np.ndarray, float]:
        """发区图（非发区置灰）→ 嵌入。返回 (768-d 归一向量, 发区占比)。"""
        h, w = bgr.shape[:2]
        m = cv2.resize(hair_mask_q.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        hair = m > self.cfg.hair_mask_thr
        frac = float(hair.mean())
        img = bgr.copy()
        img[~hair] = 128  # 置灰：避免背景/人脸污染发型嵌入（§4.3）
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = (x.transpose(2, 0, 1)[None] if self.nchw else x[None])
        out = self.sess.run([self.out_name], {self.in_name: x})[0][0]
        v = out.mean(axis=0) if out.ndim == 2 else out  # [1,197,768] → token 均值池化
        v = np.asarray(v, dtype=np.float32)
        return v / max(1e-6, float(np.linalg.norm(v))), frac


def text_embeds(cfg: Config, prompts: list[str], log=print) -> np.ndarray:
    """文本塔批量编码（仅模式 A 预计算调用；需 tokenizer + text_model.onnx）。"""
    tp = model_dir(cfg) / "onnx" / "text_model.onnx"
    if not tp.is_file():
        raise FileNotFoundError(
            f"模式 A 需要 {tp}（~1.1GB）\n"
            f"  python -m facecluster.cli fetch-model --text"
        )
    sess = _session(cfg, tp, log)
    ids_all, mask_all = _tokenize(cfg, prompts)
    feed_base = {}
    for inp in sess.get_inputs():
        n = inp.name.lower()
        if "input_ids" in n:
            feed_base[inp.name] = ids_all
        elif "attention_mask" in n:
            feed_base[inp.name] = mask_all
        elif "position" in n:
            feed_base[inp.name] = np.arange(MAX_TOKENS, dtype=np.int64)[None]
        else:  # 未知整型输入（如 type_ids）给零
            feed_base[inp.name] = np.zeros((1, MAX_TOKENS), dtype=np.int64)
    outs = []
    for i in range(ids_all.shape[0]):
        feed = {k: v[i:i + 1] for k, v in feed_base.items()}
        o = sess.run([sess.get_outputs()[0].name], feed)[0][0]
        v = o.mean(axis=0) if o.ndim == 2 else o
        outs.append(np.asarray(v, dtype=np.float32))
    E = np.stack(outs)
    return E / np.maximum(1e-6, np.linalg.norm(E, axis=1, keepdims=True))


def _tokenize(cfg: Config, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """tokenizers 库（tokenizer.json）编码 + 截断/填充到 MAX_TOKENS。"""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir(cfg) / "tokenizer.json"))
    pad_id = tok.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0
    ids = np.zeros((len(prompts), MAX_TOKENS), dtype=np.int64)
    mask = np.zeros((len(prompts), MAX_TOKENS), dtype=np.int64)
    for i, p in enumerate(prompts):
        enc = tok.encode(p)
        seq = enc.ids[:MAX_TOKENS]
        ids[i, : len(seq)] = seq
        mask[i, : len(seq)] = 1
    return ids, mask


# ---------------- 嵌入批量计算（缓存优先，三模式共用） ----------------

def ensure_embeddings(cfg: Config, log=print, progress_cb=None) -> int:
    """为所有有发区掩码的图算 hair_emb（已有则跳过）。返回本次新算数量。"""
    from .cluster import load_annotations

    records = load_annotations(cfg)
    todo = []
    for r in records:
        if r.unread:
            continue
        if load_array(cfg.cache_root, r.content_hash, "hair_emb") is None:
            todo.append(r)
    if not todo:
        log("发型嵌入全部缓存命中，零重算")
        return 0
    eng = SiglipEngine(cfg, log)
    log(f"计算发型嵌入 {len(todo)}/{len(records)} 张…")
    t0 = time.perf_counter()
    done = 0
    for i, r in enumerate(todo, 1):
        try:
            bgr = load_bgr(Path(r.path), cfg.max_side)
            mq = load_array(cfg.cache_root, r.content_hash, "mask_hair")
            if mq is None:
                log(f"✗ {Path(r.path).name}: 缺发区掩码（重跑 annotate 可修复）")
            else:
                v, frac = eng.image_embed(bgr, mq)
                # 发区占比一并入缓存：load_all 据此剔除掩码不可靠的图（重复运行零重算的前提）
                merge_arrays(cfg.cache_root, r.content_hash, {
                    "hair_emb": v.astype(np.float32),
                    "hair_frac": np.float32(frac),
                })
                done += 1
        except Exception as e:  # 单图失败不阻断
            log(f"✗ {Path(r.path).name}: {type(e).__name__}: {e}")
        if progress_cb:
            progress_cb(i, len(todo))
        if i % 100 == 0 or i == len(todo):
            log(f"  [{i}/{len(todo)}] {time.perf_counter() - t0:.1f}s")
    log(f"—— 发型嵌入完成：新增 {done}，耗时 {time.perf_counter() - t0:.1f}s（重复运行零重算）——")
    return done


def load_all_hair_emb(cfg: Config, log=print) -> tuple[dict[str, np.ndarray], list[str]]:
    """全量读 hair_emb。返回 (chash→向量, 无嵌入的 chash 列表)。"""
    from .cluster import load_annotations

    records = load_annotations(cfg)
    have: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for r in records:
        if r.unread:
            missing.append(r.content_hash)
            continue
        v = load_array(cfg.cache_root, r.content_hash, "hair_emb")
        if v is None:
            missing.append(r.content_hash)
            continue
        fr = load_array(cfg.cache_root, r.content_hash, "hair_frac")
        if fr is not None and float(fr) < HAIR_MIN_FRAC:
            missing.append(r.content_hash)  # 发区占比过低 = 掩码不可靠（§4.3）
            continue
        have[r.content_hash] = v
    return have, missing


# ---------------- 聚类工具（模式 B） ----------------

def _kmeans(X: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    """k-means++ + Lloyd，全程 float32（禁止隐式升 float64）。返回 labels。

    退化场景：k>点数时收敛为点数；点全部重合（重复图）时 kmeans++ 的距离概率
    全为 0 无法选种子，直接轮分标签——重复成员对聚类无信息量，任意分桶即可。
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(k, n)
    centers = [X[rng.integers(n)].copy()]
    d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for _ in range(k - 1):
        if d2.sum() <= 1e-12:
            return np.arange(n) % k
        p = d2 / d2.sum()
        centers.append(X[rng.choice(n, p=p)].copy())
        d2 = np.minimum(d2, np.sum((X - centers[-1]) ** 2, axis=1))
    C = np.stack(centers)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        dist = ((X[:, None, :] - C[None]) ** 2).sum(-1)  # n×k，k≤12 内存安全
        new = dist.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(0)
    return labels


def _silhouette(X: np.ndarray, labels: np.ndarray, sample: int = 5000, seed: int = 0) -> float:
    """抽样轮廓系数（样本内 O(m²) 距离矩阵，float32 分块）。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    S, L = X[idx], labels[idx]
    D = np.sqrt(np.maximum(0.0, np.maximum(
        (S * S).sum(1)[:, None] + (S * S).sum(1)[None] - 2 * (S @ S.T), 0.0)))
    np.fill_diagonal(D, 0.0)
    uniq = np.unique(L)
    if len(uniq) < 2:
        return -1.0
    a = np.zeros(len(S), dtype=np.float32)
    b = np.full(len(S), np.inf, dtype=np.float32)
    for c in uniq:
        mask = L == c
        if mask.sum() <= 1:
            a[mask] = 0.0
            continue
        sub = D[np.ix_(mask, mask)]
        a[mask] = sub.sum(1) / (mask.sum() - 1)
    for c in uniq:
        mask = L == c
        other = ~mask
        if not other.any():
            continue
        b_other = D[np.ix_(mask, other)].mean(1)
        b[mask] = np.minimum(b[mask], b_other)
    s = (b - a) / np.maximum(a, b)
    return float(np.nanmean(s))


# ---------------- 三模式主入口 ----------------

def build(
    cfg: Config,
    mode: str | None = None,
    k: int | None = None,
    examples: list[str] | None = None,
    refresh_prompts: bool = False,
    log=print,
    progress_cb=None,
) -> Path:
    """计算发型分组 → out/hairstyle.json。嵌入缓存命中时秒级完成。"""
    mode = (mode or cfg.hairstyle_mode).upper()
    k = cfg.hairstyle_k if k is None else k
    ensure_embeddings(cfg, log, progress_cb)
    embs, missing = load_all_hair_emb(cfg, log)
    chashes = sorted(embs)
    log(f"发型模式 {mode}：{len(chashes)} 张有效嵌入 / {len(missing)} 张无效（无掩码或失败）")
    if not chashes:
        log("⚠ 无有效发型嵌入，输出空分组")
        groups, group_order = {}, []
    elif mode == "A":
        groups, group_order = _mode_a(cfg, embs, refresh_prompts, log)
    elif mode == "B":
        groups, group_order = _mode_b(embs, k, log)
    elif mode == "C":
        groups, group_order = _mode_c(cfg, embs, examples or [], log)
    else:
        raise ValueError(f"未知发型模式 {mode}（A/B/C）")

    art = cfg.out_dir / "hairstyle.json"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps({
        "config_hash": stable_hash(mode=mode, k=k, examples=sorted(examples or [])),
        "mode": mode,
        "group_order": group_order,
        "groups": groups,
        "invalid": missing,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    log(f"—— 发型分组完成：{len(groups)} 组，组大小 top10 {sizes[:10]} ——")
    log(f"输出：{art}")
    return art


def _mode_a(cfg: Config, embs: dict[str, np.ndarray], refresh: bool, log=print):
    """固定清单零样本。提示词嵌入预计算进 text_prompts.npz，运行时不加载 text_model。"""
    prompt_list = load_prompt_list(cfg)
    prompts = [p for _, ps in prompt_list for p in ps]
    labels = [lab for lab, _ in prompt_list]
    pf = model_dir(cfg) / "text_prompts.npz"
    key = stable_hash(prompts=prompts).encode()
    E = None
    if pf.is_file() and not refresh:
        z = np.load(pf, allow_pickle=False)
        if z["key"].tobytes() == key:
            E = z["embs"]
    if E is None:
        log("预计算提示词嵌入（首次/清单变更时需 text_model，一次性）…")
        E = text_embeds(cfg, prompts, log)
        np.savez_compressed(pf, key=np.frombuffer(key, dtype=np.uint8), embs=E)
        log(f"✓ 提示词嵌入已缓存 {pf.name}（后续模式 A 运行不再加载 text_model）")
    X = np.stack([embs[c] for c in sorted(embs)])
    sims = X @ E.T  # n×prompt
    pi = sims.argmax(1)
    prompt_label = [lab for lab, ps in prompt_list for _ in ps]
    groups: dict[str, list[str]] = {}
    chashes = sorted(embs)
    for c, j in zip(chashes, pi):
        groups.setdefault(prompt_label[j], []).append(c)
    order = [lab for lab in labels if lab in groups]
    return groups, order


def _mode_b(embs: dict[str, np.ndarray], k: int | None, log=print):
    """嵌入聚类（默认）：K 自动轮廓系数搜索 [2,12] 或指定。"""
    X = np.stack([embs[c] for c in sorted(embs)]).astype(np.float32)
    chashes = sorted(embs)
    if k is None:
        hi = min(12, len(X) - 1)
        if hi < 2:
            k = 1  # 仅 1 个有效嵌入时无法聚类，单组即可
            log("  有效嵌入仅 1 张，K=1")
        else:
            best, best_k, best_s = None, 2, -2.0
            for kk in range(2, hi + 1):
                s = _silhouette(X, _kmeans(X, kk))
                log(f"  K={kk} 轮廓系数 {s:.3f}")
                if s > best_s:
                    best, best_k, best_s = kk, kk, s
            k = best_k
            log(f"自动选 K={k}（轮廓系数 {best_s:.3f}）")
    labels = _kmeans(X, k)
    groups: dict[str, list[str]] = {}
    for c, lb in zip(chashes, labels):
        groups.setdefault(f"发型组{lb + 1}", []).append(c)
    order = [f"发型组{i + 1}" for i in range(k) if f"发型组{i + 1}" in groups]
    return groups, order


def _mode_c(cfg: Config, embs: dict[str, np.ndarray], examples: list[str], log=print):
    """示例标注：示例图嵌入为锚点，其余按最近示例分配；示例间 cos>0.9 告警。"""
    if not examples:
        raise ValueError("模式 C 需要 --examples 至少一张示例图")
    from .io import content_hash

    anchors: list[tuple[str, np.ndarray]] = []
    for p in examples:
        c = content_hash(Path(p))
        v = embs.get(c)
        if v is None:
            v = load_array(cfg.cache_root, c, "hair_emb")
        if v is None:
            bgr = load_bgr(Path(p), cfg.max_side)
            mq = load_array(cfg.cache_root, c, "mask_hair")
            if mq is None:
                raise ValueError(f"示例图缺发区掩码（先对包含它的包跑 annotate）：{p}")
            v, _ = SiglipEngine(cfg, log).image_embed(bgr, mq)
        anchors.append((Path(p).stem, np.asarray(v, dtype=np.float32)))
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            cos = float(anchors[i][1] @ anchors[j][1])
            if cos > 0.9:
                log(f"⚠ 示例 {anchors[i][0]} 与 {anchors[j][0]} 余弦 {cos:.2f} > 0.9，考虑合并为一组")
    A = np.stack([v for _, v in anchors])
    chashes = sorted(embs)
    X = np.stack([embs[c] for c in chashes])
    which = (X @ A.T).argmax(1)
    groups: dict[str, list[str]] = {name: [] for name, _ in anchors}
    for c, j in zip(chashes, which):
        groups[anchors[j][0]].append(c)
    return groups, [name for name, _ in anchors]


# ---------------- 手工组编辑（GUI 看图合并/拆组，§4.3 模式 B 工作流） ----------------

def _load_hs(cfg: Config) -> tuple[dict, Path]:
    art = cfg.out_dir / "hairstyle.json"
    if not art.is_file():
        raise FileNotFoundError(f"先运行 hairstyle 生成 {art}")
    return json.loads(art.read_text(encoding="utf-8")), art


def _write_hs(cfg: Config, hs: dict, art: Path) -> None:
    hs["manual_edits"] = True  # 手工编辑标记：进 plan 漂移指纹，组数/标志变化即被捕捉
    art.write_text(json.dumps(hs, ensure_ascii=False, indent=1), encoding="utf-8")


def merge_groups(cfg: Config, names: list[str], log=print) -> Path:
    """把 names[1:] 的成员并入 names[0]；对任意模式的分组结果有效。"""
    if len(names) < 2:
        raise ValueError("合并至少需要两个组")
    hs, art = _load_hs(cfg)
    groups, order = hs["groups"], hs["group_order"]
    missing = [n for n in names if n not in groups]
    if missing:
        raise ValueError(f"组不存在: {missing}")
    keep = names[0]
    for n in names[1:]:
        groups[keep].extend(groups.pop(n))
        order.remove(n)
    _write_hs(cfg, hs, art)
    log(f"✓ 合并 {names} → {keep}（{len(groups[keep])} 图）；现 {len(order)} 组")
    return art


def split_group(cfg: Config, name: str, k: int = 2, log=print) -> Path:
    """按发型嵌入对组做 k-means 拆分（默认 2），新组名 = 原名·1/2…，组序位置不变。"""
    hs, art = _load_hs(cfg)
    if name not in hs["groups"]:
        raise ValueError(f"组不存在: {name}")
    members = hs["groups"][name]
    if len(members) < k:
        raise ValueError(f"组 {name} 仅 {len(members)} 图，无法拆成 {k} 组")
    embs_map: dict[str, np.ndarray] = {}
    for m in members:
        v = load_array(cfg.cache_root, m, "hair_emb")
        if v is not None:
            embs_map[m] = v
    chashes = sorted(embs_map)
    if not chashes:
        raise ValueError(f"组 {name} 成员均无发型嵌入缓存（重跑 hairstyle 或 annotate 后再拆）")
    labels = _kmeans(np.stack([embs_map[c] for c in chashes]).astype(np.float32), k)
    new_names = [f"{name}·{i + 1}" for i in range(k)]
    hs["groups"].pop(name)
    for i, nn in enumerate(new_names):
        hs["groups"][nn] = [c for c, lb in zip(chashes, labels) if lb == i]
    leftovers = [m for m in members if m not in embs_map]  # 无嵌入成员归末组
    hs["groups"][new_names[-1]].extend(leftovers)
    idx = hs["group_order"].index(name)
    hs["group_order"][idx:idx + 1] = new_names
    _write_hs(cfg, hs, art)
    sizes = [len(hs["groups"][nn]) for nn in new_names]
    log(f"✓ 拆分 {name}（{len(members)} 图）→ {dict(zip(new_names, sizes))}；现 {len(hs['group_order'])} 组")
    return art


def representative_chashes(cfg: Config, chashes: list[str], n: int = 3) -> list[str]:
    """取组内最接近组中心的前 n 张作代表图（GUI 组卡片展示用）。"""
    embs = [load_array(cfg.cache_root, c, "hair_emb") for c in chashes]
    pairs = [(c, e) for c, e in zip(chashes, embs) if e is not None]
    if len(pairs) <= n:
        return [c for c, _ in pairs] + [c for c in chashes if c not in {p[0] for p in pairs}]
    X = np.stack([e for _, e in pairs])
    center = X.mean(axis=0)
    order = np.argsort(-(X @ center))
    return [pairs[i][0] for i in order[:n]]
