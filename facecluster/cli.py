# -*- coding: utf-8 -*-
"""命令行入口（M0~M2 的交互形态：命令行 + JSON/CSV 报告）。

用法：
  python -m facecluster.cli annotate [--limit N] [--pack buffalo_l] [--no-gpu] [--mask off]
  python -m facecluster.cli report        # 生成接触表 HTML
  python -m facecluster.cli gpu-check
"""
from __future__ import annotations

import argparse
import sys

from .config import Config


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="facecluster", description="头像包分层分类排序工具（M0）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ann = sub.add_parser("annotate", help="标注管线：人脸+掩码+三属性+pHash（M0）")
    p_ann.add_argument("--limit", type=int, default=None)
    p_ann.add_argument("--pack", default=None, help="insightface 模型包，默认 buffalo_s")
    p_ann.add_argument("--no-gpu", action="store_true")
    p_ann.add_argument("--mask", default=None, choices=["mediapipe", "off"])

    p_clu = sub.add_parser("cluster", help="M1 身份聚簇 → out/clusters.json")
    p_clu.add_argument("--thr", type=float, default=None, help="同人余弦阈值，默认 0.55")

    p_ord = sub.add_parser("order", help="M2 硬分区+排序 → out/plan.json + 预览报告")
    p_ord.add_argument("--keys", default=None, help="逗号分隔键序，默认 肤色,发长,发色")
    p_ord.add_argument("--soft-hairstyle", action="store_true",
                       help="发型软排序：不改硬分区，段内同发型组相邻（需先跑 hairstyle）")

    p_ren = sub.add_parser("rename", help="M2 按 plan.json 批量重命名")
    p_ren.add_argument("--mode", default=None, choices=["copy", "inplace"])
    p_ren.add_argument("--dry-run", action="store_true", help="预演，不动文件")
    p_ren.add_argument("--natural", action="store_true", help="纯自然序号 1.jpg..N.jpg")
    p_ren.add_argument("--rollback", action="store_true", help="按 journal 回滚 inplace 改名")

    p_hs = sub.add_parser("hairstyle", help="M3 发型分组 → out/hairstyle.json（嵌入缓存共用）")
    p_hs.add_argument("--mode", default=None, choices=["A", "B", "C"],
                      help="A 固定清单零样本 | B 嵌入聚类(默认) | C 示例标注")
    p_hs.add_argument("--k", type=int, default=None, help="模式 B 指定组数（默认自动轮廓系数）")
    p_hs.add_argument("--examples", default=None, help="模式 C 示例图路径，逗号分隔")
    p_hs.add_argument("--refresh-prompts", action="store_true", help="模式 A 重算提示词嵌入")

    p_fm = sub.add_parser("fetch-model", help="下载 SigLIP2 视觉塔 ~355MB（联网一次）")
    p_fm.add_argument("--text", action="store_true",
                      help="另下文本塔 ~1.1GB（仅模式 A 提示词预计算需要）")
    p_fm.add_argument("--pack", default=None, metavar="NAME",
                      help="下载 InsightFace 模型包并解压（识别精度升级位，如 buffalo_l、antelopev2）")

    sub.add_parser("report", help="由 annotations.jsonl 生成接触表 HTML")

    sub.add_parser("gpu-check", help="DirectML 可用性自检")

    args = ap.parse_args(argv)
    cfg = Config()

    if args.cmd == "annotate":
        if args.pack:
            cfg.face_pack = args.pack
        if args.no_gpu:
            cfg.use_gpu = False
        if args.mask:
            cfg.mask_provider = args.mask
        from .annotate import annotate

        out = annotate(cfg, limit=args.limit)
        print(f"输出：{out}")
        return 0

    if args.cmd == "cluster":
        from . import cluster

        out = cluster.build(cfg, thr=args.thr)
        print(f"输出：{out}")
        return 0

    if args.cmd == "order":
        from . import partition, report

        keys = [k.strip() for k in args.keys.split(",")] if args.keys else None
        if args.soft_hairstyle:
            cfg.hairstyle_soft = True
        out = partition.build(cfg, keys=keys)
        html, csv = report.build(cfg)
        print(f"输出：{out}")
        print(f"输出：{html}")
        print(f"输出：{csv}")
        return 0

    if args.cmd == "rename":
        from . import rename

        if args.mode:
            cfg.rename_mode = args.mode
        if args.natural:
            cfg.natural_order = True
        if args.rollback:
            rename.rollback(cfg)
            return 0
        out = rename.run(cfg, dry_run=args.dry_run)
        if out:
            print(f"输出：{out}")
        return 0

    if args.cmd == "hairstyle":
        from . import hairstyle

        examples = [p.strip() for p in args.examples.split(",")] if args.examples else None
        out = hairstyle.build(cfg, mode=args.mode, k=args.k, examples=examples,
                              refresh_prompts=args.refresh_prompts)
        print(f"输出：{out}")
        return 0

    if args.cmd == "fetch-model":
        from . import modelhub
        from .hairstyle import model_dir

        if args.pack:
            pack_dir = cfg.insightface_root / "models" / args.pack
            modelhub.download_and_unpack("public-data/insightface", f"models/{args.pack}.zip",
                                         pack_dir, min_bytes=10_000_000)
            print(f"✓ 模型包 {args.pack} 就绪（annotate --pack {args.pack} 启用）")
            return 0

        base = model_dir(cfg)
        modelhub.download(cfg.siglip_repo, "onnx/vision_model.onnx",
                          base / "onnx" / "vision_model.onnx", min_bytes=100_000_000)
        modelhub.download(cfg.siglip_repo, "preprocessor_config.json",
                          base / "preprocessor_config.json", min_bytes=64)
        modelhub.download(cfg.siglip_repo, "tokenizer.json",
                          base / "tokenizer.json", min_bytes=100_000)
        if args.text:
            modelhub.download(cfg.siglip_repo, "onnx/text_model.onnx",
                              base / "onnx" / "text_model.onnx", min_bytes=500_000_000)
        print("✓ 模型就绪")
        return 0

    if args.cmd == "report":
        from tools.contact_sheet import build

        out = build(cfg)
        print(f"输出：{out}")
        return 0

    if args.cmd == "gpu-check":
        from tools.gpu_check import run

        return run(cfg)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
