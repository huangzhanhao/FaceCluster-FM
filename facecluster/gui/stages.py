# -*- coding: utf-8 -*-
"""六个阶段工作区：准备 / 标注 / 聚簇 / 发型 / 分区排序 / 输出。

阶段只负责"参数编排 + 请求发出 + 结果呈现"，后台调用一律经 ctx.run（app 保证
单任务互斥与进度回传）。所有后台任务写成带 log/progress_cb 形参的模块级函数，
由 Worker 按签名注入——阶段里不出现任何 threading 细节。
"""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QRadioButton,
    QSizePolicy, QSpinBox, QTableView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..config import KEY_REGISTRY, Config
from . import theme
from .panels import (
    FValuePanel, HBarChart, KeyChainPanel, ThumbGrid, heading, muted, subheading,
)
from .table_model import PlanTableModel

PAL = theme.PALETTE
HAIRSTYLE_KEY = "发型"


# ---------- 后台任务（形参带 log / progress_cb → Worker 按签名注入）----------
def task_annotate(cfg: Config, limit, log=print):
    from ..annotate import annotate

    return annotate(cfg, limit=limit, log=log)


def task_cluster(cfg: Config, log=print, progress_cb=None):
    from .. import cluster

    return cluster.build(cfg, log=log, progress_cb=progress_cb)


def task_order(cfg: Config, keys, log=print):
    from .. import partition

    return partition.build(cfg, keys=keys, log=log)


def task_hairstyle(cfg: Config, mode, k, examples, refresh, log=print, progress_cb=None):
    from .. import hairstyle

    return hairstyle.build(cfg, mode=mode, k=k, examples=examples,
                           refresh_prompts=refresh, log=log, progress_cb=progress_cb)


def task_merge_clusters(cfg: Config, cids, log=print):
    from .. import adjust

    return adjust.merge(cfg, cids, log=log)


def task_split_cluster(cfg: Config, cid, k, log=print):
    from .. import adjust

    return adjust.split(cfg, cid, k=k, log=log)


def task_merge_hs(cfg: Config, names, log=print):
    from .. import hairstyle

    return hairstyle.merge_groups(cfg, names, log=log)


def task_split_hs(cfg: Config, name, log=print):
    from .. import hairstyle

    return hairstyle.split_group(cfg, name, k=2, log=log)


def task_rename(cfg: Config, dry, log=print):
    from .. import rename

    return rename.run(cfg, dry_run=dry, log=log)


def task_rollback(cfg: Config, log=print):
    from .. import rename

    return rename.rollback(cfg, log=log)


def task_fetch_pack(cfg: Config, pack, log=print):
    from .. import modelhub

    dest = cfg.insightface_root / "models" / pack
    modelhub.download_pack(pack, dest, log=log)
    return dest


def task_fetch_siglip(cfg: Config, with_text, log=print):
    from .. import modelhub
    from ..hairstyle import model_dir

    base = model_dir(cfg)
    repo = cfg.siglip_repo
    modelhub.download(repo, "onnx/vision_model.onnx", base / "onnx" / "vision_model.onnx",
                      log, 100_000_000)
    modelhub.download(repo, "preprocessor_config.json", base / "preprocessor_config.json", log, 64)
    modelhub.download(repo, "tokenizer.json", base / "tokenizer.json", log, 100_000)
    if with_text:
        modelhub.download(repo, "onnx/text_model.onnx", base / "onnx" / "text_model.onnx",
                          log, 500_000_000)
    return base


def task_gpu_check(cfg: Config, log=print):
    """gpu_check 打印到 stdout（CLI 工具），此处重定向到日志再回传结论。"""
    from tools.gpu_check import run as gpu_run

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = gpu_run(cfg)
    for line in buf.getvalue().splitlines():
        log(line)
    return "GPU 自检通过" if rc == 0 else "GPU 自检发现问题（见日志）"


def task_report(cfg: Config, which, log=print):
    from .. import report
    from tools import contact_sheet

    if which == "sort":
        html, _csv = report.build(cfg)
        log(f"排序报告：{html}")
        return html
    out = contact_sheet.build(cfg)
    log(f"接触表：{out}")
    return out


def task_full(cfg: Config, keys, limit, log=print, progress_cb=None):
    """一键全流程：标注 → 聚簇 → 排序（发型需模型，不纳入自动链路）。"""
    from ..annotate import annotate
    from .. import cluster, partition

    log("═══ 全流程 · 1/3 标注 ═══")
    annotate(cfg, limit=limit, log=log)
    log("═══ 全流程 · 2/3 身份聚簇 ═══")
    cluster.build(cfg, log=log, progress_cb=progress_cb)
    log("═══ 全流程 · 3/3 分区排序 ═══")
    partition.build(cfg, keys=keys, log=log)
    return "全流程完成"


def _open(path: Path) -> None:
    if path.exists():
        os.startfile(str(path))  # noqa: S606 — Windows 本机自用工具


def _dlg_dir(parent, title: str, cur: Path) -> str | None:
    d = QFileDialog.getExistingDirectory(parent, title, str(cur))
    return d or None


# ============================ ① 准备 ============================
class PrepareStage(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        cfg = ctx.cfg
        v = QVBoxLayout(self)
        v.setSpacing(14)
        v.addWidget(heading("① 准备 · 数据目录与环境"))

        # 数据目录
        dirs = QWidget()
        dv = QVBoxLayout(dirs)
        dv.setContentsMargins(0, 0, 0, 0)
        dv.addWidget(subheading("数据目录"))
        self.sample_edit = QLineEdit(str(cfg.sample_dir))
        self.sample_edit.setReadOnly(True)
        b1 = QPushButton("选择…")
        b1.clicked.connect(self._pick_sample)
        dv.addWidget(_row(self.sample_edit, b1, labels=("样本目录",), fill=True))
        self.out_edit = QLineEdit(str(cfg.out_dir))
        self.out_edit.setReadOnly(True)
        b2 = QPushButton("选择…")
        b2.clicked.connect(self._pick_out)
        dv.addWidget(_row(self.out_edit, b2, labels=("输出目录",), fill=True))
        v.addWidget(dirs)

        # 模型与环境：表格一次性建好（固定 5 行），refresh_env 只更新状态/说明/按钮文案。
        # 每次刷新重建单元格控件会留下"待析构"的旧控件（Qt deleteLater 未及时处理时
        # 仍挂在 viewport 上，停在上次位置），从而在表格左上角叠出错位残影。
        v.addWidget(subheading("模型与环境"))
        self.env = QTableWidget(5, 4)
        self.env.setHorizontalHeaderLabels(["组件", "状态", "说明", "操作"])
        hh = self.env.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        # 操作列按内容撑不高（ResizeToContents 只看 QTableWidgetItem，不看单元格控件），
        # 定宽才能容下"下载 buffalo_l"这类最长按钮文字
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.env.setColumnWidth(3, 132)
        self.env.verticalHeader().setVisible(False)
        self.env.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.env.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.env.setShowGrid(False)
        self.env.verticalHeader().setDefaultSectionSize(38)
        self._env_pill: list[QLabel] = []
        self._env_note: list[QTableWidgetItem] = []
        self._env_btn: list[QPushButton | None] = []
        actions = (self._dl_pack, None, lambda: self._dl_siglip(False),
                   lambda: self._dl_siglip(True), self._gpu_check)
        for i in range(5):
            self.env.setItem(i, 0, QTableWidgetItem(""))
            p = QLabel()
            self.env.setCellWidget(i, 1, p)
            self._env_pill.append(p)
            note = QTableWidgetItem("")
            self.env.setItem(i, 2, note)
            self._env_note.append(note)
            slot = actions[i]
            if slot is None:
                self.env.setCellWidget(i, 3, QLabel("—"))
                self._env_btn.append(None)
            else:
                b = QPushButton()
                b.clicked.connect(slot)
                self.env.setCellWidget(i, 3, b)
                self._env_btn.append(b)
        self.env.setFixedHeight(self.env.horizontalHeader().height() + 38 * 5 + 4)
        v.addWidget(self.env)

        # 一键全流程
        v.addWidget(subheading("一键全流程"))
        v.addWidget(muted("按 标注 → 聚簇 → 分区排序 顺序串行执行；已标注/已缓存的图秒过。"))
        run_all = QPushButton("▶ 一键全流程")
        run_all.setProperty("role", "primary")
        run_all.clicked.connect(self._run_all)
        v.addWidget(run_all, alignment=Qt.AlignmentFlag.AlignLeft)
        v.addStretch(1)
        self.refresh_env()

    # ---- 目录选择 ----
    def _pick_sample(self):
        d = _dlg_dir(self, "选择样本目录", self.ctx.cfg.sample_dir)
        if d:
            self.ctx.set_sample_dir(Path(d))
            self.sample_edit.setText(d)

    def _pick_out(self):
        d = _dlg_dir(self, "选择输出目录", self.ctx.cfg.out_dir)
        if d:
            self.ctx.set_out_dir(Path(d))
            self.out_edit.setText(d)

    # ---- 环境状态（只更新内容，控件复用）----
    def refresh_env(self):
        cfg = self.ctx.cfg
        dml = _dml_available()
        pack_ok = (cfg.insightface_root / "models" / cfg.face_pack).is_dir()
        states = (
            (f"人脸识别 {cfg.face_pack}", pack_ok,
             f"~/.insightface/models/{cfg.face_pack}"),
            ("掩码 mediapipe", (cfg.models_dir / "selfie_multiclass_256x256.onnx").is_file(),
             "缺失时自动降级无掩码回退档，发长失效"),
            ("SigLIP2 视觉塔",
             (cfg.models_dir / "siglip2" / "onnx" / "vision_model.onnx").is_file(),
             "355MB · 发型嵌入必需"),
            ("SigLIP2 文本塔",
             (cfg.models_dir / "siglip2" / "onnx" / "text_model.onnx").is_file(),
             "1.1GB · 仅模式 A 需要"),
            ("DirectML", bool(dml), "DML 可用" if dml else "未检测到 DmlExecutionProvider"),
        )
        for i, (name, ok, note) in enumerate(states):
            self.env.item(i, 0).setText(name)
            self._env_pill[i].setText("● 就绪" if ok else "● 缺失")
            self._env_pill[i].setStyleSheet(
                f"color:{PAL['ok'] if ok else PAL['warn']}; font-weight:600;")
            self._env_note[i].setText(note)
            b = self._env_btn[i]
            if b is not None:
                if i == 0:
                    b.setText(f"下载 {'buffalo_l' if pack_ok else cfg.face_pack}")
                elif i == 4:
                    b.setText("自检")
                else:
                    b.setText("重新下载" if ok else "下载")

    # ---- 下载 / 自检：槽一次性绑定，参数在点击时按当前状态计算 ----
    def _dl_pack(self):
        cfg = self.ctx.cfg
        ready = (cfg.insightface_root / "models" / cfg.face_pack).is_dir()
        pack = "buffalo_l" if ready else cfg.face_pack
        self.ctx.run(task_fetch_pack, f"下载 {pack}", args=(cfg, pack),
                     on_done=self.refresh_env)

    def _dl_siglip(self, with_text: bool):
        self.ctx.run(task_fetch_siglip, "下载 SigLIP2 模型",
                     args=(self.ctx.cfg, with_text), on_done=self.refresh_env)

    def _gpu_check(self):
        self.ctx.run(task_gpu_check, "GPU 自检", args=(self.ctx.cfg,))

    def _run_all(self):
        keys = list(self.ctx.cfg.key_order)
        self.ctx.run(task_full, "一键全流程", args=(self.ctx.cfg, keys, None),
                     on_done=lambda _r: self.ctx.load_plan())


# ============================ ② 标注 ============================
class AnnotateStage(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        v = QVBoxLayout(self)
        v.setSpacing(12)
        v.addWidget(heading("② 标注 · 每图一次，特征进缓存"))

        params = QWidget()
        pv = QVBoxLayout(params)
        pv.setContentsMargins(0, 0, 0, 0)
        self.limit = QSpinBox()
        self.limit.setRange(0, 1_000_000)
        self.limit.setSpecialValueText("全部")
        self.limit.setValue(0)
        self.limit.setFixedWidth(112)
        self.pack = QComboBox()
        self.pack.addItems(["buffalo_s", "buffalo_l", "antelopev2"])
        self.pack.setCurrentText(ctx.cfg.face_pack)
        self.pack.setFixedWidth(120)
        self.mask = QComboBox()
        self.mask.addItems(["mediapipe", "off"])
        self.mask.setFixedWidth(120)
        self.gpu = QCheckBox("GPU(DirectML)")
        self.gpu.setChecked(ctx.cfg.use_gpu)
        pv.addWidget(_row(self.limit, self.pack, self.mask, self.gpu,
                          labels=("图片上限", "识别包", "掩码", "")))
        self.run_btn = QPushButton("▶ 运行标注")
        self.run_btn.setProperty("role", "primary")
        self.run_btn.clicked.connect(self._run)
        pv.addWidget(self.run_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        v.addWidget(params)

        v.addWidget(subheading("结果概览"))
        self.summary = muted("尚无标注结果")
        v.addWidget(self.summary)
        self.bar_color = HBarChart()
        self.bar_skin = HBarChart()
        self.bar_len = HBarChart()
        v.addWidget(self.bar_color)
        v.addWidget(self.bar_skin)
        v.addWidget(self.bar_len)
        v.addStretch(1)
        self.refresh()

    def _run(self):
        cfg = self.ctx.cfg
        cfg.face_pack = self.pack.currentText()
        cfg.mask_provider = self.mask.currentText()
        cfg.use_gpu = self.gpu.isChecked()
        limit = self.limit.value() or None
        self.ctx.run(task_annotate, "标注", args=(cfg, limit), on_done=lambda _r: self.refresh())

    def refresh(self):
        recs = self.ctx.annotations()
        if not recs:
            self.summary.setText("尚无标注结果（先运行标注）")
            for b in (self.bar_color, self.bar_skin, self.bar_len):
                b.set_data("", [])
            return
        from ..cluster import face_quality_fail

        total = len(recs)
        unread = sum(1 for r in recs if r.unread)
        noface = sum(1 for r in recs if r.face is None and not r.unread)
        bad = sum(1 for r in recs if r.face is not None
                  and face_quality_fail(r, self.ctx.cfg) is not None)
        self.summary.setText(
            f"总计 {total} · 有脸 {total - noface - bad - unread} · 坏脸 {bad} · "
            f"无人脸 {noface} · 失败 {unread}")

        def counter(fn) -> list[tuple[str, int]]:
            d: dict[str, int] = {}
            for r in recs:
                if not r.unread:
                    d[fn(r)] = d.get(fn(r), 0) + 1
            return sorted(d.items(), key=lambda kv: -kv[1])

        self.bar_color.set_data("发色", counter(lambda r: r.attrs.hair_color or "无效"))
        self.bar_len.set_data("发长", counter(lambda r: r.attrs.hair_length or "无效"))
        skin = counter(lambda r: r.attrs.skin_word or "无效")
        self.bar_skin.set_data("肤色", skin)
        self.ctx.mark_step(1, "✓" if total else "○")


# ============================ ③ 聚簇 ============================
class ClusterStage(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        v = QVBoxLayout(self)
        v.setSpacing(12)
        v.addWidget(heading("③ 聚簇 · 簇 = 同一个人"))
        self.info = muted("尚无聚簇结果")
        v.addWidget(self.info)

        self.thr = QDoubleSpinBox()
        self.thr.setRange(0.30, 0.95)
        self.thr.setSingleStep(0.01)
        self.thr.setValue(self.ctx.cfg.identity_thr)
        self.thr.setFixedWidth(110)
        self.run_btn = QPushButton("▶ 运行聚簇")
        self.run_btn.setProperty("role", "primary")
        self.run_btn.clicked.connect(self._run)
        v.addWidget(_row(self.thr, self.run_btn, labels=("同人阈值",)))

        self.grid = ThumbGrid(self.ctx.thumbs, icon=84)
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        v.addWidget(self.grid, 1)

        self.merge_btn = QPushButton("合并选中")
        self.merge_btn.clicked.connect(self._merge)
        self.split_btn = QPushButton("拆分")
        self.split_btn.clicked.connect(self._split)
        self.split_k = QSpinBox()
        self.split_k.setRange(2, 8)
        self.split_k.setValue(2)
        self.split_k.setFixedWidth(64)
        v.addWidget(_row(self.merge_btn, self.split_k, self.split_btn,
                         labels=("", "拆成", "")))
        v.addWidget(muted("ⓘ 合并/拆分是当前 clusters.json 的会话级修正（纠正识别错分）；"
                          "重跑「运行聚簇」会覆盖。"))
        self._cids: list[int] = []  # 网格行 → cid

    def _run(self):
        cfg = self.ctx.cfg
        cfg.identity_thr = self.thr.value()
        self.ctx.run(task_cluster, "身份聚簇", args=(cfg,),
                     on_done=lambda _r: self.ctx.after_clusters_changed())

    def refresh(self, clusters: list[dict], tail: dict | None = None):
        self.grid.reset_grid()
        self._cids = []
        for c in clusters:
            self.grid.add(c.get("medoid"), c.get("medoid_path"),
                          f"簇{c['cid']}·{len(c['members'])}",
                          tip=f"簇 {c['cid']} 共 {len(c['members'])} 图")
            self._cids.append(c["cid"])
        n_tail = sum(len(v) for v in (tail or {}).values())
        biggest = max((len(c["members"]) for c in clusters), default=0)
        self.info.setText(f"{len(clusters)} 簇 · 最大簇 {biggest} 图 · 尾部 {n_tail} 图")
        self.ctx.mark_step(2, "✓" if clusters else "○")

    def _selected(self) -> list[int]:
        return [self._cids[i.row()] for i in self.grid.selectedIndexes()]

    def _merge(self):
        cids = self._selected()
        if len(cids) >= 2:
            self.ctx.run(task_merge_clusters, "合并簇", args=(self.ctx.cfg, cids),
                         on_done=lambda _r: self.ctx.after_clusters_changed())

    def _split(self):
        cids = self._selected()
        if len(cids) == 1:
            self.ctx.run(task_split_cluster, "拆分簇",
                         args=(self.ctx.cfg, cids[0], self.split_k.value()),
                         on_done=lambda _r: self.ctx.after_clusters_changed())


# ============================ ④ 发型 ============================
class HairstyleStage(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        cfg = ctx.cfg
        v = QVBoxLayout(self)
        v.setSpacing(10)
        v.addWidget(heading("④ 发型 · SigLIP2 嵌入三模式"))

        modes = QWidget()
        mh = QHBoxLayout(modes)
        mh.setContentsMargins(0, 0, 0, 0)
        self.mode_btns = []
        for i, t in enumerate(("B 嵌入聚类", "A 固定清单", "C 示例标注")):
            rb = QRadioButton(t)
            rb.setChecked(i == 0)
            rb.toggled.connect(self._mode_changed)
            self.mode_btns.append(rb)
            mh.addWidget(rb)
        mh.addStretch(1)
        v.addWidget(modes)

        opt = QWidget()
        ov = QVBoxLayout(opt)
        ov.setContentsMargins(0, 0, 0, 0)
        self.k_spin = QSpinBox()
        self.k_spin.setRange(0, 12)
        self.k_spin.setSpecialValueText("自动（轮廓系数）")
        self.k_spin.setFixedWidth(190)
        self.k_row = _row(self.k_spin, labels=("组数 K",))
        ov.addWidget(self.k_row)
        self.ex_edit = QLineEdit()
        self.ex_edit.setPlaceholderText("示例图路径，逗号分隔")
        self.ex_btn = QPushButton("选择示例…")
        self.ex_btn.clicked.connect(self._pick_examples)
        self.ex_row = _row(self.ex_edit, self.ex_btn, labels=("示例图",), fill=True)
        ov.addWidget(self.ex_row)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("模式 A 清单，每行：类目 | 英文提示词1; 英文提示词2")
        self.prompt_edit.setFixedHeight(120)
        self.prompt_save = QPushButton("保存清单并重算（模式 A）")
        self.prompt_save.clicked.connect(self._save_prompts)
        ov.addWidget(self.prompt_edit)
        ov.addWidget(self.prompt_save, alignment=Qt.AlignmentFlag.AlignLeft)
        v.addWidget(opt)

        self.run_btn = QPushButton("▶ 计算发型分组")
        self.run_btn.setProperty("role", "primary")
        self.run_btn.clicked.connect(self._run)
        self.apply_btn = QPushButton("应用为分区键")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.ctx.apply_hairstyle_key)
        v.addWidget(_row(self.run_btn, self.apply_btn))

        self.soft_cb = QCheckBox("软排序：不进分区键，段内同发型组相邻")
        v.addWidget(self.soft_cb)
        self.hint = muted("")
        v.addWidget(self.hint)

        self.grid = ThumbGrid(self.ctx.thumbs, icon=64)
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.grid.setFixedHeight(200)  # 容两行（icon 64 + 标签），避免第二行标签被裁掉
        v.addWidget(self.grid)
        self.merge_btn = QPushButton("合并选中组")
        self.merge_btn.clicked.connect(self._merge)
        self.split_btn = QPushButton("拆分组 k=2")
        self.split_btn.clicked.connect(self._split)
        v.addWidget(_row(self.merge_btn, self.split_btn))

        self.suspect_label = muted("盘束疑似（updo_suspect）：尚无数据")
        v.addWidget(self.suspect_label)
        self.suspects = ThumbGrid(self.ctx.thumbs, icon=56)
        self.suspects.setFixedHeight(170)
        v.addWidget(self.suspects)
        v.addStretch(1)

        self._names: list[str] = []
        self._gchash: dict[int, str] = {}
        self._suspect_rows: list[str] = []
        self._mode_changed()
        self.refresh()

    # ---- 模式切换 ----
    def _mode_index(self) -> int:
        return next(i for i, rb in enumerate(self.mode_btns) if rb.isChecked())

    def _mode_changed(self, *_):
        i = self._mode_index()
        self.k_row.setVisible(i == 0)
        self.ex_row.setVisible(i == 2)
        self.prompt_edit.setVisible(i == 1)
        self.prompt_save.setVisible(i == 1)
        if i == 1:
            self._load_editor()

    def _pick_examples(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择示例图", "",
                                                "图片 (*.jpg *.jpeg *.png *.webp *.bmp)")
        if paths:
            self.ex_edit.setText(",".join(paths))

    def _load_editor(self):
        from ..hairstyle import load_prompt_list

        lines = [f"{lab} | {'; '.join(ps)}" for lab, ps in load_prompt_list(self.ctx.cfg)]
        self.prompt_edit.setPlainText("\n".join(lines))

    def _save_prompts(self):
        import json

        from ..hairstyle import prompts_file

        items = []
        for raw in self.prompt_edit.toPlainText().splitlines():
            if not raw.strip():
                continue
            label, _, prompts = raw.partition("|")
            ps = [p.strip() for p in prompts.split(";") if p.strip()]
            if label.strip() and ps:
                items.append({"label": label.strip(), "prompts": ps})
        if not items:
            self.hint.setText("清单为空——至少一行『标签 | 提示词』")
            return
        prompts_file(self.ctx.cfg).write_text(
            json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        self.hint.setText(f"清单已保存（{len(items)} 类），重算提示词嵌入…")
        self.ctx.run(task_hairstyle, "发型分组",
                     args=(self.ctx.cfg, "A", None, [], True),
                     on_done=lambda _r: self.ctx.after_hairstyle_changed())

    # ---- 运行 ----
    def _run(self):
        mode = "BAC"[self._mode_index()]
        k = self.k_spin.value() or None
        examples = ([p.strip() for p in self.ex_edit.text().split(",") if p.strip()]
                    if mode == "C" else [])
        self.ctx.run(task_hairstyle, "发型分组",
                     args=(self.ctx.cfg, mode, k, examples, False),
                     on_done=lambda _r: self.ctx.after_hairstyle_changed())

    def refresh(self):
        cfg = self.ctx.cfg
        p = cfg.out_dir / "hairstyle.json"
        if not p.is_file():
            self.hint.setText("尚无发型分组结果——点「计算发型分组」")
            self.grid.reset_grid()
            self._names = []
            self.apply_btn.setEnabled(False)
            self.ctx.mark_step(3, "○")
            return
        import json

        from ..hairstyle import representative_chashes

        hs = json.loads(p.read_text(encoding="utf-8"))
        self.grid.reset_grid()
        self._names, self._gchash = [], {}
        paths = self.ctx.chash_paths()
        for row, g in enumerate(hs["group_order"]):
            members = hs["groups"].get(g) or []
            reps = representative_chashes(cfg, members, 3)
            ch = reps[0] if reps else None
            tip = "、".join(Path(paths.get(c, c)).name for c in reps)
            self.grid.add(ch, paths.get(ch) if ch else None, f"{g}（{len(members)}）", tip)
            self._names.append(g)
            self._gchash[row] = ch
        edited = "，含手工编辑" if hs.get("manual_edits") else ""
        sizes = sorted((len(v) for v in hs["groups"].values()), reverse=True)
        self.hint.setText(f"模式 {hs['mode']} · {len(hs['group_order'])} 组{edited}"
                          f"（top: {sizes[:5]}）")
        self.apply_btn.setEnabled(True)
        self.ctx.mark_step(3, "✓" if hs["group_order"] else "○")

    def show_suspects(self, items: list[tuple[str, str]]):
        self.suspects.reset_grid()
        self._suspect_rows = []
        if not items:
            self.suspect_label.setText("盘束疑似（updo_suspect）：本包无疑似样本")
            return
        self.suspect_label.setText(
            f"盘束疑似（updo_suspect，{len(items)} 张）· 与上方分组交叉核对，不自动改判")
        for chash, path in items:
            self.suspects.add(chash, path, Path(path).name, tip=path)

    def _selected(self) -> list[str]:
        return [self._names[i.row()] for i in self.grid.selectedIndexes()]

    def _merge(self):
        names = self._selected()
        if len(names) >= 2:
            self.ctx.run(task_merge_hs, "合并发型组", args=(self.ctx.cfg, names),
                         on_done=lambda _r: self.ctx.after_hairstyle_changed())

    def _split(self):
        names = self._selected()
        if len(names) == 1:
            self.ctx.run(task_split_hs, "拆分发型组", args=(self.ctx.cfg, names[0]),
                         on_done=lambda _r: self.ctx.after_hairstyle_changed())


# ============================ ⑤ 分区排序 ============================
class SortStage(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        v = QVBoxLayout(self)
        v.setSpacing(10)
        v.addWidget(heading("⑤ 分区排序 · 键序即分区序（纯内存秒级重放）"))

        split = QHBoxLayout()
        split.setSpacing(12)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(subheading("键序"))
        self.key_panel = KeyChainPanel(self._keys(), self.ctx.cfg.key_order,
                                       parent=left)
        lv.addWidget(self.key_panel)
        self.soft_cb = QCheckBox("软排序：发型不硬分区，段内同组相邻")
        lv.addWidget(self.soft_cb)
        self.replan_btn = QPushButton("▶ 重新排序")
        self.replan_btn.setProperty("role", "primary")
        self.replan_btn.clicked.connect(self.ctx.replan)
        lv.addWidget(self.replan_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        lv.addStretch(1)
        left.setFixedWidth(300)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(subheading("F 值（越大段内越连贯）"))
        self.fpanel = FValuePanel()
        rv.addWidget(self.fpanel)
        split.addWidget(right, 1)
        v.addLayout(split)

        head = QHBoxLayout()
        head.addWidget(subheading("预览"))
        self.count = muted("")
        head.addWidget(self.count)
        head.addStretch(1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索 原名 / 新名 / 段 / 簇号…")
        self.search.setFixedWidth(240)
        self.search.textChanged.connect(self._on_search)
        head.addWidget(self.search)
        v.addLayout(head)

        self.model = PlanTableModel(self.ctx.thumbs, self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setDefaultSectionSize(100)  # ≥ 缩略图边长(96)，否则图标被行高裁掉
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        for col, w in ((0, 100), (1, 52), (4, 130), (5, 74)):
            self.table.setColumnWidth(col, w)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table, 1)

        self.key_panel.keysChanged.connect(self.ctx.on_keys_changed)

    def _keys(self) -> list[str]:
        has_hs = (self.ctx.cfg.out_dir / "hairstyle.json").is_file()
        return list(KEY_REGISTRY) + ([HAIRSTYLE_KEY] if has_hs else [])

    def rebuild_keys(self):
        """发型键注册与否会变（生成 hairstyle.json 后），重建键列表面板。

        以当前勾选与顺序为初始态重建，避免"生成发型 → 键序被重置"的意外。
        """
        parent = self.key_panel.parentWidget()
        layout = parent.layout()
        idx = layout.indexOf(self.key_panel)
        selected = self.current_keys() or self.ctx.cfg.key_order
        self.key_panel.setParent(None)
        self.key_panel = KeyChainPanel(self._keys(), selected, parent=parent)
        self.key_panel.keysChanged.connect(self.ctx.on_keys_changed)
        layout.insertWidget(idx, self.key_panel)

    def current_keys(self) -> list[str]:
        return self.key_panel.current_keys()

    def ensure_key(self, key: str):
        self.key_panel.ensure_key(key)

    def refresh(self, plan: dict | None):
        if plan is None:
            self.count.setText("无 plan.json——先完成 聚簇 与 排序")
            self.ctx.mark_step(4, "○")
            return
        self.model.set_plan(plan)
        self.fpanel.refresh(plan)
        self._on_search(self.search.text())
        self.ctx.mark_step(4, "✓")

    def _on_search(self, text: str):
        self.model.set_filter(text)
        self.count.setText(f"{self.model.visible_count} / {len(self.model.items)} 条目")


# ============================ ⑥ 输出 ============================
class OutputStage(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        cfg = ctx.cfg
        v = QVBoxLayout(self)
        v.setSpacing(10)
        v.addWidget(heading("⑥ 输出 · 重命名与报告"))

        opts = QWidget()
        oh = QHBoxLayout(opts)
        oh.setContentsMargins(0, 0, 0, 0)
        self.copy_rb = QRadioButton("copy 副本（原包不动）")
        self.inplace_rb = QRadioButton("inplace 原地改名（可回滚）")
        (self.copy_rb if cfg.rename_mode == "copy" else self.inplace_rb).setChecked(True)
        self.dry_cb = QCheckBox("预演（不动文件）")
        self.dry_cb.setChecked(True)
        self.natural_cb = QCheckBox("纯自然序号 1..N")
        oh.addWidget(self.copy_rb)
        oh.addWidget(self.inplace_rb)
        oh.addSpacing(12)
        oh.addWidget(self.dry_cb)
        oh.addWidget(self.natural_cb)
        oh.addStretch(1)
        v.addWidget(opts)

        split = QHBoxLayout()
        split.setSpacing(12)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["原文件", "新名"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        split.addWidget(self.table, 1)

        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(8)
        sv.addWidget(subheading("重命名"))
        self.dry_btn = QPushButton("预演")
        self.dry_btn.clicked.connect(lambda: self._rename(True))
        self.exec_btn = QPushButton("执行重命名")
        self.exec_btn.setProperty("role", "primary")
        self.exec_btn.clicked.connect(lambda: self._rename(False))
        self.rollback_btn = QPushButton("回滚 inplace")
        self.rollback_btn.setProperty("role", "danger")
        self.rollback_btn.clicked.connect(self._rollback)
        sv.addWidget(self.dry_btn)
        sv.addWidget(self.exec_btn)
        sv.addWidget(self.rollback_btn)
        sv.addSpacing(10)
        sv.addWidget(subheading("报告"))
        self.rep_sort_btn = QPushButton("生成排序报告")
        self.rep_sort_btn.clicked.connect(lambda: self._report("sort"))
        self.open_sort_btn = QPushButton("打开排序报告")
        self.open_sort_btn.clicked.connect(lambda: _open(cfg.out_dir / "sort_report.html"))
        self.rep_match_btn = QPushButton("生成接触表")
        self.rep_match_btn.clicked.connect(lambda: self._report("contact"))
        self.open_match_btn = QPushButton("打开接触表")
        self.open_match_btn.clicked.connect(lambda: _open(cfg.out_dir / "contact_sheet.html"))
        for b in (self.rep_sort_btn, self.open_sort_btn, self.rep_match_btn, self.open_match_btn):
            sv.addWidget(b)
        sv.addStretch(1)
        side.setFixedWidth(220)
        split.addWidget(side)
        v.addLayout(split)

        self.warn = muted("")
        v.addWidget(self.warn)
        self.refresh()

    def _rename(self, dry: bool):
        cfg = self.ctx.cfg
        cfg.rename_mode = "inplace" if self.inplace_rb.isChecked() else "copy"
        cfg.natural_order = self.natural_cb.isChecked()
        dry = dry or self.dry_cb.isChecked()
        self.ctx.run(task_rename, "重命名（预演）" if dry else "重命名",
                     args=(cfg, dry), on_done=lambda _r: self.refresh())

    def _rollback(self):
        self.ctx.run(task_rollback, "回滚 inplace", args=(self.ctx.cfg,),
                     on_done=lambda _r: self.refresh())

    def _report(self, which: str):
        self.ctx.run(task_report, "生成报告", args=(self.ctx.cfg, which))

    def refresh(self):
        from .. import rename

        cfg = self.ctx.cfg
        try:
            plan = rename.load_plan(cfg)
        except FileNotFoundError as e:
            self.warn.setText(str(e))
            self.table.setRowCount(0)
            self.ctx.mark_step(5, "○")
            for b in (self.dry_btn, self.exec_btn):
                b.setEnabled(False)
            return
        self.dry_btn.setEnabled(True)
        self.exec_btn.setEnabled(True)
        try:
            conflicts = rename.check_conflicts(cfg, plan)
        except OSError:
            conflicts = []
        drift = [it for it in plan["sequence"] if rename._drifted(it)]
        seq = plan["sequence"]
        self.table.setRowCount(len(seq))
        for r, it in enumerate(seq):
            self.table.setItem(r, 0, QTableWidgetItem(Path(it["path"]).name))
            self.table.setItem(r, 1, QTableWidgetItem(it["new_name"]))
            if it in drift:
                for c in (0, 1):
                    self.table.item(r, c).setBackground(QColor(PAL["warn"]))
        mode = cfg.rename_mode
        target = rename.target_dir(cfg) if mode == "copy" else cfg.sample_dir
        self.warn.setText(f"模式 {mode} · 冲突 {len(conflicts)} · 漂移 {len(drift)} · 目标 {target}")
        self.warn.setStyleSheet(f"color:{PAL['danger'] if (conflicts or drift) else PAL['muted']};")
        ok = not conflicts
        self.exec_btn.setEnabled(ok)
        self.exec_btn.setToolTip("" if ok else "存在冲突，无法执行（详见重命名对话框）")
        self.ctx.mark_step(5, "✓")


# ---------- 布局小工具 ----------
def _row(*widgets, labels=(), spacing=8, fill=False) -> QWidget:
    """把若干控件横向排成一行，包成容器控件返回（labels 在对应控件前插入同级标签）。

    容器必须包一层：若把 QHBoxLayout 直接嵌进父布局，父布局给的一整行宽度里的剩余
    空白会全部分配给行内某个子控件——全定宽的行里会落到第一个 QLabel 上（实测「组数 K」
    被撑到 640px，把后面的输入框顶到很远处）。包一层后在行尾加 stretch 吸收空白即可。
    fill=True 时行尾不加 stretch，留给需要占满整行的输入控件（目录/示例图行）。
    """
    box = QWidget()
    h = QHBoxLayout(box)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(spacing)
    for i, w in enumerate(widgets):
        if i < len(labels) and labels[i]:
            lab = QLabel(labels[i])
            lab.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            h.addWidget(lab)
        if not fill and w.minimumWidth() == w.maximumWidth() and w.minimumWidth() > 0:
            w.setSizePolicy(QSizePolicy.Policy.Fixed, w.sizePolicy().verticalPolicy())
        h.addWidget(w)
    if not fill:
        h.addStretch(1)
    return box


def _dml_available():
    try:
        import onnxruntime as ort

        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False
