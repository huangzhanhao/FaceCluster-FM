# -*- coding: utf-8 -*-
"""控制面板与对话框：KeyOrderPanel / FValuePanel / ClusterPanel / HairstylePanel / RenameDialog。"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit,
    QPushButton, QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .. import rename
from ..config import Config
from .thumbs import ThumbStore


class KeyOrderPanel(QWidget):
    """分区键编辑：勾选 = 参与分区，拖拽 = 键序。任何变更即时发 keysChanged。"""

    keysChanged = Signal(list)

    def __init__(self, keys: list[str], selected: list[str], parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.addWidget(QLabel("分区键（勾选参与，拖拽排序）"))
        self.listw = QListWidget()
        self.listw.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.listw.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.listw.setFixedHeight(28 * (len(keys) + 1))
        for k in keys:
            it = QListWidgetItem(k)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            it.setCheckState(Qt.Checked if k in selected else Qt.Unchecked)
            self.listw.addItem(it)
        self.listw.model().rowsMoved.connect(self._emit)
        self.listw.itemChanged.connect(self._emit)
        v.addWidget(self.listw)

    def current_keys(self) -> list[str]:
        return [self.listw.item(i).text() for i in range(self.listw.count())
                if self.listw.item(i).checkState() == Qt.Checked]

    def ensure_key(self, key: str) -> None:
        """键纳入面板并勾选（无则追加）；勾选变更经 itemChanged → 自动重排。"""
        for i in range(self.listw.count()):
            if self.listw.item(i).text() == key:
                if self.listw.item(i).checkState() != Qt.Checked:
                    self.listw.item(i).setCheckState(Qt.Checked)
                return
        it = QListWidgetItem(key)
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
        it.setCheckState(Qt.Checked)
        self.listw.addItem(it)
        self.listw.setFixedHeight(28 * (self.listw.count() + 1))

    def _emit(self, *_):
        self.keysChanged.emit(self.current_keys())


class FValuePanel(QWidget):
    """F 值（R11）：每级段内相邻 cos 之和 + 最差相邻对提示。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.addWidget(QLabel("F 值（越大段内越连贯）"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["级别", "键", "F 值", "段数"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(140)
        v.addWidget(self.table)
        self.worst = QLabel("")
        self.worst.setWordWrap(True)
        v.addWidget(self.worst)

    def refresh(self, plan: dict):
        f = plan.get("fvalue", {})
        rows = f.get("per_level", [])
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, val in enumerate((f"L{r['level']}", "→".join(r["keys"]),
                                     f"{r['f']:.1f}", str(r["n_seg"]))):
                self.table.setItem(i, j, QTableWidgetItem(val))
        wp = f.get("worst_pairs") or []
        if wp:
            w = wp[0]
            self.worst.setText(
                f"最差相邻对：{Path(w['a']).name} ↔ {Path(w['b']).name}（cos={w['sim']}）")
        else:
            self.worst.setText("")


class ClusterPanel(QWidget):
    """身份簇面板：medoid 代表图网格，多选合并 / 单选拆分(k=2)。"""

    mergeRequested = Signal(list)
    splitRequested = Signal(int)

    def __init__(self, thumbs: ThumbStore, parent=None):
        super().__init__(parent)
        self.thumbs = thumbs
        v = QVBoxLayout(self)
        v.addWidget(QLabel("身份簇（多选→合并，单选→拆分）"))
        self.listw = QListWidget()
        self.listw.setViewMode(QListWidget.ViewMode.IconMode)
        self.listw.setIconSize(QSize(72, 72))
        self.listw.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.listw.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        v.addWidget(self.listw, 1)
        h = QHBoxLayout()
        self.merge_btn = QPushButton("合并选中")
        self.split_btn = QPushButton("拆分(k=2)")
        h.addWidget(self.merge_btn)
        h.addWidget(self.split_btn)
        v.addLayout(h)
        self.merge_btn.clicked.connect(self._merge)
        self.split_btn.clicked.connect(self._split)
        thumbs.ready.connect(self._on_thumb)
        self._meta: dict[int, tuple[int, str, str | None]] = {}

    def refresh(self, clusters: list[dict]):
        self.listw.clear()
        self._meta = {}
        for row, c in enumerate(clusters):
            chash, path = c["medoid"], c.get("medoid_path")
            it = QListWidgetItem(QIcon(self.thumbs.pixmap(chash, path)),
                                 f"簇{c['cid']}·{len(c['members'])}")
            self.listw.addItem(it)
            self._meta[row] = (c["cid"], chash, path)

    def _on_thumb(self, chash: str):
        for row in range(self.listw.count()):
            cid, ch, p = self._meta.get(row, (None, None, None))
            if ch == chash:
                self.listw.item(row).setIcon(QIcon(self.thumbs.pixmap(ch, p)))

    def _merge(self):
        cids = [self._meta[i.row()][0] for i in self.listw.selectedIndexes()]
        if len(cids) >= 2:
            self.mergeRequested.emit(cids)

    def _split(self):
        sel = self.listw.selectedIndexes()
        if len(sel) == 1:
            self.splitRequested.emit(self._meta[sel[0].row()][0])


class HairstylePanel(QWidget):
    """发型三模式 + 手工组编辑：A 清单可改 / B 聚类看图合并拆组 / C 示例；软排序开关。"""

    runRequested = Signal(str, object, list, bool)   # mode, k, examples, refresh_prompts
    applyRequested = Signal()
    mergeRequested = Signal(list)
    splitRequested = Signal(str)

    def __init__(self, cfg: Config, thumbs: ThumbStore, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.thumbs = thumbs
        self._chash_path: dict[str, str] = {}  # annotations.jsonl 的 chash→path，供组卡片缩略图
        self._meta: dict[int, str] = {}
        self._group_chash: dict[int, str] = {}   # 组卡片行 → 代表图 chash
        self._suspect_items: dict[int, tuple[str, str]] = {}  # 盘束行 → (chash, path)
        v = QVBoxLayout(self)
        v.addWidget(QLabel("发型分组（SigLIP2 嵌入三模式共用）"))
        self.mode = QComboBox()
        self.mode.addItems(["B 嵌入聚类（推荐）", "A 固定清单零样本", "C 示例标注"])
        v.addWidget(self.mode)
        self.k_label = QLabel("组数 K")
        self.k_spin = QSpinBox()
        self.k_spin.setRange(0, 12)
        self.k_spin.setValue(0)
        self.k_spin.setSpecialValueText("K 自动（轮廓系数）")
        hk = QHBoxLayout()
        hk.addWidget(self.k_label)
        hk.addWidget(self.k_spin)
        v.addLayout(hk)
        self.soft_cb = QCheckBox("软排序：不进分区键，段内同发型组相邻（§4.3）")
        v.addWidget(self.soft_cb)
        self.ex_edit = QLineEdit()
        self.ex_edit.setPlaceholderText("示例图路径，逗号分隔")
        self.ex_btn = QPushButton("选择示例…")
        exh = QHBoxLayout()
        exh.addWidget(self.ex_edit)
        exh.addWidget(self.ex_btn)
        v.addLayout(exh)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("模式 A 清单，每行：类目 | 英文提示词1; 英文提示词2\n保存后重算提示词嵌入（需 text_model，一次性）")
        self.prompt_edit.setMaximumHeight(130)
        v.addWidget(self.prompt_edit)
        self.prompt_save = QPushButton("保存清单并重算（模式 A）")
        v.addWidget(self.prompt_save)
        self.run_btn = QPushButton("计算发型分组")
        v.addWidget(self.run_btn)
        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        v.addWidget(self.hint)
        self.apply_btn = QPushButton("应用为分区键")
        self.apply_btn.setEnabled(False)
        v.addWidget(self.apply_btn)

        v.addWidget(QLabel("分组结果（多选→合并，单选→拆组；改动写入 hairstyle.json）"))
        self.group_list = QListWidget()
        self.group_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.group_list.setIconSize(QSize(64, 64))
        self.group_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.group_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.group_list.setFixedHeight(120)
        v.addWidget(self.group_list)
        gh = QHBoxLayout()
        self.merge_btn = QPushButton("合并选中组")
        self.split_btn = QPushButton("拆分组(k=2)")
        gh.addWidget(self.merge_btn)
        gh.addWidget(self.split_btn)
        v.addLayout(gh)

        self.suspect_label = QLabel("盘束疑似（updo_suspect，与上方分组交叉核对，不自动改判）")
        self.suspect_list = QListWidget()
        self.suspect_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.suspect_list.setIconSize(QSize(56, 56))
        self.suspect_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.suspect_list.setFixedHeight(100)
        v.addWidget(self.suspect_label)
        v.addWidget(self.suspect_list)
        v.addStretch(1)

        self.ex_edit.hide()
        self.ex_btn.hide()
        self.prompt_edit.hide()
        self.prompt_save.hide()
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.ex_btn.clicked.connect(self._pick)
        self.run_btn.clicked.connect(self._emit_run)
        self.apply_btn.clicked.connect(self.applyRequested.emit)
        self.prompt_save.clicked.connect(self._save_prompts)
        self.merge_btn.clicked.connect(self._merge)
        self.split_btn.clicked.connect(self._split)
        # 组卡片/盘束列表的缩略图未命中时走后台生成，完成后经 ready 局部补绘
        self.thumbs.ready.connect(self._on_thumb_ready)

    # ---- 模式切换 ----
    def _mode_changed(self, i):
        self.ex_edit.setVisible(i == 2)
        self.ex_btn.setVisible(i == 2)
        self.k_spin.setVisible(i == 0)
        self.k_label.setVisible(i == 0)
        self.prompt_edit.setVisible(i == 1)
        self.prompt_save.setVisible(i == 1)
        if i == 1:
            self._load_editor()

    def _pick(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择示例图", "", "图片 (*.jpg *.jpeg *.png *.webp *.bmp)")
        if paths:
            self.ex_edit.setText(",".join(paths))

    # ---- 模式 A 清单编辑（§4.3：用户可增删改）----
    def _load_editor(self):
        from ..hairstyle import load_prompt_list

        lines = [f"{lab} | {'; '.join(ps)}" for lab, ps in load_prompt_list(self.cfg)]
        self.prompt_edit.setPlainText("\n".join(lines))

    def _save_prompts(self):
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
        prompts_file(self.cfg).write_text(
            json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        self.hint.setText(f"清单已保存（{len(items)} 类），重算提示词嵌入…")
        self.runRequested.emit("A", None, [], True)

    def _emit_run(self):
        mode = "ABC"[self.mode.currentIndex()]
        k = self.k_spin.value() or None
        examples = ([p.strip() for p in self.ex_edit.text().split(",") if p.strip()]
                    if mode == "C" else [])
        self.runRequested.emit(mode, k, examples, False)

    # ---- 状态 ----
    def set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        if running:
            self.hint.setText("计算中（嵌入缓存命中时仅模式计算）…")

    def show_result(self, hs: dict):
        sizes = sorted((len(v) for v in hs["groups"].values()), reverse=True)
        edited = "，含手工编辑" if hs.get("manual_edits") else ""
        self.hint.setText(
            f"模式 {hs['mode']} 已生成 {len(hs['group_order'])} 组{edited}（top: {sizes[:5]}）")
        self.apply_btn.setEnabled(True)

    # ---- 组卡片（看图合并/拆组）----
    def set_chash_paths(self, mapping: dict[str, str]):
        self._chash_path = mapping

    def refresh_groups(self, hs: dict):
        from ..hairstyle import representative_chashes

        self.group_list.clear()
        self._meta = {}
        self._group_chash = {}
        for row, g in enumerate(hs["group_order"]):
            members = hs["groups"].get(g) or []
            reps = representative_chashes(self.cfg, members, 3)
            icon_chash = reps[0] if reps else None
            icon = QIcon(self.thumbs.pixmap(icon_chash, self._chash_path.get(icon_chash))) \
                if icon_chash else QIcon()
            tip = "、".join(Path(self._chash_path.get(c, c)).name for c in reps)
            it = QListWidgetItem(icon, f"{g}（{len(members)}）")
            it.setToolTip(tip)
            self.group_list.addItem(it)
            self._meta[row] = g
            self._group_chash[row] = icon_chash

    def _on_thumb_ready(self, chash: str):
        """后台缩略图生成完成：补绘对应组卡片与盘束疑似行（未命中时行图标先为空）。"""
        for row, c in self._group_chash.items():
            if c == chash:
                self.group_list.item(row).setIcon(
                    QIcon(self.thumbs.pixmap(chash, self._chash_path.get(chash))))
        for row, (c, path) in self._suspect_items.items():
            if c == chash:
                self.suspect_list.item(row).setIcon(QIcon(self.thumbs.pixmap(chash, path)))

    def _merge(self):
        names = [self._meta[i.row()] for i in self.group_list.selectedIndexes()]
        if len(names) >= 2:
            self.mergeRequested.emit(names)

    def _split(self):
        sel = self.group_list.selectedIndexes()
        if len(sel) == 1:
            self.splitRequested.emit(self._meta[sel[0].row()])

    # ---- 盘束疑似交叉核对（§4.2：标记不自动改判）----
    def show_suspects(self, items: list[tuple[str, str]]):
        self.suspect_list.clear()
        self._suspect_items = {}
        if not items:
            self.suspect_label.setText("盘束疑似（updo_suspect）：本包无疑似样本")
            return
        self.suspect_label.setText(
            f"盘束疑似（updo_suspect，{len(items)} 张，与上方分组交叉核对，不自动改判）")
        for row, (chash, path) in enumerate(items):
            it = QListWidgetItem(QIcon(self.thumbs.pixmap(chash, path)),
                                 Path(path).name)
            it.setToolTip(Path(path).name)
            self.suspect_list.addItem(it)
            self._suspect_items[row] = (chash, path)


class RenameDialog(QDialog):
    """重命名：plan 前后名对照（漂移红标）→ 模式/预演选择 → 确认执行。"""

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量重命名")
        self.cfg = cfg
        self.resize(760, 560)
        self.plan = rename.load_plan(cfg)
        self.conflicts = rename.check_conflicts(cfg, self.plan)
        self.drift = {it["chash"]: d for it in self.plan["sequence"]
                      if (d := rename._drifted(it))}

        v = QVBoxLayout(self)
        mode = cfg.rename_mode
        banner = (f"模式 {mode}：{len(self.plan['sequence'])} 条 → "
                  f"{rename.target_dir(cfg) if mode == 'copy' else cfg.sample_dir}")
        if self.conflicts:
            banner += f"　⚠ 冲突 {len(self.conflicts)} 个：{'；'.join(self.conflicts[:5])}"
        if self.drift:
            banner += f"　⚠ 漂移 {len(self.drift)} 条（将跳过）"
        lab = QLabel(banner)
        lab.setWordWrap(True)
        if self.conflicts or self.drift:
            lab.setStyleSheet("color:#c33")
        v.addWidget(lab)

        self.table = QTableWidget(len(self.plan["sequence"]), 2)
        self.table.setHorizontalHeaderLabels(["原文件", "新名"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for r, it in enumerate(self.plan["sequence"]):
            self.table.setItem(r, 0, QTableWidgetItem(Path(it["path"]).name))
            self.table.setItem(r, 1, QTableWidgetItem(it["new_name"]))
            if it["chash"] in self.drift:
                for cidx in (0, 1):
                    self.table.item(r, cidx).setBackground(Qt.red)
        v.addWidget(self.table, 1)

        h = QHBoxLayout()
        self.copy_rb = QRadioButton("copy 副本（默认，原包不动）")
        self.inplace_rb = QRadioButton("inplace 改名（可回滚）")
        (self.copy_rb if mode == "copy" else self.inplace_rb).setChecked(True)
        self.dry_cb = QCheckBox("预演（不动文件）")
        self.dry_cb.setChecked(True)
        self.run_btn = QPushButton("执行")
        self.run_btn.setEnabled(not self.conflicts)
        h.addWidget(self.copy_rb)
        h.addWidget(self.inplace_rb)
        h.addWidget(self.dry_cb)
        h.addWidget(self.run_btn)
        v.addLayout(h)

        self.run_btn.clicked.connect(self.accept)

    @property
    def chosen_mode(self) -> str:
        return "inplace" if self.inplace_rb.isChecked() else "copy"

    @property
    def dry_run(self) -> bool:
        return self.dry_cb.isChecked()
