# -*- coding: utf-8 -*-
"""主窗口：左控制面板（键序/F 值｜发型｜聚簇 三选项卡）｜中央预览表｜下日志坞。

任务互斥：同一时刻仅一个 Worker。键序拖拽/勾选经 300ms 防抖后自动重排
（改键序只重算 partition，纯内存秒级）；聚簇合并/拆分与发型分组完成后自动重排。
"""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QMainWindow, QPlainTextEdit,
    QProgressBar, QPushButton, QSplitter, QTabWidget, QTableView, QVBoxLayout,
    QWidget,
)

from .. import adjust, hairstyle, partition, rename
from ..annotate import annotate
from ..config import KEY_REGISTRY, Config
from .panels import ClusterPanel, FValuePanel, HairstylePanel, KeyOrderPanel, RenameDialog
from .table_model import PlanTableModel
from .thumbs import ThumbStore
from .workers import Worker

HAIRSTYLE_KEY = "发型"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = Config()
        self.setWindowTitle("FaceCluster-FM")
        self.resize(1280, 800)
        self.worker: Worker | None = None

        self.thumbs = ThumbStore(self.cfg.out_dir, self)
        self.model = PlanTableModel(self.thumbs, self)

        # ---- 左：三选项卡 ----
        has_hs = (self.cfg.out_dir / "hairstyle.json").is_file()
        self.key_panel = KeyOrderPanel(list(KEY_REGISTRY) + ([HAIRSTYLE_KEY] if has_hs else []),
                                       self.cfg.key_order)
        self.f_panel = FValuePanel()
        tab1 = QWidget()
        tv = QVBoxLayout(tab1)
        tv.addWidget(self.key_panel)
        tv.addWidget(self.f_panel)
        tv.addStretch(1)

        self.hs_panel = HairstylePanel(self.cfg, self.thumbs)
        self.cluster_panel = ClusterPanel(self.thumbs)

        tabs = QTabWidget()
        tabs.addTab(tab1, "键序")
        tabs.addTab(self.hs_panel, "发型")
        tabs.addTab(self.cluster_panel, "聚簇")

        self.annotate_btn = QPushButton("标注(annotate)")
        self.annotate_btn.clicked.connect(self._run_annotate)
        self.replan_btn = QPushButton("按当前键序重排")
        self.replan_btn.clicked.connect(self._replan)
        self.rename_btn = QPushButton("批量重命名…")
        self.rename_btn.clicked.connect(self._open_rename)
        btns = QHBoxLayout()
        btns.addWidget(self.annotate_btn)
        btns.addWidget(self.replan_btn)
        btns.addWidget(self.rename_btn)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(tabs, 1)
        lv.addLayout(btns)

        # ---- 中：预览表 ----
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setDefaultSectionSize(100)
        self.table.setColumnWidth(0, 108)
        self.table.setColumnWidth(1, 48)
        self.table.setColumnWidth(2, 220)
        self.table.setColumnWidth(4, 130)
        self.table.setColumnWidth(5, 70)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(left)
        top.addWidget(self.table)
        top.setStretchFactor(1, 1)
        top.setSizes([330, 950])

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        bottom = QSplitter(Qt.Orientation.Vertical)
        bottom.addWidget(top)
        bottom.addWidget(self.log_view)
        bottom.setSizes([620, 180])
        self.setCentralWidget(bottom)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._replan)
        self.key_panel.keysChanged.connect(self._on_keys_changed)

        # 发型：运行 → 展示结果 + 组卡片；应用 → 勾选发型键（触发自动重排）
        self.hs_panel.runRequested.connect(self._run_hairstyle)
        self.hs_panel.applyRequested.connect(lambda: self.key_panel.ensure_key(HAIRSTYLE_KEY))
        self.hs_panel.mergeRequested.connect(self._merge_hairstyle_groups)
        self.hs_panel.splitRequested.connect(self._split_hairstyle_group)

        # 聚簇：合并/拆分 → worker → 刷新面板 + 自动重排
        self.cluster_panel.mergeRequested.connect(self._merge_clusters)
        self.cluster_panel.splitRequested.connect(self._split_cluster)

        self._load_plan()
        self._refresh_clusters()

    # ---- 数据加载 ----
    def _on_keys_changed(self, keys: list):
        """发型键与软排序互斥：发型已作硬键时软排序无意义，禁用开关避免静默忽略。"""
        hs_key_on = HAIRSTYLE_KEY in keys
        cb = self.hs_panel.soft_cb
        cb.setEnabled(not hs_key_on)
        cb.setToolTip("发型已作为分区键，软排序不生效" if hs_key_on else "")
        if hs_key_on and cb.isChecked():
            cb.setChecked(False)
        self._debounce.start()

    def _load_plan(self):
        pf = self.cfg.out_dir / "plan.json"
        if not pf.is_file():
            self.log_view.appendPlainText(
                f"未找到 {pf}——先在命令行跑 annotate/cluster，或点\"按当前键序重排\"（需 clusters.json）")
            return
        plan = json.loads(pf.read_text(encoding="utf-8"))
        self.model.set_plan(plan)
        self.f_panel.refresh(plan)
        self.statusBar().showMessage(
            f"{plan['n_items']} 条目 / {plan['n_clusters']} 簇 / 键序 {'→'.join(plan['key_order'])}")

    def _refresh_clusters(self):
        p = self.cfg.out_dir / "clusters.json"
        if p.is_file():
            self.cluster_panel.refresh(json.loads(p.read_text(encoding="utf-8"))["clusters"])

    def _chash_paths(self) -> dict[str, str]:
        """annotations.jsonl 的 chash→path（发型组卡片/盘束疑似的缩略图来源），惰性缓存。"""
        if not hasattr(self, "_paths_map") or self._paths_map is None:
            self._paths_map: dict[str, str] = {}
            jl = self.cfg.out_dir / "annotations.jsonl"
            if jl.is_file():
                for line in jl.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        d = json.loads(line)
                        self._paths_map[d["content_hash"]] = d["path"]
        return self._paths_map

    # ---- 任务执行（单任务互斥）----
    def _busy(self) -> bool:
        if self.worker is not None and self.worker.isRunning():
            self.log_view.appendPlainText("⏳ 已有任务进行中，忽略新请求")
            return True
        return False

    def _run(self, fn, label: str, on_done=None):
        if self._busy():
            return
        self.progress.setRange(0, 0)
        self.progress.show()
        self.annotate_btn.setEnabled(False)
        self.replan_btn.setEnabled(False)
        self.rename_btn.setEnabled(False)
        self.hs_panel.set_running(True)
        w = Worker(fn)
        self.worker = w
        w.log.connect(self.log_view.appendPlainText)
        w.progress.connect(self._on_progress)
        w.done.connect(lambda r: self._task_done(label, on_done, r))
        w.error.connect(self._task_error)
        w.start()

    def _on_progress(self, i, n):
        self.progress.setRange(0, max(1, n))
        self.progress.setValue(i)

    def _task_done(self, label, on_done, result):
        self.progress.hide()
        self.annotate_btn.setEnabled(True)
        self.replan_btn.setEnabled(True)
        self.rename_btn.setEnabled(True)
        self.hs_panel.set_running(False)
        self.log_view.appendPlainText(f"✓ {label}完成")
        if on_done:
            on_done(result)

    def _task_error(self, msg):
        self.progress.hide()
        self.annotate_btn.setEnabled(True)
        self.replan_btn.setEnabled(True)
        self.rename_btn.setEnabled(True)
        self.hs_panel.set_running(False)
        self.log_view.appendPlainText(f"✗ 任务失败：{msg}")
        self.statusBar().showMessage(f"任务失败：{msg}", 8000)

    # ---- 标注（M0，GUI 入口）----
    def _run_annotate(self):
        if self._busy():
            return
        self.log_view.appendPlainText(
            f"—— 标注管线启动（{self.cfg.sample_dir}；有缓存的图秒过）——")
        self._run(lambda: annotate(self.cfg), "标注",
                  on_done=lambda _r: self._annotate_done())

    def _annotate_done(self):
        self._paths_map = None  # 新图进入 chash→path 映射，组卡片/盘束疑似重建时生效
        self.log_view.appendPlainText("下一步：跑 cluster 重新聚簇（新图才会生效）")

    # ---- 分区排序 ----
    def _replan(self):
        keys = self.key_panel.current_keys()
        if not keys:
            self.statusBar().showMessage("至少勾选一个分区键", 4000)
            return
        if self._busy():
            return
        # 软排序开关随每次重排同步（发型分组结果存在时生效）
        self.cfg.hairstyle_soft = self.hs_panel.soft_cb.isChecked()
        self.log_view.appendPlainText(f"—— 分区排序：键序 {'→'.join(keys)} ——")
        self._run(lambda: partition.build(self.cfg, keys=keys), "分区排序",
                  on_done=lambda _r: self._load_plan())

    # ---- 发型 ----
    def _run_hairstyle(self, mode: str, k, examples: list, refresh_prompts: bool):
        if self._busy():
            return
        self._run(lambda: hairstyle.build(self.cfg, mode=mode, k=k, examples=examples,
                                          refresh_prompts=refresh_prompts),
                  "发型分组",
                  on_done=lambda _r: self._hairstyle_done())

    def _hairstyle_done(self):
        p = self.cfg.out_dir / "hairstyle.json"
        if p.is_file():
            hs = json.loads(p.read_text(encoding="utf-8"))
            self.hs_panel.set_chash_paths(self._chash_paths())
            self.hs_panel.show_result(hs)
            self.hs_panel.refresh_groups(hs)
            self._show_suspects()

    def _show_suspects(self):
        """盘束疑似（updo_suspect）缩略图，与发型分组交叉核对（§4.2，不自动改判）。"""
        items = []
        jl = self.cfg.out_dir / "annotations.jsonl"
        if jl.is_file():
            for line in jl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    if (d.get("attrs") or {}).get("updo_suspect"):
                        items.append((d["content_hash"], d["path"]))
        self.hs_panel.show_suspects(items)

    def _merge_hairstyle_groups(self, names: list):
        if self._busy():
            return
        self._run(lambda: hairstyle.merge_groups(self.cfg, names), "合并发型组",
                  on_done=lambda _r: self._hairstyle_done())

    def _split_hairstyle_group(self, name: str):
        if self._busy():
            return
        self._run(lambda: hairstyle.split_group(self.cfg, name, k=2), "拆分发型组",
                  on_done=lambda _r: self._hairstyle_done())

    # ---- 聚簇合并/拆分 ----
    def _merge_clusters(self, cids: list):
        if self._busy():
            return
        self._run(lambda: adjust.merge(self.cfg, cids), "合并簇",
                  on_done=lambda _r: self._clusters_changed())

    def _split_cluster(self, cid: int):
        if self._busy():
            return
        self._run(lambda: adjust.split(self.cfg, cid, k=2), "拆分簇",
                  on_done=lambda _r: self._clusters_changed())

    def _clusters_changed(self):
        self._refresh_clusters()
        self._replan()

    # ---- 批量重命名 ----
    def _open_rename(self):
        if self._busy():
            return
        try:
            dlg = RenameDialog(self.cfg, self)
        except FileNotFoundError as e:
            self.log_view.appendPlainText(f"✗ {e}")
            return
        if dlg.exec() and dlg.chosen_mode:
            mode, dry = dlg.chosen_mode, dlg.dry_run
            self.log_view.appendPlainText(
                f"—— 重命名（mode={mode}{'，预演' if dry else ''}）——")

            def task():
                self.cfg.rename_mode = mode
                return rename.run(self.cfg, dry_run=dry)

            self._run(task, "重命名")


def main():
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("FaceCluster-FM")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
