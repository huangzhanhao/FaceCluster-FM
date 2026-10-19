# -*- coding: utf-8 -*-
"""主窗口：顶栏（项目/环境）｜左侧六步流程导航｜中央阶段工作区｜底部可折叠日志。

阶段只做界面编排；全部后台任务经 MainWindow.run 派发，单任务互斥与进度回传集中
在此。阶段切换时按需刷新对应工作区（懒刷新，避免 30k 规模下每次切换都重建表格）。
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit, QProgressBar, QPushButton,
    QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from ..config import Config
from . import theme
from .panels import StepNav
from .stages import (
    AnnotateStage, ClusterStage, HairstyleStage, OutputStage, PrepareStage, SortStage,
    task_order,
)
from .thumbs import ThumbStore
from .workers import Worker

HAIRSTYLE_KEY = "发型"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = Config()
        self.setWindowTitle("FaceCluster-FM")
        self.resize(1360, 860)
        self.worker: Worker | None = None
        self._ann_cache: list | None = None
        self._paths_map: dict[str, str] | None = None

        self.thumbs = ThumbStore(self.cfg.out_dir, self)

        root = QWidget()
        root.setObjectName("Root")
        rv = QVBoxLayout(root)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        rv.addWidget(self._build_topbar())

        body = QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(self._build_nav())
        body.addWidget(self._build_stages())
        body.setStretchFactor(1, 1)
        body.setSizes([190, 1170])

        bottom = QSplitter(Qt.Orientation.Vertical)
        bottom.addWidget(body)
        bottom.addWidget(self._build_logdock())
        bottom.setSizes([660, 180])
        rv.addWidget(bottom, 1)
        self.setCentralWidget(root)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._replan)

        self._refresh_all()
        self.nav.setCurrentRow(0)

    # ---------- 顶栏 ----------
    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(50)
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 16, 0)
        h.setSpacing(14)
        brand = QLabel("FaceCluster-FM")
        brand.setProperty("role", "brand")
        h.addWidget(brand)
        self.chip_sample = QLabel()
        self.chip_sample.setObjectName("Chip")
        self.chip_out = QLabel()
        self.chip_out.setObjectName("Chip")
        h.addWidget(self.chip_sample)
        h.addWidget(self.chip_out)
        h.addStretch(1)
        self.chip_env = QLabel()
        self.chip_env.setObjectName("Chip")
        h.addWidget(self.chip_env)
        refresh = QPushButton("↻ 刷新")
        refresh.clicked.connect(self._refresh_all)
        h.addWidget(refresh)
        return bar

    # ---------- 导航 ----------
    def _build_nav(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("NavRail")
        v = QVBoxLayout(rail)
        v.setContentsMargins(0, 8, 0, 8)
        v.setSpacing(6)
        self.nav = StepNav()
        self.nav.currentRowChanged.connect(self._on_stage)
        v.addWidget(self.nav, 1)
        return rail

    # ---------- 阶段工作区 ----------
    def _build_stages(self) -> QWidget:
        self.stack = QStackedWidget()
        self.prepare = PrepareStage(self)
        self.annotate = AnnotateStage(self)
        self.cluster = ClusterStage(self)
        self.hairstyle = HairstyleStage(self)
        self.sort = SortStage(self)
        self.output = OutputStage(self)
        for w in (self.prepare, self.annotate, self.cluster, self.hairstyle, self.sort, self.output):
            self.stack.addWidget(w)
        return self.stack

    # ---------- 日志坞 ----------
    def _build_logdock(self) -> QWidget:
        dock = QWidget()
        dock.setObjectName("LogDock")  # 独立底色，勿与顶栏混用同一 objectName
        v = QVBoxLayout(dock)
        v.setContentsMargins(12, 6, 12, 8)
        v.setSpacing(6)
        head = QHBoxLayout()
        self.log_toggle = QPushButton("▾ 日志")
        self.log_toggle.setFixedWidth(90)
        self.log_toggle.clicked.connect(self._toggle_log)
        clear = QPushButton("清空")
        clear.clicked.connect(lambda: self.log_view.clear())
        head.addWidget(self.log_toggle)
        head.addStretch(1)
        head.addWidget(clear)
        v.addLayout(head)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(4000)
        self.log_view.setPlaceholderText("运行日志将显示在此…")
        self.log_view.setProperty("role", "mono")
        self.log_view.setStyleSheet(f"font-family:{theme.MONO_FONT}; color:{theme.PALETTE['muted']};")
        v.addWidget(self.log_view)
        return dock

    def _toggle_log(self):
        vis = not self.log_view.isVisible()
        self.log_view.setVisible(vis)
        self.log_toggle.setText("▾ 日志" if vis else "▸ 日志")

    def log(self, msg: str):
        self.log_view.appendPlainText(msg)

    # ---------- 阶段切换（懒刷新） ----------
    def _on_stage(self, row: int):
        self.stack.setCurrentIndex(row)
        {
            0: self.prepare.refresh_env,
            1: self.annotate.refresh,
            2: self.reload_clusters,
            3: self.reload_hairstyle,
            4: self.load_plan,
            5: self.output.refresh,
        }.get(row, lambda: None)()

    def mark_step(self, i: int, badge: str):
        self.nav.set_badge(i, badge)

    # ---------- 项目 / 环境 ----------
    def set_sample_dir(self, p: Path):
        self.cfg.sample_dir = p
        self._refresh_all(keep_log=True)
        self.log(f"样本目录 → {p}")

    def set_out_dir(self, p: Path):
        self.cfg.out_dir = p
        self.thumbs.dir = p / "thumbs"
        self.thumbs.dir.mkdir(parents=True, exist_ok=True)
        self._ann_cache = self._paths_map = None
        self._refresh_all(keep_log=True)
        self.log(f"输出目录 → {p}")

    def _refresh_all(self, keep_log: bool = False):
        cfg = self.cfg
        self.chip_sample.setText(f"样本 {cfg.sample_dir.name or cfg.sample_dir} "
                                 f"({self._sample_count()})")
        self.chip_sample.setToolTip(str(cfg.sample_dir))
        self.chip_out.setText(f"输出 {cfg.out_dir.name or cfg.out_dir}")
        self.chip_out.setToolTip(str(cfg.out_dir))
        dml = _dml_available()
        self.chip_env.setText("● DirectML 就绪" if dml else "○ CPU 模式")
        self.chip_env.setStyleSheet(
            f"color:{theme.PALETTE['ok'] if dml else theme.PALETTE['warn']};")
        self.prepare.sample_edit.setText(str(cfg.sample_dir))
        self.prepare.out_edit.setText(str(cfg.out_dir))
        self.prepare.refresh_env()
        self.annotate.refresh()
        self.reload_clusters()
        self.reload_hairstyle()
        self.load_plan()

    def _sample_count(self) -> int:
        recs = self.annotations()
        return len(recs) if recs else "—"

    # ---- 数据源 ----
    def annotations(self) -> list:
        if self._ann_cache is None:
            try:
                from ..cluster import load_annotations

                self._ann_cache = load_annotations(self.cfg)
            except FileNotFoundError:
                self._ann_cache = []
        return self._ann_cache

    def chash_paths(self) -> dict[str, str]:
        if self._paths_map is None:
            self._paths_map = {r.content_hash: r.path for r in self.annotations()}
        return self._paths_map

    # ---- 工件刷新 ----
    def load_plan(self):
        pf = self.cfg.out_dir / "plan.json"
        plan = json.loads(pf.read_text(encoding="utf-8")) if pf.is_file() else None
        self.sort.refresh(plan)
        if plan:
            # 前缀点明这是磁盘上 plan.json 的快照：其键序可能还停留在上次运行的结果，
            # 与左侧键序面板的当前勾选未必一致，不加前缀会被误读成"当前键序"。
            self.statusBar().showMessage(
                f"plan.json：{plan['n_items']} 条目 / {plan['n_clusters']} 簇 / 键序 "
                f"{'→'.join(plan['key_order'])}")

    def reload_clusters(self):
        p = self.cfg.out_dir / "clusters.json"
        if p.is_file():
            clu = json.loads(p.read_text(encoding="utf-8"))
            self.cluster.refresh(clu["clusters"], clu.get("tail"))
        else:
            self.cluster.refresh([], None)

    def reload_hairstyle(self):
        self.hairstyle.refresh()
        items = []
        for r in self.annotations():
            if (r.attrs and getattr(r.attrs, "updo_suspect", False)):
                items.append((r.content_hash, r.path))
        self.hairstyle.show_suspects(items)

    def after_clusters_changed(self):
        """聚簇合并/拆分完成：刷新面板并重排（人工修正反映到 plan）。"""
        self.reload_clusters()
        self._replan()

    def after_hairstyle_changed(self):
        self.reload_hairstyle()
        self.sort.rebuild_keys()
        if HAIRSTYLE_KEY in self.sort.current_keys():
            self._replan()

    def apply_hairstyle_key(self):
        """把发型纳入分区键（勾选变更经 keysChanged → 自动重排）。"""
        self.sort.ensure_key(HAIRSTYLE_KEY)

    # ---------- 键序 → 重排 ----------
    def on_keys_changed(self, keys: list):
        """发型键与软排序互斥：发型已作硬键时软排序无意义，禁用开关以免静默忽略。"""
        hs_on = HAIRSTYLE_KEY in keys
        cb = self.sort.soft_cb
        cb.setEnabled(not hs_on)
        cb.setToolTip("发型已作分区键，软排序不生效" if hs_on else "")
        if hs_on and cb.isChecked():
            cb.setChecked(False)
        self._debounce.start()

    def replan(self):
        self._debounce.stop()
        self._replan()

    def _replan(self):
        keys = self.sort.current_keys()
        if not keys:
            self.statusBar().showMessage("至少勾选一个分区键", 4000)
            return
        self.cfg.hairstyle_soft = self.sort.soft_cb.isChecked()
        self.log(f"—— 分区排序：键序 {'→'.join(keys)} ——")
        self.run(task_order, "分区排序", args=(self.cfg, keys), on_done=lambda _r: self.load_plan())

    # ---------- 后台任务 ----------
    def _busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def run(self, fn, label: str, args=(), on_done=None) -> bool:
        if self._busy():
            self.log(f"⏳ 已有任务进行中，忽略「{label}」")
            return False
        self.progress.setRange(0, 0)
        self.progress.show()
        self._set_running(True)
        w = Worker(fn, *args)
        self.worker = w
        w.log.connect(self.log)
        w.progress.connect(self._on_progress)
        w.done.connect(lambda r: self._task_done(label, on_done, r))
        w.error.connect(self._task_error)
        w.start()
        return True

    def _set_running(self, running: bool):
        # 每次现场遍历：刷新环境表格/重建键序面板会销毁并替换旧按钮，
        # 缓存按钮列表会留下已析构的 C++ 包装对象（property 调用即崩溃）。
        for b in self.findChildren(QPushButton):
            if b.property("role") == "primary":
                b.setEnabled(not running)

    def _on_progress(self, i: int, n: int):
        self.progress.setRange(0, max(1, n))
        self.progress.setValue(i)

    def _task_done(self, label: str, on_done, result):
        self.progress.hide()
        self._set_running(False)
        self.log(f"✓ {label}完成")
        if on_done:
            on_done(result)

    def _task_error(self, msg: str):
        self.progress.hide()
        self._set_running(False)
        self.log(f"✗ 任务失败：{msg}")
        self.statusBar().showMessage(f"任务失败：{msg}", 8000)


def _dml_available() -> bool:
    try:
        import onnxruntime as ort

        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def main():
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("FaceCluster-FM")
    theme.apply(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
