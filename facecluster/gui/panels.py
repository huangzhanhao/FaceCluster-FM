# -*- coding: utf-8 -*-
"""共享控件库：流程导航 / 键序链 / F 值 / 缩略图网格 / 条形图 / 重命名对话框。

阶段工作区（stages.py）只做"编排 + 请求发出"，可复用控件集中在此，避免同一套
缩略图网格/表格在聚簇、发型、预览三处各写一遍。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..gui.thumbs import ThumbStore
from . import theme

PAL = theme.PALETTE


# ---------- 小工具 ----------
def heading(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("role", "h1")
    return lab


def subheading(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("role", "h2")
    return lab


def muted(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("role", "muted")
    lab.setWordWrap(True)
    return lab


# ---------- 流程导航 ----------
class StepNav(QListWidget):
    """六步流程步骤条：选中即切页；徽标表达完成态（✓ 完成 / ● 当前 / ○ 待办）。"""

    STEPS = ("① 准备", "② 标注", "③ 聚簇", "④ 发型", "⑤ 分区排序", "⑥ 输出")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NavList")
        self._badges = ["○"] * len(self.STEPS)
        for t in self.STEPS:
            it = QListWidgetItem(t)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.addItem(it)
        self._render()
        self.currentRowChanged.connect(lambda _r: self._render())

    def set_badge(self, i: int, badge: str) -> None:
        """badge ∈ {✓, ●, ○}；仅刷新文本，不触发选中态变化。"""
        if 0 <= i < len(self._badges) and self._badges[i] != badge:
            self._badges[i] = badge
            self._render()

    def _render(self):
        # 仅"已完成"追加对勾；待办/当前靠选中高亮与弱化配色表达。步骤序号（①②…）
        # 本身已表达顺序，前缀再放 ○/● 会得到"○ ① 准备"这样的双符号噪声。
        for i, t in enumerate(self.STEPS):
            mark = "✓ " if self._badges[i] == "✓" else ""
            self.item(i).setText(f"{mark}{t}")


# ---------- 键序链 ----------
class KeyChainPanel(QWidget):
    """分区键编排：勾选=参与分区，拖拽=键序；另有上移/下移按钮兜底（拖拽不便时）。"""

    keysChanged = Signal(list)

    def __init__(self, keys: list[str], selected: list[str], parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        v.addWidget(muted("勾选 = 参与分区 · 拖拽 = 调键序（自上而下即分区层级）"))
        self.listw = QListWidget()
        self.listw.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.listw.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.listw.setFixedHeight(30 * max(3, len(keys)) + 12)
        # 已勾选键按用户既有顺序在前，其余按注册表顺序补后（保留拖拽结果）
        ordered = [k for k in selected if k in keys] + [k for k in keys if k not in selected]
        for k in ordered:
            self.listw.addItem(self._mk_item(k, k in selected))
        v.addWidget(self.listw)

        h = QHBoxLayout()
        self.up_btn = QPushButton("↑ 上移")
        self.down_btn = QPushButton("↓ 下移")
        self.up_btn.clicked.connect(lambda: self._move(-1))
        self.down_btn.clicked.connect(lambda: self._move(1))
        h.addWidget(self.up_btn)
        h.addWidget(self.down_btn)
        h.addStretch(1)
        v.addLayout(h)

        self.listw.model().rowsMoved.connect(self._emit)
        self.listw.itemChanged.connect(self._emit)

    @staticmethod
    def _mk_item(text: str, checked: bool) -> QListWidgetItem:
        it = QListWidgetItem(text)
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
        it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        return it

    def current_keys(self) -> list[str]:
        return [self.listw.item(i).text() for i in range(self.listw.count())
                if self.listw.item(i).checkState() == Qt.Checked]

    def ensure_key(self, key: str) -> None:
        """键纳入面板并勾选（无则追加）；勾选变更经 itemChanged 自动重排。"""
        for i in range(self.listw.count()):
            if self.listw.item(i).text() == key:
                if self.listw.item(i).checkState() != Qt.Checked:
                    self.listw.item(i).setCheckState(Qt.Checked)
                return
        self.listw.addItem(self._mk_item(key, True))
        self.listw.setFixedHeight(30 * max(3, self.listw.count()) + 12)

    def _move(self, delta: int):
        r = self.listw.currentRow()
        t = r + delta
        if r < 0 or not (0 <= t < self.listw.count()):
            return
        it = self.listw.takeItem(r)
        self.listw.insertItem(t, it)
        self.listw.setCurrentRow(t)
        self._emit()

    def _emit(self, *_):
        self.keysChanged.emit(self.current_keys())


# ---------- F 值 ----------
class FValuePanel(QWidget):
    """F 值（R11）：每级段内相邻 cos 之和 + 最差相邻对。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["级别", "键", "段数", "F 值"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setFixedHeight(128)
        v.addWidget(self.table)
        self.worst = muted("")
        v.addWidget(self.worst)

    def refresh(self, plan: dict):
        rows = (plan.get("fvalue") or {}).get("per_level", [])
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            cells = (f"L{r['level']}", " → ".join(r["keys"]), str(r["n_seg"]), f"{r['f']:.1f}")
            for j, val in enumerate(cells):
                item = QTableWidgetItem(val)
                if j == 3:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, item)
        wp = (plan.get("fvalue") or {}).get("worst_pairs") or []
        if wp:
            w = wp[0]
            self.worst.setText(f"最差相邻对 cos={w['sim']}：{Path(w['a']).name} ↔ {Path(w['b']).name}")
            self.worst.setStyleSheet(f"color:{PAL['warn']};")
        else:
            self.worst.setText("无数据")
            self.worst.setStyleSheet("")


# ---------- 缩略图网格 ----------
class ThumbGrid(QListWidget):
    """medoid/组代表图网格：图标 + 标签；缩略图后台生成完成时自动补绘。

    网格本身不关心业务，行 → 外部标识的映射由调用方维护。
    """

    def __init__(self, thumbs: ThumbStore, icon: int = 84, parent=None):
        super().__init__(parent)
        self.thumbs = thumbs
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(icon, icon))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setSpacing(4)
        self.setUniformItemSizes(True)
        self._rows_of: dict[str, list[int]] = {}
        thumbs.ready.connect(self._on_ready)

    def add(self, chash: str | None, src: str | None, label: str, tip: str = "") -> int:
        it = QListWidgetItem(QIcon(self.thumbs.pixmap(chash, src)) if chash else QIcon(), label)
        it.setData(Qt.ItemDataRole.UserRole, src)  # 补绘时回填源路径
        if tip:
            it.setToolTip(tip)
        self.addItem(it)
        row = self.count() - 1
        if chash:
            self._rows_of.setdefault(chash, []).append(row)
        return row

    def reset_grid(self):
        self.clear()
        self._rows_of = {}

    def _on_ready(self, chash: str):
        for row in self._rows_of.get(chash, ()):
            item = self.item(row)
            src = (item.data(Qt.ItemDataRole.UserRole) or None) if item else None
            if item:
                item.setIcon(QIcon(self.thumbs.pixmap(chash, src)))


# ---------- 横向条形图 ----------
class HBarChart(QWidget):
    """属性分布横条图（发色/发长桶计数）：左标签 + 条 + 右数值，纯绘制无依赖。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list[tuple[str, int]] = []
        self._title = ""
        self.setMinimumHeight(24)

    def set_data(self, title: str, pairs: list[tuple[str, int]]):
        self._title = title
        self._data = [p for p in pairs if p[1] > 0]
        self.setFixedHeight(22 + max(1, len(self._data)) * 20)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        label_w, val_w = 66, 52
        if self._title:
            p.setPen(QColor(PAL["muted"]))
            p.drawText(4, 15, self._title)
        top = 22 if self._title else 0
        mx = max((v for _, v in self._data), default=1)
        for i, (name, v) in enumerate(self._data):
            y = top + i * 20
            p.setPen(QColor(PAL["text"]))
            p.drawText(4, y + 14, name)
            bx = label_w
            bw = w - label_w - val_w
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(PAL["surface3"]))
            p.drawRoundedRect(QRect(bx, y + 3, int(bw), 10), 5, 5)
            p.setBrush(QColor(PAL["accent"]))
            p.drawRoundedRect(QRect(bx, y + 3, int(bw * v / mx), 10), 5, 5)
            p.setPen(QColor(PAL["muted"]))
            p.drawText(w - val_w + 4, y + 14, str(v))
