# -*- coding: utf-8 -*-
"""预览表模型：行 = plan.json sequence 条目（450~30k 行虚拟化天然成立）。

列：缩略图 | 簇号 | 段路径 | 原文件 | 新名 | 相似度。
缩略图未命中时返回空图，ThumbStore ready 信号触发 dataChanged 补绘。
"""
from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QPixmap

from .thumbs import ThumbStore

COLS = ("缩略图", "簇", "段", "原文件", "新名", "相似度")


class PlanTableModel(QAbstractTableModel):
    def __init__(self, thumbs: ThumbStore, parent=None):
        super().__init__(parent)
        self.thumbs = thumbs
        self.items: list[dict] = []
        self.rows_of: dict[str, list[int]] = {}
        thumbs.ready.connect(self._thumb_ready)

    def set_plan(self, plan: dict):
        self.beginResetModel()
        self.items = plan["sequence"]
        self.rows_of = {}
        for i, it in enumerate(self.items):
            self.rows_of.setdefault(it["chash"], []).append(i)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.items)

    def columnCount(self, parent=QModelIndex()):
        return len(COLS)

    def headerData(self, s, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return COLS[s]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        it = self.items[index.row()]
        c = index.column()
        if role == Qt.DecorationRole and c == 0:
            return self.thumbs.pixmap(it["chash"], it.get("path"))
        if role == Qt.DisplayRole:
            if c == 1:
                return it["cid"]
            if c == 2:
                return " → ".join(map(str, it["seg"]))
            if c == 3:
                return it["path"]
            if c == 4:
                return it["new_name"]
            if c == 5:
                sim = it.get("sim")
                return "" if sim is None else f"{sim:.3f}"
        return None

    def _thumb_ready(self, chash: str):
        for r in self.rows_of.get(chash, ()):
            self.dataChanged.emit(self.index(r, 0), self.index(r, 0))
