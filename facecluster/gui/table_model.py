# -*- coding: utf-8 -*-
"""预览表模型：行 = plan.json sequence 条目（450~30k 行虚拟化天然成立）。

列：缩略图 | 簇 | 段 | 原文件 | 新名 | 相似度。支持关键字过滤（原名/新名/段/簇号）。
缩略图未命中时返回空图，ThumbStore ready 信号触发 dataChanged 补绘。
"""
from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from .thumbs import ThumbStore

COLS = ("缩略图", "簇", "段", "原文件", "新名", "相似度")


class PlanTableModel(QAbstractTableModel):
    def __init__(self, thumbs: ThumbStore, parent=None):
        super().__init__(parent)
        self.thumbs = thumbs
        self.items: list[dict] = []
        self._rows: list[int] = []            # 可见行 → items 下标
        self._vis_of_item: dict[int, int] = {}  # items 下标 → 可见行（缩略图补绘用）
        self._rofs: dict[str, list[int]] = {}   # chash → items 下标集合
        self._filter = ""
        thumbs.ready.connect(self._thumb_ready)

    # ---- 数据装载 ----
    def set_plan(self, plan: dict):
        self.beginResetModel()
        self.items = plan["sequence"]
        self._rofs = {}
        for i, it in enumerate(self.items):
            self._rofs.setdefault(it["chash"], []).append(i)
        self._rebuild()
        self.endResetModel()

    def set_filter(self, text: str):
        text = (text or "").strip().lower()
        if text == self._filter:
            return
        self.beginResetModel()
        self._filter = text
        self._rebuild()
        self.endResetModel()

    def _rebuild(self):
        if not self._filter:
            self._rows = list(range(len(self.items)))
        else:
            f = self._filter
            self._rows = [
                i for i, it in enumerate(self.items)
                if f in it["new_name"].lower() or f in it["path"].lower()
                or f in " → ".join(map(str, it["seg"])).lower() or f == str(it["cid"])
            ]
        self._vis_of_item = {item: row for row, item in enumerate(self._rows)}

    @property
    def visible_count(self) -> int:
        return len(self._rows)

    # ---- Qt 模型接口 ----
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return len(COLS)

    def headerData(self, s, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return COLS[s]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        it = self.items[self._rows[index.row()]]
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
        # chash → items 下标集合：逐个换算可见行补绘（过滤态下只刷新可见的那部分）
        for item in self._rofs.get(chash, ()):
            row = self._vis_of_item.get(item)
            if row is not None:
                self.dataChanged.emit(self.index(row, 0), self.index(row, 0))
