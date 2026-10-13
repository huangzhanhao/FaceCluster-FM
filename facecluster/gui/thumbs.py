# -*- coding: utf-8 -*-
"""缩略图：out/thumbs/{chash}.jpg 磁盘缓存 + 内存 LRU + QThreadPool(max=2) 异步生成。

pixmap() 永不阻塞：命中直接返回，未命中触发后台生成并返回空图，完成后经
ready(chash) 信号由模型刷新对应行。缓存身份 = 内容哈希，改名/挪目录不失效。
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap

THUMB_SIDE = 96
MEM_CAP = 1024  # 96px QImage 约 27KB/张，1024 张 ≈ 28MB


def _make_thumb(src: Path, dst: Path) -> bool:
    from PIL import Image, ImageOps

    try:
        im = ImageOps.exif_transpose(Image.open(src))
        im.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.BILINEAR)
        im.convert("RGB").save(dst, "JPEG", quality=80)
        return True
    except Exception:
        return False


class _Job(QRunnable):
    def __init__(self, store: "ThumbStore", chash: str, src: Path):
        super().__init__()
        self.store, self.chash, self.src = store, chash, src

    def run(self):
        dst = self.store.dir / f"{self.chash}.jpg"
        self.store._finished(self.chash, dst if _make_thumb(self.src, dst) else None)


class ThumbStore(QObject):
    ready = Signal(str)  # chash：缩略图已生成

    def __init__(self, out_dir: Path, parent=None):
        super().__init__(parent)
        self.dir = out_dir / "thumbs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.mem: OrderedDict[str, QImage] = OrderedDict()
        self.pending: set[str] = set()

    def pixmap(self, chash: str, src: str | None) -> QPixmap:
        """同步取图；未命中返回空 QPixmap 并触发异步生成。仅主线程调用。"""
        img = self.mem.get(chash)
        if img is None:
            dst = self.dir / f"{chash}.jpg"
            if dst.is_file():
                img = QImage(str(dst))
            elif chash not in self.pending and src:
                self.pending.add(chash)
                self.pool.start(_Job(self, chash, Path(src)))
        if img is None:
            return QPixmap()
        self.mem[chash] = img
        if len(self.mem) > MEM_CAP:
            self.mem.popitem(last=False)
        return QPixmap.fromImage(img)

    def _finished(self, chash: str, path: Path | None):
        self.pending.discard(chash)
        if path is not None:
            self.mem[chash] = QImage(str(path))
            self.ready.emit(chash)
