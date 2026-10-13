# -*- coding: utf-8 -*-
"""后台任务线程：同进程直调后端纯函数，信号回传日志/进度/结果（GUI 主线程只做展示）。

后端函数签名各异（有的带 log/progress_cb，有的不带），按 inspect 签名注入——
调用方不必为线程适配写包装函数。同一时刻仅一个主任务，由 app 的互斥保证。
"""
from __future__ import annotations

import inspect
import traceback

from PySide6.QtCore import QThread, Signal


class Worker(QThread):
    progress = Signal(int, int)
    log = Signal(str)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self):
        try:
            params = inspect.signature(self.fn).parameters
            kw = dict(self.kwargs)
            if "log" in params:
                kw.setdefault("log", self.log.emit)
            if "progress_cb" in params:
                kw.setdefault("progress_cb", lambda i, n: self.progress.emit(i, n))
            self.done.emit(self.fn(*self.args, **kw))
        except BaseException as e:  # rename 冲突用 SystemExit 拒绝，不能漏接
            traceback.print_exc()
            self.error.emit(str(e) or type(e).__name__)
