# -*- coding: utf-8 -*-
"""深色专业工具主题：集中调色板 + 全局 QSS。

单一强调色（teal）承担"可操作/被选中"，琥珀仅用于警示、红仅用于危险，
避免多彩堆叠；面板用三层深浅（bg < surface < surface2）表达层级，边框统一 1px。
调色板修改只需改 PALETTE 一处，QSS 经 % 占位代入（不改用 f-string：QSS 里的
花括号是语法，f-string 需转义反而易错）。
"""
from __future__ import annotations

from pathlib import Path

_ICONS = Path(__file__).parent / "icons"


def _icon(name: str) -> str:
    """QSS 的 url() 需要绝对路径；Windows 的反斜杠要转正斜杠否则解析失败。"""
    return (_ICONS / name).as_posix()


PALETTE = {
    "bg": "#0e1116",
    "surface": "#151922",
    "surface2": "#1c2230",
    "surface3": "#232b3b",
    "border": "#2b3445",
    "border_soft": "#212936",
    "text": "#e7ebf2",
    "muted": "#8892a6",
    "accent": "#5ad1b0",
    "accent_hover": "#72dfc0",
    "accent_dim": "#2f7d6b",
    "warn": "#e2b23c",
    "danger": "#e5645b",
    "ok": "#63c07a",
    "sel": "#1c3634",
}

UI_FONT = '"Segoe UI", "Microsoft YaHei UI", sans-serif'
MONO_FONT = '"Cascadia Code", "Consolas", monospace'

_QSS = """
* { font-family: %(ui)s; font-size: 13px; }

QMainWindow, #Root { background: %(bg)s; }
QLabel { color: %(text)s; background: transparent; }
QLabel[role="muted"] { color: %(muted)s; }
QLabel[role="h1"] { font-size: 19px; font-weight: 600; letter-spacing: 0.4px; }
QLabel[role="h2"] { font-size: 15px; font-weight: 600; }
QLabel[role="mono"] { font-family: %(mono)s; color: %(muted)s; }
QLabel[role="warn"] { color: %(warn)s; }
QLabel[role="danger"] { color: %(danger)s; }

/* ---- 顶栏 / 日志坞 ---- */
#TopBar { background: %(surface)s; border-bottom: 1px solid %(border)s; }
#LogDock { background: %(surface)s; border-top: 1px solid %(border)s; }
#TopBar QLabel[role="brand"] { font-size: 15px; font-weight: 600; letter-spacing: 0.6px; }
#Chip { background: %(surface2)s; border: 1px solid %(border)s; border-radius: 6px; padding: 3px 10px; color: %(muted)s; }

/* ---- 流程导航 ---- */
#NavRail { background: %(surface)s; border-right: 1px solid %(border)s; }
#NavList { background: transparent; border: none; outline: none; padding: 6px; }
#NavList::item { color: %(muted)s; padding: 9px 12px; margin: 3px 6px; border-radius: 8px; }
#NavList::item:hover { background: %(surface2)s; color: %(text)s; }
#NavList::item:selected { background: %(sel)s; color: %(text)s; border-left: 2px solid %(accent)s; }

/* ---- 卡片 ---- */
QFrame[role="card"], QGroupBox { background: %(surface)s; border: 1px solid %(border)s; border-radius: 10px; }
QGroupBox { margin-top: 12px; padding: 12px 10px 10px 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: %(muted)s; font-weight: 600; }

/* ---- 按钮 ---- */
QPushButton { background: %(surface3)s; color: %(text)s; border: 1px solid %(border)s;
              border-radius: 7px; padding: 6px 14px; }
QPushButton:hover { border-color: %(accent_dim)s; }
QPushButton:pressed { background: %(surface2)s; }
QPushButton:disabled { color: %(muted)s; border-color: %(border_soft)s; background: %(surface2)s; }
QPushButton[role="primary"] { background: %(accent)s; color: #0b0f14; border: none; font-weight: 600; }
QPushButton[role="primary"]:hover { background: %(accent_hover)s; }
QPushButton[role="primary"]:disabled { background: %(surface3)s; color: %(muted)s; }
QPushButton[role="danger"] { color: %(danger)s; border-color: %(danger)s; }
QPushButton[role="danger"]:hover { background: %(danger)s; color: #0b0f14; }

/* ---- 输入控件 ---- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {
    background: %(surface2)s; border: 1px solid %(border)s; border-radius: 7px;
    padding: 4px 8px; color: %(text)s; selection-background-color: %(accent)s; selection-color: #0b0f14; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus { border-color: %(accent)s; }
QLineEdit:read-only { color: %(muted)s; }
QComboBox::drop-down { border: none; width: 22px; }
/* 给控件设样式表后 Qt 不再画原生下拉箭头；border 拼三角在 Qt 里只会画出实心方块，
   故用矢量图标资源（icons/）——这也是本主题唯一需要外部资源的地方。 */
QComboBox::down-arrow { image: url(%(caret)s); width: 10px; height: 6px; }
QComboBox QAbstractItemView { background: %(surface2)s; border: 1px solid %(border)s;
    border-radius: 7px; padding: 4px; selection-background-color: %(sel)s; color: %(text)s; }
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 16px; background: %(surface3)s; border: none; }

/* ---- 勾选 / 单选 ---- */
QCheckBox, QRadioButton { color: %(text)s; spacing: 7px; }
QCheckBox::indicator, QRadioButton::indicator { width: 14px; height: 14px;
    border: 1px solid %(border)s; background: %(surface2)s; }
QCheckBox::indicator { border-radius: 4px; }
QRadioButton::indicator { border-radius: 7px; }
QCheckBox::indicator:checked { background: %(accent)s; border-color: %(accent)s;
    image: url(%(check)s); }
QRadioButton::indicator:checked { background: %(accent)s; border-color: %(accent)s; }
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled { border-color: %(border_soft)s; }

/* ---- 列表 ---- */
QListWidget { background: %(surface)s; border: 1px solid %(border)s; border-radius: 9px; outline: none; }
QListWidget::item { color: %(text)s; border-radius: 8px; }
QListWidget::item:selected { background: %(sel)s; border: 1px solid %(accent_dim)s; }
QListWidget::item:hover { background: %(surface2)s; }
/* 列表项自带的勾选框由基类 style 绘制（Win11 风格恒为白底方框，深色主题下未勾选也
   像已勾选）。必须显式接管 ::indicator，否则键序的"参与分区"勾选态无法分辨。 */
QListWidget::indicator { width: 14px; height: 14px; border: 1px solid %(border)s;
    border-radius: 4px; background: %(surface2)s; }
QListWidget::indicator:checked { background: %(accent)s; border-color: %(accent)s;
    image: url(%(check)s); }
QListWidget::indicator:hover { border-color: %(accent_dim)s; }

/* ---- 表格 ---- */
QTableView, QTableWidget { background: %(surface)s; gridline-color: %(border_soft)s;
    border: 1px solid %(border)s; border-radius: 9px; color: %(text)s;
    selection-background-color: %(sel)s; selection-color: %(text)s;
    alternate-background-color: %(surface2)s; }
QTableView::item, QTableWidget::item { padding: 3px 6px; }
QHeaderView::section { background: %(surface2)s; color: %(muted)s; padding: 7px 8px;
    border: none; border-bottom: 1px solid %(border)s; }
QTableCornerButton::section { background: %(surface2)s; border: none; }

/* ---- 滚动条 ---- */
QScrollBar:vertical { background: transparent; width: 11px; margin: 2px; }
QScrollBar::handle:vertical { background: %(surface3)s; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: %(border)s; }
QScrollBar:horizontal { background: transparent; height: 11px; margin: 2px; }
QScrollBar::handle:horizontal { background: %(surface3)s; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent; height: 0; width: 0; }

/* ---- 进度 / 状态栏 / 分隔条 ---- */
QProgressBar { background: %(surface2)s; border: 1px solid %(border)s; border-radius: 7px;
    height: 16px; text-align: center; color: %(text)s; }
QProgressBar::chunk { background: %(accent)s; border-radius: 6px; }
QStatusBar { background: %(surface)s; color: %(muted)s; border-top: 1px solid %(border)s; }
QStatusBar::item { border: none; }
QSplitter::handle { background: %(border_soft)s; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
QToolTip { background: %(surface3)s; color: %(text)s; border: 1px solid %(border)s; padding: 4px 7px; }
"""


def stylesheet() -> str:
    """全局 QSS（把调色板、字体与图标路径代入占位符）。"""
    return _QSS % {**PALETTE, "ui": UI_FONT, "mono": MONO_FONT,
                   "caret": _icon("caret_down.svg"), "check": _icon("check.svg")}


def apply(app) -> None:
    """把主题应用到 QApplication（深色调色板 + 字体 + QSS）。

    调色板必须一并设深：QSS 只覆盖显式声明的子控件，而组合框下拉箭头、列表勾选框等
    仍由 style 依调色板绘制——留着默认浅色调色板会画出深色箭头/深色勾选框，在深底上
    几乎不可见。
    """
    from PySide6.QtGui import QColor, QFont, QPalette

    f = QFont("Segoe UI", 9)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(f)

    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: PALETTE["bg"],
        QPalette.ColorRole.WindowText: PALETTE["text"],
        QPalette.ColorRole.Base: PALETTE["surface2"],
        QPalette.ColorRole.AlternateBase: PALETTE["surface"],
        QPalette.ColorRole.Text: PALETTE["text"],
        QPalette.ColorRole.Button: PALETTE["surface3"],
        QPalette.ColorRole.ButtonText: PALETTE["text"],
        QPalette.ColorRole.ToolTipBase: PALETTE["surface3"],
        QPalette.ColorRole.ToolTipText: PALETTE["text"],
        QPalette.ColorRole.Highlight: PALETTE["accent"],
        QPalette.ColorRole.HighlightedText: "#0b0f14",
        QPalette.ColorRole.PlaceholderText: PALETTE["muted"],
    }
    for role, col in roles.items():
        p.setColor(role, QColor(col))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(PALETTE["muted"]))
    app.setPalette(p)
    app.setStyleSheet(stylesheet())
