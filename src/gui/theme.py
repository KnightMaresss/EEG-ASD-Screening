"""Design tokens and global stylesheet for the application.

All colors, fonts, and QSS live here. No widget should hardcode a hex color.
"""
from pathlib import Path


class Theme:
    """Design tokens — one source of truth for the whole UI."""

    # Backgrounds
    BG_APP        = "#0F1115"
    BG_CARD       = "#171A21"
    BG_CARD_ALT   = "#1E222B"
    BG_ELEVATED   = "#232733"
    BG_CONSOLE    = "#0B0D11"

    # Borders
    BORDER        = "#2A2F3A"
    BORDER_FOCUS  = "#3B82F6"

    # Text
    TEXT          = "#E6E9EF"
    TEXT_MUTED    = "#8B93A7"
    TEXT_DIM      = "#5C6478"
    TEXT_CONSOLE  = "#A5D6A7"

    # Brand
    PRIMARY       = "#3B82F6"
    PRIMARY_HOVER = "#2563EB"
    PRIMARY_DIM   = "#1E3A8A"

    # Semantic
    SUCCESS       = "#10B981"
    SUCCESS_DIM   = "#065F46"
    WARNING       = "#F59E0B"
    DANGER        = "#EF4444"
    DANGER_DIM    = "#7F1D1D"

    # Fonts
    FONT_UI   = "'Inter', 'Segoe UI', system-ui, sans-serif"
    FONT_MONO = "'JetBrains Mono', 'Consolas', monospace"


def load_stylesheet() -> str:
    """Return the global QSS string.

    If a file named `theme.qss` sits next to this module, its contents are
    used instead — useful for tweaking styles without touching Python.
    """
    qss_path = Path(__file__).with_suffix(".qss")
    if qss_path.exists():
        return qss_path.read_text(encoding="utf-8")
    return _INLINE_QSS


_INLINE_QSS = f"""
/* ---------- Base ---------- */
QMainWindow, QWidget {{
    background-color: {Theme.BG_APP};
    color: {Theme.TEXT};
    font-family: {Theme.FONT_UI};
    font-size: 13px;
}}

/* ---------- Tabs ---------- */
QTabWidget::pane {{ border: none; background-color: {Theme.BG_APP}; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent;
    color: {Theme.TEXT_MUTED};
    padding: 10px 22px;
    margin-right: 4px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 600;
    font-size: 13px;
}}
QTabBar::tab:hover {{ color: {Theme.TEXT}; }}
QTabBar::tab:selected {{
    color: {Theme.PRIMARY};
    border-bottom: 2px solid {Theme.PRIMARY};
}}

/* ---------- Cards ---------- */
QFrame#Card {{
    background-color: {Theme.BG_CARD};
    border: 1px solid {Theme.BORDER};
    border-radius: 12px;
}}
QFrame#CardHeader {{ background: transparent; border: none; }}

/* ---------- Labels ---------- */
QLabel {{ color: {Theme.TEXT}; background: transparent; }}
QLabel#PageTitle {{ font-size: 22px; font-weight: 700; color: {Theme.TEXT}; }}
QLabel#PageSubtitle {{ font-size: 12px; color: {Theme.TEXT_MUTED}; font-weight: 500; }}
QLabel#CardTitle {{ font-size: 13px; font-weight: 600; color: {Theme.TEXT}; letter-spacing: 0.3px; }}
QLabel#CardCaption {{
    font-size: 11px; color: {Theme.TEXT_DIM};
    text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
}}
QLabel#MutedLabel {{ color: {Theme.TEXT_MUTED}; font-size: 12px; }}

/* ---------- Buttons ---------- */
QPushButton {{
    background-color: {Theme.BG_ELEVATED};
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER};
    border-radius: 8px;
    padding: 9px 16px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton:hover {{ background-color: {Theme.BG_CARD_ALT}; border-color: {Theme.BORDER_FOCUS}; }}
QPushButton:pressed {{ background-color: {Theme.BG_CARD}; }}

QPushButton#PrimaryButton {{
    background-color: {Theme.PRIMARY};
    color: #FFFFFF;
    border: 1px solid {Theme.PRIMARY};
    padding: 14px 20px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.6px;
    border-radius: 10px;
    min-height: 22px;
}}
QPushButton#PrimaryButton:hover {{ background-color: {Theme.PRIMARY_HOVER}; border-color: {Theme.PRIMARY_HOVER}; }}
QPushButton#PrimaryButton:pressed {{ background-color: {Theme.PRIMARY_DIM}; border-color: {Theme.PRIMARY_DIM}; }}
QPushButton#PrimaryButton:disabled {{
    background-color: {Theme.BG_ELEVATED};
    color: {Theme.TEXT_DIM};
    border-color: {Theme.BORDER};
}}

QPushButton#GhostButton {{
    background-color: {Theme.BG_CARD_ALT};
    border: 1px solid {Theme.BORDER};
    color: {Theme.TEXT};
    padding: 10px 16px;
    font-weight: 600;
    border-radius: 8px;
}}
QPushButton#GhostButton:hover {{
    background-color: {Theme.BG_ELEVATED};
    color: {Theme.TEXT};
    border-color: {Theme.BORDER_FOCUS};
}}
QPushButton#GhostButton:pressed {{ background-color: {Theme.BG_CARD}; }}

QPushButton#TrainButton {{
    background-color: {Theme.PRIMARY};
    color: #FFFFFF;
    border: 1px solid {Theme.PRIMARY};
    padding: 15px;
    font-size: 14px;
    font-weight: 700;
    border-radius: 10px;
}}
QPushButton#TrainButton:hover {{ background-color: {Theme.PRIMARY_HOVER}; }}
QPushButton#TrainButton:disabled {{
    background-color: {Theme.BG_ELEVATED};
    color: {Theme.TEXT_DIM};
    border-color: {Theme.BORDER};
}}

/* ---------- Inputs ---------- */
QComboBox, QSpinBox, QLineEdit {{
    background-color: {Theme.BG_CARD_ALT};
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {Theme.PRIMARY};
}}
QComboBox:focus, QSpinBox:focus, QLineEdit:focus {{ border-color: {Theme.BORDER_FOCUS}; }}

/* ---------- Progress ---------- */
QProgressBar {{
    background-color: {Theme.BG_CARD_ALT};
    border: none;
    border-radius: 6px;
    text-align: center;
    color: {Theme.TEXT};
    font-size: 11px;
    height: 8px;
}}
QProgressBar::chunk {{ background-color: {Theme.PRIMARY}; border-radius: 6px; }}

/* ---------- Log / Console ---------- */
QTextEdit {{
    background-color: {Theme.BG_CONSOLE};
    color: {Theme.TEXT_CONSOLE};
    border: 1px solid {Theme.BORDER};
    border-radius: 8px;
    font-family: {Theme.FONT_MONO};
    font-size: 12px;
    padding: 8px;
    selection-background-color: {Theme.PRIMARY_DIM};
}}

/* ---------- Table ---------- */
QTableWidget {{
    background-color: {Theme.BG_CARD};
    alternate-background-color: {Theme.BG_CARD_ALT};
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER};
    border-radius: 10px;
    gridline-color: {Theme.BORDER};
    selection-background-color: {Theme.PRIMARY_DIM};
    selection-color: {Theme.TEXT};
}}
QHeaderView::section {{
    background-color: {Theme.BG_ELEVATED};
    color: {Theme.TEXT_MUTED};
    padding: 10px 12px;
    border: none;
    border-bottom: 1px solid {Theme.BORDER};
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}
QTableWidget::item {{ padding: 8px 12px; }}

/* ---------- Splitter ---------- */
QSplitter::handle {{ background-color: transparent; width: 8px; }}
QSplitter::handle:hover {{ background-color: {Theme.BORDER}; }}

/* ---------- Scrollbars ---------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {Theme.BG_ELEVATED}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {Theme.BORDER_FOCUS}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

/* ---------- Status bar ---------- */
QStatusBar {{
    background-color: {Theme.BG_CARD};
    color: {Theme.TEXT_MUTED};
    border-top: 1px solid {Theme.BORDER};
    padding: 4px 12px;
    font-size: 12px;
}}
"""