"""LBOT 深色 ttk 主题；中文优先 Tk 矢量字体（Noto file=）。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from lbot_ui_font import (
    CjkLabel,
    UI_USE_PIL,
    _widget_bg,
    apply_ui_fonts,
    cjk_button,
    resolve_ui_font,
    ui_font,
)

COLORS = {
    "bg": "#12141c",
    "surface": "#1a1d28",
    "border": "#2d3344",
    "text": "#e8eaed",
    "text_muted": "#8b92a8",
    "primary": "#3b82f6",
    "primary_hover": "#2563eb",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "secondary": "#3f4658",
    "accent": "#8b5cf6",
    "canvas": "#0e1018",
    "log_bar": "#161922",
    "entry_bg": "#222633",
}

FONT_SM, FONT_MD, FONT_LG, FONT_XL = 12, 13, 15, 17
FONT_BASE = FONT_MD
FONT_SMALL = FONT_SM

_BTN_STYLE = {
    "primary": "Primary.TButton",
    "success": "Success.TButton",
    "warning": "Warning.TButton",
    "danger": "Danger.TButton",
    "secondary": "Secondary.TButton",
    "accent": "Accent.TButton",
    "neutral": "Secondary.TButton",
}


def init_theme(root=None):
    if root is not None:
        resolve_ui_font(root)
    style = apply_ui_fonts(root) if root else None
    if style is not None:
        _apply_dark_style(style)
        try:
            root.configure(bg=COLORS["bg"])
        except tk.TclError:
            pass
    return style


def _apply_dark_style(style: ttk.Style):
    style.theme_use("clam")
    c = COLORS
    style.configure(".", background=c["bg"], foreground=c["text"])
    style.configure("TFrame", background=c["bg"])
    style.configure("TLabel", background=c["bg"], foreground=c["text"])
    style.configure(
        "TLabelframe", background=c["surface"], foreground=c["text"],
        bordercolor=c["border"], relief="flat",
    )
    style.configure(
        "TLabelframe.Label", background=c["surface"], foreground=c["text"],
        font=ui_font(FONT_MD, True),
    )
    style.configure("TNotebook", background=c["bg"], borderwidth=0, tabmargins=(2, 2, 2, 0))
    style.configure(
        "TNotebook.Tab", background=c["secondary"], foreground=c["text_muted"],
        padding=(10, 4), font=ui_font(FONT_SM),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", c["primary"])],
        foreground=[("selected", "#ffffff")],
    )
    style.configure(
        "TEntry", fieldbackground=c["entry_bg"], foreground=c["text"],
        insertcolor=c["text"], bordercolor=c["border"],
    )
    style.configure(
        "TCombobox", fieldbackground=c["entry_bg"], background=c["entry_bg"],
        foreground=c["text"],
    )
    style.configure("TCheckbutton", background=c["bg"], foreground=c["text"])
    style.configure(
        "TSpinbox", fieldbackground=c["entry_bg"], foreground=c["text"],
        background=c["entry_bg"], arrowcolor=c["text"],
    )
    for name, bg, fg in (
        ("Primary.TButton", c["primary"], "#fff"),
        ("Success.TButton", c["success"], "#fff"),
        ("Warning.TButton", c["warning"], "#111"),
        ("Danger.TButton", c["danger"], "#fff"),
        ("Secondary.TButton", c["secondary"], c["text"]),
        ("Accent.TButton", c["accent"], "#fff"),
    ):
        style.configure(
            name, background=bg, foreground=fg, borderwidth=0, padding=(12, 6),
            font=ui_font(FONT_MD),
        )
        style.map(name, background=[("active", bg), ("pressed", bg)])


def create_root(title="LBOT", geometry=None, minsize=None):
    root = tk.Tk()
    root.title(title)
    if geometry:
        root.geometry(geometry)
    if minsize:
        root.minsize(*minsize)
    init_theme(root)
    return root


def create_toplevel(parent, title="", geometry=None, minsize=None):
    win = tk.Toplevel(parent)
    if title:
        win.title(title)
    if geometry:
        win.geometry(geometry)
    if minsize:
        win.minsize(*minsize)
    win.configure(bg=COLORS["bg"])
    return win


def frame(parent, **kwargs):
    kwargs.pop("transparent", None)
    kwargs.pop("padding", None)
    return ttk.Frame(parent, **kwargs)


def card(parent, title="", padding=8):
    box = ttk.LabelFrame(parent, text=title, padding=padding)
    box.body = box
    return box


def card_body(card_widget):
    return getattr(card_widget, "body", card_widget)


def label(parent, text="", textvariable=None, size=FONT_MD, bold=False, muted=False, **kwargs):
    if UI_USE_PIL:
        fg = kwargs.pop("fg", COLORS["text_muted"] if muted else COLORS["text"])
        bg = kwargs.pop("bg", None) or _widget_bg(parent, COLORS["bg"])
        return CjkLabel(
            parent, text=text, textvariable=textvariable, size=size, bold=bold,
            fg=fg, bg=bg, **kwargs,
        )
    kw = {"font": ui_font(size, bold)}
    if muted:
        kw["foreground"] = COLORS["text_muted"]
    kw.update(kwargs)
    return ttk.Label(parent, text=text, textvariable=textvariable, **kw)


def button(parent, text, command=None, style="primary", size=FONT_MD, **kwargs):
    kwargs.pop("padx", None)
    kwargs.pop("pady", None)
    kwargs.pop("width", None)
    if UI_USE_PIL:
        colors = {
            "primary": (COLORS["primary"], COLORS["primary_hover"]),
            "success": (COLORS["success"], "#16a34a"),
            "warning": (COLORS["warning"], "#d97706"),
            "danger": (COLORS["danger"], "#dc2626"),
            "secondary": (COLORS["secondary"], "#4b5563"),
            "accent": (COLORS["accent"], "#7c3aed"),
            "neutral": (COLORS["secondary"], "#4b5563"),
        }
        bg, hover = colors.get(style, colors["primary"])
        return cjk_button(
            parent, text, command=command, size=size, bg=bg,
            activebackground=hover, **kwargs,
        )
    return ttk.Button(
        parent, text=text, command=command,
        style=_BTN_STYLE.get(style, "Primary.TButton"), **kwargs,
    )


def checkbox(parent, text, variable, **kwargs):
    return ttk.Checkbutton(parent, text=text, variable=variable, **kwargs)


def spinbox(parent, textvariable, from_, to, increment=0.05, width=7, fmt="%.3f"):
    return ttk.Spinbox(
        parent, from_=from_, to=to, increment=increment, width=width,
        textvariable=textvariable, format=fmt,
    )


def entry(parent, textvariable=None, width=200, **kwargs):
    return ttk.Entry(parent, textvariable=textvariable, width=max(8, width // 10), **kwargs)


def tabview(parent):
    return ttk.Notebook(parent)


def add_tab(notebook, name, padding=4):
    page = ttk.Frame(notebook, padding=padding)
    notebook.add(page, text=name)
    return page


def separator(parent, orient=tk.HORIZONTAL):
    return ttk.Separator(parent, orient=orient)


def canvas(parent, width, height, bg=None, **kwargs):
    return tk.Canvas(
        parent, width=width, height=height, bg=bg or COLORS["canvas"],
        highlightthickness=0, **kwargs,
    )


def scale(parent, variable, from_=0, to=255, length=170, command=None):
    return tk.Scale(
        parent, from_=from_, to=to, orient=tk.HORIZONTAL, variable=variable,
        showvalue=False, length=length, bg=COLORS["surface"], fg=COLORS["text"],
        troughcolor=COLORS["entry_bg"], highlightthickness=0, bd=0,
        command=command,
    )


def combobox(parent, variable, values, width=8):
    return ttk.Combobox(
        parent, textvariable=variable, values=list(values),
        state="readonly", width=width,
    )


def log_bar(parent, textvariable, **kwargs):
    wrap = kwargs.pop("wraplength", 1200)
    bar = tk.Frame(parent, bg=COLORS["log_bar"], highlightthickness=0)
    if UI_USE_PIL:
        lbl = CjkLabel(
            bar, textvariable=textvariable, size=FONT_MD,
            fg=COLORS["text_muted"], bg=COLORS["log_bar"],
            wraplength=wrap, padx=8, pady=6,
        )
        lbl.pack(fill=tk.X)
        return bar
    lbl = ttk.Label(
        bar, textvariable=textvariable, foreground=COLORS["text_muted"],
        font=ui_font(FONT_MD), wraplength=wrap,
    )
    lbl.pack(fill=tk.X, padx=8, pady=6)
    return bar


def status_header(parent, title, subtitle=""):
    row = ttk.Frame(parent)
    label(row, text=title, size=FONT_XL, bold=True).pack(side=tk.LEFT)
    if subtitle:
        label(row, text=subtitle, size=FONT_MD, muted=True).pack(side=tk.RIGHT)
    return row
