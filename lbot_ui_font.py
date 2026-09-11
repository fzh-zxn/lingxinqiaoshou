"""中文界面字体：标签/按钮用 Noto 2x 超采样；输入框等用 Tk 族名。"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import ttk

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:
    Image = ImageDraw = ImageFont = ImageTk = None

_CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
)

_UI_FONT_PATH = None
_UI_FONT_BOLD_PATH = None
_PIL_OK = False
_PHOTO_CACHE: dict = {}
_FONT_CACHE: dict = {}
UI_FONT = "Sans"
UI_USE_PIL = False
UI_SHARP_PIL = False
UI_SCALE = 1.0

_UI_INVALID_BG = frozenset({"transparent", "Transparent", ""})


def _family_from_font_file(path: str) -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["fc-query", "-f", "%{family}\n", path],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            for part in result.stdout.replace(",", "\n").splitlines():
                name = part.strip()
                if name:
                    return name
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _find_cjk_font_file():
    for path in _CJK_FONT_CANDIDATES:
        if Path(path).is_file():
            index = 0 if path.endswith(".ttc") else 0
            return path, index
    return None, 0


def _normalize_tk_scaling(root) -> float:
    """把 1.357 这类 DPI 缩放对齐到最近的标准分数，减轻 ttk 文字发糊。"""
    try:
        scale = float(root.tk.call("tk", "scaling"))
        candidates = (1.0, 1.25, 1.3333333333333333, 1.5, 1.75, 2.0)
        best = min(candidates, key=lambda target: abs(target - scale))
        if abs(best - scale) < 0.12:
            root.tk.call("tk", "scaling", best)
            return best
        return scale
    except tk.TclError:
        return 1.0


def _pil_font(size: int, bold: bool = False):
    global _UI_FONT_PATH
    key = (size, bold, _UI_FONT_PATH)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    if _UI_FONT_PATH is None:
        return None
    path = _UI_FONT_BOLD_PATH if bold and _UI_FONT_BOLD_PATH else _UI_FONT_PATH
    try:
        if path.endswith(".ttc"):
            font = ImageFont.truetype(path, size=size, index=0)
        else:
            font = ImageFont.truetype(path, size=size)
    except OSError:
        font = ImageFont.truetype(_UI_FONT_PATH, size=size)
    _FONT_CACHE[key] = font
    return font


def _wrap_text(text: str, font, wrap_px: int | None) -> str:
    if not wrap_px or wrap_px <= 0:
        return text
    lines = []
    for paragraph in str(text).split("\n"):
        if not paragraph:
            lines.append("")
            continue
        line = ""
        for ch in paragraph:
            trial = line + ch
            try:
                w = font.getlength(trial)
            except AttributeError:
                w = font.getsize(trial)[0]
            if line and w > wrap_px:
                lines.append(line)
                line = ch
            else:
                line = trial
        lines.append(line)
    return "\n".join(lines)


def render_cjk_photo(text, size=14, fill="#1a1a1a", bg="#f0f0f0", bold=False,
                     pad_x=8, pad_y=4, wrap=None, scale=None):
    if not _PIL_OK or Image is None:
        return None
    supersample = 2 if UI_SHARP_PIL else 1
    px = max(11, int(round(size))) * supersample
    font = _pil_font(px, bold=bold)
    if font is None:
        return None
    text = "" if text is None else str(text)
    wrap_px = None if not wrap else int(round(wrap)) * supersample
    text = _wrap_text(text, font, wrap_px)
    cache_key = (text, px, fill, bg, bold, pad_x, pad_y, wrap_px, supersample)
    hit = _PHOTO_CACHE.get(cache_key)
    if hit is not None:
        return hit
    dummy = Image.new("RGB", (8, 8), bg)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    img = Image.new("RGB", (tw + 2 * pad_x, th + 2 * pad_y), bg)
    draw = ImageDraw.Draw(img)
    draw.multiline_text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=fill, spacing=4)
    if supersample > 1:
        img = img.resize(
            (max(1, img.width // supersample), max(1, img.height // supersample)),
            Image.Resampling.LANCZOS,
        )
    photo = ImageTk.PhotoImage(image=img)
    _PHOTO_CACHE[cache_key] = photo
    if len(_PHOTO_CACHE) > 256:
        for _ in range(64):
            _PHOTO_CACHE.pop(next(iter(_PHOTO_CACHE)), None)
    return photo


def _normalize_bg(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, (tuple, list)):
        value = value[0] if value else fallback
    value = str(value).strip()
    if value in _UI_INVALID_BG:
        return fallback
    return value


def _widget_attrs_for_bg(widget):
    if "customtkinter" in getattr(type(widget), "__module__", ""):
        return ("fg_color",)
    return ("bg", "background")


def _widget_bg(widget, fallback="#1a1d26"):
    current = widget
    while current is not None:
        for attr in _widget_attrs_for_bg(current):
            try:
                value = _normalize_bg(current.cget(attr), None)
                if value:
                    return value
            except (tk.TclError, AttributeError, ValueError):
                pass
        try:
            current = current.master
        except AttributeError:
            break
    try:
        return _normalize_bg(widget.winfo_toplevel().cget("bg"), fallback)
    except tk.TclError:
        return fallback


def _tk_font_measures_cjk(font: tkfont.Font) -> bool:
    try:
        actual = font.actual()
        family = str(actual.get("family", "")).lower()
        if family in {"fixed", "nil", "courier", "monospace"}:
            return False
        w_cjk = font.measure("中文测试")
        w_ascii = font.measure("ABCD")
        return w_cjk >= max(int(w_ascii * 0.85), 28)
    except tk.TclError:
        return False


def _tk_family_renders_cjk(root, family: str, size: int = 13) -> bool:
    try:
        return _tk_font_measures_cjk(tkfont.Font(root=root, family=family, size=size))
    except tk.TclError:
        return False


def _collect_tk_family_candidates() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str | None):
        if not name or name in seen:
            return
        seen.add(name)
        names.append(name)

    for name in (
        "clearlyu", "fangsong ti", "song ti",
        "AR PL UMing CN", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
    ):
        add(name)
    if _UI_FONT_PATH:
        try:
            import subprocess

            result = subprocess.run(
                ["fc-query", "-f", "%{family}\n", _UI_FONT_PATH],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0:
                for part in result.stdout.replace(",", "\n").splitlines():
                    add(part.strip())
        except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
            add(_family_from_font_file(_UI_FONT_PATH))
    for name in (
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
        "Droid Sans Fallback", "Source Han Sans SC", "Microsoft YaHei",
    ):
        add(name)
    return names


def _pick_tk_family(root) -> str | None:
    for name in _collect_tk_family_candidates():
        if _tk_family_renders_cjk(root, name):
            return name
    return None


def ui_font(size=14, bold=False):
    size = int(size)
    weight = "bold" if bold else "normal"
    return (UI_FONT, size, weight)


class CjkLabel(tk.Label):
    """Pillow 栅格化标签（UI_USE_PIL 时使用）。"""

    def __init__(self, master, text="", textvariable=None, size=13, fg="#e8eaed",
                 bg=None, bold=False, wraplength=0, anchor="w", justify=tk.LEFT,
                 padx=0, pady=0, **kwargs):
        bg = _normalize_bg(bg if bg is not None else _widget_bg(master), "#1a1d26")
        super().__init__(master, bg=bg, fg=fg, bd=0, highlightthickness=0,
                         anchor=anchor, justify=justify, padx=padx, pady=pady, **kwargs)
        self._cjk_size = size
        self._cjk_fg = fg
        self._cjk_bg = bg
        self._cjk_bold = bold
        self._cjk_wrap = wraplength if wraplength else None
        self._cjk_var = textvariable
        self._cjk_static = text
        self._cjk_img = None
        self._cjk_last = None
        if textvariable is not None:
            textvariable.trace_add("write", self._on_var)
        self._refresh()

    def _on_var(self, *_args):
        self._refresh()

    def set_text(self, text):
        self._cjk_static = text
        self._refresh()

    def _refresh(self):
        text = self._cjk_var.get() if self._cjk_var is not None else self._cjk_static
        if text == self._cjk_last and self._cjk_img is not None:
            return
        self._cjk_last = text
        photo = render_cjk_photo(text, size=self._cjk_size, fill=self._cjk_fg,
                                 bg=self._cjk_bg, bold=self._cjk_bold, wrap=self._cjk_wrap)
        if photo is not None:
            self._cjk_img = photo
            self.configure(image=photo, text="")
        else:
            self.configure(image="", text=text, font=ui_font(self._cjk_size, self._cjk_bold))


def cjk_button(master, text, command=None, size=13, fg="#ffffff", bg="#3498db",
               activebackground=None, bold=True, padx=12, pady=6, **kwargs):
    activebackground = activebackground or bg
    photo = render_cjk_photo(text, size=size, fill=fg, bg=bg, bold=bold, pad_x=padx, pad_y=pady)
    w = tk.Label(
        master,
        image=photo if photo else None,
        text="" if photo else text,
        font=ui_font(size, bold) if not photo else None,
        bg=bg,
        fg=fg,
        bd=0,
        highlightthickness=0,
        cursor="hand2",
        **kwargs,
    )
    w._cjk_img = photo
    w.bind("<Button-1>", lambda _e: command() if command else None)
    return w


def cjk_labelwidget(parent, text, size=12, bold=True, fg="#e8eaed"):
    bg = _widget_bg(parent, "#1a1d26")
    return CjkLabel(parent, text=text, size=size, bold=bold, fg=fg, bg=bg, padx=4)


def resolve_ui_font(root):
    global UI_FONT, UI_USE_PIL, UI_SHARP_PIL, UI_SCALE, _UI_FONT_PATH, _UI_FONT_BOLD_PATH, _PIL_OK

    UI_SCALE = _normalize_tk_scaling(root)

    _UI_FONT_PATH, _ = _find_cjk_font_file()
    bold = _UI_FONT_PATH.replace("Regular", "Bold") if _UI_FONT_PATH else ""
    _UI_FONT_BOLD_PATH = bold if bold != _UI_FONT_PATH and Path(bold).is_file() else None
    _PIL_OK = Image is not None and _UI_FONT_PATH is not None

    tk_family = _pick_tk_family(root)
    UI_FONT = tk_family or "Sans"

    # conda tk 无 font -file，song ti 矢量在 1.36 DPI 下仍糊；
    # 标签/按钮改用 Noto 文件 + 2x 超采样，输入框等仍用 Tk 族名。
    if _PIL_OK:
        UI_USE_PIL = True
        UI_SHARP_PIL = True
        return UI_FONT

    UI_USE_PIL = False
    UI_SHARP_PIL = False
    return UI_FONT


def apply_ui_fonts(root, family=None):
    global UI_FONT
    resolve_ui_font(root)
    family = family or UI_FONT
    UI_FONT = family
    size = 13
    root.option_add("*Font", ui_font(size))
    style = ttk.Style(root)
    base = ui_font(size)
    style.configure(".", font=base)
    for name in ("TLabel", "TButton", "TLabelframe.Label", "TNotebook.Tab",
                 "TCheckbutton", "TRadiobutton", "TEntry", "TCombobox", "TSpinbox"):
        style.configure(name, font=base)
    return style


def font_status_line():
    if UI_USE_PIL and UI_SHARP_PIL:
        mode = "Noto 2x超采样"
    elif UI_USE_PIL:
        mode = "Pillow位图"
    else:
        mode = "Tk矢量族名"
    return f"界面字体: {mode} tk={UI_FONT} file={_UI_FONT_PATH or '—'} scale={UI_SCALE:.2f}"
