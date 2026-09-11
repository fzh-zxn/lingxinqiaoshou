#!/usr/bin/env python3
"""
右臂眼在手上 · 单窗口监控 + 抓取

入口已瘦身：实现见 pick_app/（按职责 Mixin 分包）。

用法：
  python demo_pick_square.py --dry-run
  python demo_pick_square.py --pick --confirm
  python demo_pick_square.py --preview
"""

from __future__ import annotations

from pick_app.main import main

if __name__ == "__main__":
    main()
