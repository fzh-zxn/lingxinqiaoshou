#!/usr/bin/env python3
"""
用 RGB（蓝色）+ 深度找蓝色长方形筐（独立检验窗口）。

核心算法见 lbot_basket_rgbd.py（与 demo_pick_square 共用）。

用法：
  python demo_find_basket_rgbd.py
  python demo_find_basket_rgbd.py --image shot.jpg --depth d.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lbot_basket_rgbd import (  # noqa: E402
    DEFAULT_HSV_HIGH,
    DEFAULT_HSV_LOW,
    draw_basket,
    find_blue_basket_rgbd,
)
from lbot_grasp_utils import (  # noqa: E402
    depth_colormap_view,
    start_depth_processes,
    stop_process,
)


def _safe_imread(path: Path, flags=cv2.IMREAD_COLOR):
    if path is None or not path.is_file() or path.stat().st_size < 64:
        return None
    return cv2.imread(str(path), flags)


def main():
    ap = argparse.ArgumentParser(description="RGB+深度找蓝色长方形筐（旋转紧框）")
    ap.add_argument("--image", type=Path, default=None, help="离线彩色图")
    ap.add_argument("--depth", type=Path, default=None, help="离线深度 png(uint16 mm)")
    ap.add_argument("--hsv-low", type=int, nargs=3, default=list(DEFAULT_HSV_LOW))
    ap.add_argument("--hsv-high", type=int, nargs=3, default=list(DEFAULT_HSV_HIGH))
    ap.add_argument("--min-rim-delta-mm", type=float, default=20.0)
    ap.add_argument("--require-depth", action="store_true")
    ap.add_argument("--color-topic", default="/camera/color/image_raw")
    ap.add_argument("--depth-topic", default="/camera/depth/image_raw")
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args()

    hsv_low = tuple(args.hsv_low)
    hsv_high = tuple(args.hsv_high)

    def _run_once(color, depth):
        hit, mask, cands = find_blue_basket_rgbd(
            color,
            depth,
            hsv_low=hsv_low,
            hsv_high=hsv_high,
            min_rim_delta_mm=args.min_rim_delta_mm,
            require_depth=bool(args.require_depth),
        )
        title = (
            f"cands={len(cands)}  rimΔ≥{args.min_rim_delta_mm:.0f}mm  OBB  q=quit"
        )
        vis = draw_basket(color, hit, mask=mask, title=title)
        if hit:
            print(
                f"HIT {hit.fit_mode} size={hit.size_wh[0]:.0f}x{hit.size_wh[1]:.0f} "
                f"ang={hit.angle_deg:.1f} score={hit.score:.1f} rimΔ={hit.rim_delta_mm}  "
                f"{hit.reason}",
                flush=True,
            )
        else:
            print(f"MISS cands={len(cands)}", flush=True)
        return vis, hit

    if args.image is not None:
        color = cv2.imread(str(args.image))
        if color is None:
            raise SystemExit(f"读图失败: {args.image}")
        depth = None
        if args.depth is not None:
            depth = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise SystemExit(f"读深度失败: {args.depth}")
        vis, hit = _run_once(color, depth)
        out = args.save or (ROOT / "_basket_rgbd_preview.jpg")
        cv2.imwrite(str(out), vis)
        print(f"已写 {out.resolve()}", flush=True)
        try:
            if depth is not None:
                dv, _ = depth_colormap_view(depth, color.shape[1], color.shape[0])
                cv2.imshow("basket depth", dv)
            cv2.imshow("basket rgbd", vis)
            print("按任意键关闭…", flush=True)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"无 GUI，跳过 imshow（{exc}）", flush=True)
        return

    print("启动相机…", flush=True)
    launch = relay = None
    color_file = depth_file = log_path = None
    try:
        launch, relay, depth_file, color_file, log_path = start_depth_processes(
            args.depth_topic, color_topic=args.color_topic,
        )
        print(f"color={color_file} depth={depth_file} log={log_path}", flush=True)
        t0 = time.time()
        frame = None
        while time.time() - t0 < 20.0:
            frame = _safe_imread(color_file)
            if frame is not None:
                break
            time.sleep(0.1)
        if frame is None:
            raise SystemExit(f"等不到彩色首帧。见 {log_path}")
        print(f"首帧 OK {frame.shape[1]}x{frame.shape[0]}", flush=True)

        win = "basket rgbd"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        last_print = 0.0
        while True:
            color = _safe_imread(color_file)
            if color is None:
                time.sleep(0.02)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            depth = _safe_imread(depth_file, cv2.IMREAD_UNCHANGED)
            hit, mask, cands = find_blue_basket_rgbd(
                color,
                depth,
                hsv_low=hsv_low,
                hsv_high=hsv_high,
                min_rim_delta_mm=args.min_rim_delta_mm,
                require_depth=bool(args.require_depth),
            )
            title = (
                f"cands={len(cands)}  rimΔ≥{args.min_rim_delta_mm:.0f}mm  "
                f"depth={'ok' if depth is not None else 'none'}  OBB  q=quit s=save"
            )
            vis = draw_basket(color, hit, mask=mask, title=title)
            now = time.time()
            if now - last_print > 1.0:
                if hit:
                    print(
                        f"HIT {hit.fit_mode} {hit.size_wh[0]:.0f}x{hit.size_wh[1]:.0f} "
                        f"ang={hit.angle_deg:.1f} rimΔ={hit.rim_delta_mm} "
                        f"score={hit.score:.1f}",
                        flush=True,
                    )
                else:
                    print(f"MISS cands={len(cands)} depth={depth is not None}", flush=True)
                last_print = now
            cv2.imshow(win, vis)
            if depth is not None:
                dv, _ = depth_colormap_view(depth, color.shape[1], color.shape[0])
                if hit is not None:
                    import numpy as np
                    c = np.asarray(hit.corners, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(dv, [c], True, (255, 255, 255), 2)
                cv2.imshow("basket depth", dv)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = args.save or (ROOT / f"_basket_rgbd_{int(now)}.jpg")
                cv2.imwrite(str(out), vis)
                print(f"已保存 {out}", flush=True)
    finally:
        cv2.destroyAllWindows()
        stop_process(relay)
        stop_process(launch)
        for p in (color_file, depth_file):
            if p is not None:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
