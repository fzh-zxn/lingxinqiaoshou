#!/usr/bin/env python3
"""
单独检验放置/筐盘 YOLO（默认 框best.pt），不连机械臂。

用法（在 lbot_pick_demo 目录）：
  python demo_detect_basket.py
  python demo_detect_basket.py --model 框best.pt --conf 0.25
  python demo_detect_basket.py --model 盘best.pt
  python demo_detect_basket.py --image /path/to.jpg

按 q / Esc 退出。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lbot_grasp_utils import (  # noqa: E402
    draw_detections,
    filter_place_detections,
    load_yolo_model,
    start_depth_processes,
    stop_process,
    yolo_class_names_brief,
)


def _safe_imread(path: Path, flags=cv2.IMREAD_COLOR):
    if path is None or not path.is_file() or path.stat().st_size < 64:
        return None
    return cv2.imread(str(path), flags)


def _predict(yolo, frame_bgr, conf: float, imgsz: int = 640):
    """返回 (annotated_bgr, n_all, n_place)。"""
    yolo.model.conf = float(conf)
    results = yolo.predict([frame_bgr], size=int(imgsz))
    names = getattr(getattr(yolo, "model", None), "names", {}) or {}
    dets = None
    try:
        pred0 = results.pred[0]
        if pred0 is not None and len(pred0):
            dets = pred0.detach().cpu().numpy()
    except Exception:
        dets = None

    vis = frame_bgr.copy()
    n_all = 0
    n_place = 0
    if dets is not None and dets.size:
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        n_all = int(dets.shape[0])
        draw_detections(vis, dets, names)
        place = filter_place_detections(dets, names)
        n_place = len(place) if place is not None else 0
        # 单类「框」模型：filter 应全收；若类名未登记则 n_place 可能为 0，改用全部
        if n_place == 0 and n_all > 0 and len(names) <= 1:
            n_place = n_all
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), (40, 40, 40), -1)
        cv2.putText(
            vis,
            f"basket  dets={n_all} place={n_place}  conf>={conf:.2f}  q=quit s=save",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            vis,
            f"no detection  conf>={conf:.2f}  q=quit",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    return vis, n_all, n_place


def main():
    ap = argparse.ArgumentParser(description="筐子 YOLO 单独检验")
    ap.add_argument(
        "--model",
        type=Path,
        default=ROOT / "9.4.筐子.pt",
        help="放置权重（默认 9.4.筐子.pt；也可 框best.pt / 盘best.pt）",
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--image", type=Path, default=None, help="离线单张图")
    ap.add_argument(
        "--color-topic",
        default="/camera/color/image_raw",
    )
    ap.add_argument(
        "--depth-topic",
        default="/camera/depth/image_raw",
    )
    ap.add_argument("--save", type=Path, default=None, help="保存一帧标注图后退出")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model_path = args.model.expanduser()
    if not model_path.is_file():
        cand = ROOT / str(args.model)
        if cand.is_file():
            model_path = cand
    if not model_path.is_file():
        raise SystemExit(f"找不到模型: {args.model}")

    print(f"加载 YOLO: {model_path.resolve()} …", flush=True)
    yolo = load_yolo_model(model_path, args.device, args.conf)
    print(
        f"就绪 classes={yolo_class_names_brief(yolo)} conf={args.conf:.2f}",
        flush=True,
    )

    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            raise SystemExit(f"读图失败: {args.image}")
        vis, n_all, n_place = _predict(yolo, img, args.conf, args.imgsz)
        print(f"检测: 全部={n_all} 放置类={n_place}", flush=True)
        out = args.save or (ROOT / "_basket_detect_preview.jpg")
        cv2.imwrite(str(out), vis)
        print(f"已写 {out.resolve()}", flush=True)
        cv2.imshow("basket detect", vis)
        print("按任意键关闭窗口…", flush=True)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print("启动相机（复用已有 Orbbec 话题则不二次 launch）…", flush=True)
    launch = relay = None
    color_file = depth_file = log_path = None
    try:
        launch, relay, depth_file, color_file, log_path = start_depth_processes(
            args.depth_topic, color_topic=args.color_topic,
        )
        print(f"彩色文件 {color_file}  日志 {log_path}", flush=True)
        t0 = time.time()
        frame = None
        while time.time() - t0 < 20.0:
            frame = _safe_imread(color_file)
            if frame is not None:
                break
            time.sleep(0.1)
        if frame is None:
            raise SystemExit(
                f"等不到彩色首帧（>20s）。看日志 {log_path}；"
                "若 pick 正在占相机，可先停 pick 或确认 /camera/color/image_raw 有数据。"
            )
        print(f"首帧 OK {frame.shape[1]}x{frame.shape[0]}", flush=True)

        win = "place detect"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        last_print = 0.0
        while True:
            frame = _safe_imread(color_file)
            if frame is None:
                time.sleep(0.02)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            vis, n_all, n_place = _predict(yolo, frame, args.conf, args.imgsz)
            now = time.time()
            if now - last_print > 1.0:
                print(f"dets={n_all} place={n_place}", flush=True)
                last_print = now
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = args.save or (ROOT / f"_basket_detect_{int(now)}.jpg")
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
