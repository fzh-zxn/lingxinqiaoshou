#!/usr/bin/env python3
"""
螺丝板孔位标定：红点实测 → 板局部归一化 (u,v)。

默认：每个孔独立测量，不做左右镜像（板不完全对称时必须如此）。
可选多张不同角度照片，对同一 id 的 (u,v) 取中位数，压噪声。

布局（--expect 8 --per-corner 2）：
  id 0,1  左上 TL（距角近→远）
  id 2,3  右上 TR
  id 4,5  左下 BL
  id 6,7  右下 BR

用法：
  python calibrate_board_holes.py photo.png --expect 8 --per-corner 2 --write-yaml
  python calibrate_board_holes.py a.png b.png --expect 8 --per-corner 2 --write-yaml
  python calibrate_board_holes.py a.png b.png --expect 8 --per-corner 2 --apply-config
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """四点 → TL, TR, BR, BL。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(4)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl], axis=0)


def detect_board_from_overlay(bgr, dots_xy=None):
    """
    UI 截图上的黄/橙 YOLO 板框 → 板四角。
    用「黄像素贴在矩形周界上」打分，并要求框尺寸贴近红点跨度（避免并进左侧黄字）。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ym = cv2.inRange(hsv, (12, 80, 100), (45, 255, 255))
    ym = cv2.dilate(ym, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    h, w = bgr.shape[:2]
    contours, _ = cv2.findContours(ym, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    dots = None
    span_x = span_y = None
    if dots_xy is not None and len(dots_xy) >= 4:
        dots = np.asarray(dots_xy, dtype=np.float64).reshape(-1, 2)
        span_x = float(np.ptp(dots[:, 0]))
        span_y = float(np.ptp(dots[:, 1]))
        dc = dots.mean(axis=0)

    def _perimeter_yellow_ratio(box: np.ndarray) -> float:
        """矩形周界采样点落在黄掩膜上的比例（线框高、实心块低）。"""
        ordered = _order_quad(box)
        pts = []
        for i in range(4):
            a = ordered[i]
            b = ordered[(i + 1) % 4]
            for t in np.linspace(0.0, 1.0, 48, endpoint=False):
                pts.append(a + (b - a) * t)
        hit = 0
        for p in pts:
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            if 0 <= x < w and 0 <= y < h and ym[y, x] > 0:
                hit += 1
        return hit / max(len(pts), 1)

    for c in contours:
        peri = float(cv2.arcLength(c, True))
        if peri < 280:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = float(rect[1][0]), float(rect[1][1])
        short, long = sorted((rw, rh))
        if short < 80 or long < 120:
            continue
        ar = long / max(short, 1.0)
        if ar < 1.05 or ar > 2.8:
            continue
        box = np.array(cv2.boxPoints(rect), dtype=np.float32)
        area = float(rw * rh)
        frac = area / float(h * w)
        if frac < 0.04 or frac > 0.55:
            continue
        if dots is not None:
            # 框宽/高相对红点跨度不能过大（否则吞进左侧黄字）
            if rw > span_x * 1.65 or rh > span_y * 1.65:
                # 旋转矩形 wh 可能对调
                if max(rw, rh) > max(span_x, span_y) * 1.65 and min(rw, rh) > min(span_x, span_y) * 1.65:
                    continue
            inside = sum(
                1 for x, y in dots
                if cv2.pointPolygonTest(box.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0
            )
            if inside < max(4, int(0.7 * len(dots))):
                continue
            # 左右相对红点的外扩应大致对称，禁止单侧狂扩
            left_pad = float(dots[:, 0].min() - box[:, 0].min())
            right_pad = float(box[:, 0].max() - dots[:, 0].max())
            if left_pad > right_pad * 2.5 + 25 or right_pad > left_pad * 2.5 + 25:
                continue
            top_pad = float(dots[:, 1].min() - box[:, 1].min())
            bot_pad = float(box[:, 1].max() - dots[:, 1].max())
            if top_pad > bot_pad * 2.5 + 25 or bot_pad > top_pad * 2.5 + 25:
                continue
        yratio = _perimeter_yellow_ratio(box)
        if yratio < 0.18:
            continue
        score = peri * yratio * min(frac / 0.12, 1.5)
        if dots is not None:
            bc = box.mean(axis=0)
            dist = float(np.linalg.norm(bc - dc)) / max(float(np.hypot(w, h)), 1.0)
            score *= (0.4 + 0.6 * (inside / max(len(dots), 1))) * (1.0 - min(dist, 0.35))
            # 尺寸贴近红点跨度加分
            size_pen = abs(max(rw, rh) / max(span_x, span_y, 1.0) - 1.25)
            score *= max(0.2, 1.0 - 0.5 * size_pen)
        if score > best_score:
            best_score = score
            best = box
    if best is not None:
        return _order_quad(best)
    return None


def detect_board_dark_containing_dots(bgr, dots_xy, max_gray=75, max_area_frac=0.5):
    """全图暗色板 OBB，必须包住红点；比 dots_roi 更贴真实板外沿。"""
    h, w = bgr.shape[:2]
    dots = np.asarray(dots_xy, dtype=np.float64).reshape(-1, 2)
    if len(dots) < 4:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # 避开顶部状态条
    mask = cv2.inRange(gray, 0, int(max_gray))
    mask[: max(1, h // 14), :] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    dc = dots.mean(axis=0)
    for c in contours:
        area = float(cv2.contourArea(c))
        frac = area / float(h * w)
        if frac < 0.04 or frac > float(max_area_frac):
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = float(rect[1][0]), float(rect[1][1])
        short, long = sorted((rw, rh))
        if short < 40 or long / max(short, 1.0) < 1.05:
            continue
        box = np.array(cv2.boxPoints(rect), dtype=np.float32)
        inside = sum(
            1 for x, y in dots
            if cv2.pointPolygonTest(box.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0
        )
        if inside < max(4, int(0.75 * len(dots))):
            continue
        # 红点不应贴死边框（外圈孔相对板缘有 inset）；过紧的框扣分
        dists = [
            abs(cv2.pointPolygonTest(box.reshape(-1, 1, 2), (float(x), float(y)), True))
            for x, y in dots
        ]
        min_inset = float(np.percentile(dists, 20)) if dists else 0.0
        bc = box.mean(axis=0)
        dist = float(np.linalg.norm(bc - dc)) / max(float(np.hypot(w, h)), 1.0)
        score = (
            (inside / max(len(dots), 1))
            * min(frac / 0.12, 1.2)
            * (1.0 - min(dist, 0.3))
            * (1.0 + min(min_inset, 40.0) / 40.0)
        )
        if score > best_score:
            best_score = score
            best = box
    if best is None:
        return None
    return _order_quad(best)


def detect_board_obb(bgr, max_gray=90, max_area_frac=0.45):
    """暗色板 minAreaRect；限制最大面积，避免把地板当板。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.inRange(gray, 0, int(max_gray))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1,
    )
    h, w = gray.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    for c in contours:
        area = float(cv2.contourArea(c))
        frac = area / float(h * w)
        if frac < 0.03 or frac > float(max_area_frac):
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw * rh < 1:
            continue
        short, long = sorted((float(rw), float(rh)))
        if short < 20 or long / max(short, 1) < 1.05:
            continue
        fill = area / max(float(rw * rh), 1.0)
        score = fill * min(frac / 0.15, 1.0)
        if score > best_score:
            best_score = score
            best = np.array(cv2.boxPoints(rect), dtype=np.float32)
    if best is None:
        return None
    return _order_quad(best)


def detect_board_from_red_hull(dots_xy, pad_frac=0.18, img_wh=None):
    """红点凸包外扩成近似板框。"""
    pts = np.asarray(dots_xy, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return None
    hull = cv2.convexHull(pts)
    rect = cv2.minAreaRect(hull)
    box = np.array(cv2.boxPoints(rect), dtype=np.float32)
    c = box.mean(axis=0)
    box = c + (box - c) * (1.0 + float(pad_frac))
    if img_wh is not None:
        w, h = img_wh
        box[:, 0] = np.clip(box[:, 0], 0, w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, h - 1)
    return _order_quad(box)


def detect_red_dots(bgr, board_poly=None, min_area=4, max_area=2500, expect=12):
    """HSV 红点中心；可选限制在板多边形内。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 50, 50), (15, 255, 255))
    m2 = cv2.inRange(hsv, (165, 50, 50), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
    )
    if board_poly is not None:
        roi = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(roi, np.int32(board_poly), 255)
        roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        mask = cv2.bitwise_and(mask, roi)

    dots = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < 1.2 or radius > 45:
            continue
        dots.append((float(cx), float(cy), float(radius), area))
    if expect > 0 and len(dots) > expect + 2:
        dots = sorted(dots, key=lambda d: d[3], reverse=True)[:expect]
    return dots


def pixel_to_uv(pts_xy, quad_tl_tr_br_bl):
    """像素 → 板局部归一化 uv（透视）。"""
    src = np.asarray(quad_tl_tr_br_bl, dtype=np.float32)
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out


def filter_dot_outliers(xy, keep=12, max_dev_frac=0.55):
    """去掉远离主簇的误检。"""
    del max_dev_frac
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) <= keep:
        med = np.median(xy, axis=0)
        d = np.linalg.norm(xy - med, axis=1)
        thr = float(np.median(d) + 2.5 * (np.median(np.abs(d - np.median(d))) + 1e-6))
        thr = max(thr, float(np.percentile(d, 75) * 1.8))
        mask = d <= thr
        if int(mask.sum()) >= max(4, keep - 2):
            return xy[mask]
        return xy
    med = np.median(xy, axis=0)
    d = np.linalg.norm(xy - med, axis=1)
    idx = np.argsort(d)[:keep]
    return xy[idx]


def detect_board_around_dots(bgr, dots_xy, max_gray=80, pad_px=80):
    """在红点包围盒（较大外扩）内做暗色板 OBB；兜底用，优先 overlay/全图暗色。"""
    h, w = bgr.shape[:2]
    xy = np.asarray(dots_xy, dtype=np.float64)
    # 外圈孔距板缘通常 >40px，pad 过小会把下/上沿裁掉 → 绿框「切孔」
    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1.0))
    pad = int(max(pad_px, 0.18 * span))
    x0 = int(max(0, xy[:, 0].min() - pad))
    y0 = int(max(0, xy[:, 1].min() - pad))
    x1 = int(min(w, xy[:, 0].max() + pad))
    y1 = int(min(h, xy[:, 1].max() + pad))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    roi = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.inRange(gray, 0, int(max_gray))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:
        return None
    box = np.array(cv2.boxPoints(cv2.minAreaRect(c)), dtype=np.float32)
    box[:, 0] += float(x0)
    box[:, 1] += float(y0)
    return _order_quad(box)


def cluster_corners_kmeans(xy, k=4):
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if len(xy) < k:
        return None, None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.05)
    _compact, labels, centers = cv2.kmeans(
        xy, k, None, crit, 12, cv2.KMEANS_PP_CENTERS,
    )
    return labels.ravel().astype(int), centers.reshape(k, 2)


def sort_holes_corners_measured(uvs, xy, per_corner=2):
    """
    四角分组 + 组内按距角点由近到远编号。
    每个点的 (u,v) 保持实测，绝不左右镜像。
    """
    uvs = np.asarray(uvs, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    corner_uv = {
        "TL": np.array([0.0, 0.0]),
        "TR": np.array([1.0, 0.0]),
        "BL": np.array([0.0, 1.0]),
        "BR": np.array([1.0, 1.0]),
    }
    corner_order = ("TL", "TR", "BL", "BR")

    labels, _centers = cluster_corners_kmeans(xy, k=4)
    groups = {name: [] for name in corner_order}

    if labels is not None and len(np.unique(labels)) == 4:
        cen_uv = []
        for ci in range(4):
            pts_uv = uvs[labels == ci]
            cen_uv.append(pts_uv.mean(axis=0) if len(pts_uv) else np.array([0.5, 0.5]))
        cen_uv = np.asarray(cen_uv, dtype=np.float64)
        used = set()
        for name in corner_order:
            target = corner_uv[name]
            best_i, best_d = None, 1e9
            for ci in range(4):
                if ci in used:
                    continue
                d = float(np.linalg.norm(cen_uv[ci] - target))
                if d < best_d:
                    best_d, best_i = d, ci
            used.add(best_i)
            groups[name] = np.where(labels == best_i)[0].tolist()
    else:
        corners = np.stack([corner_uv[n] for n in corner_order], axis=0)
        d2 = ((uvs[:, None, :] - corners[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)
        for ci, name in enumerate(corner_order):
            groups[name] = np.where(assign == ci)[0].tolist()

    ordered_uv = []
    ordered_xy = []
    labels_out = []
    meta = []
    for name in corner_order:
        idxs = list(groups[name])
        if not idxs:
            print(f"警告: 角 {name} 没有分到点", file=sys.stderr)
            continue
        target = corner_uv[name]
        idxs.sort(key=lambda i: float(np.linalg.norm(uvs[i] - target)))
        if len(idxs) != per_corner:
            print(
                f"警告: 角 {name} 分到 {len(idxs)} 点（期望 {per_corner}）",
                file=sys.stderr,
            )
        for j, i in enumerate(idxs):
            ordered_uv.append(uvs[i])
            ordered_xy.append(xy[i])
            labels_out.append(len(ordered_uv) - 1)
            meta.append({"corner": name, "local": j})
    return (
        np.asarray(ordered_uv, dtype=np.float64),
        np.asarray(ordered_xy, dtype=np.float64),
        labels_out,
        meta,
    )


def enforce_lr_symmetry(uvs, meta, per_corner=2):
    """不推荐：仅当板确认左右对称时才用。"""
    uvs = np.asarray(uvs, dtype=np.float64).copy()
    groups = {"TL": [], "TR": [], "BL": [], "BR": []}
    for i, m in enumerate(meta):
        groups[m["corner"]].append(i)
    for left, right in (("TL", "TR"), ("BL", "BR")):
        li, ri = groups[left], groups[right]
        if len(li) != per_corner or len(ri) != per_corner:
            continue
        for k in range(per_corner):
            iL, iR = li[k], ri[k]
            mirror = np.array([1.0 - uvs[iL, 0], uvs[iL, 1]], dtype=np.float64)
            uvs[iR] = 0.3 * mirror + 0.7 * uvs[iR]
    return uvs


def draw_preview(bgr, quad, dots_xy, uvs, ids, meta=None):
    vis = bgr.copy()
    if quad is not None:
        cv2.polylines(vis, [np.int32(quad)], True, (0, 255, 0), 2, cv2.LINE_AA)
    for i, ((x, y), uv) in enumerate(zip(dots_xy, uvs)):
        hid = ids[i] if i < len(ids) else i
        tag = meta[i]["corner"] if meta and i < len(meta) else ""
        cv2.circle(vis, (int(x), int(y)), 8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            vis, f"{hid}:{tag}", (int(x) + 6, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            vis, f"({uv[0]:.2f},{uv[1]:.2f})", (int(x) + 6, int(y) + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA,
        )
    return vis


def layout_name_for(expect: int) -> str:
    if int(expect) == 8:
        return "corners_inner8_measured"
    if int(expect) == 12:
        return "corners_3x4_measured"
    return f"corners_measured_{int(expect)}"


def write_holes_snippet(holes: list, layout: str):
    block = yaml.safe_dump(
        {
            "holes_layout": layout,
            "holes_count": len(holes),
            "holes": holes,
        },
        allow_unicode=True, sort_keys=False, default_flow_style=False,
    )
    print("\n—— 粘贴到 grasp_config.yaml → detection.screw_board: 下 ——")
    print(block)


def calibrate_one_image(path: Path, expect: int, per_corner: int, max_gray: int):
    """单张图 → (holes_list, preview_bgr, meta)。"""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"无法读取: {path}")
    h, w = bgr.shape[:2]
    print(f"\n=== {path.name} ===")

    dots_raw = detect_red_dots(bgr, board_poly=None, expect=expect + 4)
    print(f"初检红点: {len(dots_raw)}")
    xy0 = np.array([(d[0], d[1]) for d in dots_raw], dtype=np.float64)
    if len(xy0) < 4:
        raise RuntimeError(f"{path.name}: 红点太少")
    xy0 = filter_dot_outliers(xy0, keep=expect)
    print(f"去离群后: {len(xy0)}")

    # 优先级：全图暗色板(贴真实外沿) > YOLO橙框(校验后) > ROI暗色 > 红点凸包
    # 勿优先旧 dots_roi（pad 小会切孔）；勿盲信全黄像素 overlay（易吞左侧黄字）。
    quad = detect_board_dark_containing_dots(
        bgr, xy0, max_gray=min(int(max_gray), 78), max_area_frac=0.5,
    )
    src = "dark_full"
    if quad is None:
        quad = detect_board_from_overlay(bgr, dots_xy=xy0)
        src = "overlay"
    if quad is None:
        quad = detect_board_obb(bgr, max_gray=max_gray, max_area_frac=0.45)
        src = "dark"
    if quad is None:
        quad = detect_board_around_dots(bgr, xy0, max_gray=max_gray)
        src = "dots_roi_dark"
    if quad is None:
        quad = detect_board_from_red_hull(xy0, pad_frac=0.28, img_wh=(w, h))
        src = "red_hull"
    if quad is None:
        raise RuntimeError(f"{path.name}: 未检出板框")
    print(f"板框来源: {src}")
    _uv_chk = pixel_to_uv(xy0, quad)
    _edge = float(np.min(np.minimum(_uv_chk, 1.0 - _uv_chk)))
    if _edge < 0.04:
        print(
            f"警告: 红点距板框过近(min_inset_uv={_edge:.3f})，框可能仍偏小",
            file=sys.stderr,
        )
    # 左右不应严重失衡（吞黄字时左孔 u 会到 0.4）
    _ul = float(np.min(_uv_chk[:, 0]))
    _ur = float(np.max(_uv_chk[:, 0]))
    if _ul > 0.28 or _ur < 0.72:
        print(
            f"警告: 红点 u 范围异常([{_ul:.3f},{_ur:.3f}])，板框可能左右偏了",
            file=sys.stderr,
        )

    dots = detect_red_dots(bgr, board_poly=quad, expect=expect)
    xy = np.array([(d[0], d[1]) for d in dots], dtype=np.float64)
    xy = filter_dot_outliers(xy, keep=expect)
    print(f"板内红点: {len(xy)}（期望 {expect}）")
    if len(xy) < max(4, expect - 2):
        raise RuntimeError(f"{path.name}: 板内红点不足 ({len(xy)})")

    uvs = pixel_to_uv(xy, quad)
    uvs_s, xy_s, labels, meta = sort_holes_corners_measured(
        uvs, xy, per_corner=per_corner,
    )
    holes = []
    for i, uv in enumerate(uvs_s):
        m = meta[i] if i < len(meta) else {"corner": "?", "local": 0}
        holes.append({
            "id": i,
            "u": float(np.clip(uv[0], -0.05, 1.05)),
            "v": float(np.clip(uv[1], -0.05, 1.05)),
            "corner": m["corner"],
            "local": int(m["local"]),
        })
        print(
            f"  hole[{i}] {m['corner']}.{m['local']}  "
            f"u={holes[-1]['u']:.4f} v={holes[-1]['v']:.4f}"
        )
    preview = draw_preview(bgr, quad, xy_s, uvs_s, labels, meta)
    return holes, preview, meta


def merge_holes_median(per_image_holes: list) -> list:
    """多图同一 id 的 (u,v) 取中位数；corner/local 以首张为准。"""
    if not per_image_holes:
        return []
    by_id = {}
    for holes in per_image_holes:
        for h in holes:
            by_id.setdefault(int(h["id"]), []).append(h)
    out = []
    for hid in sorted(by_id.keys()):
        items = by_id[hid]
        us = [float(x["u"]) for x in items]
        vs = [float(x["v"]) for x in items]
        base = items[0]
        out.append({
            "id": hid,
            "u": round(float(np.median(us)), 4),
            "v": round(float(np.median(vs)), 4),
            "corner": base["corner"],
            "local": int(base["local"]),
        })
    return out


def print_asymmetry_report(holes: list):
    """仅报告实测不对称量，不修正。"""
    print("\n实测左右差异（仅供参考，不强制对称；Δu=|uL+uR-1| Δv=|vL-vR|）:")
    by = {}
    for hitem in holes:
        by.setdefault(hitem["corner"], []).append(hitem)
    for L, R in (("TL", "TR"), ("BL", "BR")):
        if L not in by or R not in by:
            continue
        for a, b in zip(by[L], by[R]):
            du = abs(a["u"] + b["u"] - 1.0)
            dv = abs(a["v"] - b["v"])
            print(
                f"  {L}.{a['local']}↔{R}.{b['local']}: "
                f"Δu={du:.3f} Δv={dv:.3f}  "
                f"(L={a['u']:.3f},{a['v']:.3f} R={b['u']:.3f},{b['v']:.3f})"
            )


def main():
    ap = argparse.ArgumentParser(
        description="红点标定孔位：每孔独立实测（默认不做左右对称）"
    )
    ap.add_argument(
        "images", nargs="+", type=Path,
        help="带红点的板照片/截图（可多张，不同角度取中位数）",
    )
    ap.add_argument("--expect", type=int, default=8, help="期望孔数（默认8）")
    ap.add_argument("--per-corner", type=int, default=2, help="每角孔数（8孔=2）")
    ap.add_argument("--max-gray", type=int, default=90, help="黑板灰度上限")
    ap.add_argument(
        "--sym-smooth", action="store_true",
        help="【不推荐】左右镜像平滑；仅当板确认对称时使用",
    )
    ap.add_argument("--write-yaml", action="store_true", help="打印可粘贴 yaml 片段")
    ap.add_argument(
        "--apply-config", action="store_true",
        help="直接写回 grasp_config.yaml 的 detection.screw_board.holes",
    )
    ap.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parent / "grasp_config.yaml",
    )
    ap.add_argument("--preview", type=Path, default=None)
    args = ap.parse_args()

    per_image = []
    last_preview = None
    last_name = "holes"
    for img in args.images:
        if not img.is_file():
            print(f"找不到图片: {img}", file=sys.stderr)
            sys.exit(1)
        holes, preview, meta = calibrate_one_image(
            img, args.expect, int(args.per_corner), args.max_gray,
        )
        if args.sym_smooth:
            print(
                "警告: --sym-smooth 会拉右半向左镜像，不对称板请勿使用",
                file=sys.stderr,
            )
            uvs = np.array([[h["u"], h["v"]] for h in holes], dtype=np.float64)
            uvs = enforce_lr_symmetry(uvs, meta, per_corner=int(args.per_corner))
            for i, h in enumerate(holes):
                h["u"] = round(float(uvs[i, 0]), 4)
                h["v"] = round(float(uvs[i, 1]), 4)
        per_image.append(holes)
        last_preview = preview
        last_name = img.stem

    if len(per_image) == 1:
        holes = per_image[0]
        print("\n单图实测（无对称修正）")
    else:
        holes = merge_holes_median(per_image)
        print(f"\n{len(per_image)} 图中位数合并（每孔独立，无对称修正）:")
        for h in holes:
            print(
                f"  hole[{h['id']}] {h['corner']}.{h['local']}  "
                f"u={h['u']:.4f} v={h['v']:.4f}"
            )

    for h in holes:
        h["u"] = round(float(h["u"]), 4)
        h["v"] = round(float(h["v"]), 4)

    print_asymmetry_report(holes)
    layout = layout_name_for(args.expect)

    out_dir = Path(__file__).resolve().parent
    if last_preview is not None:
        out_prev = args.preview or (out_dir / f"{last_name}_holes_preview.png")
        cv2.imwrite(str(out_prev), last_preview)
        print(f"\n预览: {out_prev}")

    frag = out_dir / f"{last_name}_holes.yaml"
    with frag.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "holes_layout": layout,
                "holes_count": len(holes),
                "holes": holes,
            },
            f, allow_unicode=True, sort_keys=False,
        )
    print(f"片段: {frag}")

    if args.write_yaml:
        write_holes_snippet(holes, layout)

    if args.apply_config:
        cfg_path = args.config
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sb = (cfg.setdefault("detection", {})).setdefault("screw_board", {})
        sb["holes_layout"] = layout
        sb["holes_count"] = len(holes)
        sb["holes"] = holes
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"已写回: {cfg_path}")


if __name__ == "__main__":
    main()
