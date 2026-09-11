#!/usr/bin/env python3
"""矩形螺丝板检测：深度主平面 + 深色/浅色金属，多候选打分。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from lbot_basket_rgbd import _align_depth, _largest_cc, _median_depth_mm, tight_oriented_rect


@dataclass
class ScrewBoardHit:
    box_xyxy: Tuple[int, int, int, int]
    corners: np.ndarray  # (4,2)
    size_wh: Tuple[float, float]
    center_uv: Tuple[float, float]
    angle_deg: float
    area_px: float
    score: float
    surface_mm: Optional[float]
    n_holes: int
    reason: str


def _morph_clean(mask, open_k=5, close_k=7):
    if mask is None:
        return None
    k_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_o, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_c, iterations=2)
    return out


def _light_metal_mask(bgr, hsv_low=(0, 0, 90), hsv_high=(180, 70, 255)):
    """浅灰/银金属：低饱和、中高亮度。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array(hsv_low, dtype=np.uint8), np.array(hsv_high, dtype=np.uint8),
    )
    return _morph_clean(mask)


def _dark_plate_mask(bgr, v_max=95, s_max=90):
    """深灰/黑钢板：HSV 低亮 + 灰度阈值（白纸上黑板）。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(
        hsv,
        np.array((0, 0, 0), dtype=np.uint8),
        np.array((180, int(s_max), int(v_max)), dtype=np.uint8),
    )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 相对画面偏暗：压过桌面浅色，保留黑板
    thr = float(np.clip(np.percentile(gray, 35.0), 40.0, 120.0))
    mask_gray = (gray.astype(np.float32) < thr).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask_hsv, mask_gray)
    return _morph_clean(mask, open_k=3, close_k=9)


def _border_touch_ratio(mask):
    """掩膜贴边比例；桌面/机架常贴边，板子通常在画面中部。"""
    if mask is None:
        return 1.0
    h, w = mask.shape[:2]
    m = mask > 0
    n = int(m.sum())
    if n < 1:
        return 1.0
    edge = np.zeros_like(m)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    return float((m & edge).sum()) / float(n)


def _depth_plane_mask(depth, min_depth_mm, max_depth_mm, band_mm=40.0, pct=20.0):
    """近景主平面（深度直方图靠前分位）。"""
    if depth is None:
        return None
    valid = (
        (depth >= float(min_depth_mm))
        & (depth <= float(max_depth_mm))
        & np.isfinite(depth)
    )
    vals = depth[valid]
    if vals.size < 200:
        return None
    near = float(np.percentile(vals, float(pct)))
    plane = valid & (np.abs(depth - near) <= float(band_mm))
    return (plane.astype(np.uint8) * 255)


def _count_holes_in_rect(depth, corners, surface_mm, hole_deeper_mm=6.0, min_hole_px=8):
    """框内相对表面更深的连通域数（螺孔近似）。"""
    if depth is None or surface_mm is None or not np.isfinite(surface_mm):
        return 0
    h, w = depth.shape[:2]
    poly = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi, [poly], 255)
    d = depth.astype(np.float32)
    hole = (
        (roi > 0)
        & np.isfinite(d)
        & (d > float(surface_mm) + float(hole_deeper_mm))
        & (d < float(surface_mm) + 80.0)
    )
    if int(hole.sum()) < min_hole_px:
        return 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), 8)
    count = 0
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_hole_px:
            count += 1
    return count


def _count_dark_holes_bgr(bgr, corners, min_hole_px=10):
    """彩色上框内更暗的小圆斑（螺孔在深色板上常更黑或透白）。"""
    if bgr is None:
        return 0
    h, w = bgr.shape[:2]
    poly = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi, [poly], 255)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    vals = gray[roi > 0]
    if vals.size < 80:
        return 0
    med = float(np.median(vals))
    # 相对板面明显更亮（透白纸）或更暗的斑
    spot = (roi > 0) & (
        (gray.astype(np.float32) > med + 35.0)
        | (gray.astype(np.float32) < med - 25.0)
    )
    if int(spot.sum()) < min_hole_px:
        return 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(spot.astype(np.uint8), 8)
    count = 0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if min_hole_px <= a <= 2500:
            count += 1
    return count


def _hit_from_mask(
    mask,
    bgr,
    depth,
    *,
    reason,
    min_area_frac,
    max_area_frac,
    aspect_min,
    aspect_max,
    min_rect_fill,
    min_depth_mm,
    max_depth_mm,
):
    if mask is None or int(np.count_nonzero(mask)) < 80:
        return None
    h, w = bgr.shape[:2]
    cc = _largest_cc(mask)
    if cc is None or int(np.count_nonzero(cc)) < 80:
        return None
    ys, xs = np.where(cc > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if pts.shape[0] < 80:
        return None

    (center, (rw, rh), angle), corners = tight_oriented_rect(
        pts, aspect_prior=None, img_wh=(w, h),
    )
    aw, ah = float(max(rw, rh)), float(min(rw, rh))
    if ah < 1.0:
        return None
    aspect = aw / ah
    area_rect = aw * ah
    area_px = float(pts.shape[0])
    fill = area_px / max(area_rect, 1.0)
    area_frac = area_px / float(h * w)

    if area_frac < float(min_area_frac) or area_frac > float(max_area_frac):
        return None
    if aspect < float(aspect_min) or aspect > float(aspect_max):
        return None
    if fill < float(min_rect_fill):
        return None

    x1 = int(np.clip(np.min(corners[:, 0]), 0, w - 1))
    y1 = int(np.clip(np.min(corners[:, 1]), 0, h - 1))
    x2 = int(np.clip(np.max(corners[:, 0]), 0, w - 1))
    y2 = int(np.clip(np.max(corners[:, 1]), 0, h - 1))
    surf, _n = _median_depth_mm(
        depth, cc, min_mm=min_depth_mm, max_mm=max_depth_mm,
    )
    n_holes_d = (
        _count_holes_in_rect(depth, corners, surf) if depth is not None else 0
    )
    n_holes_c = _count_dark_holes_bgr(bgr, corners)
    n_holes = max(int(n_holes_d), int(n_holes_c))

    score = float(fill) * min(aspect, 2.5) / 2.5
    # RGB 优先：深度平面与桌面易混，降权
    if reason in ("dark_only", "light_only"):
        score += 0.28
    elif reason.startswith("depth_"):
        score -= 0.35
    if "dark" in reason:
        score += 0.10
    if n_holes >= 2:
        score += 0.18 * min(n_holes, 8)
    # 贴边多 → 像机架/桌面而非板
    br = _border_touch_ratio(cc)
    if br > 0.02:
        score -= min(0.45, br * 8.0)
    score = float(np.clip(score, 0.0, 2.0))

    return ScrewBoardHit(
        box_xyxy=(x1, y1, x2, y2),
        corners=np.asarray(corners, dtype=np.float64).reshape(4, 2),
        size_wh=(aw, ah),
        center_uv=(float(center[0]), float(center[1])),
        angle_deg=float(angle),
        area_px=area_px,
        score=score,
        surface_mm=float(surf) if surf is not None else None,
        n_holes=int(n_holes),
        reason=reason,
    )


def find_screw_board_rgbd(
    bgr,
    depth_mm=None,
    *,
    min_area_frac=0.02,
    max_area_frac=0.65,
    aspect_min=1.15,
    aspect_max=3.8,
    min_rect_fill=0.55,
    min_depth_mm=80.0,
    max_depth_mm=900.0,
    hsv_low=(0, 0, 90),
    hsv_high=(180, 70, 255),
    dark_v_max=95,
    dark_s_max=90,
    depth_band_mm=40.0,
    prefer_depth_plane=False,
    prefer_rgb=True,
    max_border_touch=0.08,
):
    """
    检测矩形螺丝板。

    默认以 RGB 为主（深色板 / 浅灰金属）：板相对桌面抬升很浅时，
    深度平面容易和桌面糊成一片。深度仅用于孔洞/表面距离估计；
    prefer_depth_plane=True 时才把深度平面列入候选（并降权）。
    """
    if bgr is None or bgr.size < 16:
        return None
    h, w = bgr.shape[:2]
    depth = None
    if depth_mm is not None:
        depth = _align_depth(np.asarray(depth_mm), w, h)

    light = _light_metal_mask(bgr, hsv_low=hsv_low, hsv_high=hsv_high)
    dark = _dark_plate_mask(bgr, v_max=dark_v_max, s_max=dark_s_max)

    candidates: List[Tuple[str, np.ndarray]] = []
    # RGB 优先
    if dark is not None:
        candidates.append(("dark_only", dark))
    if light is not None:
        candidates.append(("light_only", light))

    if prefer_depth_plane and depth is not None:
        plane = _depth_plane_mask(
            depth, min_depth_mm, max_depth_mm, band_mm=depth_band_mm,
        )
        if plane is not None:
            if dark is not None:
                candidates.append(("depth_dark", cv2.bitwise_and(plane, dark)))
            if light is not None:
                candidates.append(("depth_light", cv2.bitwise_and(plane, light)))
            # 纯深度最易混桌面：默认不进候选；仅非 prefer_rgb 时启用
            if not prefer_rgb:
                candidates.append(("depth_only", plane))

    kw = dict(
        min_area_frac=min_area_frac,
        max_area_frac=max_area_frac,
        aspect_min=aspect_min,
        aspect_max=aspect_max,
        min_rect_fill=min_rect_fill,
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
    )
    best = None
    for reason, mask in candidates:
        if int(np.count_nonzero(mask)) < int(0.008 * h * w):
            continue
        # 贴边过多：机架/桌面边缘，直接跳过
        if _border_touch_ratio(mask) > float(max_border_touch):
            # 仍允许再试 largest_cc 后的贴边；此处用全 mask 粗滤
            cc = _largest_cc(mask)
            if cc is None or _border_touch_ratio(cc) > float(max_border_touch):
                continue
            mask = cc
        hit = _hit_from_mask(mask, bgr, depth, reason=reason, **kw)
        if hit is None:
            continue
        if best is None or hit.score > best.score:
            best = hit
    return best


def screw_board_hit_from_yolo(det, depth_mm=None, reason="yolo"):
    """YOLO xyxy 框 → ScrewBoardHit（轴对齐角点；深度中位数估表面）。"""
    if det is None or len(det) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in det[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    w = x2 - x1
    h = y2 - y1
    cu = 0.5 * (x1 + x2)
    cv_ = 0.5 * (y1 + y2)
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32,
    )
    surf = None
    if depth_mm is not None:
        d = np.asarray(depth_mm)
        ih, iw = d.shape[:2]
        xi1 = int(max(0, min(iw - 1, round(x1))))
        xi2 = int(max(0, min(iw, round(x2))))
        yi1 = int(max(0, min(ih - 1, round(y1))))
        yi2 = int(max(0, min(ih, round(y2))))
        if xi2 > xi1 and yi2 > yi1:
            patch = d[yi1:yi2, xi1:xi2].astype(np.float32)
            valid = patch[(patch > 50.0) & (patch < 2000.0)]
            if valid.size >= 8:
                surf = float(np.median(valid))
    conf = float(det[4]) if len(det) > 4 else 0.5
    return ScrewBoardHit(
        box_xyxy=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
        corners=corners,
        size_wh=(w, h),
        center_uv=(cu, cv_),
        angle_deg=0.0,
        area_px=float(w * h),
        score=conf,
        surface_mm=surf,
        n_holes=0,
        reason=str(reason or "yolo"),
    )


def _fill_internal_holes(mask):
    """填充与画面边界不连通的白色空洞（板内螺孔不打断主体）。"""
    if mask is None:
        return None
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def _edge_support(gray, points):
    """旋转矩形四边附近的 Canny 支撑；允许一条边被遮挡（取最稳两条边均值）。"""
    edges = cv2.Canny(gray, 35, 110)
    edges = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    scores = []
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    for index in range(4):
        side = np.zeros_like(gray)
        p1 = tuple(int(v) for v in pts[index])
        p2 = tuple(int(v) for v in pts[(index + 1) % 4])
        cv2.line(side, p1, p2, 255, 3, cv2.LINE_AA)
        side_pixels = cv2.countNonZero(side)
        supported = cv2.countNonZero(cv2.bitwise_and(edges, side))
        scores.append(supported / max(side_pixels, 1))
    return float(np.mean(sorted(scores, reverse=True)[:2]))


@dataclass
class ScrewBoardRefineState:
    """YOLO+RGB 精修的短时遮挡记忆。"""
    last_corners: Optional[np.ndarray] = None
    occlusion_frames: int = 0


def refine_screw_board_from_yolo_box(
    bgr,
    box_xyxy,
    depth_mm=None,
    *,
    pad_frac=0.15,
    max_gray=75,
    min_area_frac_roi=0.08,
    min_dark_coverage=0.24,
    min_edge_support=0.10,
    aspect_min=1.10,
    aspect_max=4.0,
    min_rect_fill=0.34,
    min_depth_mm=80.0,
    max_depth_mm=900.0,
    yolo_conf=0.5,
    occlusion_hold_frames=24,
    state: Optional[ScrewBoardRefineState] = None,
    reason="yolo+rgb",
) -> Optional[ScrewBoardHit]:
    """
    在 YOLO 轴对齐框（外扩）内用 RGB 精修旋转矩形。

    思路来自 offline `RGBBlackRectangleDetector`：
    gray≤max_gray → 形态学 → 填孔 → minAreaRect + 暗色覆盖/边支撑打分。
    失败返回 None（调用方保留纯 YOLO 框）。
    """
    if bgr is None or bgr.size < 16 or box_xyxy is None:
        return None
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * float(pad_frac)
    pad_y = bh * float(pad_frac)
    rx1 = int(max(0, math.floor(x1 - pad_x)))
    ry1 = int(max(0, math.floor(y1 - pad_y)))
    rx2 = int(min(w, math.ceil(x2 + pad_x)))
    ry2 = int(min(h, math.ceil(y2 + pad_y)))
    if rx2 - rx1 < 24 or ry2 - ry1 < 24:
        return None

    roi = bgr[ry1:ry2, rx1:rx2]
    rh, rw = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    max_g = int(np.clip(max_gray, 5, 180))
    raw_mask = cv2.inRange(gray, 0, max_g)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel, iterations=1)
    hole_filled = _fill_internal_holes(mask)
    bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge, iterations=1)

    depth = None
    if depth_mm is not None:
        depth = _align_depth(np.asarray(depth_mm), w, h)

    roi_area = float(rh * rw)
    min_area = max(100.0, float(min_area_frac_roi) * roi_area)
    best = None  # (score, corners_roi, conf, area_px, angle, size_wh)

    contours, _ = cv2.findContours(
        grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > roi_area * 0.98:
            continue
        rect = cv2.minAreaRect(contour)
        rw_r, rh_r = rect[1]
        rect_area = float(rw_r * rh_r)
        if rect_area <= 1.0:
            continue
        short_side, long_side = sorted((float(rw_r), float(rh_r)))
        if short_side < 10.0:
            continue
        aspect = long_side / max(short_side, 1.0)
        if aspect < float(aspect_min) or aspect > float(aspect_max):
            continue
        rectangularity = area / rect_area
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / max(hull_area, 1.0)
        if rectangularity < float(min_rect_fill) or solidity < 0.46:
            continue

        rect_points = np.int32(np.round(cv2.boxPoints(rect)))
        # 轮廓外接框贴 ROI 边过多时通常是半截机架，跳过
        bx, by, bbw, bbh = cv2.boundingRect(contour)
        if bx <= 0 or by <= 0 or bx + bbw >= rw - 1 or by + bbh >= rh - 1:
            # YOLO ROI 本身就裁切板，允许贴边；仅当几乎整边贴满才拒
            touch = (
                (bx <= 0) + (by <= 0) + (bx + bbw >= rw - 1) + (by + bbh >= rh - 1)
            )
            if touch >= 3:
                continue

        inside = np.zeros_like(gray)
        cv2.drawContours(inside, [contour], -1, 255, thickness=cv2.FILLED)
        mean_gray = float(cv2.mean(gray, mask=inside)[0])
        darkness = 1.0 - mean_gray / 255.0

        rect_mask = np.zeros_like(gray)
        cv2.fillConvexPoly(rect_mask, rect_points, 255)
        rect_pixels = int(cv2.countNonZero(rect_mask))
        dark_pixels = int(
            cv2.countNonZero(cv2.bitwise_and(hole_filled, rect_mask))
        )
        dark_coverage = dark_pixels / max(rect_pixels, 1)
        edge_support = _edge_support(gray, rect_points)
        if (
            dark_coverage < float(min_dark_coverage)
            or edge_support < float(min_edge_support)
        ):
            continue

        geometry_score = float(np.clip((rectangularity - 0.30) / 0.60, 0.0, 1.0))
        coverage_score = float(np.clip((dark_coverage - 0.20) / 0.65, 0.0, 1.0))
        edge_score = float(np.clip((edge_support - 0.08) / 0.55, 0.0, 1.0))
        confidence = float(np.clip(
            0.27 * geometry_score
            + 0.23 * coverage_score
            + 0.18 * solidity
            + 0.16 * darkness
            + 0.16 * edge_score,
            0.0,
            0.99,
        ))
        # YOLO conf 作弱先验：略抬精修分
        confidence = float(np.clip(
            0.85 * confidence + 0.15 * float(np.clip(yolo_conf, 0.0, 1.0)),
            0.0, 0.99,
        ))
        if best is None or confidence > best[0]:
            angle = float(rect[2])
            best = (
                confidence,
                rect_points.astype(np.float32),
                confidence,
                area,
                angle,
                (long_side, short_side),
            )

    used_prev = False
    hold_n = max(0, int(occlusion_hold_frames))
    if state is not None and state.last_corners is not None and hold_n > 0:
        old_full = np.asarray(state.last_corners, dtype=np.float32).reshape(4, 2)
        old_roi = old_full.copy()
        old_roi[:, 0] -= float(rx1)
        old_roi[:, 1] -= float(ry1)
        # 旧框是否仍大部分落在当前 ROI 内
        if (
            np.all(old_roi[:, 0] >= -8)
            and np.all(old_roi[:, 1] >= -8)
            and np.all(old_roi[:, 0] < rw + 8)
            and np.all(old_roi[:, 1] < rh + 8)
        ):
            old_pts = np.int32(np.round(np.clip(
                old_roi, [0, 0], [rw - 1, rh - 1],
            )))
            old_mask = np.zeros_like(gray)
            cv2.fillConvexPoly(old_mask, old_pts, 255)
            old_pixels = cv2.countNonZero(old_mask)
            visible = (
                cv2.countNonZero(cv2.bitwise_and(hole_filled, old_mask))
                / max(old_pixels, 1)
            )
            old_edges = _edge_support(gray, old_pts)
            if best is not None:
                new_pts = best[1]
                old_area = abs(float(cv2.contourArea(old_pts)))
                new_area = abs(float(cv2.contourArea(new_pts)))
                old_c = np.mean(old_pts.astype(np.float32), axis=0)
                new_c = np.mean(new_pts.astype(np.float32), axis=0)
                old_diag = float(
                    np.linalg.norm(np.ptp(old_pts.astype(np.float32), axis=0))
                )
                center_shift = float(np.linalg.norm(new_c - old_c))
                if (
                    old_area > 1.0
                    and new_area / old_area < 0.92
                    and center_shift < old_diag * 0.22
                    and visible >= 0.16
                    and old_edges >= 0.06
                    and state.occlusion_frames < hold_n
                ):
                    recovered = float(np.clip(
                        0.50 + 0.18 * visible + 0.12 * old_edges, 0.50, 0.78,
                    ))
                    aw = float(np.linalg.norm(old_pts[0] - old_pts[1]))
                    ah = float(np.linalg.norm(old_pts[1] - old_pts[2]))
                    long_s, short_s = sorted((aw, ah), reverse=True)
                    best = (
                        recovered, old_pts.astype(np.float32), recovered,
                        float(old_area), 0.0, (long_s, short_s),
                    )
                    used_prev = True
            elif (
                visible >= 0.16
                and old_edges >= 0.06
                and state.occlusion_frames < hold_n
            ):
                recovered = float(np.clip(
                    0.50 + 0.18 * visible + 0.12 * old_edges, 0.50, 0.78,
                ))
                aw = float(np.linalg.norm(old_pts[0] - old_pts[1]))
                ah = float(np.linalg.norm(old_pts[1] - old_pts[2]))
                long_s, short_s = sorted((aw, ah), reverse=True)
                best = (
                    recovered, old_pts.astype(np.float32), recovered,
                    abs(float(cv2.contourArea(old_pts))), 0.0, (long_s, short_s),
                )
                used_prev = True

    if best is None:
        if state is not None:
            state.last_corners = None
            state.occlusion_frames = 0
        return None

    conf, corners_roi, conf2, area_px, angle, size_wh = best
    corners = corners_roi.copy()
    corners[:, 0] += float(rx1)
    corners[:, 1] += float(ry1)
    cu = float(np.mean(corners[:, 0]))
    cv_ = float(np.mean(corners[:, 1]))
    bx1 = int(np.clip(np.min(corners[:, 0]), 0, w - 1))
    by1 = int(np.clip(np.min(corners[:, 1]), 0, h - 1))
    bx2 = int(np.clip(np.max(corners[:, 0]), 0, w - 1))
    by2 = int(np.clip(np.max(corners[:, 1]), 0, h - 1))

    surf = None
    n_holes = 0
    if depth is not None:
        poly = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        roi_m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_m, [poly], 255)
        surf, _n = _median_depth_mm(
            depth, roi_m, min_mm=min_depth_mm, max_mm=max_depth_mm,
        )
        n_holes_d = _count_holes_in_rect(depth, corners, surf) if surf else 0
        n_holes_c = _count_dark_holes_bgr(bgr, corners)
        n_holes = max(int(n_holes_d), int(n_holes_c))
    else:
        n_holes = int(_count_dark_holes_bgr(bgr, corners))

    if state is not None:
        state.last_corners = corners.copy()
        state.occlusion_frames = (
            state.occlusion_frames + 1 if used_prev else 0
        )

    tag = reason
    if used_prev:
        tag = f"{reason}+hold"
    return ScrewBoardHit(
        box_xyxy=(bx1, by1, bx2, by2),
        corners=np.asarray(corners, dtype=np.float64).reshape(4, 2),
        size_wh=(float(size_wh[0]), float(size_wh[1])),
        center_uv=(cu, cv_),
        angle_deg=float(angle),
        area_px=float(area_px),
        score=float(conf2),
        surface_mm=float(surf) if surf is not None else None,
        n_holes=int(n_holes),
        reason=str(tag),
    )


def draw_screw_board(image, hit: ScrewBoardHit, color=(0, 200, 255)):
    if image is None or hit is None:
        return
    pts = np.asarray(hit.corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)
    cu, cv_ = int(round(hit.center_uv[0])), int(round(hit.center_uv[1]))
    cv2.drawMarker(image, (cu, cv_), color, cv2.MARKER_TILTED_CROSS, 18, 2)
    dtxt = f"{hit.surface_mm/10:.1f}cm" if hit.surface_mm else "?"
    label = (
        f"board {hit.size_wh[0]:.0f}x{hit.size_wh[1]:.0f} "
        f"holes={hit.n_holes} {dtxt} [{hit.reason}]"
    )
    cv2.putText(
        image, label, (max(8, cu - 100), max(20, cv_ - 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
    )


@dataclass
class BoardHoleHit:
    id: int
    u: float
    v: float
    xy: Tuple[float, float]
    occupied: bool
    score: float
    corner: str = ""
    local: int = 0
    radius_px: float = 8.0
    # 有螺后精修中心（像素）；未精修时与 xy 相同
    xy_refined: Optional[Tuple[float, float]] = None
    refine_score: float = 0.0


def _order_quad_tl_tr_br_bl(pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(4)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl], axis=0)


def project_calib_holes(corners, holes_cfg) -> List[dict]:
    """标定 holes[{id,u,v,...}] + 当前板四角 → 像素点列表。"""
    if corners is None or not holes_cfg:
        return []
    quad = _order_quad_tl_tr_br_bl(corners)
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, quad)
    out = []
    for h in holes_cfg:
        try:
            hid = int(h.get("id", len(out)))
            u = float(h["u"])
            v = float(h["v"])
        except (KeyError, TypeError, ValueError):
            continue
        pt = cv2.perspectiveTransform(
            np.array([[[u, v]]], dtype=np.float32), H,
        ).reshape(2)
        out.append({
            "id": hid,
            "u": u,
            "v": v,
            "xy": (float(pt[0]), float(pt[1])),
            "corner": str(h.get("corner", "")),
            "local": int(h.get("local", 0) or 0),
        })
    return out


def _hole_window_features(bgr, cx, cy, radius_px):
    """孔窗内灰度/饱和/top-hat 特征，用于判有无银亮螺母。"""
    h, w = bgr.shape[:2]
    r = max(3, int(round(radius_px)))
    x0 = max(0, int(cx) - r * 2)
    y0 = max(0, int(cy) - r * 2)
    x1 = min(w, int(cx) + r * 2 + 1)
    y1 = min(h, int(cy) + r * 2 + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    roi = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yy_l, xx_l = np.ogrid[0:y1 - y0, 0:x1 - x0]
    cx_l, cy_l = float(cx) - x0, float(cy) - y0
    d2 = (xx_l - cx_l) ** 2 + (yy_l - cy_l) ** 2
    core_m = d2 <= (r * r)
    ring_m = (d2 > (r * r)) & (d2 <= ((2.2 * r) ** 2))
    if int(core_m.sum()) < 6:
        return None
    mean_core = float(gray[core_m].mean())
    mean_ring = float(gray[ring_m].mean()) if int(ring_m.sum()) >= 6 else mean_core
    sat = float(hsv[:, :, 1][core_m].mean())
    ksz = int(np.clip(int(round(r * 2.5)) | 1, 9, 31))
    if ksz % 2 == 0:
        ksz += 1
    top = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)),
    )
    contrast = float(top[core_m].mean())
    std_g = float(gray[core_m].std())
    return {
        "mean_core": mean_core,
        "mean_ring": mean_ring,
        "delta": mean_core - mean_ring,
        "sat": sat,
        "contrast": contrast,
        "std": std_g,
    }


def classify_hole_occupied(
    bgr,
    cx,
    cy,
    radius_px,
    *,
    min_delta=5.0,
    min_contrast=5.0,
    max_sat=140.0,
    min_mean=40.0,
    max_mean=235.0,
    score_min=0.30,
):
    """
    孔窗相对周围更亮 + 局部对比 → 有螺母/螺丝头。
    空孔偏暗或透白均匀；有螺为银亮凸起（阈值可略高，不单独一票否决）。
    """
    feat = _hole_window_features(bgr, cx, cy, radius_px)
    if feat is None:
        return False, 0.0
    if feat["mean_core"] < min_mean or feat["mean_core"] > max_mean:
        return False, 0.0
    # 高饱和仅在「不够亮」时否决（避免拒掉略偏暖的金属）
    if feat["sat"] > max_sat and feat["delta"] < float(min_delta):
        return False, 0.0
    textured = (
        feat["mean_core"] >= 65.0
        and feat["std"] >= 10.0
        and feat["delta"] >= max(2.0, float(min_delta) * 0.4)
    )
    bright = (
        feat["delta"] >= float(min_delta)
        or feat["contrast"] >= float(min_contrast)
        or textured
    )
    if not bright:
        return False, float(np.clip(feat["delta"] / 35.0, 0.0, 0.40))
    score = float(np.clip(
        0.40 * np.clip(feat["delta"] / 28.0, 0.0, 1.0)
        + 0.30 * np.clip(feat["contrast"] / 32.0, 0.0, 1.0)
        + 0.15 * np.clip(feat["std"] / 25.0, 0.0, 1.0)
        + 0.15 * np.clip((120.0 - min(feat["sat"], 120.0)) / 120.0, 0.0, 1.0),
        0.0, 0.99,
    ))
    return score >= float(score_min), score


def evaluate_board_holes(
    bgr,
    corners,
    holes_cfg,
    *,
    radius_frac=0.048,
    min_radius_px=5.0,
    max_radius_px=28.0,
    min_delta=5.0,
    min_contrast=5.0,
    max_sat=140.0,
    score_min=0.30,
) -> List[BoardHoleHit]:
    """投影标定孔并判有无螺母。"""
    if bgr is None or corners is None or not holes_cfg:
        return []
    projected = project_calib_holes(corners, holes_cfg)
    if not projected:
        return []
    quad = _order_quad_tl_tr_br_bl(corners)
    short = min(
        float(np.linalg.norm(quad[0] - quad[1])),
        float(np.linalg.norm(quad[1] - quad[2])),
    )
    radius = float(np.clip(short * float(radius_frac), min_radius_px, max_radius_px))
    hits = []
    for p in projected:
        cx, cy = p["xy"]
        occ, score = classify_hole_occupied(
            bgr, cx, cy, radius,
            min_delta=min_delta,
            min_contrast=min_contrast,
            max_sat=max_sat,
            score_min=score_min,
        )
        hits.append(BoardHoleHit(
            id=int(p["id"]),
            u=float(p["u"]),
            v=float(p["v"]),
            xy=(cx, cy),
            occupied=bool(occ),
            score=float(score),
            corner=str(p.get("corner") or ""),
            local=int(p.get("local") or 0),
            radius_px=radius,
        ))
    hits.sort(key=lambda h: h.id)
    return hits


def refine_screw_center(
    bgr,
    depth,
    cx,
    cy,
    radius_px,
    *,
    search_scale=1.85,
    max_shift_frac=0.85,
) -> Tuple[Tuple[float, float], float]:
    """
    有螺孔内精修银亮螺头中心：
    标定孔心作先验 + top-hat/低饱和 + 圆度(Hough) + 可选近端深度。
    """
    if bgr is None:
        return (float(cx), float(cy)), 0.0
    h, w = bgr.shape[:2]
    r = max(4, int(round(radius_px)))
    R = max(r + 2, int(round(r * float(search_scale))))
    x0 = max(0, int(cx) - R)
    y0 = max(0, int(cy) - R)
    x1 = min(w, int(cx) + R + 1)
    y1 = min(h, int(cy) + R + 1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return (float(cx), float(cy)), 0.0
    roi = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    ksz = int(np.clip((r * 2) | 1, 7, 21))
    if ksz % 2 == 0:
        ksz += 1
    top = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)),
    ).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    yy, xx = np.ogrid[0:y1 - y0, 0:x1 - x0]
    cx_l, cy_l = float(cx) - x0, float(cy) - y0
    dist = np.sqrt((xx - cx_l) ** 2 + (yy - cy_l) ** 2)
    disk = dist <= float(R)
    # 先验：越靠近标定孔心权重越高（抑制反光/邻孔）
    prior = np.exp(-(dist ** 2) / (2.0 * max(2.5, r * 0.55) ** 2)).astype(np.float32)

    metal = disk & (sat <= 95) & (val >= 60) & (val <= 235)
    thr = max(7.0, float(np.percentile(top[disk], 78)) if int(disk.sum()) else 7.0)
    bright = metal & (top >= thr)
    if int(bright.sum()) < 5:
        bright = metal & (blur.astype(np.float32) >= float(np.median(blur[disk])) + 8.0)
    if int(bright.sum()) < 4:
        return (float(cx), float(cy)), 0.0

    wgt = (
        0.40 * np.clip(top / max(1.0, thr * 1.4), 0.0, 2.0)
        + 0.20 * np.clip((110.0 - sat) / 110.0, 0.0, 1.0)
        + 0.15 * np.clip(val / 200.0, 0.0, 1.0)
        + 0.25 * prior
    )
    wgt = np.where(bright, wgt, 0.0)

    # Hough 圆：有则强拉向圆心
    hough_xy = None
    try:
        g8 = np.clip(blur, 0, 255).astype(np.uint8)
        circles = cv2.HoughCircles(
            g8, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(6.0, r * 0.8),
            param1=80, param2=12,
            minRadius=max(3, int(r * 0.35)),
            maxRadius=max(5, int(r * 1.15)),
        )
        if circles is not None and len(circles) > 0:
            best = None
            best_sc = -1e9
            for c in circles[0]:
                hx, hy, hr = float(c[0]), float(c[1]), float(c[2])
                d0 = math.hypot(hx - cx_l, hy - cy_l)
                if d0 > r * 1.1:
                    continue
                sc = -d0 + 0.15 * hr
                if sc > best_sc:
                    best_sc = sc
                    best = (hx, hy)
            if best is not None:
                hough_xy = best
    except Exception:
        hough_xy = None

    if depth is not None:
        dfull = np.asarray(depth)
        if dfull.shape[:2] == (h, w):
            d = dfull[y0:y1, x0:x1].astype(np.float32)
            valid = np.isfinite(d) & (d > 50) & (d < 1200)
            ring = disk & (dist >= r * 0.85) & (dist <= R) & valid
            core_v = bright & valid & (dist <= r * 1.05)
            if int(ring.sum()) >= 8 and int(core_v.sum()) >= 3:
                board_z = float(np.median(d[ring]))
                nearer = core_v & (d <= board_z - 1.0) & (d >= board_z - 22.0)
                if int(nearer.sum()) >= 3:
                    wgt = np.where(nearer, wgt * 1.40, wgt)

    sw = float(wgt.sum())
    if sw < 1e-3:
        return (float(cx), float(cy)), 0.0
    rx = float((wgt * xx).sum() / sw)
    ry = float((wgt * yy).sum() / sw)
    if hough_xy is not None:
        # 圆检测与亮度质心融合
        rx = 0.55 * hough_xy[0] + 0.45 * rx
        ry = 0.55 * hough_xy[1] + 0.45 * ry
    rx += x0
    ry += y0
    # 向标定孔心轻度回拉，防漂移
    rx = 0.72 * rx + 0.28 * float(cx)
    ry = 0.72 * ry + 0.28 * float(cy)
    max_shift = float(r) * float(max_shift_frac)
    if (rx - cx) ** 2 + (ry - cy) ** 2 > max_shift ** 2:
        # 超出则只信标定 + 小幅质心
        rx = 0.85 * float(cx) + 0.15 * rx
        ry = 0.85 * float(cy) + 0.15 * ry
        if (rx - cx) ** 2 + (ry - cy) ** 2 > max_shift ** 2:
            return (float(cx), float(cy)), 0.0
    score = float(np.clip(
        0.5 * (sw / max(12.0, bright.sum()))
        + (0.25 if hough_xy is not None else 0.0)
        + 0.25 * float(prior[int(np.clip(ry - y0, 0, prior.shape[0]-1)),
                             int(np.clip(rx - x0, 0, prior.shape[1]-1))]),
        0.0, 1.0,
    ))
    return (rx, ry), score


def refine_occupied_hole_centers(
    bgr,
    depth,
    holes: List[BoardHoleHit],
) -> List[BoardHoleHit]:
    """仅对 occupied 孔精修 xy_refined。"""
    if not holes:
        return holes
    out = []
    for h in holes:
        if not h.occupied:
            h.xy_refined = (float(h.xy[0]), float(h.xy[1]))
            h.refine_score = 0.0
            out.append(h)
            continue
        xy_r, sc = refine_screw_center(
            bgr, depth, h.xy[0], h.xy[1], h.radius_px,
        )
        h.xy_refined = (float(xy_r[0]), float(xy_r[1]))
        h.refine_score = float(sc)
        out.append(h)
    return out


def draw_board_holes(
    image,
    holes: List[BoardHoleHit],
    selected_id=None,
):
    """绿=有螺母，灰=空；选中加粗黄圈；有螺精修中心画青十字。"""
    if image is None or not holes:
        return
    for h in holes:
        x, y = int(round(h.xy[0])), int(round(h.xy[1]))
        r = max(4, int(round(h.radius_px)))
        if h.occupied:
            color = (0, 220, 0)
            thick = 2
        else:
            color = (120, 120, 120)
            thick = 1
        cv2.circle(image, (x, y), r, color, thick, cv2.LINE_AA)
        if selected_id is not None and int(selected_id) == int(h.id):
            cv2.circle(image, (x, y), r + 4, (0, 220, 255), 2, cv2.LINE_AA)
        if h.occupied and h.xy_refined is not None:
            rx, ry = int(round(h.xy_refined[0])), int(round(h.xy_refined[1]))
            mark = (0, 255, 255) if (
                selected_id is not None and int(selected_id) == int(h.id)
            ) else (255, 200, 0)
            cv2.drawMarker(
                image, (rx, ry), mark, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA,
            )
            if abs(rx - x) + abs(ry - y) >= 2:
                cv2.line(image, (x, y), (rx, ry), mark, 1, cv2.LINE_AA)
        cv2.putText(
            image, str(h.id), (x + r + 2, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
