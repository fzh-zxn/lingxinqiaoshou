#!/usr/bin/env python3
"""蓝色筐 RGB+深度检测（旋转紧框 OBB）。供 demo_pick_square / demo_find_basket_rgbd 共用。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# OpenCV HSV：该蓝筐实测 H≈107–109、S≈150–190
DEFAULT_HSV_LOW = (95, 70, 40)
DEFAULT_HSV_HIGH = (130, 255, 255)
# 蓝三格筐典型外廓长宽比（长边/短边）；截断时用来补全被裁掉的那一侧
DEFAULT_ASPECT_PRIOR = 1.45
_BORDER_MARGIN = 6


@dataclass
class BasketHit:
    box_xyxy: Tuple[int, int, int, int]
    corners: np.ndarray  # (4,2) float，旋转框顶点
    size_wh: Tuple[float, float]
    center_uv: Tuple[float, float]
    angle_deg: float
    area_px: float
    score: float
    rim_mm: Optional[float]
    outside_mm: Optional[float]
    interior_mm: Optional[float]
    rim_delta_mm: Optional[float]  # outside - rim（正=口沿更近）
    reason: str
    table_mm: Optional[float] = None
    fit_mode: str = "obb"


@dataclass
class BasketSlot:
    """筐沿长边三等分之一格。label: big|medium|small（世界前→大、中、近→小）。"""
    label: str
    center_uv: Tuple[float, float]
    corners: np.ndarray  # (4,2) float
    index: int  # 0=前/大, 1=中, 2=近/小
    world_xyz: Optional[np.ndarray] = None
    depth_mm: Optional[float] = None


def _align_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.shape[1] != width or depth.shape[0] != height:
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return depth


def _median_depth_mm(depth: np.ndarray, mask: np.ndarray, min_mm=80.0, max_mm=1200.0):
    if depth is None or mask is None or not np.any(mask):
        return None, 0
    vals = depth[mask > 0].astype(np.float32)
    vals = vals[(vals >= min_mm) & (vals <= max_mm) & np.isfinite(vals)]
    if vals.size < 20:
        return None, int(vals.size)
    return float(np.median(vals)), int(vals.size)


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    if mask is None or not np.any(mask):
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return (mask > 0).astype(np.uint8) * 255
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == idx] = 255
    return out


def blue_mask_bgr(
    bgr: np.ndarray,
    hsv_low=DEFAULT_HSV_LOW,
    hsv_high=DEFAULT_HSV_HIGH,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.asarray(hsv_low, np.uint8), np.asarray(hsv_high, np.uint8)
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)
    return _largest_cc(mask)


def refine_mask_depth_spill(
    blue: np.ndarray,
    depth: Optional[np.ndarray],
    *,
    elevate_mm: float = 22.0,
    spill_tol_mm: float = 12.0,
    depth_min_mm: float = 80.0,
    depth_max_mm: float = 1200.0,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    去掉落在桌面深度上的蓝溢色；保留相对桌面抬升的蓝筐主体。
    返回 (refined_mask, table_mm)。
    """
    m = _largest_cc(blue)
    if depth is None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, k), None

    d = depth.astype(np.float32)
    valid = (d >= depth_min_mm) & (d <= depth_max_mm) & np.isfinite(d)
    nb = (m == 0) & valid
    if int(nb.sum()) > 800:
        table = float(np.median(d[nb]))
    elif valid.any():
        table = float(np.median(d[valid]))
    else:
        return m, None

    elevated = valid & (d <= table - float(elevate_mm))
    elev_blue = np.zeros_like(m)
    elev_blue[(m > 0) & elevated] = 255
    elev_blue = _largest_cc(elev_blue) if np.any(elev_blue) else elev_blue
    protect = cv2.dilate(
        elev_blue,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )

    spill = (m > 0) & valid & (d >= table - float(spill_tol_mm)) & (protect == 0)
    m2 = m.copy()
    m2[spill] = 0
    far = (m2 > 0) & (~valid) & (protect == 0)
    m2[far] = 0
    m2 = _largest_cc(m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, k, iterations=2)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, k3, iterations=1)
    return m2, table


def _order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def mask_touches_border(fill: np.ndarray, margin: int = _BORDER_MARGIN) -> bool:
    """掩膜是否贴到图像边界（部分出画）。"""
    if fill is None or not np.any(fill):
        return False
    m = max(1, int(margin))
    h, w = fill.shape[:2]
    return bool(
        np.any(fill[:m] > 0)
        or np.any(fill[-m:] > 0)
        or np.any(fill[:, :m] > 0)
        or np.any(fill[:, -m:] > 0)
    )


def _interior_points(
    pts: np.ndarray, width: int, height: int, margin: int = _BORDER_MARGIN
) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return pts
    m = float(margin)
    keep = (
        (pts[:, 0] >= m)
        & (pts[:, 0] < width - m)
        & (pts[:, 1] >= m)
        & (pts[:, 1] < height - m)
    )
    return pts[keep]


def estimate_rect_angle_hough(
    fill: np.ndarray, border_margin: int = 12
) -> Optional[float]:
    """
    用可见内缘直线估矩形朝向（忽略贴边，避免把图框当筐边）。
    返回 OpenCV minAreaRect 风格角度（度，约 [-90, 0)），失败 None。
    """
    if fill is None or not np.any(fill):
        return None
    h, w = fill.shape[:2]
    m = max(4, int(border_margin))
    edge = cv2.Canny(fill, 40, 120)
    edge[:m] = 0
    edge[-m:] = 0
    edge[:, :m] = 0
    edge[:, -m:] = 0
    band = max(8, int(0.035 * min(h, w)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 2 + 1, band * 2 + 1))
    outer = cv2.bitwise_and(fill, cv2.bitwise_not(cv2.erode(fill, k)))
    edge = cv2.bitwise_and(edge, outer)

    min_len = max(36, int(0.06 * min(h, w)))
    lines = cv2.HoughLinesP(
        edge, 1, np.pi / 180.0, threshold=28,
        minLineLength=min_len, maxLineGap=18,
    )
    if lines is None or len(lines) == 0:
        return None

    votes = np.zeros(90, dtype=np.float64)
    for ln in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in ln]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len * 0.8:
            continue
        ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang < 0:
            ang += 180.0
        bin_a = int(np.clip(np.round(ang % 90.0), 0, 89))
        votes[bin_a] += length

    if votes.max() < 1.0:
        return None
    kernel = np.array([0.15, 0.2, 0.3, 0.2, 0.15], dtype=np.float64)
    smooth = np.convolve(votes, kernel, mode="same")
    best = int(np.argmax(smooth))
    angle = float(best)
    if angle > 0:
        angle = angle - 90.0
    return angle


def tight_oriented_rect(
    points_xy: np.ndarray,
    *,
    angle_deg: Optional[float] = None,
    q_low: float = 3.0,
    q_high: float = 97.0,
    aspect_prior: Optional[float] = None,
    img_wh: Optional[Tuple[int, int]] = None,
    truncated: bool = False,
) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float], float], np.ndarray]:
    """
    抗毛刺旋转外接框。可固定朝向；截断时可按 aspect_prior 补全被裁边。
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 20:
        rect = cv2.minAreaRect(pts)
        return rect, cv2.boxPoints(rect)

    if angle_deg is None:
        (_cx, _cy), (_w0, _h0), angle = cv2.minAreaRect(pts)
        angle = float(angle)
    else:
        angle = float(angle_deg)

    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s], [-s, c]], dtype=np.float32)
    mean = pts.mean(axis=0)
    local = (pts - mean) @ R.T
    x0, x1 = np.percentile(local[:, 0], [q_low, q_high])
    y0, y1 = np.percentile(local[:, 1], [q_low, q_high])
    w = float(max(x1 - x0, 1.0))
    h = float(max(y1 - y0, 1.0))
    cx_l = 0.5 * (x0 + x1)
    cy_l = 0.5 * (y0 + y1)

    if truncated and img_wh is not None and aspect_prior is not None:
        width_i, height_i = int(img_wh[0]), int(img_wh[1])
        corners_l = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
        )
        corners_img = corners_l @ np.linalg.inv(R.T) + mean
        m = float(_BORDER_MARGIN + 2)
        near = (
            float(corners_img[:, 0].min()) <= m
            or float(corners_img[:, 0].max()) >= width_i - 1 - m
            or float(corners_img[:, 1].min()) <= m
            or float(corners_img[:, 1].max()) >= height_i - 1 - m
        )
        ap = float(aspect_prior)
        if near and ap > 1.05:
            short_v, long_v = (w, h) if w <= h else (h, w)
            vis_asp = long_v / max(short_v, 1e-6)
            # 可见长宽比过小：缺长边 → 补长边
            if vis_asp < ap * 0.92:
                target_long = short_v * ap
                if w >= h:
                    grow = 0.5 * (target_long - w)
                    cx_l += -grow if mean[0] < width_i * 0.5 else grow
                    w = target_long
                else:
                    grow = 0.5 * (target_long - h)
                    cy_l += -grow if mean[1] < height_i * 0.5 else grow
                    h = target_long
            # 可见长宽比过大：短边被压扁（贴边点丢掉后常见）→ 补短边
            elif vis_asp > ap * 1.15:
                target_short = long_v / ap
                if w <= h:
                    grow = 0.5 * (target_short - w)
                    cx_l += -grow if mean[0] < width_i * 0.5 else grow
                    w = target_short
                else:
                    grow = 0.5 * (target_short - h)
                    cy_l += -grow if mean[1] < height_i * 0.5 else grow
                    h = target_short

    center_local = np.array([cx_l, cy_l], dtype=np.float32)
    center = mean + (np.linalg.inv(R.T) @ center_local)
    rect = ((float(center[0]), float(center[1])), (w, h), float(angle))
    return rect, cv2.boxPoints(rect)


def fit_basket_obb(
    fill: np.ndarray,
    depth: Optional[np.ndarray] = None,
    table_mm: Optional[float] = None,
    *,
    elevate_mm: float = 22.0,
    depth_min_mm: float = 80.0,
    depth_max_mm: float = 1200.0,
    aspect_prior: float = DEFAULT_ASPECT_PRIOR,
) -> Tuple[dict, str]:
    """
    优先：抬升口沿 → 百分位收紧 OBB。
    截断出画：Hough 定朝向 + 丢弃贴边点 + 长宽比先验补全。
    """
    cnts, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("empty fill")
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    h, w = fill.shape[:2]
    truncated = mask_touches_border(fill)
    fit_mode = "obb"

    pts_fit = None
    if depth is not None and table_mm is not None:
        d = depth.astype(np.float32)
        valid = (d >= depth_min_mm) & (d <= depth_max_mm)
        elevated = valid & (d <= float(table_mm) - float(elevate_mm))
        band = max(10, int(0.04 * min(h, w)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 2 + 1, band * 2 + 1))
        outer = cv2.bitwise_and(fill, cv2.bitwise_not(cv2.erode(fill, k)))
        rim = outer.copy()
        rim[~elevated] = 0
        nz = cv2.findNonZero(rim)
        if nz is not None and len(nz) >= 80:
            pts_fit = nz.reshape(-1, 2).astype(np.float32)
            fit_mode = "rim%obb"
        else:
            elev_m = fill.copy()
            elev_m[~elevated] = 0
            nz2 = cv2.findNonZero(elev_m)
            if nz2 is not None and len(nz2) >= 200:
                pts_fit = nz2.reshape(-1, 2).astype(np.float32)
                fit_mode = "elev%obb"

    if pts_fit is None:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and not truncated:
            pts_fit = approx.reshape(-1, 2).astype(np.float32)
            fit_mode = "quad"
            rect = cv2.minAreaRect(pts_fit)
            corners = _order_corners_clockwise(cv2.boxPoints(rect))
            (cx, cy), (rw, rh), angle = rect
            return {
                "center": (float(cx), float(cy)),
                "size_wh": (float(rw), float(rh)),
                "angle": float(angle),
                "corners": corners.astype(np.float32),
                "area": area,
                "contour": cnt,
                "truncated": False,
            }, fit_mode
        pts_fit = cnt.reshape(-1, 2).astype(np.float32)
        fit_mode = "cnt%obb"

    angle_fix = None
    if truncated:
        # 朝向：Hough（忽略贴边）；尺寸仍用全部点，避免短边被掏空
        angle_fix = estimate_rect_angle_hough(fill, border_margin=12)
        if angle_fix is None:
            pts_in = _interior_points(pts_fit, w, h, margin=_BORDER_MARGIN + 4)
            if pts_in.shape[0] >= 60:
                (_c, _s), _wh, ang0 = cv2.minAreaRect(pts_in)
                angle_fix = float(ang0)
                fit_mode = fit_mode.replace("%obb", "%trunc") + "+inAng"
            else:
                fit_mode = fit_mode.replace("%obb", "%trunc")
        else:
            fit_mode = fit_mode.replace("%obb", "%trunc") + "+hough"

    rect, corners = tight_oriented_rect(
        pts_fit,
        angle_deg=angle_fix,
        q_low=2.0 if truncated else 4.0,
        q_high=98.0 if truncated else 96.0,
        aspect_prior=aspect_prior if truncated else None,
        img_wh=(w, h),
        truncated=truncated,
    )

    (cx, cy), (rw, rh), angle = rect
    corners = _order_corners_clockwise(corners)
    return {
        "center": (float(cx), float(cy)),
        "size_wh": (float(rw), float(rh)),
        "angle": float(angle),
        "corners": corners.astype(np.float32),
        "area": area,
        "contour": cnt,
        "truncated": truncated,
    }, fit_mode


def find_blue_basket_rgbd(
    bgr: np.ndarray,
    depth_mm: Optional[np.ndarray] = None,
    *,
    hsv_low=DEFAULT_HSV_LOW,
    hsv_high=DEFAULT_HSV_HIGH,
    min_area_frac: float = 0.02,
    max_area_frac: float = 0.55,
    aspect_min: float = 0.55,
    aspect_max: float = 2.4,
    rect_min: float = 0.55,
    rim_band_px: int = 14,
    outside_band_px: int = 18,
    interior_erode_px: int = 28,
    min_rim_delta_mm: float = 20.0,
    require_depth: bool = False,
    depth_min_mm: float = 80.0,
    depth_max_mm: float = 1200.0,
    elevate_mm: float = 22.0,
    aspect_prior: float = DEFAULT_ASPECT_PRIOR,
) -> Tuple[Optional[BasketHit], np.ndarray, list]:
    """
    返回 (最佳命中或 None, 精炼后的 mask, 候选列表)。
    hit.corners 为旋转框四顶点；绘制请用 OBB 而非 box_xyxy。
    """
    h, w = bgr.shape[:2]
    raw = blue_mask_bgr(bgr, hsv_low=hsv_low, hsv_high=hsv_high)
    depth = None
    if depth_mm is not None:
        depth = _align_depth(np.asarray(depth_mm), w, h)

    mask, table_mm = refine_mask_depth_spill(
        raw,
        depth,
        elevate_mm=elevate_mm,
        spill_tol_mm=12.0,
        depth_min_mm=depth_min_mm,
        depth_max_mm=depth_max_mm,
    )

    min_area = float(min_area_frac) * w * h
    max_area = float(max_area_frac) * w * h
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[BasketHit] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue

        fill = np.zeros((h, w), np.uint8)
        cv2.drawContours(fill, [cnt], -1, 255, thickness=-1)
        truncated = mask_touches_border(fill)

        try:
            geom, fit_mode = fit_basket_obb(
                fill,
                depth,
                table_mm,
                elevate_mm=elevate_mm,
                depth_min_mm=depth_min_mm,
                depth_max_mm=depth_max_mm,
                aspect_prior=aspect_prior,
            )
        except ValueError:
            continue

        cx, cy = geom["center"]
        rw, rh = geom["size_wh"]
        angle = geom["angle"]
        corners = geom["corners"]
        if rw < 8 or rh < 8:
            continue
        aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
        if aspect < aspect_min or aspect > aspect_max:
            continue
        rect_area = max(rw * rh, 1.0)
        rectangularity = float(np.clip(area / rect_area, 0.0, 1.5))
        # 截断时可见面积必然小于完整 OBB，放宽门槛
        min_rect = float(rect_min) * (0.45 if truncated else 1.0)
        if rectangularity < min_rect:
            continue

        x1 = int(np.clip(corners[:, 0].min(), 0, w - 1))
        y1 = int(np.clip(corners[:, 1].min(), 0, h - 1))
        x2 = int(np.clip(corners[:, 0].max(), 0, w - 1))
        y2 = int(np.clip(corners[:, 1].max(), 0, h - 1))

        rim_mm = outside_mm = interior_mm = rim_delta = None
        depth_ok = depth is None
        reason_bits = [
            f"rect={rectangularity:.2f}",
            f"asp={aspect:.2f}",
            fit_mode,
        ]
        if truncated:
            reason_bits.append("trunc")

        if depth is not None:
            k_rim = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (rim_band_px * 2 + 1, rim_band_px * 2 + 1)
            )
            eroded = cv2.erode(fill, k_rim)
            rim = cv2.bitwise_and(fill, cv2.bitwise_not(eroded))

            k_out = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (outside_band_px * 2 + 1, outside_band_px * 2 + 1),
            )
            dilated = cv2.dilate(fill, k_out)
            outside = cv2.bitwise_and(dilated, cv2.bitwise_not(fill))

            k_in = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (interior_erode_px * 2 + 1, interior_erode_px * 2 + 1),
            )
            interior = cv2.erode(fill, k_in)

            rim_mm, n_rim = _median_depth_mm(
                depth, rim, min_mm=depth_min_mm, max_mm=depth_max_mm
            )
            outside_mm, n_out = _median_depth_mm(
                depth, outside, min_mm=depth_min_mm, max_mm=depth_max_mm
            )
            interior_mm, n_in = _median_depth_mm(
                depth, interior, min_mm=depth_min_mm, max_mm=depth_max_mm
            )
            reason_bits.append(f"nR/O/I={n_rim}/{n_out}/{n_in}")

            if rim_mm is not None and outside_mm is not None:
                rim_delta = float(outside_mm - rim_mm)
                reason_bits.append(f"dRim={rim_delta:.0f}mm")
                depth_ok = rim_delta >= float(min_rim_delta_mm)
                if interior_mm is not None and interior_mm > rim_mm + 10.0:
                    reason_bits.append("deepIn")
            else:
                depth_ok = False
                reason_bits.append("depthSparse")

        if require_depth and not depth_ok:
            continue

        score = rectangularity * 40.0 + min(area / (w * h), 0.4) * 80.0
        score += max(0.0, rectangularity - 0.75) * 30.0
        if rim_delta is not None:
            score += float(np.clip(rim_delta, -20.0, 80.0)) * 0.8
            if not depth_ok:
                score -= 25.0

        hit = BasketHit(
            box_xyxy=(x1, y1, x2, y2),
            corners=corners,
            size_wh=(rw, rh),
            center_uv=(float(cx), float(cy)),
            angle_deg=float(angle),
            area_px=area,
            score=float(score),
            rim_mm=rim_mm,
            outside_mm=outside_mm,
            interior_mm=interior_mm,
            rim_delta_mm=rim_delta,
            reason=",".join(reason_bits),
            table_mm=table_mm,
            fit_mode=fit_mode,
        )
        candidates.append(hit)

    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0] if candidates else None
    if depth is not None and not require_depth:
        ok = [
            c
            for c in candidates
            if c.rim_delta_mm is not None and c.rim_delta_mm >= min_rim_delta_mm
        ]
        if ok:
            best = max(ok, key=lambda c: c.score)
    return best, mask, candidates


def _basket_long_edge_pair(corners: np.ndarray):
    """
    从顺时针四顶点取两条长边端点对 (A→B, D→C)，使参数 t 沿同一方向。
    返回 (A, B, D, C) 各为 (2,)；失败返回 None。
    """
    pts = _order_corners_clockwise(corners)
    if pts.shape[0] != 4:
        return None
    e01 = float(np.linalg.norm(pts[1] - pts[0]))
    e12 = float(np.linalg.norm(pts[2] - pts[1]))
    if e01 < 1.0 and e12 < 1.0:
        return None
    if e01 >= e12:
        # 长边 0→1 与 3→2
        return pts[0], pts[1], pts[3], pts[2]
    # 长边 1→2 与 0→3
    return pts[1], pts[2], pts[0], pts[3]


def split_basket_slots(
    corners: np.ndarray,
    *,
    world_xyz_list: Optional[List[Optional[np.ndarray]]] = None,
    depth_list: Optional[List[Optional[float]]] = None,
    forward: Optional[np.ndarray] = None,
) -> List[BasketSlot]:
    """
    沿 OBB 长边均匀分成 3 格，再按世界「前→近」标大/中/小。

    - 最前（相对世界系，默认 +X）→ big
    - 中间 → medium
    - 最近 → small

    world_xyz_list：三格中心的世界坐标（与未排序的几何格一一对应）；
    缺省时用 depth_list（深度大=更远≈前）或图像 v 小≈上≈前。
    """
    pair = _basket_long_edge_pair(corners)
    if pair is None:
        return []
    a, b, d, c = pair
    raw = []
    for i in range(3):
        t0 = i / 3.0
        t1 = (i + 1) / 3.0
        p0 = (1.0 - t0) * a + t0 * b
        p1 = (1.0 - t1) * a + t1 * b
        p2 = (1.0 - t1) * d + t1 * c
        p3 = (1.0 - t0) * d + t0 * c
        poly = np.stack([p0, p1, p2, p3], axis=0).astype(np.float32)
        cu = float(poly[:, 0].mean())
        cv = float(poly[:, 1].mean())
        wxyz = None
        if world_xyz_list is not None and i < len(world_xyz_list):
            w = world_xyz_list[i]
            if w is not None:
                wxyz = np.asarray(w, dtype=np.float64).reshape(3)
        dmm = None
        if depth_list is not None and i < len(depth_list) and depth_list[i] is not None:
            dmm = float(depth_list[i])
        raw.append(
            {
                "corners": poly,
                "center_uv": (cu, cv),
                "world_xyz": wxyz,
                "depth_mm": dmm,
                "geom_i": i,
            }
        )

    fwd = np.asarray(
        forward if forward is not None else [1.0, 0.0, 0.0],
        dtype=np.float64,
    ).reshape(3)
    fn = float(np.linalg.norm(fwd))
    fwd = fwd / fn if fn > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def _front_key(item):
        w = item["world_xyz"]
        if w is not None and np.all(np.isfinite(w)):
            return float(np.dot(w, fwd))
        if item["depth_mm"] is not None and np.isfinite(item["depth_mm"]):
            return float(item["depth_mm"])  # 更远≈更前
        # 图像上方常对应远处
        return -float(item["center_uv"][1])

    ordered = sorted(raw, key=_front_key, reverse=True)
    labels = ("big", "medium", "small")
    out: List[BasketSlot] = []
    for idx, lab in enumerate(labels):
        it = ordered[idx]
        out.append(
            BasketSlot(
                label=lab,
                center_uv=it["center_uv"],
                corners=it["corners"],
                index=idx,
                world_xyz=it["world_xyz"],
                depth_mm=it["depth_mm"],
            )
        )
    return out


def split_basket_slots_from_hit(
    hit: BasketHit,
    *,
    world_of_uv=None,
    forward: Optional[np.ndarray] = None,
) -> List[BasketSlot]:
    """
    从 BasketHit 切三格。world_of_uv(u,v)->(xyz|None) 可选，用于世界前/近排序。
    深度统一用 hit.rim_mm。
    """
    if hit is None or hit.corners is None:
        return []
    corners = np.asarray(hit.corners, dtype=np.float32).reshape(4, 2)
    # 先按几何切出未排序三格中心，再填世界/深度后排序
    pair = _basket_long_edge_pair(corners)
    if pair is None:
        return []
    a, b, d, c = pair
    centers = []
    for i in range(3):
        t = (i + 0.5) / 3.0
        p_ab = (1.0 - t) * a + t * b
        p_dc = (1.0 - t) * d + t * c
        centers.append(0.5 * (p_ab + p_dc))
    rim = hit.rim_mm
    world_list: List[Optional[np.ndarray]] = []
    depth_list: List[Optional[float]] = []
    for cxy in centers:
        u, v = float(cxy[0]), float(cxy[1])
        depth_list.append(float(rim) if rim is not None else None)
        w = None
        if world_of_uv is not None and rim is not None:
            try:
                w = world_of_uv(u, v, float(rim))
            except Exception:
                w = None
        world_list.append(w)
    return split_basket_slots(
        corners,
        world_xyz_list=world_list,
        depth_list=depth_list,
        forward=forward,
    )


def slot_for_size_class(
    slots: List[BasketSlot],
    size_class: Optional[str],
) -> Optional[BasketSlot]:
    """按螺母档位选格：big/large→前，medium→中，small→近。无尺寸时返回 None（勿默认中格）。"""
    if not slots:
        return None
    cls = str(size_class or "").strip().lower()
    if not cls:
        return None
    want = None
    if cls in ("big", "large"):
        want = "big"
    elif cls == "small":
        want = "small"
    elif cls == "medium":
        want = "medium"
    else:
        return None
    for s in slots:
        if s.label == want:
            return s
    by_idx = {0: "big", 1: "medium", 2: "small"}
    for s in slots:
        if by_idx.get(s.index) == want:
            return s
    return None


_SLOT_DRAW = {
    "big": ((40, 80, 255), "大"),
    "medium": ((40, 220, 255), "中"),
    "small": ((80, 255, 120), "小"),
}


def draw_basket_slots(
    bgr: np.ndarray,
    slots: List[BasketSlot],
    *,
    highlight_label: Optional[str] = None,
) -> np.ndarray:
    """在图上标注三格分区与大/中/小。"""
    if bgr is None or not slots:
        return bgr
    vis = bgr
    for s in slots:
        color, zh = _SLOT_DRAW.get(s.label, ((200, 200, 200), s.label))
        thick = 3 if highlight_label and s.label == highlight_label else 2
        poly = np.asarray(s.corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], True, color, thick, cv2.LINE_AA)
        cu, cv_ = int(round(s.center_uv[0])), int(round(s.center_uv[1]))
        cv2.drawMarker(vis, (cu, cv_), color, cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(
            vis, zh, (cu - 8, cv_ - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
    return vis


def draw_basket(
    bgr: np.ndarray,
    hit: Optional[BasketHit],
    mask: Optional[np.ndarray] = None,
    title: str = "",
) -> np.ndarray:
    vis = bgr.copy()
    if mask is not None:
        overlay = vis.copy()
        overlay[mask > 0] = (
            0.55 * overlay[mask > 0] + 0.45 * np.array([255, 120, 40])
        ).astype(np.uint8)
        vis = overlay
    if hit is not None:
        corners = np.asarray(hit.corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [corners], True, (0, 255, 80), 2, cv2.LINE_AA)
        x1, y1, x2, y2 = hit.box_xyxy
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 80, 80), 1)
        cu, cv_ = int(hit.center_uv[0]), int(hit.center_uv[1])
        cv2.drawMarker(vis, (cu, cv_), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        lines = [
            f"basket {hit.fit_mode} score={hit.score:.1f}",
            hit.reason,
        ]
        if hit.rim_mm is not None:
            lines.append(
                f"rim={hit.rim_mm:.0f} out={hit.outside_mm or 0:.0f} "
                f"d={hit.rim_delta_mm or 0:.0f}mm"
            )
        y = max(28, y1 - 8)
        for i, t in enumerate(lines):
            cv2.putText(
                vis, t, (x1, y + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv2.LINE_AA,
            )
    else:
        cv2.putText(
            vis, "no blue basket", (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 255), 2, cv2.LINE_AA,
        )
    if title:
        cv2.putText(
            vis, title, (12, vis.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )
    return vis
