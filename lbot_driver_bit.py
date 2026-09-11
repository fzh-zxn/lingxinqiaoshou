#!/usr/bin/env python3
"""电动批头尖端像素估计：YOLO 框 + 深度机身 + RGB 细杆/金属，抗手遮挡。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DriverBitTip:
    u: float
    v: float
    depth_mm: Optional[float]
    axis_uv: Tuple[float, float]  # tip ← body 单位方向（图像）
    conf: float
    method: str
    chuck_uv: Optional[Tuple[float, float]] = None


class DriverBitTipTracker:
    """EMA + 跳变确认，抑制手晃/反光闪点。"""

    def __init__(
        self,
        ema=0.40,
        jump_px=28.0,
        jump_confirm=3,
        lost_max=12,
    ):
        self.ema = float(ema)
        self.jump_px = float(jump_px)
        self.jump_confirm = int(jump_confirm)
        self.lost_max = int(lost_max)
        self._uv = None
        self._depth = None
        self._axis = None
        self._conf = 0.0
        self._method = ""
        self._pending = None
        self._pending_n = 0
        self._lost = 0
        self._chuck = None

    def reset(self):
        self._uv = None
        self._depth = None
        self._axis = None
        self._conf = 0.0
        self._method = ""
        self._pending = None
        self._pending_n = 0
        self._lost = 0
        self._chuck = None

    def update(self, tip: Optional[DriverBitTip]) -> Optional[DriverBitTip]:
        if tip is None:
            self._lost += 1
            if self._lost >= self.lost_max:
                self.reset()
                return None
            if self._uv is None:
                return None
            return DriverBitTip(
                u=float(self._uv[0]),
                v=float(self._uv[1]),
                depth_mm=self._depth,
                axis_uv=self._axis or (0.0, 1.0),
                conf=max(0.15, self._conf * 0.92),
                method=f"{self._method}|hold",
                chuck_uv=self._chuck,
            )
        self._lost = 0
        nu = np.array([tip.u, tip.v], dtype=np.float64)
        if self._uv is None:
            self._uv = nu.copy()
            self._depth = tip.depth_mm
            self._axis = tip.axis_uv
            self._conf = float(tip.conf)
            self._method = tip.method
            self._chuck = tip.chuck_uv
            self._pending = None
            self._pending_n = 0
            return tip
        dist = float(np.linalg.norm(nu - self._uv))
        if dist > self.jump_px:
            if self._pending is None:
                self._pending = nu.copy()
                self._pending_n = 1
            else:
                if float(np.linalg.norm(nu - self._pending)) < self.jump_px * 0.7:
                    self._pending_n += 1
                    self._pending = 0.5 * self._pending + 0.5 * nu
                else:
                    self._pending = nu.copy()
                    self._pending_n = 1
            if self._pending_n < self.jump_confirm:
                # 拒绝野值，保留旧尖端
                return DriverBitTip(
                    u=float(self._uv[0]),
                    v=float(self._uv[1]),
                    depth_mm=self._depth,
                    axis_uv=self._axis or tip.axis_uv,
                    conf=self._conf,
                    method=f"{self._method}|rej_jump",
                    chuck_uv=self._chuck,
                )
            self._uv = self._pending.copy()
            self._pending = None
            self._pending_n = 0
        else:
            self._pending = None
            self._pending_n = 0
            a = float(np.clip(self.ema, 0.05, 0.95))
            self._uv = (1.0 - a) * self._uv + a * nu
        if tip.depth_mm is not None and np.isfinite(tip.depth_mm):
            if self._depth is None:
                self._depth = float(tip.depth_mm)
            else:
                self._depth = 0.65 * float(self._depth) + 0.35 * float(tip.depth_mm)
        self._axis = tip.axis_uv
        self._conf = 0.7 * self._conf + 0.3 * float(tip.conf)
        self._method = tip.method
        if tip.chuck_uv is not None:
            self._chuck = tip.chuck_uv
        return DriverBitTip(
            u=float(self._uv[0]),
            v=float(self._uv[1]),
            depth_mm=self._depth,
            axis_uv=self._axis or (0.0, 1.0),
            conf=float(self._conf),
            method=self._method,
            chuck_uv=self._chuck,
        )


def _clip_box(box, w, h, pad_frac=0.12):
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    px, py = bw * float(pad_frac), bh * float(pad_frac)
    x0 = int(max(0, np.floor(x0 - px)))
    y0 = int(max(0, np.floor(y0 - py)))
    x1 = int(min(w, np.ceil(x1 + px)))
    y1 = int(min(h, np.ceil(y1 + py)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def _hand_mask(bgr_roi):
    """白/肤色手：高亮低纹理，从批头搜索中剔除。"""
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    white = (v >= 185) & (s <= 55)
    # 肤色粗拒（偏黄粉）
    skin = (h <= 25) & (s >= 30) & (s <= 160) & (v >= 80) & (v <= 230)
    m = (white | skin).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.dilate(m, k, iterations=1)
    return m > 0


def _depth_body_mask(depth_roi, min_mm=80.0, max_mm=900.0, band_mm=55.0):
    """机身深度带：抑制桌面/远景。"""
    if depth_roi is None:
        return None, None
    d = np.asarray(depth_roi, dtype=np.float32)
    valid = np.isfinite(d) & (d >= min_mm) & (d <= max_mm) & (d > 1.0)
    if int(valid.sum()) < 30:
        return None, None
    # 中心权重中位数，抗边缘噪声
    hh, ww = d.shape[:2]
    yy, xx = np.ogrid[:hh, :ww]
    cx, cy = (ww - 1) * 0.5, (hh - 1) * 0.5
    wgt = np.exp(-(((xx - cx) / max(8.0, ww * 0.28)) ** 2
                   + ((yy - cy) / max(8.0, hh * 0.28)) ** 2))
    vals = d[valid]
    wg = wgt[valid]
    order = np.argsort(vals)
    vals_s, wg_s = vals[order], wg[order]
    csum = np.cumsum(wg_s)
    mid = csum[-1] * 0.5
    body_z = float(vals_s[int(np.searchsorted(csum, mid))])
    near = valid & (np.abs(d - body_z) <= float(band_mm))
    # 再拒「明显更远」桌面
    near &= d <= (body_z + band_mm * 0.85)
    return near, body_z


def _metal_shaft_mask(bgr_roi, hand_m):
    """低饱和金属杆/夹头 + 暗色细杆（批头常偏暗）。"""
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)
    metal = (s <= 90) & (v >= 70) & (v <= 230)
    # 暗杆：相对 ROI 更暗
    med = float(np.median(gray))
    dark = gray <= max(40.0, med * 0.72)
    # top-hat：细亮金属反光
    ksz = 15
    top = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)),
    )
    bright_thin = top >= max(12, int(np.percentile(top, 88)))
    m = (metal | dark | bright_thin) & (~hand_m)
    m_u8 = m.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, k, iterations=1)
    m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, k, iterations=1)
    return m_u8 > 0, gray, top


def _axis_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) < 40:
        return None
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = centered.T @ centered / max(1, len(pts) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    nrm = float(np.linalg.norm(axis))
    if nrm < 1e-6:
        return None
    axis = axis / nrm
    # 投影端点
    proj = centered @ axis
    i0, i1 = int(np.argmin(proj)), int(np.argmax(proj))
    p0 = pts[i0]
    p1 = pts[i1]
    length = float(np.linalg.norm(p1 - p0))
    if length < 12.0:
        return None
    return {
        "mean": mean,
        "axis": axis,
        "p0": p0,
        "p1": p1,
        "length": length,
        "pts": pts,
        "proj": proj,
    }


def _cross_section_width(mask, point, axis, half_len=8):
    """沿垂直于轴方向量局部宽度（越小越像批头）。"""
    perp = np.array([-axis[1], axis[0]], dtype=np.float64)
    hh, ww = mask.shape[:2]
    hits = 0
    for s in range(-half_len, half_len + 1):
        x = int(round(point[0] + perp[0] * s))
        y = int(round(point[1] + perp[1] * s))
        if 0 <= x < ww and 0 <= y < hh and mask[y, x]:
            hits += 1
    return float(hits)


def _score_tip_end(gray, top, mask, hand_m, end_pt, other_pt, axis):
    """尖端打分：更细、更金属、更远离手、更靠 YOLO 外沿。"""
    hh, ww = gray.shape[:2]
    x, y = int(round(end_pt[0])), int(round(end_pt[1]))
    if not (1 <= x < ww - 1 and 1 <= y < hh - 1):
        return -1e9
    # 局部窗
    r = 7
    x0, x1 = max(0, x - r), min(ww, x + r + 1)
    y0, y1 = max(0, y - r), min(hh, y + r + 1)
    loc_hand = float(hand_m[y0:y1, x0:x1].mean()) if (x1 > x0 and y1 > y0) else 1.0
    if loc_hand > 0.45:
        return -1e9
    w_end = _cross_section_width(mask, end_pt, axis)
    w_other = _cross_section_width(mask, other_pt, axis)
    metal = float(np.median(top[y0:y1, x0:x1])) if (x1 > x0 and y1 > y0) else 0.0
    gmed = float(np.median(gray[y0:y1, x0:x1])) if (x1 > x0 and y1 > y0) else 0.0
    # 倾向「细端」
    thin = (w_other + 1.0) / (w_end + 1.0)
    # 图像中更偏「远离机身中心」已由端点保证；额外惩罚手
    score = (
        2.2 * thin
        + 0.04 * metal
        + 0.01 * max(0.0, 160.0 - abs(gmed - 110.0))
        - 3.5 * loc_hand
    )
    return float(score)


def _walk_to_tip(mask, start, direction, max_step=80):
    """从端点沿轴再走，落到最后仍在 mask 上的点。"""
    hh, ww = mask.shape[:2]
    p = np.array(start, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return start
    d = d / n
    last = p.copy()
    for _ in range(int(max_step)):
        p = p + d
        x, y = int(round(p[0])), int(round(p[1]))
        if not (0 <= x < ww and 0 <= y < hh):
            break
        if mask[y, x]:
            last = p.copy()
        else:
            # 允许 1px 间隙
            ok = False
            for t in (1.0, 2.0):
                q = p + d * t
                x2, y2 = int(round(q[0])), int(round(q[1]))
                if 0 <= x2 < ww and 0 <= y2 < hh and mask[y2, x2]:
                    ok = True
                    break
            if not ok:
                break
    return last


def _depth_at_shaft(depth_roi, tip_xy, axis_from_tip, body_z, inset_px=10):
    """尖端深度不可靠：沿轴退回 inset 采中位。"""
    if depth_roi is None or body_z is None:
        return None
    hh, ww = depth_roi.shape[:2]
    dvec = -np.asarray(axis_from_tip, dtype=np.float64)
    n = float(np.linalg.norm(dvec))
    if n < 1e-6:
        dvec = np.array([0.0, -1.0])
    else:
        dvec = dvec / n
    samples = []
    for t in np.linspace(inset_px * 0.5, inset_px * 1.8, 8):
        p = tip_xy + dvec * t
        x, y = int(round(p[0])), int(round(p[1]))
        if not (0 <= x < ww and 0 <= y < hh):
            continue
        z = float(depth_roi[y, x])
        if np.isfinite(z) and 50.0 < z < 1200.0 and abs(z - body_z) < 80.0:
            samples.append(z)
    if not samples:
        return float(body_z)
    return float(np.median(samples))


def estimate_driver_bit_tip(
    bgr,
    depth,
    box_xyxy,
    *,
    pad_frac=0.14,
    depth_band_mm=55.0,
    min_conf=0.22,
) -> Optional[DriverBitTip]:
    """
    YOLO 框内：深度机身带 ∩ (金属/暗杆) − 手 → PCA 轴 → 细端为批头。
    深度取夹头侧，不取尖端像素。
    """
    if bgr is None or box_xyxy is None:
        return None
    h, w = bgr.shape[:2]
    clipped = _clip_box(box_xyxy, w, h, pad_frac=pad_frac)
    if clipped is None:
        return None
    x0, y0, x1, y1 = clipped
    roi = bgr[y0:y1, x0:x1]
    depth_roi = None
    if depth is not None:
        d = np.asarray(depth)
        if d.shape[:2] == (h, w):
            depth_roi = d[y0:y1, x0:x1]
        elif d.shape[:2] == roi.shape[:2]:
            depth_roi = d

    hand_m = _hand_mask(roi)
    metal_m, gray, top = _metal_shaft_mask(roi, hand_m)
    depth_m, body_z = _depth_body_mask(depth_roi, band_mm=depth_band_mm)

    if depth_m is not None:
        # 深度机身 ∪（深度邻域内的金属细杆）；手一律剔除
        dil = cv2.dilate(
            depth_m.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        ) > 0
        comb = ((depth_m | (metal_m & dil)) & (~hand_m))
    else:
        comb = metal_m & (~hand_m)

    # 去小碎块
    comb_u8 = comb.astype(np.uint8) * 255
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(comb_u8, connectivity=8)
    if nlab <= 1:
        return None
    # 选面积最大且不太贴手的连通域
    best_lab, best_sc = -1, -1.0
    for lab in range(1, nlab):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 60:
            continue
        m = labels == lab
        hand_frac = float(hand_m[m].mean()) if int(m.sum()) else 1.0
        sc = area * (1.0 - 0.85 * hand_frac)
        if sc > best_sc:
            best_sc = sc
            best_lab = lab
    if best_lab < 0:
        return None
    shaft = labels == best_lab

    ax = _axis_from_mask(shaft)
    if ax is None:
        # 回退：取 mask 最「细长方向」极值 — 用图像下方暗/金属点
        ys, xs = np.where(shaft & metal_m)
        if len(xs) < 8:
            ys, xs = np.where(shaft)
        if len(xs) < 8:
            return None
        # 倾向 v 更大（朝下批头常见）且 top-hat 高
        scores = top[ys, xs].astype(np.float64) + 0.15 * ys
        i = int(np.argmax(scores))
        tip_l = np.array([xs[i], ys[i]], dtype=np.float64)
        mean_yx = np.array([np.mean(xs), np.mean(ys)], dtype=np.float64)
        tip_xy = tip_l
        axis = tip_xy - mean_yx
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 1e-6 else np.array([0.0, 1.0])
        depth_mm = _depth_at_shaft(depth_roi, tip_xy, axis, body_z)
        tip = DriverBitTip(
            u=float(tip_xy[0] + x0),
            v=float(tip_xy[1] + y0),
            depth_mm=depth_mm,
            axis_uv=(float(axis[0]), float(axis[1])),
            conf=0.28,
            method="rgb_fallback",
            chuck_uv=(float(mean_yx[0] + x0), float(mean_yx[1] + y0)),
        )
        return tip if tip.conf >= min_conf else None

    p0, p1 = ax["p0"], ax["p1"]
    axis = ax["axis"]
    s0 = _score_tip_end(gray, top, shaft, hand_m, p0, p1, axis)
    s1 = _score_tip_end(gray, top, shaft, hand_m, p1, p0, axis)
    # 竖持电批常见：尖端更靠图像下方；给 v 更大端加分
    s0 += 0.55 * float(p0[1]) / max(1.0, gray.shape[0])
    s1 += 0.55 * float(p1[1]) / max(1.0, gray.shape[0])
    # 细杆：宽度比再加权一次
    w0 = _cross_section_width(shaft, p0, axis) + 1.0
    w1 = _cross_section_width(shaft, p1, axis) + 1.0
    s0 += 0.8 * (w1 / w0)
    s1 += 0.8 * (w0 / w1)
    if s1 >= s0:
        tip0, other, score = p1, p0, s1
        tip_dir = p1 - p0
    else:
        tip0, other, score = p0, p1, s0
        tip_dir = p0 - p1
    n = float(np.linalg.norm(tip_dir))
    tip_dir = tip_dir / n if n > 1e-6 else axis
    tip_l = _walk_to_tip(shaft, tip0, tip_dir, max_step=int(ax["length"] * 0.25) + 6)
    # 限制在 ROI 内，并略偏「细金属」质心拉回
    hh, ww = shaft.shape[:2]
    tip_l[0] = float(np.clip(tip_l[0], 1, ww - 2))
    tip_l[1] = float(np.clip(tip_l[1], 1, hh - 2))
    # 尖端邻域金属质心微调（抗轴端点落在机身角上）
    r_tip = 9
    tx0 = max(0, int(tip_l[0]) - r_tip)
    ty0 = max(0, int(tip_l[1]) - r_tip)
    tx1 = min(ww, int(tip_l[0]) + r_tip + 1)
    ty1 = min(hh, int(tip_l[1]) + r_tip + 1)
    local = metal_m[ty0:ty1, tx0:tx1] & (~hand_m[ty0:ty1, tx0:tx1])
    if int(local.sum()) >= 4:
        lys, lxs = np.where(local)
        # 沿 tip_dir 加权，偏向更远端
        lx = lxs.astype(np.float64) + tx0
        ly = lys.astype(np.float64) + ty0
        proj = (lx - tip_l[0]) * tip_dir[0] + (ly - tip_l[1]) * tip_dir[1]
        ww_l = np.clip(proj + 3.0, 0.2, None) * (1.0 + 0.05 * top[lys + ty0, lxs + tx0])
        tip_l = np.array([
            float(np.average(lx, weights=ww_l)),
            float(np.average(ly, weights=ww_l)),
        ])

    # 置信度
    conf = float(np.clip(
        0.25
        + 0.12 * np.clip(score / 4.0, 0.0, 1.5)
        + 0.15 * np.clip(ax["length"] / 80.0, 0.0, 1.0)
        + (0.12 if depth_m is not None else 0.0),
        0.0, 0.95,
    ))
    depth_mm = _depth_at_shaft(depth_roi, tip_l, tip_dir, body_z)
    tip = DriverBitTip(
        u=float(tip_l[0] + x0),
        v=float(tip_l[1] + y0),
        depth_mm=depth_mm,
        axis_uv=(float(tip_dir[0]), float(tip_dir[1])),
        conf=conf,
        method="yolo+depth+rgb" if depth_m is not None else "yolo+rgb",
        chuck_uv=(float(other[0] + x0), float(other[1] + y0)),
    )
    if tip.conf < min_conf:
        return None
    return tip


def draw_driver_bit_tip(image, tip: Optional[DriverBitTip], box=None):
    """青十字=批头；细线=刀轴；橙点=夹头侧。"""
    if image is None or tip is None:
        return
    u, v = int(round(tip.u)), int(round(tip.v))
    cv2.drawMarker(
        image, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA,
    )
    cv2.circle(image, (u, v), 5, (0, 255, 255), 1, cv2.LINE_AA)
    if tip.chuck_uv is not None:
        cu, cv_ = int(round(tip.chuck_uv[0])), int(round(tip.chuck_uv[1]))
        cv2.line(image, (cu, cv_), (u, v), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.circle(image, (cu, cv_), 4, (0, 140, 255), -1, cv2.LINE_AA)
    dtxt = f"{tip.depth_mm/10:.1f}cm" if tip.depth_mm else "?"
    cv2.putText(
        image,
        f"bit {tip.conf:.2f} {dtxt}",
        (u + 10, max(16, v - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
    )
