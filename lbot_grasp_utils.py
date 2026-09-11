#!/usr/bin/env python3
"""视觉抓取共用工具：检测、深度、坐标变换。"""

import math
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
# 抓取：single.pt（螺母，类名 nut）→ best.pt …
# 放置：pan_best.pt（盘）→ 盘best.pt → 框best.pt → plate.pt → nut.pt
_DEFAULT_CANDIDATES = (
    PROJECT_ROOT / "single.pt",
    PROJECT_ROOT.parent / "single.pt",
    PROJECT_ROOT / "best.pt",
    PROJECT_ROOT.parent / "best.pt",
    PROJECT_ROOT / "three_color.pt",
)
DEFAULT_MODEL = next((p for p in _DEFAULT_CANDIDATES if p.is_file()), _DEFAULT_CANDIDATES[0])
_DEFAULT_PLACE_CANDIDATES = (
    PROJECT_ROOT / "pan_best.pt",
    PROJECT_ROOT.parent / "pan_best.pt",
    PROJECT_ROOT / "盘best.pt",
    PROJECT_ROOT.parent / "盘best.pt",
    PROJECT_ROOT / "框best.pt",
    PROJECT_ROOT.parent / "框best.pt",
    PROJECT_ROOT / "plate.pt",
    PROJECT_ROOT.parent / "plate.pt",
    PROJECT_ROOT / "nut.pt",
    PROJECT_ROOT.parent / "nut.pt",
)
DEFAULT_PLACE_MODEL = next(
    (p for p in _DEFAULT_PLACE_CANDIDATES if p.is_file()), _DEFAULT_PLACE_CANDIDATES[0]
)
# 额外筐子检验（与盘并行画框；类名 kuang）
_DEFAULT_BASKET_CANDIDATES = (
    PROJECT_ROOT / "9.4.筐子.pt",
    PROJECT_ROOT.parent / "9.4.筐子.pt",
    PROJECT_ROOT / "框best.pt",
    PROJECT_ROOT.parent / "框best.pt",
)
DEFAULT_BASKET_MODEL = next(
    (p for p in _DEFAULT_BASKET_CANDIDATES if p.is_file()), _DEFAULT_BASKET_CANDIDATES[0]
)
DEFAULT_CONFIG = Path(__file__).with_name("grasp_config.yaml")

# single.pt: nut；best.pt: big/medium/small；three_color: red/green/blue
TRACK_COLOR_ALIASES = {
    "nut": {"nut", "螺母", "nuts", "hexnut", "hex_nut"},
    "big": {"big", "large", "大", "大螺母"},
    "medium": {"medium", "mid", "中", "中螺母"},
    "small": {"small", "小", "小螺母"},
    "red": {"red", "红"},
    "green": {"green", "绿"},
    "blue": {"blue", "蓝"},
}
# 放置目标：盘 / 框 / 9.4.筐子.pt(kuang)
PLACE_CLASS_ALIASES = {
    "plate", "basket", "bin", "bowl", "tray", "box",
    "kuang", "frame",
    "框", "螺母框", "料框",
    "筐", "筐子", "盘", "盆", "盒", "篮子",
}
# 仅 three_color 权重需要；螺母尺寸模型不要开
MODEL_SWAP_RED_BLUE = False
CLASS_BGR = {
    "nut": (200, 200, 40),
    "框": (200, 200, 40),
    "big": (40, 40, 255),
    "medium": (60, 200, 60),
    "small": (255, 120, 40),
    "大": (40, 40, 255),
    "中": (60, 200, 60),
    "小": (255, 120, 40),
    "red": (40, 40, 255),
    "green": (60, 200, 60),
    "blue": (255, 120, 40),
    "红": (40, 40, 255),
    "绿": (60, 200, 60),
    "蓝": (255, 120, 40),
    "plate": (40, 200, 255),
    "basket": (255, 80, 255),
    "bin": (40, 200, 255),
    "kuang": (255, 80, 255),
    "筐": (255, 80, 255),
    "筐子": (255, 80, 255),
    "盘": (40, 200, 255),
}


def camera_source(value):
    return int(value) if str(value).isdigit() else value


def load_grasp_config(path: Path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("需要 PyYAML: pip install pyyaml") from exc
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# 示教器「软限位设置」最小/最大限位（度）。J1/J2 非对称。
DEFAULT_SOFT_JOINT_LIMITS_DEG = {
    "lower": [-166.73, -183.35, -154.13, -117.46, -154.13, -91.10, -91.10],
    "upper": [114.59, 4.01, 154.13, 117.46, 154.13, 91.10, 91.10],
}


def resolve_soft_joint_limits_rad(config):
    """
    从 config['robot']['soft_joint_limits_deg'] 读软限位，返回 (lo_rad, hi_rad)。
    未配置则用 DEFAULT_SOFT_JOINT_LIMITS_DEG。
    """
    robot = {}
    if isinstance(config, dict):
        robot = config.get("robot") or {}
        if "soft_joint_limits_deg" not in robot and "soft_joint_limits_deg" in config:
            robot = config
    raw = None
    if isinstance(robot, dict):
        raw = robot.get("soft_joint_limits_deg")
    if not isinstance(raw, dict):
        raw = DEFAULT_SOFT_JOINT_LIMITS_DEG
    lo_deg = list(raw.get("lower", DEFAULT_SOFT_JOINT_LIMITS_DEG["lower"]))
    hi_deg = list(raw.get("upper", DEFAULT_SOFT_JOINT_LIMITS_DEG["upper"]))
    while len(lo_deg) < 7:
        lo_deg.append(DEFAULT_SOFT_JOINT_LIMITS_DEG["lower"][len(lo_deg)])
    while len(hi_deg) < 7:
        hi_deg.append(DEFAULT_SOFT_JOINT_LIMITS_DEG["upper"][len(hi_deg)])
    lo = np.array([math.radians(float(v)) for v in lo_deg[:7]], dtype=np.float64)
    hi = np.array([math.radians(float(v)) for v in hi_deg[:7]], dtype=np.float64)
    return lo, hi


def joints_soft_limit_violation(joints, lo_rad, hi_rad, margin_rad=0.0, eps_rad=0.0):
    """
    若有关节越软限位（超出 eps），返回 (index, value_rad, lo, hi)；否则 None。
    margin_rad>0 收紧可用区间；eps_rad 为边界容差（贴边不算越界）。
    """
    if joints is None or lo_rad is None or hi_rad is None:
        return None
    js = [float(j) for j in list(joints)[:7]]
    m = float(margin_rad)
    eps = float(eps_rad)
    for i, q in enumerate(js):
        lo = float(lo_rad[i]) + m
        hi = float(hi_rad[i]) - m
        if q < lo - eps or q > hi + eps:
            return i, q, lo, hi
    return None


def clamp_joints_to_soft_limits(joints, lo_rad, hi_rad, margin_rad=0.0):
    """把关节钳到软限位内，返回 list。"""
    js = [float(j) for j in list(joints)[:7]]
    while len(js) < 7:
        js.append(0.0)
    m = float(margin_rad)
    out = []
    for i in range(7):
        lo = float(lo_rad[i]) + m
        hi = float(hi_rad[i]) - m
        if hi < lo:
            lo, hi = float(lo_rad[i]), float(hi_rad[i])
        out.append(float(np.clip(js[i], lo, hi)))
    return out


def soft_limit_inward_seed(joints, lo_rad, hi_rad, pull_rad=0.08):
    """
    越限位的轴往区间内侧拉 pull_rad，作 IK 重解种子（7 轴冗余下常能换支）。
    """
    js = [float(j) for j in list(joints)[:7]]
    while len(js) < 7:
        js.append(0.0)
    pull = abs(float(pull_rad))
    out = []
    for i in range(7):
        lo = float(lo_rad[i])
        hi = float(hi_rad[i])
        q = js[i]
        if q > hi - 1e-9:
            q = hi - pull
        elif q < lo + 1e-9:
            q = lo + pull
        out.append(float(np.clip(q, lo, hi)))
    return out


def euler_rpy_to_matrix(roll, pitch, yaw):
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _wrap_angle_diff(delta: float) -> float:
    """将角度差归一化到 (-π, π]。"""
    return (float(delta) + math.pi) % (2.0 * math.pi) - math.pi


def _rpy_equivalent_candidates(roll, pitch, yaw):
    """与 euler_rpy_to_matrix (Rz@Ry@Rx) 等价的常见 RPY 分支。"""
    r_ref = euler_rpy_to_matrix(roll, pitch, yaw)
    seen = set()
    out = []

    def add(r, p, y):
        key = (round(r, 6), round(p, 6), round(y, 6))
        if key in seen:
            return
        R = euler_rpy_to_matrix(r, p, y)
        if not np.allclose(R, r_ref, atol=1e-4, rtol=1e-4):
            return
        seen.add(key)
        out.append((float(r), float(p), float(y)))

    add(roll, pitch, yaw)
    for sr in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            add(roll + sr * math.pi, math.pi - pitch, yaw + sy * math.pi)
    return out


def unwrap_rpy_for_display(prev_rpy, new_rpy):
    """
    显示用 RPY 解缠绕：在等价欧拉角中选与上一帧最接近的一支，避免 180° 突跳。
    prev_rpy/new_rpy: (roll, pitch, yaw) 弧度；prev 为 None 时直接返回 new。
    """
    new = np.asarray(new_rpy, dtype=np.float64).reshape(3)
    if prev_rpy is None:
        return new.tolist()
    prev = np.asarray(prev_rpy, dtype=np.float64).reshape(3)
    best = prev.copy()
    best_cost = float("inf")
    for r, p, y in _rpy_equivalent_candidates(new[0], new[1], new[2]):
        cand = prev + np.array(
            [_wrap_angle_diff(r - prev[0]),
             _wrap_angle_diff(p - prev[1]),
             _wrap_angle_diff(y - prev[2])],
            dtype=np.float64,
        )
        cost = float(np.sum((cand - prev) ** 2))
        if cost < best_cost:
            best_cost = cost
            best = cand
    return best.tolist()


# O6 手 L6 通道 → MuJoCF 主关节；dip/ip 由 mimic 从动（与 workstation.mjcf 一致）
O6_L6_MASTER_JOINTS = (
    (0, "hand_right_rh_thumb_cmc_pitch"),
    (1, "hand_right_rh_thumb_cmc_yaw"),
    (2, "hand_right_rh_index_mcp_pitch"),
    (3, "hand_right_rh_middle_mcp_pitch"),
    (4, "hand_right_rh_ring_mcp_pitch"),
    (5, "hand_right_rh_pinky_mcp_pitch"),
)
O6_L6_MIMIC_JOINTS = (
    ("hand_right_rh_thumb_ip", "hand_right_rh_thumb_cmc_pitch", 1.86),
    ("hand_right_rh_index_dip", "hand_right_rh_index_mcp_pitch", 0.89),
    ("hand_right_rh_middle_dip", "hand_right_rh_middle_mcp_pitch", 0.89),
    ("hand_right_rh_ring_dip", "hand_right_rh_ring_mcp_pitch", 0.89),
    ("hand_right_rh_pinky_dip", "hand_right_rh_pinky_mcp_pitch", 0.89),
)


def l6_value_to_joint_angle(lo: float, hi: float, l6_val: float) -> float:
    """L6: 255=张开→lo，0=握紧→hi（同 LinkerHand mapping.py o6_*_derict=-1）。"""
    t = 1.0 - float(l6_val) / 255.0
    return float(lo + t * (hi - lo))


def l6_value_to_joint_angle_mj(lo: float, hi: float, l6_val: float, mirror_qpos: bool = False) -> float:
    """
    L6→MuJoCo qpos。
    实机 L6 语义不变；mirror 在关节量程内做 ang→hi+lo-ang，修正 MJCF 网格正方向与电机相反。
    """
    ang = l6_value_to_joint_angle(lo, hi, l6_val)
    if mirror_qpos:
        ang = hi + lo - ang
    return float(np.clip(ang, lo, hi))


def palm_relative_direction(vec_world, palm_pos, palm_rot):
    """世界系向量 → 掌心系 (azimuth°, elevation°, length m)。azimuth=绕掌法向, elevation=仰角。"""
    v = np.asarray(vec_world, dtype=np.float64).reshape(3)
    local = np.asarray(palm_rot, dtype=np.float64).reshape(3, 3).T @ v
    length = float(np.linalg.norm(local))
    if length < 1e-9:
        return 0.0, 0.0, 0.0
    az = math.degrees(math.atan2(local[1], local[0]))
    el = math.degrees(math.atan2(local[2], math.hypot(local[0], local[1])))
    return az, el, length


def _grasp_frame_from_x(origin, x_axis, approach_hint=None, palm_rot=None):
    """由原点、X(闭合) 与接近 hint 构造 task frame 旋转部分。"""
    x_axis = np.asarray(x_axis, dtype=np.float64).reshape(3)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-9:
        return None
    x_axis = x_axis / x_norm

    if approach_hint is None:
        if palm_rot is not None:
            approach_hint = np.asarray(palm_rot, dtype=np.float64).reshape(3, 3)[:, 2]
        else:
            approach_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    hint = np.asarray(approach_hint, dtype=np.float64).reshape(3)
    hint_raw = hint.copy()
    z_axis = hint - np.dot(hint, x_axis) * x_axis
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-9:
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(fallback, x_axis))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        z_axis = fallback - np.dot(fallback, x_axis) * x_axis
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm < 1e-9:
            return None
    z_axis /= z_norm
    if float(np.dot(z_axis, hint_raw)) < 0.0:
        z_axis = -z_axis

    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-9:
        return None
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-9)

    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return {
        "origin_m": origin.tolist(),
        "R_world": R.tolist(),
        "x_close": x_axis.tolist(),
        "y_axis": y_axis.tolist(),
        "z_approach": z_axis.tolist(),
        "pinch_sep_m": x_norm,
    }


def triangle_incenter(p0, p1, p2):
    """
    三角形内切圆心（边长加权质心）与内切半径。
    返回 (center_xyz, radius_m)；退化时回退质心、r=0。
    """
    a = np.asarray(p0, dtype=np.float64).reshape(3)
    b = np.asarray(p1, dtype=np.float64).reshape(3)
    c = np.asarray(p2, dtype=np.float64).reshape(3)
    la = float(np.linalg.norm(b - c))
    lb = float(np.linalg.norm(a - c))
    lc = float(np.linalg.norm(a - b))
    peri = la + lb + lc
    if peri < 1e-9:
        mid = (a + b + c) / 3.0
        return mid, 0.0
    center = (la * a + lb * b + lc * c) / peri
    # 面积 (Heron) → r = A / s
    s = 0.5 * peri
    area2 = max(s * (s - la) * (s - lb) * (s - lc), 0.0)
    area = math.sqrt(area2)
    radius = float(area / s) if s > 1e-12 else 0.0
    return center, radius


def triangle_circumcenter(p0, p1, p2):
    """
    三角形外接圆圆心与半径（三 tip 共圆中心 ≈ 螺母中心）。
    近共线时返回 (None, 0)。
    """
    a = np.asarray(p0, dtype=np.float64).reshape(3)
    b = np.asarray(p1, dtype=np.float64).reshape(3)
    c = np.asarray(p2, dtype=np.float64).reshape(3)
    ab = b - a
    ac = c - a
    n = np.cross(ab, ac)
    n2 = float(np.dot(n, n))
    if n2 < 1e-16:
        return None, 0.0
    ab2 = float(np.dot(ab, ab))
    ac2 = float(np.dot(ac, ac))
    # O = A + (|AC|^2 (N×AB) + |AB|^2 (AC×N)) / (2 |N|^2), N=AB×AC
    center = a + (ac2 * np.cross(n, ab) + ab2 * np.cross(ac, n)) / (2.0 * n2)
    radius = float(np.linalg.norm(center - a))
    return center, radius


def build_grasp_task_frame(thumb_tip, index_tip, middle_tip, approach_hint=None, palm_rot=None):
    """
    三指夹取 task frame（拇指+食指+中指，与 grasp_close 预设一致）。

    - 原点：三指腹质心
    - X：拇指 → (食指+中指)/2
    - Z：接近方向（hint 投影到 X 法平面）
    """
    thumb = np.asarray(thumb_tip, dtype=np.float64).reshape(3)
    index = np.asarray(index_tip, dtype=np.float64).reshape(3)
    middle = np.asarray(middle_tip, dtype=np.float64).reshape(3)
    origin = (thumb + index + middle) / 3.0
    x_close = 0.5 * (index + middle) - thumb
    return _grasp_frame_from_x(origin, x_close, approach_hint=approach_hint, palm_rot=palm_rot)


def build_grasp_task_frame_power(
    thumb_tip, middle_tip, ring_tip, approach_hint=None, palm_rot=None,
):
    """
    五指主抓 task frame（拇指+中指+无名指）。

    - 原点：三 tip **外接圆圆心**（circumcenter）≈ 螺母中心
      近共线或外接半径异常大时，回退闭合轴中点（拇↔中无名中点）
    - X：拇指 → (中指+无名)/2
    - pinch_sep_m：拇到中/无名中点距离
    """
    thumb = np.asarray(thumb_tip, dtype=np.float64).reshape(3)
    middle = np.asarray(middle_tip, dtype=np.float64).reshape(3)
    ring = np.asarray(ring_tip, dtype=np.float64).reshape(3)
    mid_pair = 0.5 * (middle + ring)
    x_close = mid_pair - thumb
    sep = float(np.linalg.norm(x_close))
    axis_mid = 0.5 * (thumb + mid_pair)
    circ, circ_r = triangle_circumcenter(thumb, middle, ring)
    # 外心飞离（扁平三角常见）：|R|≫sep/2 或外心远离闭合轴中点 → 回退
    use_circ = (
        circ is not None
        and circ_r > 1e-6
        and circ_r < max(0.08, 1.25 * max(sep, 1e-6))
        and float(np.linalg.norm(circ - axis_mid)) < max(0.03, 0.55 * max(sep, 1e-6))
    )
    if use_circ:
        origin = np.asarray(circ, dtype=np.float64).reshape(3)
        origin_mode = "circumcenter"
    else:
        origin = axis_mid
        origin_mode = "close_axis_mid_fallback"
    in_c, in_r = triangle_incenter(thumb, middle, ring)
    frame = _grasp_frame_from_x(
        origin, x_close, approach_hint=approach_hint, palm_rot=palm_rot,
    )
    if frame is None:
        return None
    frame["pinch_sep_m"] = sep
    frame["circumradius_m"] = float(circ_r) if circ is not None else 0.0
    frame["inradius_m"] = float(in_r)
    frame["incenter_m"] = np.asarray(in_c, dtype=np.float64).reshape(3).tolist()
    if circ is not None:
        frame["circumcenter_m"] = np.asarray(circ, dtype=np.float64).reshape(3).tolist()
    frame["tcp_origin_mode"] = origin_mode
    return frame


def rotation_matrix_to_rpy(R):
    """3x3 旋转矩阵 → (roll, pitch, yaw) rad，ZYX 外旋。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def pose_to_matrix(position, euler_rpy):
    """位置 + (roll,pitch,yaw) rad → 4x4 齐次矩阵（工作系）。"""
    if hasattr(position, "x"):
        t = np.array([position.x, position.y, position.z], dtype=np.float64)
    else:
        t = np.asarray(position, dtype=np.float64).reshape(3)
    if hasattr(euler_rpy, "x"):
        rpy = (euler_rpy.x, euler_rpy.y, euler_rpy.z)
    else:
        rpy = tuple(euler_rpy)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_rpy_to_matrix(*rpy)
    T[:3, 3] = t
    return T


def matrix_to_rpy_near(T, ref_rpy=None):
    """4x4 → (xyz, (r,p,y))；RPY 选与 ref 最接近的等价角。"""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    pos = T[:3, 3].copy()
    rpy0 = rotation_matrix_to_rpy(T[:3, :3])
    if ref_rpy is None:
        return pos, rpy0
    ref = np.asarray(ref_rpy, dtype=np.float64).reshape(3)
    best = rpy0
    best_cost = float("inf")
    for cand in _rpy_equivalent_candidates(rpy0[0], rpy0[1], rpy0[2]):
        cost = sum(_wrap_angle_diff(cand[i] - ref[i]) ** 2 for i in range(3))
        if cost < best_cost:
            best_cost = cost
            best = cand
    return pos, best


def rpy_delta_deg(rpy_a, rpy_b):
    """两姿态各轴最短角差（度），返回 list[3]。"""
    out = []
    for a, b in zip(rpy_a, rpy_b):
        d = (float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi
        out.append(math.degrees(d))
    return out


def rotation_geodesic_deg(rpy_a, rpy_b):
    """两 RPY 之间的测地线转角（度）。"""
    Ra = euler_rpy_to_matrix(float(rpy_a[0]), float(rpy_a[1]), float(rpy_a[2]))
    Rb = euler_rpy_to_matrix(float(rpy_b[0]), float(rpy_b[1]), float(rpy_b[2]))
    R_rel = Ra.T @ Rb
    # angle from rotation matrix: acos((tr-1)/2)
    tr = float(np.trace(R_rel))
    c = float(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(c))


def lerp_pose_rpy(pos0, rpy0, pos1, rpy1, t):
    """位置线性 + 姿态测地线插值（等价最短路径 SLERP）。t∈[0,1]。"""
    t = float(np.clip(t, 0.0, 1.0))
    p0 = np.asarray(pos0, dtype=np.float64).reshape(3)
    p1 = np.asarray(pos1, dtype=np.float64).reshape(3)
    pos = p0 + t * (p1 - p0)
    R0 = euler_rpy_to_matrix(float(rpy0[0]), float(rpy0[1]), float(rpy0[2]))
    R1 = euler_rpy_to_matrix(float(rpy1[0]), float(rpy1[1]), float(rpy1[2]))
    R_rel = R0.T @ R1
    tr = float(np.clip((float(np.trace(R_rel)) - 1.0) * 0.5, -1.0, 1.0))
    ang = math.acos(tr)
    if ang < 1e-8:
        R = R0
    else:
        # Rodrigues from relative rotation * t
        k = np.array([
            R_rel[2, 1] - R_rel[1, 2],
            R_rel[0, 2] - R_rel[2, 0],
            R_rel[1, 0] - R_rel[0, 1],
        ], dtype=np.float64)
        kn = float(np.linalg.norm(k))
        if kn < 1e-10:
            R = R0
        else:
            axis = k / kn
            a = ang * t
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ], dtype=np.float64)
            R_step = (
                np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)
            )
            R = R0 @ R_step
    rpy = rotation_matrix_to_rpy(R)
    # unwrap near start
    _, rpy = matrix_to_rpy_near(
        np.block([[R, pos.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]]),
        ref_rpy=rpy0,
    )
    return pos, rpy


def pinch_dict_to_matrix(pinch_info):
    """MuJoCo pinch_info → 4x4 世界系捏取 frame。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(pinch_info["R_world"], dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(pinch_info["pinch_mid_m"], dtype=np.float64).reshape(3)
    return T


def detection_point_from_camera_pose(
    cam_pos,
    cam_rot,
    u,
    v,
    depth_mm,
    intrinsics,
    xy_rotate_deg=0.0,
    flip_x=False,
    flip_y=False,
    depth_sign=1.0,
):
    """
    检测点 = 相机光心 + R @ 射线。
    射线：OpenCV 像素反投影 → depth_sign(前后) → 与对准伺服相同的 XY remap(90°等)。
    """
    point_cam = pixel_depth_to_camera_point(u, v, depth_mm, intrinsics)
    if point_cam is None:
        return None
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    cam_rot = np.asarray(cam_rot, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(cam_pos)) or not np.all(np.isfinite(cam_rot)):
        return None
    point_cam = np.array(point_cam, dtype=np.float64, copy=True)
    point_cam[2] *= float(depth_sign)
    point_cam = _servo_xy_remap(point_cam, xy_rotate_deg, flip_x, flip_y)
    out = cam_pos + cam_rot @ point_cam
    return out if np.all(np.isfinite(out)) else None


def camera_pose_in_world(R_ee_cam, t_ee_cam, ee_pose):
    """末端/tool0 位姿 + 相机外参 → 相机光心在世界/工作系下的位姿。"""
    if ee_pose is None:
        return None, None
    pos, euler = ee_pose
    R_we = euler_rpy_to_matrix(euler.x, euler.y, euler.z)
    t_we = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
    t_ee_cam = np.asarray(t_ee_cam, dtype=np.float64).reshape(3)
    R_ee_cam = np.asarray(R_ee_cam, dtype=np.float64).reshape(3, 3)
    cam_rot = R_we @ R_ee_cam
    cam_pos = R_we @ t_ee_cam + t_we
    return cam_pos, cam_rot


def detection_point_in_world(
    u, v, depth_mm, intrinsics, R_ee_cam, t_ee_cam, ee_pose,
    xy_rotate_deg=90.0, flip_x=True, flip_y=False, depth_sign=-1.0,
):
    """检测点：像素+深度 → 工作系 3D（相机光心 + 射线 + remap）。"""
    cam_pos, cam_rot = camera_pose_in_world(R_ee_cam, t_ee_cam, ee_pose)
    if cam_pos is None:
        return None
    return detection_point_from_camera_pose(
        cam_pos, cam_rot, u, v, depth_mm, intrinsics,
        xy_rotate_deg=xy_rotate_deg, flip_x=flip_x, flip_y=flip_y,
        depth_sign=depth_sign,
    )


def approach_unit_toward_object(ee_pos_world, target_point_world, pinch_z_away=None):
    """
    世界系单位向量：末端向 target 靠近（⑦ 最终靠近为正方向）。
    pinch_z_away: build_pinch_frame_at_object 的 Z 轴 (物体→法兰)，与本函数相反。
    """
    ee = np.asarray(ee_pos_world, dtype=np.float64).reshape(3)
    tgt = np.asarray(target_point_world, dtype=np.float64).reshape(3)
    toward = tgt - ee
    tn = float(np.linalg.norm(toward))
    if tn > 1e-6:
        return toward / tn
    if pinch_z_away is not None:
        z = np.asarray(pinch_z_away, dtype=np.float64).reshape(3)
        zn = float(np.linalg.norm(z))
        if zn > 1e-6:
            return -z / zn
    return None


def pinch_standoff_away_unit(
    obj_world,
    ee_pos_world,
    pinch_info=None,
    use_hand_z=False,
    world_up=None,
    level_finger_plane=False,
):
    """
    预捏合 standoff 方向（物体→法兰，单位向量）。

    level_finger_plane：沿 ±重力（与法兰同侧），使黄球在物体正上方，⑦ 竖直下落。
    默认：几何方向 ee−obj。
    use_hand_z=True：手系 z 与几何夹角不太大才用手系。
    """
    obj = np.asarray(obj_world, dtype=np.float64).reshape(3)
    ee = np.asarray(ee_pos_world, dtype=np.float64).reshape(3)
    ee_dir = ee - obj
    en = float(np.linalg.norm(ee_dir))
    if en < 1e-6:
        return None
    ee_dir /= en

    if level_finger_plane:
        up = np.asarray(
            world_up if world_up is not None else (0.0, 0.0, 1.0),
            dtype=np.float64,
        ).reshape(3)
        un = float(np.linalg.norm(up))
        if un < 1e-9:
            return ee_dir
        up = up / un
        if float(np.dot(up, ee_dir)) < 0.0:
            up = -up
        return up

    if use_hand_z and pinch_info is not None:
        z_hand = np.asarray(pinch_info.get("z_approach"), dtype=np.float64).reshape(3)
        zn = float(np.linalg.norm(z_hand))
        if zn > 1e-6:
            z_hand = z_hand / zn
            if float(np.dot(z_hand, ee_dir)) < 0.0:
                z_hand = -z_hand
            if float(np.dot(z_hand, ee_dir)) >= 0.75:
                return z_hand
    return ee_dir


def pinch_standoff_target(
    obj_world,
    ee_pos_world,
    standoff_m,
    pinch_info=None,
    use_hand_z=False,
    world_up=None,
    level_finger_plane=False,
):
    """预捏合黄球位置：沿 standoff 方向距物体点 standoff_m。"""
    z_away = pinch_standoff_away_unit(
        obj_world,
        ee_pos_world,
        pinch_info=pinch_info,
        use_hand_z=use_hand_z,
        world_up=world_up,
        level_finger_plane=level_finger_plane,
    )
    if z_away is None:
        return None
    obj = np.asarray(obj_world, dtype=np.float64).reshape(3)
    return obj + z_away * float(standoff_m)


def object_up_direction_world(
    R_world_cam,
    grasp_point_mode="bbox_center",
    grasp_point_side="right",
    xy_rotate_deg=0.0,
    flip_x=False,
    flip_y=False,
    world_up_fallback=None,
):
    """
    物体「上」方向（世界系）：与抓取点语义一致。

    - hand_top / world_up：直接用 pinch_world_up（默认世界 +Z）；side=left/down 取反
    - top_face / top：画面上缘（OpenCV −Y）经 cam→world（旧 servo_xy 已并入 R_ee_cam）
    - 其它：回退 pinch_world_up
    """
    R = np.asarray(R_world_cam, dtype=np.float64).reshape(3, 3)
    mode = str(grasp_point_mode or "bbox_center").strip().lower()
    side = str(grasp_point_side or "right").strip().lower()

    fb = world_up_fallback
    if fb is None:
        fb = (0.0, 0.0, 1.0)
    world_up = np.asarray(fb, dtype=np.float64).reshape(3)
    wn = float(np.linalg.norm(world_up))
    if wn < 1e-9:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        world_up = world_up / wn

    if mode in ("hand_top", "hand_upper", "side_top", "world_top", "world_up"):
        if side in ("left", "l", "down", "bottom"):
            return -world_up
        return world_up

    if mode in ("top_face", "top", "upper_third"):
        d_cam = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        d_cam = _servo_xy_remap(
            d_cam, float(xy_rotate_deg), bool(flip_x), bool(flip_y),
        )
        up = R @ d_cam
        n = float(np.linalg.norm(up))
        if n > 1e-9:
            return up / n

    return world_up


def _outer_af_mm_from_size(size_info=None, eye=None):
    """外径对边 AF (mm)：优先 size_info.hex_mm，否则按档标称，再退 yaml。"""
    eye = eye or {}
    if isinstance(size_info, dict):
        for key in ("hex_mm", "long_mm", "hex_cls_mm", "hex_raw_mm"):
            v = size_info.get(key)
            if v is not None and np.isfinite(float(v)) and float(v) > 1.0:
                return float(v)
        cls = str(size_info.get("class") or "").strip().lower()
        nom = eye.get("pinch_grasp_below_surface_nominal_mm") or {
            "big": 70.0, "medium": 50.0, "small": 41.0,
        }
        if cls in nom:
            return float(nom[cls])
    fb = eye.get("pinch_grasp_below_surface_outer_mm")
    if fb is not None and np.isfinite(float(fb)) and float(fb) > 1.0:
        return float(fb)
    return None


def _nut_height_mm_from_size(size_info=None, eye=None, detection=None):
    """
    螺母厚度 H (mm)。优先 size_info，再 detection.size_metric / eye 的 nominal_height_mm。
    （与 AF 定档分开：侧面捏取应对准 H/2，不是 0.2×AF。）
    """
    eye = eye or {}
    det = detection or {}
    if isinstance(size_info, dict):
        for key in ("height_mm", "H_mm", "thickness_mm", "nominal_height_mm"):
            v = size_info.get(key)
            if v is not None and np.isfinite(float(v)) and float(v) > 0.5:
                return float(v)
        cls = str(size_info.get("class") or "").strip().lower()
    else:
        cls = ""
    sm = det.get("size_metric") if isinstance(det.get("size_metric"), dict) else {}
    nom = (
        eye.get("pinch_grasp_nominal_height_mm")
        or det.get("nominal_height_mm")
        or sm.get("nominal_height_mm")
        or {}
    )
    if cls and isinstance(nom, dict) and cls in nom:
        v = nom[cls]
        if v is not None and np.isfinite(float(v)) and float(v) > 0.5:
            return float(v)
    return None


def apply_pinch_grasp_bias(
    obj_world, ee_pos_world, object_up, eye, log=None, size_info=None,
    detection=None,
):
    """
    锁定/规划抓取点修正（与放下 place_bias 同一世界口径）。

    - right / forward：工作系 右=−Y、前=+X（place_slot_forward），勿用接近系
    - up：沿 pinch_world_up（少用；高度主要靠 world_down）
    - along：仍为接近轴（物−法兰），仅特殊补偿；一般保持 0
    - below_surface / world_down：沿 −pinch_world_up 下偏到侧中心
    """
    obj = np.asarray(obj_world, dtype=np.float64).reshape(3).copy()
    ee = np.asarray(ee_pos_world, dtype=np.float64).reshape(3)
    cls = ""
    if isinstance(size_info, dict):
        cls = str(size_info.get("class") or "").strip().lower()
        if cls == "large":
            cls = "big"

    def _by_size(key_base, key_by, default=0.0):
        val = float(eye.get(key_base, default) or 0.0)
        by = eye.get(key_by) or {}
        if cls and isinstance(by, dict) and by.get(cls) is not None:
            try:
                val = float(by[cls])
            except (TypeError, ValueError):
                pass
        return val

    br = _by_size("pinch_grasp_bias_right_m", "pinch_grasp_bias_right_by_size")
    bf = _by_size("pinch_grasp_bias_forward_m", "pinch_grasp_bias_forward_by_size")
    bu = float(eye.get("pinch_grasp_bias_up_m", 0.0) or 0.0)
    ba = float(eye.get("pinch_grasp_bias_along_m", 0.0) or 0.0)
    down_m, down_src = pinch_grasp_down_m(
        eye, size_info=size_info, detection=detection, with_source=True,
    )
    need_side = (
        abs(br) >= 1e-6 or abs(bf) >= 1e-6 or abs(bu) >= 1e-6 or abs(ba) >= 1e-6
    )
    need_down = abs(down_m) >= 1e-6
    if not need_side and not need_down:
        return obj

    wu = np.asarray(
        eye.get("pinch_world_up", [0.0, 0.0, 1.0]), dtype=np.float64
    ).reshape(3)
    wn = float(np.linalg.norm(wu))
    wu = wu / wn if wn > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    out = obj
    delta_side = np.zeros(3, dtype=np.float64)
    if need_side:
        # 与放下同口径：前 = place_slot_forward，右 = forward × world_up
        fwd = np.asarray(
            eye.get("place_slot_forward", [1.0, 0.0, 0.0]), dtype=np.float64
        ).reshape(3)
        fn = float(np.linalg.norm(fwd))
        fwd = fwd / fn if fn > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fwd = fwd - float(np.dot(fwd, wu)) * wu
        fn = float(np.linalg.norm(fwd))
        fwd = fwd / fn if fn > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right = np.cross(fwd, wu)
        rn = float(np.linalg.norm(right))
        right = (
            right / rn if rn > 1e-9
            else np.array([0.0, -1.0, 0.0], dtype=np.float64)
        )

        delta_side = br * right + bf * fwd + bu * wu

        # along：接近轴（物−法兰），与世界前/右独立
        if abs(ba) >= 1e-6:
            approach = obj - ee
            an = float(np.linalg.norm(approach))
            if an >= 1e-9:
                delta_side = delta_side + ba * (approach / an)

        out = out + delta_side

    delta_down = np.zeros(3, dtype=np.float64)
    if abs(down_m) >= 1e-6:
        # 世界向下 = −world_up（指尖中点略低于上表面识别点）
        delta_down = -wu * down_m
        out = out + delta_down

    if log is not None:
        parts = []
        if need_side:
            parts.append(
                f"右{br*1000:.0f}mm 前{bf*1000:.0f}mm "
                f"上{bu*1000:.0f}mm 接近{ba*1000:.0f}mm(世界前/右)"
            )
        if abs(down_m) >= 1e-6:
            parts.append(f"world_down={down_m*1000:.1f}mm({down_src})")
        d = delta_side + delta_down
        log(
            f"抓取点修正 {' '.join(parts)} → "
            f"Δxyz[{d[0]*1000:.1f},{d[1]*1000:.1f},{d[2]*1000:.1f}]mm "
            f"(+X前 +Y左 −Y右)"
        )
    return out


def estimate_hex_nut_height_m(outer_af_mm=None, height_mm=None):
    """厚度优先用实测/标称 height_mm；否则粗估 H≈0.55×AF。"""
    if height_mm is not None and np.isfinite(float(height_mm)) and float(height_mm) > 0.5:
        return float(height_mm) * 1e-3
    if outer_af_mm is None or not np.isfinite(float(outer_af_mm)) or float(outer_af_mm) <= 1.0:
        return 0.008
    return float(outer_af_mm) * 0.55 * 1e-3


def pinch_grasp_down_m(eye, size_info=None, detection=None, with_source=False):
    """
    世界向下偏量（米）。
    mode=height：height_frac × 厚度 H（侧中心 ≈0.5H）
    mode=af：frac × 外径 AF（可按尺寸 by_size）
    方向在 apply_pinch_grasp_bias 里乘 −pinch_world_up（+Z 为上时即降低）。
    """
    eye = eye or {}
    fixed = float(eye.get("pinch_grasp_below_surface_m", 0.0) or 0.0)
    mode = str(eye.get("pinch_grasp_below_mode", "height")).strip().lower()
    h_mm = _nut_height_mm_from_size(size_info, eye, detection)
    outer_mm = _outer_af_mm_from_size(size_info, eye)

    down_m = fixed
    src = "fixed" if abs(fixed) >= 1e-6 else "none"
    if mode in ("height", "h", "thickness", "side_center") and h_mm is not None:
        h_frac = float(eye.get("pinch_grasp_below_height_frac", 0.5))
        by_h = eye.get("pinch_grasp_below_height_frac_by_size") or {}
        if isinstance(size_info, dict) and isinstance(by_h, dict):
            cls = str(size_info.get("class") or "").strip().lower()
            keys = ["big", "large"] if cls in ("big", "large") else ([cls] if cls else [])
            for k in keys:
                if k in by_h and by_h[k] is not None:
                    try:
                        h_frac = float(by_h[k])
                        break
                    except (TypeError, ValueError):
                        pass
        down_m = fixed + h_frac * float(h_mm) * 1e-3
        src = f"H{h_mm:.1f}mm×{h_frac:.2f}"
    else:
        frac = resolve_pinch_below_surface_frac(eye, size_info)
        if frac > 1e-9 and outer_mm is not None and outer_mm > 1.0:
            down_m = fixed + frac * float(outer_mm) * 1e-3
            src = f"AF{outer_mm:.0f}mm×{frac:.2f}"
        elif h_mm is not None:
            h_frac = float(eye.get("pinch_grasp_below_height_frac", 0.5))
            down_m = fixed + h_frac * float(h_mm) * 1e-3
            src = f"H{h_mm:.1f}mm×{h_frac:.2f}(fallback)"

    if with_source:
        return float(down_m), src
    return float(down_m), outer_mm


def diagnose_level_pinch_top_rub(
    obj_world,
    pinch_info,
    eye,
    size_info=None,
    grasp_down_m=None,
    detection=None,
    pinch_info_pad=None,
    pinch_info_open=None,
):
    """
    level 捏取高度诊断（相对顶面，用标称厚度；勿把「mid≈青」再推成侧中心差）。
    - mid_below_top：闭合 mid 在顶面下多少（期望 ≈ H/2）
    - tip_vs_pad_z：同关节 tip mid 与 pad mid 的高度差（接触模型是否偏）
    - open_above_top：张手指尖相对顶面（合拢前是否已蹭顶）
    """
    if not resolve_pinch_level_finger_plane(eye, size_info=size_info):
        return {"level_mode": False}
    if pinch_info is None or obj_world is None:
        return {"level_mode": True, "risk": None}

    obj = np.asarray(obj_world, dtype=np.float64).reshape(3)
    wu = np.asarray(eye.get("pinch_world_up", [0.0, 0.0, 1.0]), dtype=np.float64).reshape(3)
    wn = float(np.linalg.norm(wu))
    wu = wu / wn if wn > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if grasp_down_m is None:
        grasp_down_m, _src = pinch_grasp_down_m(
            eye, size_info, detection=detection, with_source=True,
        )
    elif isinstance(grasp_down_m, (tuple, list)):
        grasp_down_m = float(grasp_down_m[0])

    h_mm = _nut_height_mm_from_size(size_info, eye, detection)
    outer_mm = _outer_af_mm_from_size(size_info, eye)
    nut_h = estimate_hex_nut_height_m(outer_mm, height_mm=h_mm)
    # 顶面 = 青球（已下偏）沿 +up 回退 grasp_down
    nut_top = obj + wu * float(grasp_down_m)
    side_center = nut_top - wu * (nut_h * 0.5)
    expect_below_top_mm = nut_h * 1000.0 * 0.5

    mid = np.asarray(pinch_info.get("pinch_mid_m", obj), dtype=np.float64).reshape(3)
    mid_below_top_mm = float(np.dot(nut_top - mid, wu)) * 1000.0
    mid_above_side_mm = float(np.dot(mid - side_center, wu)) * 1000.0

    tip_vs_pad_z_mm = None
    if pinch_info_pad is not None and pinch_info_pad.get("pinch_mid_m") is not None:
        mid_pad = np.asarray(
            pinch_info_pad["pinch_mid_m"], dtype=np.float64
        ).reshape(3)
        # pad 更高 → 正值（tip 规划会让真实指腹偏高）
        tip_vs_pad_z_mm = float(np.dot(mid_pad - mid, wu)) * 1000.0

    open_above_top_mm = None
    open_mid_below_top_mm = None
    if pinch_info_open is not None:
        if pinch_info_open.get("pinch_mid_m") is not None:
            mid_o = np.asarray(
                pinch_info_open["pinch_mid_m"], dtype=np.float64
            ).reshape(3)
            open_mid_below_top_mm = float(np.dot(nut_top - mid_o, wu)) * 1000.0
        zs = []
        for key in ("thumb_tip_m", "index_tip_m", "middle_tip_m"):
            t = pinch_info_open.get(key)
            if t is not None:
                tw = np.asarray(t, dtype=np.float64).reshape(3)
                # 指尖在顶面之上多少（>0 = 还在顶面上方）
                zs.append(float(np.dot(tw - nut_top, wu)) * 1000.0)
        if zs:
            open_above_top_mm = max(zs)

    tips = []
    tip_labels = []
    for key, lab in (
        ("thumb_tip_m", "拇"), ("index_tip_m", "食"), ("middle_tip_m", "中"),
    ):
        t = pinch_info.get(key)
        if t is not None:
            tips.append(np.asarray(t, dtype=np.float64).reshape(3))
            tip_labels.append(lab)
    tip_below_top_mm = [
        float(np.dot(nut_top - t, wu)) * 1000.0 for t in tips
    ]

    z_ax = np.asarray(pinch_info.get("z_approach", wu), dtype=np.float64).reshape(3)
    zn = float(np.linalg.norm(z_ax))
    z_ax = z_ax / zn if zn > 1e-9 else wu
    z_tilt_deg = math.degrees(
        math.acos(float(np.clip(abs(np.dot(z_ax, wu)), -1.0, 1.0)))
    )

    sep_mm = float(pinch_info.get("pinch_sep_mm", 0.0))
    af_mm = float(outer_mm) if outer_mm is not None else float("nan")

    risk = None
    hint = None
    # 注意：mid≈青 且青=顶−down 时，mid−侧中 ≈ H/2−down，是恒等式，不能单独当「蹭顶证据」。
    # 蹭顶应看：mid 相对「真顶」是否仍接近 0、张手是否已扫过顶面、tip/pad 高差。
    top_clearance_mm = mid_below_top_mm  # >0 = mid 在顶面之下
    if top_clearance_mm < 2.0:
        risk = "high"
        hint = (
            f"闭合 mid 只在顶面下 {top_clearance_mm:.0f}mm"
            f"（下偏标称 {float(grasp_down_m)*1000:.0f}mm）；"
            f"更像深度/锁定点偏高，而非「没对准侧中心」"
        )
    elif top_clearance_mm < 5.0:
        risk = "med"
        hint = (
            f"闭合 mid 顶面下仅 {top_clearance_mm:.0f}mm，"
            f"接近上沿，合拢易蹭倒角/顶缘"
        )
    if tip_vs_pad_z_mm is not None and tip_vs_pad_z_mm > 4.0 and risk != "high":
        risk = risk or "med"
        hint = (
            (hint + "；" if hint else "")
            + f"pad mid 比 tip mid 高 {tip_vs_pad_z_mm:.0f}mm（tip 规划→实指腹偏高）"
        )
    if (
        open_mid_below_top_mm is not None
        and open_mid_below_top_mm < 3.0
        and open_mid_below_top_mm > -5.0
        and top_clearance_mm < 8.0
    ):
        # 张手 mid 已在顶附近且闭合也不深 → 横扫顶面
        risk = "high"
        hint = (
            (hint + "；" if hint else "")
            + f"张手 mid 几乎在顶面高度（顶面下 {open_mid_below_top_mm:.0f}mm），"
            f"合拢时易横扫蹭顶"
        )
    # mid−侧中 只作信息，不升 risk（避免与「青=顶−0.2AF」同义反复）
    shallow_vs_side = expect_below_top_mm - mid_below_top_mm

    return {
        "level_mode": True,
        "nut_top_m": nut_top.tolist(),
        "side_center_m": side_center.tolist(),
        "nut_height_mm": nut_h * 1000.0,
        "grasp_down_mm": float(grasp_down_m) * 1000.0,
        "expect_below_top_mm": expect_below_top_mm,
        "mid_below_top_mm": mid_below_top_mm,
        "mid_above_side_mm": mid_above_side_mm,
        "shallow_vs_side_mm": shallow_vs_side,
        "tip_vs_pad_z_mm": tip_vs_pad_z_mm,
        "open_mid_below_top_mm": open_mid_below_top_mm,
        "open_tip_above_top_mm": open_above_top_mm,
        "outer_af_mm": af_mm,
        "tip_below_top_mm": tip_below_top_mm,
        "tip_labels": tip_labels,
        "z_tilt_from_up_deg": z_tilt_deg,
        "pinch_sep_mm": sep_mm,
        "risk": risk,
        "hint": hint,
    }


def _nearest_pm_tilt_z(z0, tilt_deg, world_up, z_ref=None, plane="world_up"):
    """
    将 z0 偏开 |tilt_deg|，在 ± 两候选里选更靠近 z_ref 的一个。

    plane:
      - world_up / gravity / vertical：绕「重力×水平接近方向」俯仰（推荐）
        接近轴近乎竖直时，旧实现 lat≈水平 → 看起来像左右倾；现改为用水平投影定俯仰轴
      - object_up / toward_up：用传入的 world_up（若为 hand_top 水平「上边」会变成左右倾）
      - horizontal：绕竖直左右偏（易变成「左/右45°」，一般不要）
      - toward_ref：贴当前手 Z
    """
    z0 = np.asarray(z0, dtype=np.float64).reshape(3)
    z0 = z0 / max(float(np.linalg.norm(z0)), 1e-12)
    up = np.asarray(world_up, dtype=np.float64).reshape(3)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    a = math.radians(abs(float(tilt_deg)))
    plane = str(plane or "world_up").strip().lower()

    ref = None
    if z_ref is not None:
        r = np.asarray(z_ref, dtype=np.float64).reshape(3)
        rn = float(np.linalg.norm(r))
        if rn > 1e-9:
            r = r / rn
            if float(np.dot(r, z0)) < 0.0:
                r = -r
            ref = r

    def _rodrigues(v, axis, ang):
        ax = axis / max(float(np.linalg.norm(axis)), 1e-12)
        c, s = math.cos(ang), math.sin(ang)
        return (
            v * c
            + np.cross(ax, v) * s
            + ax * float(np.dot(ax, v)) * (1.0 - c)
        )

    # world_up：真俯仰 = 绕水平轴（⊥重力、⊥水平接近方向）旋转
    if plane in ("world_up", "gravity", "vertical", "up", "object_up", "toward_up"):
        # 水平接近分量；过陡则借 z_ref / 世界前方，避免 lat 退化成「左右倾」
        horiz = z0 - float(np.dot(z0, up)) * up
        hn = float(np.linalg.norm(horiz))
        if hn < 0.28 and ref is not None:
            horiz = ref - float(np.dot(ref, up)) * up
            hn = float(np.linalg.norm(horiz))
        if hn < 1e-6:
            # 最后手段：世界 X 在水平面的投影
            xw = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            horiz = xw - float(np.dot(xw, up)) * up
            hn = float(np.linalg.norm(horiz))
            if hn < 1e-6:
                return z0, 0.0
        horiz = horiz / hn
        # 俯仰轴：水平且 ⊥ 接近方向的水平投影（= 左右轴）
        pitch_axis = np.cross(up, horiz)
        pn = float(np.linalg.norm(pitch_axis))
        if pn < 1e-9:
            return z0, 0.0
        pitch_axis = pitch_axis / pn
        z_pos = _rodrigues(z0, pitch_axis, a)
        z_neg = _rodrigues(z0, pitch_axis, -a)
        z_pos /= max(float(np.linalg.norm(z_pos)), 1e-12)
        z_neg /= max(float(np.linalg.norm(z_neg)), 1e-12)
        score = ref if ref is not None else up
        if float(np.dot(z_pos, score)) >= float(np.dot(z_neg, score)):
            return z_pos, +abs(float(tilt_deg))
        return z_neg, -abs(float(tilt_deg))

    lat = None
    if plane in ("horizontal", "yaw", "side"):
        lat = np.cross(up, z0)
        if float(np.linalg.norm(lat)) < 1e-6:
            lat = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), z0)
    elif plane in ("toward_ref", "ref", "hand") and ref is not None:
        lat = ref - np.dot(ref, z0) * z0
        if float(np.linalg.norm(lat)) < 1e-6:
            lat = None
    if lat is None or float(np.linalg.norm(lat)) < 1e-6:
        lat = up - np.dot(up, z0) * z0
        if float(np.linalg.norm(lat)) < 1e-6:
            side = np.cross(z0, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            if float(np.linalg.norm(side)) < 1e-6:
                side = np.cross(z0, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            lat = side
    ln = float(np.linalg.norm(lat))
    if ln < 1e-9:
        return z0, 0.0
    lat_hat = lat / ln

    z_pos = math.cos(a) * z0 + math.sin(a) * lat_hat
    z_neg = math.cos(a) * z0 - math.sin(a) * lat_hat
    z_pos /= max(float(np.linalg.norm(z_pos)), 1e-12)
    z_neg /= max(float(np.linalg.norm(z_neg)), 1e-12)

    score = ref if ref is not None else lat_hat
    if float(np.dot(z_pos, score)) >= float(np.dot(z_neg, score)):
        return z_pos, +abs(float(tilt_deg))
    return z_neg, -abs(float(tilt_deg))


def build_pinch_frame_at_object(
    obj_world,
    ee_pos_world,
    x_close_hint=None,
    world_up=None,
    z_tilt_deg=0.0,
    z_tilt_ref=None,
    z_tilt_plane="world_up",
    prefer_horizontal_x=False,
    object_up=None,
    level_finger_plane=False,
):
    """
    在物体处建立目标捏取 frame：
    - 原点 = 物体点（黄球）
    - Z = 物体→法兰（张开/后退）；level_finger_plane 时 Z∥重力，使三指平面∥世界水平面
    - X = 闭合轴（拇指→食中），默认尽量水平
    """
    obj = np.asarray(obj_world, dtype=np.float64).reshape(3)
    ee = np.asarray(ee_pos_world, dtype=np.float64).reshape(3)
    z0 = ee - obj
    z_norm = float(np.linalg.norm(z0))
    if z_norm < 1e-6:
        return None
    z0 = z0 / z_norm

    up_gravity = np.asarray(
        world_up if world_up is not None else (0.0, 0.0, 1.0),
        dtype=np.float64,
    ).reshape(3)
    un = float(np.linalg.norm(up_gravity))
    if un < 1e-9:
        up_gravity = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up_gravity = up_gravity / un

    plane = str(z_tilt_plane or "world_up").strip().lower()
    level = bool(level_finger_plane) or plane in (
        "level", "level_finger", "finger_level", "horizontal_plane",
    )

    if level:
        # 三指平面（pinch XY）∥ 世界水平面 ⇒ 法向 Z ∥ ±重力
        z_axis = up_gravity.copy()
        if float(np.dot(z_axis, z0)) < 0.0:
            z_axis = -z_axis
        prefer_horizontal_x = True
    else:
        if plane in ("object_up", "toward_up") and object_up is not None:
            up = np.asarray(object_up, dtype=np.float64).reshape(3)
            un_o = float(np.linalg.norm(up))
            up = up / un_o if un_o > 1e-9 else up_gravity
        else:
            up = up_gravity

        z_axis = z0.copy()
        tilt = abs(float(z_tilt_deg))
        if tilt > 1e-3:
            z_axis, _signed = _nearest_pm_tilt_z(
                z0,
                tilt_deg=tilt,
                world_up=up,
                z_ref=z_tilt_ref,
                plane=plane,
            )

    # X：优先水平（⊥重力上），否则用闭合 hint / 默认叉乘
    x_axis = None
    if prefer_horizontal_x:
        x_h = np.cross(up_gravity, z_axis)
        xn = float(np.linalg.norm(x_h))
        if xn > 1e-6:
            x_axis = x_h / xn
            if x_close_hint is not None:
                hint = np.asarray(x_close_hint, dtype=np.float64).reshape(3)
                if float(np.dot(x_axis, hint)) < 0.0:
                    x_axis = -x_axis
    if x_axis is None:
        if x_close_hint is not None:
            hint = np.asarray(x_close_hint, dtype=np.float64).reshape(3)
            x_axis = hint - np.dot(hint, z_axis) * z_axis
        else:
            x_axis = np.cross(up_gravity, z_axis)
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm < 1e-6:
            x_axis = np.cross(z_axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            x_norm = float(np.linalg.norm(x_axis))
        if x_norm < 1e-6:
            return None
        x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-9:
        return None
    y_axis /= y_norm
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = obj
    return T


def pinch_frame_orient_diag(T_pinch, ee_pos_world, obj_world, world_up=None):
    """诊断：Z 相对几何接近轴夹角；符号相对物体上（world_up）仰(+)/俯(-)。"""
    T = np.asarray(T_pinch, dtype=np.float64).reshape(4, 4)
    z = T[:3, 2]
    x = T[:3, 0]
    ee = np.asarray(ee_pos_world, dtype=np.float64).reshape(3)
    obj = np.asarray(obj_world, dtype=np.float64).reshape(3)
    z0 = ee - obj
    n0 = float(np.linalg.norm(z0))
    ang_z = 0.0
    signed_z = 0.0
    up = np.asarray(
        world_up if world_up is not None else (0.0, 0.0, 1.0),
        dtype=np.float64,
    ).reshape(3)
    un = float(np.linalg.norm(up))
    if un > 1e-9:
        up = up / un
    if n0 > 1e-9:
        z0 = z0 / n0
        c = float(np.clip(np.dot(z, z0), -1.0, 1.0))
        ang_z = math.degrees(math.acos(c))
        if un > 1e-9:
            lat = up - np.dot(up, z0) * z0
            ln = float(np.linalg.norm(lat))
            if ln > 1e-9:
                signed_z = ang_z if float(np.dot(z - z0, lat / ln)) >= 0.0 else -ang_z
            else:
                signed_z = ang_z
        else:
            signed_z = ang_z
    ang_x = 90.0
    if un > 1e-9:
        ang_x = math.degrees(math.asin(float(np.clip(np.dot(x, up), -1.0, 1.0))))
    return ang_z, ang_x, signed_z


def ee_target_from_pinch(T_world_pinch_des, T_world_pinch_cur, T_world_ee_cur):
    """
    由闭合手 TCP↔tool0 固定关系，求 tool0 位姿使 grasp frame 到达目标。
    T_w_ee_des = T_w_pinch_des @ inv(T_w_pinch_cur) @ T_w_ee_cur

    预抓 ⑥ 标准用法：T_pinch_cur 用 grasp_close 手姿，目标原点在 obj_target。
    """
    T_rel = np.linalg.inv(T_world_pinch_cur) @ T_world_ee_cur
    return T_world_pinch_des @ T_rel


def ee_target_align_rotation_only(T_world_pinch_des, T_world_pinch_cur, T_world_ee_cur):
    """⑥ 预抓位：仅对齐捏取 frame 姿态，tool0 位置保持不动。"""
    T_ee = np.asarray(T_world_ee_cur, dtype=np.float64).copy()
    R_ee = T_ee[:3, :3]
    R_pc = np.asarray(T_world_pinch_cur, dtype=np.float64)[:3, :3]
    R_pd = np.asarray(T_world_pinch_des, dtype=np.float64)[:3, :3]
    T_ee[:3, :3] = R_pd @ R_pc.T @ R_ee
    return T_ee


def build_camera_to_robot(config):
    extrinsic = config["camera_to_robot"]
    translation = np.array(extrinsic["translation"], dtype=np.float64)
    euler = extrinsic["euler_rpy"]
    rotation = euler_rpy_to_matrix(*euler)
    return rotation, translation


def scale_camera_intrinsics(intrinsics, image_width, image_height):
    """
    将 yaml 中 camera_info 参考分辨率下的 K 缩放到当前彩色/深度图尺寸。
    intrinsics 可含 width/height（或 image_width/image_height）；缺省则视为与当前图同尺寸。
    """
    base = dict(intrinsics or {})
    w = float(image_width)
    h = float(image_height)
    ref_w = float(base.get("width") or base.get("image_width") or w)
    ref_h = float(base.get("height") or base.get("image_height") or h)
    if ref_w < 1.0:
        ref_w = w
    if ref_h < 1.0:
        ref_h = h
    sx = w / ref_w
    sy = h / ref_h
    fx0 = float(base.get("fx", 520.0))
    fy0 = float(base.get("fy", fx0))
    cx0 = float(base.get("cx", ref_w * 0.5))
    cy0 = float(base.get("cy", ref_h * 0.5))
    return {
        "fx": fx0 * sx,
        "fy": fy0 * sy,
        "cx": cx0 * sx,
        "cy": cy0 * sy,
        "width": int(round(w)),
        "height": int(round(h)),
        "ref_width": int(round(ref_w)),
        "ref_height": int(round(ref_h)),
    }


def pixel_depth_to_camera_point(u, v, depth_mm, intrinsics):
    if depth_mm is None or depth_mm <= 0:
        return None
    z = depth_mm / 1000.0
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def camera_point_to_robot(point_camera, rotation, translation):
    return rotation @ point_camera + translation


def world_up_in_image_uv(R_world_cam, world_up=(0.0, 0.0, 1.0)):
    """
    世界「上」投影到当前相机画面的单位方向 (du, dv)。
    OpenCV：+u 右、+v 下。光轴几乎平行世界竖轴时回退为画面上 (0,-1)。
    """
    R = np.asarray(R_world_cam, dtype=np.float64).reshape(3, 3)
    up = np.asarray(world_up, dtype=np.float64).reshape(3)
    un = float(np.linalg.norm(up))
    if un < 1e-12:
        return (0.0, -1.0)
    d = R.T @ (up / un)
    du, dv = float(d[0]), float(d[1])
    m = math.hypot(du, dv)
    if m < 1e-6:
        return (0.0, -1.0)
    return (du / m, dv / m)


def grasp_pixel_from_box(
    box,
    mode="bbox_center",
    top_frac=0.35,
    side="right",
    image_up_uv=None,
    xy_rotate_deg=0.0,
    flip_x=False,
    flip_y=False,
):
    """
    检测框 → 抓取语义像素 (u,v)。

    hand_top / world_up：物体在世界系更高的那一侧（image_up_uv=世界+Z 在画面投影）；
    side=left/down 取反。未给 image_up_uv 时回退画面上缘。
    （xy_rotate/flip 仅兼容旧调用，新路径应把安装角写入 camera_on_ee.euler_rpy。）
    """
    del xy_rotate_deg, flip_x, flip_y  # 兼容旧 kwargs；安装角已并入 R_ee_cam
    x1, y1, x2, y2 = map(float, box[:4])
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    mode = str(mode or "bbox_center").strip().lower()
    frac = float(top_frac)
    side = str(side or "right").strip().lower()
    inset = min(0.32, max(0.06, frac * 0.55))

    if mode == "upper_third":
        u = (x1 + x2) * 0.5
        v = y1 + bh * frac
    elif mode in ("top_face", "top"):
        u = (x1 + x2) * 0.5
        v = y1 + bh * min(0.25, max(0.08, frac * 0.6))
    elif mode in ("hand_top", "hand_upper", "side_top", "world_top", "world_up"):
        if image_up_uv is not None:
            du = float(image_up_uv[0])
            dv = float(image_up_uv[1])
        else:
            du, dv = 0.0, -1.0
        if side in ("left", "l", "down", "bottom"):
            du, dv = -du, -dv
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        if abs(du) >= abs(dv):
            if du >= 0.0:
                u = x2 - bw * inset
            else:
                u = x1 + bw * inset
            v = cy
        else:
            u = cx
            if dv >= 0.0:
                v = y2 - bh * inset
            else:
                v = y1 + bh * inset
    else:
        u = (x1 + x2) * 0.5
        v = (y1 + y2) * 0.5
    return float(u), float(v)


def grasp_depth_sample_pixel(box, grasp_u, grasp_v, mode, eye):
    """
    深度 ROI 采样像素：hand_top 等偏置抓取点时用框心深度，避免棱边深度导致前后 standoff 偏。
    """
    mode = str(mode or "bbox_center").strip().lower()
    use_center = bool(eye.get("grasp_depth_at_face_center", True))
    if use_center and mode in (
        "hand_top", "hand_upper", "side_top", "world_top", "world_up",
        "top_face", "top",
    ):
        x1, y1, x2, y2 = map(float, box[:4])
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5
    return float(grasp_u), float(grasp_v)


def _depth_eye_params(eye=None):
    """从 eye_in_hand 配置提取深度估计参数（缺省安全）。"""
    eye = eye or {}
    return {
        "roi_mode": str(eye.get("depth_roi_mode", "nut_surface")).strip().lower(),
        "outlier_mm": float(eye.get("depth_outlier_mm", 20.0)),
        "min_valid_mm": float(eye.get("depth_min_valid_mm", 80.0)),
        "max_valid_mm": float(eye.get("depth_max_valid_mm", 900.0)),
        "min_fill": float(eye.get("depth_min_fill_ratio", 0.05)),
        "min_pix": int(eye.get("depth_min_pixels", 10)),
        "cluster_band_mm": float(eye.get("depth_cluster_band_mm", 28.0)),
        "cluster_min_mass": float(eye.get("depth_cluster_min_mass", 0.10)),
        "near_percentile": float(eye.get("depth_near_percentile", 35.0)),
        "margin_frac": float(eye.get("depth_roi_margin_frac", 0.10)),
        # 空心六角螺母：排除中心孔 + 框外沿，在金属环上采深
        "hole_inner_frac": float(eye.get("depth_hole_inner_frac", 0.36)),
        "ring_outer_frac": float(eye.get("depth_ring_outer_frac", 0.92)),
        "ring_peak_frac": float(eye.get("depth_ring_peak_frac", 0.62)),
        "ring_weight_sigma": float(eye.get("depth_ring_weight_sigma", 0.18)),
        "ring_use_sectors": bool(eye.get("depth_ring_use_sectors", True)),
        "sector_spread_max_mm": float(eye.get("depth_sector_spread_max_mm", 16.0)),
        "sector_band_mm": float(eye.get("depth_sector_band_mm", 12.0)),
        "spatial_median_ksize": int(eye.get("depth_spatial_median_ksize", 3)),
        "min_confidence": float(eye.get("depth_min_confidence", 0.35)),
        # nut_surface：孔核精修 + 顶/桌双层
        "hole_sep_mm": float(eye.get("depth_hole_sep_mm", 10.0)),
        "min_layer_sep_mm": float(eye.get("depth_min_layer_sep_mm", 8.0)),
        # 大号六角孔相对外接框常 >0.55；过小会裁掉孔缘 → 环心偏
        "hole_search_frac": float(eye.get("depth_hole_search_frac", 0.70)),
        "hole_margin_frac": float(eye.get("depth_hole_margin_frac", 0.06)),
        "hole_max_shift_frac": float(eye.get("depth_hole_max_shift_frac", 0.40)),
        "surface_band_mm": float(eye.get("depth_surface_band_mm", 12.0)),
        "prior_up_tol_mm": float(eye.get("depth_prior_up_tol_mm", 16.0)),
    }


def _centroid_and_radius_norm(xx, yy, mask, hx, hy, cx0, cy0, max_shift_norm=0.40):
    """
    孔洞/远核像素质心 + 归一化半径估计。
    相对种子点的位移限制在 max_shift_norm（相对框半轴），防跳到邻物。
    返回 (cx, cy, r85_or_None)。
    """
    n = int(mask.sum()) if mask is not None else 0
    if n < 6:
        return float(cx0), float(cy0), None
    fx = xx[mask].astype(np.float64)
    fy = yy[mask].astype(np.float64)
    cx = float(np.mean(fx))
    cy = float(np.mean(fy))
    hx = max(float(hx), 1.0)
    hy = max(float(hy), 1.0)
    du = (cx - float(cx0)) / hx
    dv = (cy - float(cy0)) / hy
    shift = math.hypot(du, dv)
    cap = max(0.05, float(max_shift_norm))
    if shift > cap:
        s = cap / shift
        cx = float(cx0) + (cx - float(cx0)) * s
        cy = float(cy0) + (cy - float(cy0)) * s
    r = np.sqrt(((fx - cx) / hx) ** 2 + ((fy - cy) / hy) ** 2)
    r85 = float(np.percentile(r, 85.0))
    return cx, cy, r85


def _extract_depth_roi(depth, box, grasp_u=None, grasp_v=None, margin_frac=0.18):
    """
    框内缩边 ROI + 相对抓取点的高斯权重。
    返回 (u, v, vals, weights, fill_ratio) ；失败 vals 为空。
    """
    x1, y1, x2, y2 = map(int, box[:4])
    h, w = depth.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    u = float(grasp_u if grasp_u is not None else (x1 + x2) * 0.5)
    v = float(grasp_v if grasp_v is not None else (y1 + y2) * 0.5)
    empty = (u, v, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), 0.0)
    if x2 <= x1 or y2 <= y1:
        return empty

    d = depth.astype(np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    mx = max(1, int(bw * margin_frac))
    my = max(1, int(bh * margin_frac))
    y_lo, y_hi = y1 + my, y2 - my
    x_lo, x_hi = x1 + mx, x2 - mx
    patch = d[y_lo:y_hi, x_lo:x_hi]
    if patch.size < 9:
        patch = d[y1:y2, x1:x2]
        y_lo, x_lo = y1, x1
    if patch.size < 1:
        return empty

    ph, pw = patch.shape
    yy, xx = np.mgrid[0:ph, 0:pw]
    cu, cv = int(round(u)), int(round(v))
    pcy, pcx = (cv - y_lo), (cu - x_lo)
    if not (0 <= pcy < ph and 0 <= pcx < pw):
        pcy, pcx = ph / 2.0, pw / 2.0
    sigma = max(2.5, min(ph, pw) / 3.0)
    gw = np.exp(-((xx - pcx) ** 2 + (yy - pcy) ** 2) / (2.0 * sigma ** 2))

    vals = patch.ravel().astype(np.float32)
    weights = gw.ravel().astype(np.float32)
    return u, v, vals, weights, float(vals.size)


def _extract_depth_roi_hollow_ring(
    depth,
    box,
    grasp_u=None,
    grasp_v=None,
    margin_frac=0.08,
    hole_inner_frac=0.36,
    ring_outer_frac=0.92,
    ring_peak_frac=0.62,
    ring_weight_sigma=0.18,
):
    """
    空心六角螺母深度 ROI：椭圆环带（排除中心通孔 + 框边噪声）。

    归一化椭圆半径 r：中心=0、外接框≈1。
    保留 hole_inner_frac ≤ r ≤ ring_outer_frac，权重峰在 ring_peak_frac（金属环中部）。
    避免中心高斯把「孔→桌面」深度采进来导致跳变。
    """
    x1, y1, x2, y2 = map(int, box[:4])
    h, w = depth.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    u = float(grasp_u if grasp_u is not None else (x1 + x2) * 0.5)
    v = float(grasp_v if grasp_v is not None else (y1 + y2) * 0.5)
    empty = (u, v, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), 0.0)
    if x2 <= x1 or y2 <= y1:
        return empty

    d = depth.astype(np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    mx = max(0, int(bw * max(0.0, float(margin_frac))))
    my = max(0, int(bh * max(0.0, float(margin_frac))))
    y_lo, y_hi = y1 + my, y2 - my
    x_lo, x_hi = x1 + mx, x2 - mx
    if y_hi - y_lo < 4 or x_hi - x_lo < 4:
        y_lo, y_hi, x_lo, x_hi = y1, y2, x1, x2
    patch = d[y_lo:y_hi, x_lo:x_hi]
    if patch.size < 9:
        return empty

    ph, pw = patch.shape
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float64)
    # 椭圆归一化：相对 patch 中心（≈螺母中心），半轴=半宽/半高
    cx = (pw - 1) * 0.5
    cy = (ph - 1) * 0.5
    hx = max(cx, 1.0)
    hy = max(cy, 1.0)
    rx = (xx - cx) / hx
    ry = (yy - cy) / hy
    r = np.sqrt(rx * rx + ry * ry)

    r_in = float(np.clip(hole_inner_frac, 0.05, 0.85))
    r_out = float(np.clip(ring_outer_frac, r_in + 0.05, 1.35))
    r_peak = float(np.clip(ring_peak_frac, r_in, r_out))
    sig = max(0.05, float(ring_weight_sigma))

    ring = (r >= r_in) & (r <= r_out)
    if int(ring.sum()) < 8:
        # 框太小 / 孔参数过猛：放宽内孔
        ring = (r >= max(0.15, r_in * 0.6)) & (r <= min(1.15, r_out + 0.08))
    if int(ring.sum()) < 6:
        # 最后退回全 patch（仍用环权重，弱化中心）
        ring = np.ones_like(r, dtype=bool)

    gw = np.exp(-((r - r_peak) ** 2) / (2.0 * sig * sig))
    gw = np.where(ring, gw, 0.0)

    vals = patch.ravel().astype(np.float32)
    weights = gw.ravel().astype(np.float32)
    # n_total 用环像素数，便于 fill_ratio 语义 = 环上有效深度占比
    n_ring = float(max(int(ring.sum()), 1))
    return u, v, vals, weights, n_ring


def _foreground_cluster_depth_mm(
    vals,
    weights,
    band_mm=28.0,
    min_mass=0.10,
    near_percentile=35.0,
    min_pixels=8,
):
    """
    眼在手上：框内常混有物体表面 + 背景墙。
    取「最近且像素够」的深度簇；近簇太稀而远簇主导时返回 None（宁缺勿用墙面）。
    """
    if vals is None or vals.size < 3:
        return None
    order = np.argsort(vals)
    vals = vals[order]
    w = weights[order] if weights is not None and weights.size == vals.size else np.ones_like(vals)
    w = np.maximum(w.astype(np.float64), 1e-6)
    w_sum = float(w.sum())
    w = w / w_sum

    vmin, vmax = float(vals[0]), float(vals[-1])
    # 单峰：近端分位，天然抑制远景尾部
    if (vmax - vmin) <= band_mm * 1.8:
        return float(np.percentile(vals, near_percentile))

    bin_w = max(10.0, float(band_mm) * 0.75)
    edges = np.arange(vmin, vmax + bin_w, bin_w)
    if edges.size < 3:
        return float(np.percentile(vals, near_percentile))
    hist, edges = np.histogram(vals, bins=edges, weights=w)

    peaks = []
    for i in range(len(hist)):
        left = hist[i - 1] if i > 0 else -1.0
        right = hist[i + 1] if i + 1 < len(hist) else -1.0
        if hist[i] >= left and hist[i] >= right and hist[i] > 1e-9:
            center = 0.5 * (edges[i] + edges[i + 1])
            peaks.append((float(center), float(hist[i])))

    if not peaks:
        return float(np.percentile(vals, near_percentile))

    peaks.sort(key=lambda p: p[0])
    max_mass = max(p[1] for p in peaks)
    min_pix = int(max(3, min_pixels))
    chosen = None
    near_weak = False

    for idx, (center, mass) in enumerate(peaks):
        mask = np.abs(vals - center) <= float(band_mm) * 1.25
        n_in = int(mask.sum())
        strong = (mass >= float(min_mass)) or (n_in >= min_pix)
        # 最近峰：像素够或质量够 → 采用
        if idx == 0:
            if strong or (mass >= 0.5 * max_mass and n_in >= max(3, min_pix // 2)):
                chosen = center
                break
            near_weak = True
            continue
        # 更远峰：仅当近峰太弱且本峰明显主导时才考虑；否则宁可不信（背景）
        if near_weak and mass >= max(float(min_mass), 0.55 * max_mass) and n_in >= min_pix:
            # 近弱远强 → 多半是物体空洞+墙，拒绝远景
            return None
        if strong and not near_weak:
            chosen = center
            break

    if chosen is None:
        if near_weak:
            return None
        return float(np.percentile(vals, near_percentile))

    mask = np.abs(vals - chosen) <= float(band_mm) * 1.25
    if int(mask.sum()) >= 3:
        return float(np.percentile(vals[mask], 50.0))
    return float(chosen)


def _weighted_median(vals, weights):
    """加权中位数（比加权均值抗跳变）。"""
    if vals is None or vals.size < 1:
        return None
    v = np.asarray(vals, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel() if weights is not None else np.ones_like(v)
    if w.size != v.size:
        w = np.ones_like(v)
    w = np.maximum(w, 1e-9)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w) / float(w.sum())
    idx = int(np.searchsorted(cw, 0.5, side="left"))
    idx = min(max(idx, 0), v.size - 1)
    return float(v[idx])


def _mad_inlier_mask(vals, k=2.5):
    """MAD 去野值（工程上比固定 mm 带宽更稳）。"""
    if vals is None or vals.size < 4:
        return np.ones(vals.size if vals is not None else 0, dtype=bool)
    v = np.asarray(vals, dtype=np.float64)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad < 1.5:
        mad = max(1.5, float(np.std(v)) * 0.5)
    lim = max(6.0, k * mad * 1.4826)
    return np.abs(v - med) <= lim


def _ring_plane_depth_mm(
    vals,
    weights,
    band_mm=12.0,
    prior_mm=None,
    min_pixels=8,
    sector_spread_max_mm=22.0,
    sector_medians=None,
):
    """
    环带/金属顶面深度：MAD 去野值 → 最近显著峰 → 峰内加权中位数。
    可选六扇区中位数交叉验证（抗中心孔漏采桌面）。

    返回 (depth_mm, confidence 0~1)。
    """
    if vals is None or vals.size < 3:
        return None, 0.0

    v = np.asarray(vals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64) if weights is not None else np.ones_like(v)
    if w.size != v.size:
        w = np.ones_like(v)

    keep = _mad_inlier_mask(v)
    if int(keep.sum()) >= max(3, min_pixels // 2):
        v, w = v[keep], w[keep]

    if v.size < max(3, min_pixels // 2):
        return None, 0.0

    # 六扇区共识（每扇区独立中位数，再取中位数）
    if sector_medians is not None:
        sm = [float(x) for x in sector_medians if x is not None and np.isfinite(x)]
        if len(sm) >= 3:
            spread = float(max(sm) - min(sm))
            est = float(np.median(sm))
            if spread <= sector_spread_max_mm:
                conf = float(np.clip(1.0 - spread / max(sector_spread_max_mm, 1.0), 0.45, 1.0))
                return est, conf
            # 扇区发散：仍可用，但降置信（可能混入了桌面/侧壁）
            if spread <= sector_spread_max_mm * 1.8:
                return est, 0.28

    vmin, vmax = float(v.min()), float(v.max())
    band = max(8.0, float(band_mm))
    if (vmax - vmin) <= band * 1.6:
        est = _weighted_median(v, w)
        return (est, 0.75) if est is not None else (None, 0.0)

    # 直方图找峰：优先「最近且质量够」的峰（螺母顶面），拒绝远峰（桌面）
    bin_w = max(6.0, band * 0.65)
    edges = np.arange(vmin, vmax + bin_w, bin_w)
    if edges.size < 3:
        est = _weighted_median(v, w)
        return (est, 0.5) if est is not None else (None, 0.0)
    hist, edges = np.histogram(v, bins=edges, weights=w)
    peaks = []
    for i in range(len(hist)):
        if hist[i] < 1e-9:
            continue
        left = hist[i - 1] if i > 0 else -1.0
        right = hist[i + 1] if i + 1 < len(hist) else -1.0
        if hist[i] >= left and hist[i] >= right:
            peaks.append((0.5 * (edges[i] + edges[i + 1]), float(hist[i])))
    if not peaks:
        est = _weighted_median(v, w)
        return (est, 0.4) if est is not None else (None, 0.0)

    peaks.sort(key=lambda t: t[0])
    w_sum = float(w.sum())
    max_mass = max(p[1] for p in peaks)
    chosen = None
    near_weak = False
    min_pix = max(4, int(min_pixels) // 2)

    for idx, (center, mass) in enumerate(peaks):
        mask = np.abs(v - center) <= band * 1.15
        n_in = int(mask.sum())
        mass_frac = mass / max(w_sum, 1e-9)
        strong = (mass_frac >= 0.12) or (n_in >= min_pix)
        if idx == 0:
            if strong or (mass >= 0.45 * max_mass and n_in >= max(3, min_pix // 3)):
                chosen = center
                break
            near_weak = True
            continue
        if near_weak:
            # 近峰弱 + 远峰强 → 典型「孔漏桌面」，整帧不信
            if mass >= 0.5 * max_mass and n_in >= min_pix:
                return None, 0.0
            continue
        if strong:
            chosen = center
            break

    if chosen is None:
        if near_weak:
            return None, 0.0
        chosen = peaks[0][0]

    mask = np.abs(v - chosen) <= band * 1.15
    if prior_mm is not None and float(prior_mm) > 0:
        # 峰离记忆太远 → 更像背景跳变
        if abs(chosen - float(prior_mm)) > max(55.0, float(prior_mm) * 0.35):
            return None, 0.0

    est = _weighted_median(v[mask], w[mask]) if int(mask.sum()) >= 3 else float(chosen)
    if est is None:
        return None, 0.0
    inlier = float(mask.sum()) / max(float(v.size), 1.0)
    peak_mass = 0.0
    for center, mass in peaks:
        if abs(center - chosen) < band:
            peak_mass = mass / max(w_sum, 1e-9)
            break
    conf = float(np.clip(0.35 + 0.45 * inlier + 0.2 * peak_mass, 0.0, 1.0))
    return float(est), conf


def _depth_histogram_peaks(vals, weights, bin_w=8.0):
    """加权直方图峰列表 [(center_mm, mass), ...] 由近到远。"""
    v = np.asarray(vals, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel() if weights is not None else np.ones_like(v)
    if w.size != v.size:
        w = np.ones_like(v)
    if v.size < 3:
        return []
    vmin, vmax = float(v.min()), float(v.max())
    bw = max(5.0, float(bin_w))
    edges = np.arange(vmin, vmax + bw, bw)
    if edges.size < 3:
        return [(float(np.median(v)), float(w.sum()))]
    hist, edges = np.histogram(v, bins=edges, weights=w)
    peaks = []
    for i in range(len(hist)):
        if hist[i] < 1e-9:
            continue
        left = hist[i - 1] if i > 0 else -1.0
        right = hist[i + 1] if i + 1 < len(hist) else -1.0
        if hist[i] >= left and hist[i] >= right:
            peaks.append((0.5 * (edges[i] + edges[i + 1]), float(hist[i])))
    peaks.sort(key=lambda t: t[0])
    return peaks


def estimate_nut_surface_depth(depth, box, grasp_u=None, grasp_v=None, eye=None, prior_mm=None):
    """
    螺母金属顶面深度（主输出）+ 可选桌面参考。

    1) 孔洞远核精修环心 / r_hole
    2) 金属环上双峰：近=顶面、远=桌面
    3) 六扇区只验顶面峰
    返回 (u, v, top_mm_or_None, meta)
    """
    p = _depth_eye_params(eye)
    x1, y1, x2, y2 = map(int, box[:4])
    h, w = depth.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    seed_u = float(grasp_u if grasp_u is not None else (x1 + x2) * 0.5)
    seed_v = float(grasp_v if grasp_v is not None else (y1 + y2) * 0.5)
    meta = {
        "u": seed_u, "v": seed_v, "n_valid": 0, "fill": 0.0,
        "estimate": None, "mode": "nut_surface", "rejected": None,
        "confidence": 0.0, "layer": "rejected",
        "hole_r": None, "center_uv": (seed_u, seed_v),
        "top_mm": None, "table_mm": None, "n_sectors": 0,
    }
    if x2 <= x1 or y2 <= y1:
        meta["rejected"] = "bad_box"
        return seed_u, seed_v, None, meta

    d = depth.astype(np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    margin_frac = float(p["margin_frac"])
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    mx = max(0, int(bw * max(0.0, margin_frac)))
    my = max(0, int(bh * max(0.0, margin_frac)))
    y_lo, y_hi = y1 + my, y2 - my
    x_lo, x_hi = x1 + mx, x2 - mx
    if y_hi - y_lo < 6 or x_hi - x_lo < 6:
        y_lo, y_hi, x_lo, x_hi = y1, y2, x1, x2
    patch = d[y_lo:y_hi, x_lo:x_hi].copy()
    if patch.size < 16:
        meta["rejected"] = "tiny_patch"
        return seed_u, seed_v, None, meta

    ksize = int(p.get("spatial_median_ksize", 0) or 0)
    if ksize >= 3:
        k = int(ksize) | 1
        patch = cv2.medianBlur(patch.astype(np.uint16), k).astype(np.float32)

    ph, pw = patch.shape
    min_v = float(p["min_valid_mm"])
    max_v = float(p["max_valid_mm"])
    valid = (patch >= min_v) & (patch <= max_v) & np.isfinite(patch)
    if int(valid.sum()) < max(8, int(p["min_pix"])):
        meta["rejected"] = "few_valid"
        return seed_u, seed_v, None, meta

    # ── 1) 孔洞远核 → 精修环心（外环中位作顶面参考，避免核心全是孔）──
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float64)
    cx0 = float(np.clip(seed_u - x_lo, 0.0, pw - 1.0))
    cy0 = float(np.clip(seed_v - y_lo, 0.0, ph - 1.0))
    hx = max((pw - 1) * 0.5, 1.0)
    hy = max((ph - 1) * 0.5, 1.0)
    # 大号孔相对 AABB 常 ~0.55–0.70；旧默认 0.55 易裁孔缘 → 质心偏框心
    search_frac = float(np.clip(p.get("hole_search_frac", 0.70), 0.30, 0.92))
    max_shift = float(np.clip(p.get("hole_max_shift_frac", 0.40), 0.10, 0.60))
    hole_sep = float(p.get("hole_sep_mm", 10.0))
    hole_margin = float(p.get("hole_margin_frac", 0.06))
    r_hole_norm = float(np.clip(p.get("hole_inner_frac", 0.36), 0.15, 0.70))
    cx, cy = cx0, cy0

    def _far_core_around(cx_s, cy_s, rad):
        r_loc = np.sqrt(((xx - cx_s) / hx) ** 2 + ((yy - cy_s) / hy) ** 2)
        core = valid & (r_loc <= rad)
        outer = valid & (r_loc >= 0.55) & (r_loc <= 0.90)
        if int(outer.sum()) < 8:
            outer = valid & (r_loc >= 0.45) & (r_loc <= 0.95)
        if int(outer.sum()) >= 8:
            near_ref = float(np.percentile(patch[outer], 30.0))
        else:
            near_ref = float(np.percentile(patch[valid], 25.0))
        far = core & (patch >= near_ref + hole_sep)
        if int(far.sum()) < 6 and int(core.sum()) >= 8:
            core_med = float(np.median(patch[core]))
            if core_med >= near_ref + max(5.0, hole_sep * 0.5):
                far = core & (patch >= core_med - max(3.0, hole_sep * 0.3))
            else:
                far = core & (patch >= near_ref + max(6.0, hole_sep * 0.7))
        return far, near_ref

    far_core, _near_ref = _far_core_around(cx0, cy0, search_frac)
    if int(far_core.sum()) >= 6:
        cx, cy, r85 = _centroid_and_radius_norm(
            xx, yy, far_core, hx, hy, cx0, cy0, max_shift_norm=max_shift,
        )
        if r85 is not None:
            r_hole_norm = float(np.clip(r85 + hole_margin, 0.18, 0.78))
        # 二遍：以新环心扩大搜索，收全大孔（透视椭圆）
        rad2 = float(np.clip(max(search_frac, r_hole_norm * 1.45, 0.62), 0.40, 0.92))
        far2, _ = _far_core_around(cx, cy, rad2)
        if int(far2.sum()) >= 6:
            cx2, cy2, r85b = _centroid_and_radius_norm(
                xx, yy, far2, hx, hy, cx0, cy0, max_shift_norm=max_shift,
            )
            cx, cy = cx2, cy2
            if r85b is not None:
                r_hole_norm = float(np.clip(r85b + hole_margin, 0.18, 0.78))

    center_u = float(x_lo + cx)
    center_v = float(y_lo + cy)
    meta["center_uv"] = (center_u, center_v)
    meta["u"], meta["v"] = center_u, center_v
    meta["hole_r"] = float(r_hole_norm * min(hx, hy))  # 像素近似

    # ── 金属环 mask ──
    r = np.sqrt(((xx - cx) / hx) ** 2 + ((yy - cy) / hy) ** 2)
    r_out = float(np.clip(p.get("ring_outer_frac", 0.92), r_hole_norm + 0.08, 1.25))
    ring = valid & (r > r_hole_norm) & (r <= r_out)
    if int(ring.sum()) < max(8, int(p["min_pix"])):
        # 略放宽
        ring = valid & (r > max(0.12, r_hole_norm * 0.85)) & (r <= min(1.15, r_out + 0.08))
    if int(ring.sum()) < 6:
        meta["rejected"] = "no_metal_ring"
        return center_u, center_v, None, meta

    # 孔底/桌面深度：环上只保留「明显比孔近」的像素 → 顶面；否则易把孔漏桌面当成顶面
    # （小螺母金属点少时，24cm 顶面 vs 26cm 桌面会整段跳变 ~螺母高度）
    hole_depth = None
    r_hole_depth = float(np.clip(max(r_hole_norm * 1.15, 0.42), 0.28, 0.82))
    hole_mask = valid & (r <= r_hole_depth)
    if int(far_core.sum()) >= 6:
        hole_mask = hole_mask | far_core
    # 相对环带近端：孔应更远
    ring_near_ref = float(np.percentile(patch[ring], 25.0))
    hole_cand = hole_mask & (patch >= ring_near_ref + max(5.0, hole_sep * 0.55))
    if int(hole_cand.sum()) >= 6:
        hole_depth = float(np.median(patch[hole_cand]))
    elif int(hole_mask.sum()) >= 8:
        hm = float(np.median(patch[hole_mask]))
        if hm >= ring_near_ref + max(4.0, hole_sep * 0.4):
            hole_depth = hm
    meta["hole_mm"] = float(hole_depth) if hole_depth is not None else None

    near_vs_hole = None
    if hole_depth is not None:
        near_cut = float(hole_depth) - max(6.0, hole_sep * 0.65)
        near_vs_hole = ring & (patch <= near_cut)
        n_near = int(near_vs_hole.sum())
        if n_near >= max(5, int(p["min_pix"]) // 2):
            ring = near_vs_hole
        elif n_near < 3:
            # 看不见比孔近的金属面 → 拒帧（勿输出桌面深度）
            meta["rejected"] = "no_near_vs_hole"
            meta["table_mm"] = float(hole_depth)
            return center_u, center_v, None, meta

    r_peak = 0.5 * (r_hole_norm + r_out)
    sig = max(0.05, float(p.get("ring_weight_sigma", 0.18)))
    gw = np.exp(-((r - r_peak) ** 2) / (2.0 * sig * sig))
    vals = patch[ring].astype(np.float64)
    wts = gw[ring].astype(np.float64)
    meta["n_valid"] = int(vals.size)
    meta["fill"] = float(vals.size) / float(max(int(ring.sum()), 1))

    # 先找峰再去野值：整环 MAD 会把少数顶面峰当离群点杀掉（孔漏桌面占优时）
    band = float(p.get("surface_band_mm", p.get("sector_band_mm", 12.0)))
    peaks = _depth_histogram_peaks(vals, wts, bin_w=max(6.0, band * 0.65))
    if not peaks:
        meta["rejected"] = "no_peaks"
        return center_u, center_v, None, meta

    w_sum = float(np.sum(wts)) + 1e-9
    min_mass_frac = 0.10
    min_layer = float(p.get("min_layer_sep_mm", 8.0))
    if prior_mm is not None and float(prior_mm) > 0:
        min_layer = max(min_layer, min(14.0, float(prior_mm) * 0.04))

    # 近峰 = 顶面：即使远峰(桌面)质量更大也优先最近峰（孔漏桌面时远峰常更强）
    top_c, top_m = peaks[0]
    if hole_depth is not None and (float(hole_depth) - top_c) < max(5.0, min_layer * 0.55):
        # 近峰几乎就是孔/桌面 → 金属顶面丢失
        meta["rejected"] = "peak_is_hole_or_table"
        meta["table_mm"] = float(hole_depth)
        return center_u, center_v, None, meta
    # 无孔深估计时：若环上深度跨度像「一层螺母高」但近峰极弱，拒掉远单峰
    if hole_depth is None and len(peaks) == 1:
        spread = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        if spread >= max(10.0, min_layer) and top_m / w_sum < 0.22:
            near_p = float(np.percentile(vals, 25.0))
            if (top_c - near_p) >= max(8.0, min_layer * 0.7):
                meta["rejected"] = "unimodal_far_dominant"
                meta["table_mm"] = float(top_c)
                return center_u, center_v, None, meta
            # 有近端支撑：改用近分位作顶面种子
            top_c = near_p
            top_m = float(np.sum(wts[np.abs(vals - top_c) <= band * 1.2]))
    if len(peaks) >= 2 and (peaks[1][0] - top_c) >= min_layer:
        band = min(band, max(6.0, 0.40 * float(peaks[1][0] - top_c)))

    top_mask = np.abs(vals - top_c) <= band * 1.15
    top_n = int(top_mask.sum())
    # 近峰只要有绝对像素支持即可；勿因远峰更强而改选桌面
    top_strong = (top_n >= max(4, int(p["min_pix"]) // 3)) or (
        top_m / w_sum >= 0.04
    )
    if not top_strong and len(peaks) >= 2:
        # 近峰几乎无支撑 → 拒帧（避免把桌面当顶面）
        meta["rejected"] = "near_peak_too_weak"
        # 仍记录远峰供诊断
        far_peaks = [pk for pk in peaks[1:] if pk[0] - top_c >= min_layer]
        if far_peaks:
            tc, _ = max(far_peaks, key=lambda t: t[1])
            meta["table_mm"] = float(tc)
        elif hole_depth is not None:
            meta["table_mm"] = float(hole_depth)
        return center_u, center_v, None, meta

    if top_n >= 6:
        local = vals[top_mask]
        local_w = wts[top_mask]
        keep = _mad_inlier_mask(local)
        if int(keep.sum()) >= 3:
            top_mm_pre = _weighted_median(local[keep], local_w[keep])
        else:
            top_mm_pre = _weighted_median(local, local_w)
    else:
        top_mm_pre = (
            _weighted_median(vals[top_mask], wts[top_mask]) if top_n >= 3 else float(top_c)
        )

    table_mm = None
    far_peaks = [pk for pk in peaks[1:] if pk[0] - top_c >= min_layer]
    if far_peaks:
        # 取质量最大的远峰作桌面参考
        table_c, table_m = max(far_peaks, key=lambda t: t[1])
        tmask = np.abs(vals - table_c) <= band * 1.15
        if (table_m / w_sum >= 0.08) or (int(tmask.sum()) >= 3):
            if int(tmask.sum()) >= 3:
                table_mm = _weighted_median(vals[tmask], wts[tmask])
            else:
                table_mm = float(table_c)
    if not top_strong:
        meta["rejected"] = "mixed_or_table_only"
        meta["table_mm"] = table_mm
        return center_u, center_v, None, meta
    if table_mm is None and len(peaks) == 1:
        if prior_mm is not None and float(prior_mm) > 0:
            if top_c > float(prior_mm) + max(40.0, float(prior_mm) * 0.25):
                meta["rejected"] = "single_peak_too_far"
                return center_u, center_v, None, meta

    top_mm = float(top_mm_pre) if top_mm_pre is not None else float(top_c)
    if top_mm is None or not np.isfinite(top_mm):
        meta["rejected"] = "top_median"
        return center_u, center_v, None, meta

    # ── 3) 扇区共识（只验顶面 band）──
    ang = np.arctan2(yy - cy, xx - cx)
    sector = (np.floor((ang + np.pi) / (2.0 * np.pi / 6.0))).astype(np.int32) % 6
    sector_medians = []
    if bool(p.get("ring_use_sectors", True)):
        for s in range(6):
            # 扇区内且深度落入顶面 band 的环像素
            sm = ring & (sector == s) & (np.abs(patch - top_mm) <= band * 1.25)
            if int(sm.sum()) < 2:
                continue
            sv = patch[sm].astype(np.float64)
            sw = gw[sm].astype(np.float64)
            med = _weighted_median(sv, sw)
            if med is not None and np.isfinite(med):
                sector_medians.append(float(med))
    meta["n_sectors"] = len(sector_medians)
    spread_max = float(p.get("sector_spread_max_mm", 16.0))
    conf = 0.55
    if len(sector_medians) >= 3:
        spread = float(max(sector_medians) - min(sector_medians))
        if spread > spread_max * 1.8:
            meta["rejected"] = "sector_spread"
            meta["top_mm"] = float(top_mm)
            meta["table_mm"] = table_mm
            return center_u, center_v, None, meta
        # 扇区中位数再中位，抗单扇区漏桌面
        top_mm = float(np.median(sector_medians))
        conf = float(np.clip(1.0 - spread / max(spread_max, 1.0), 0.40, 1.0))
        if spread > spread_max:
            conf = min(conf, 0.38)
    elif len(sector_medians) >= 1:
        conf = 0.42
    else:
        conf = 0.36 if top_strong else 0.25

    if table_mm is not None and (table_mm - top_mm) >= min_layer:
        conf = min(1.0, conf + 0.08)  # 双层清晰加分
        meta["table_mm"] = float(table_mm)
    else:
        meta["table_mm"] = float(table_mm) if table_mm is not None else None

    # 相对孔底仍不够近 → 当桌面拒掉（比软融合更硬，防 24↔26cm 层跳）
    if hole_depth is not None and (float(hole_depth) - top_mm) < max(5.0, hole_sep * 0.5):
        meta["rejected"] = "top_too_close_to_hole"
        meta["table_mm"] = float(hole_depth)
        meta["top_mm"] = float(top_mm)
        return center_u, center_v, None, meta

    # prior 软抑制飙远（层差尺度，勿用 70mm 大容差放过桌面）
    if prior_mm is not None and float(prior_mm) > 0:
        prior_f = float(prior_mm)
        up_tol = float(p.get("prior_up_tol_mm", 16.0))
        up_tol = max(12.0, min(up_tol, max(14.0, prior_f * 0.06)))
        if top_mm > prior_f + up_tol:
            # 典型顶面→桌面：偏向 prior，并降置信
            top_mm = 0.15 * top_mm + 0.85 * prior_f
            conf = min(conf, 0.35)
            meta["rejected"] = "soft_far_blend"

    # ── 4) 已知顶面后，用「顶面+sep 的远核」再收一次环心（大号透视更稳）──
    r_now = np.sqrt(((xx - cx) / hx) ** 2 + ((yy - cy) / hy) ** 2)
    hole_final = valid & (patch >= float(top_mm) + hole_sep) & (
        r_now <= float(np.clip(max(r_hole_norm * 1.55, 0.55), 0.40, 0.88))
    )
    if int(hole_final.sum()) >= 8:
        cx_f, cy_f, r85f = _centroid_and_radius_norm(
            xx, yy, hole_final, hx, hy, cx0, cy0, max_shift_norm=max_shift,
        )
        # 仅当质心相对当前中心有意义位移时更新（避免噪声抖）
        if math.hypot(cx_f - cx, cy_f - cy) >= 0.8:
            cx, cy = cx_f, cy_f
            center_u = float(x_lo + cx)
            center_v = float(y_lo + cy)
            meta["center_uv"] = (center_u, center_v)
            meta["u"], meta["v"] = center_u, center_v
            if r85f is not None:
                r_hole_norm = float(np.clip(r85f + hole_margin, 0.18, 0.78))
                meta["hole_r"] = float(r_hole_norm * min(hx, hy))
            meta["center_refined"] = "post_top_hole"

    meta["estimate"] = float(top_mm)
    meta["top_mm"] = float(top_mm)
    meta["confidence"] = float(conf)
    meta["layer"] = "top"
    meta["hole_r_norm"] = float(r_hole_norm)
    return center_u, center_v, float(top_mm), meta


def _extract_hollow_ring_patch(depth, box, grasp_u, grasp_v, p):
    """
    提取环带 patch + 每像素扇区编号（0~5）。
    返回 dict 或 None。
    """
    margin_frac = float(p.get("margin_frac", 0.10))
    hole_inner = float(p.get("hole_inner_frac", 0.36))
    ring_outer = float(p.get("ring_outer_frac", 0.92))
    ring_peak = float(p.get("ring_peak_frac", 0.62))
    ring_sigma = float(p.get("ring_weight_sigma", 0.18))
    ksize = int(p.get("spatial_median_ksize", 0))

    x1, y1, x2, y2 = map(int, box[:4])
    h, w = depth.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    u = float(grasp_u if grasp_u is not None else (x1 + x2) * 0.5)
    v = float(grasp_v if grasp_v is not None else (y1 + y2) * 0.5)
    if x2 <= x1 or y2 <= y1:
        return None

    d = depth.astype(np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    mx = max(0, int(bw * max(0.0, margin_frac)))
    my = max(0, int(bh * max(0.0, margin_frac)))
    y_lo, y_hi = y1 + my, y2 - my
    x_lo, x_hi = x1 + mx, x2 - mx
    if y_hi - y_lo < 4 or x_hi - x_lo < 4:
        y_lo, y_hi, x_lo, x_hi = y1, y2, x1, x2
    patch = d[y_lo:y_hi, x_lo:x_hi].copy()
    if patch.size < 9:
        return None

    if ksize >= 3:
        k = int(ksize) | 1
        patch = cv2.medianBlur(patch.astype(np.uint16), k).astype(np.float32)

    ph, pw = patch.shape
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float64)
    cx, cy = (pw - 1) * 0.5, (ph - 1) * 0.5
    hx, hy = max(cx, 1.0), max(cy, 1.0)
    r = np.sqrt(((xx - cx) / hx) ** 2 + ((yy - cy) / hy) ** 2)

    r_in = float(np.clip(hole_inner, 0.05, 0.88))
    r_out = float(np.clip(ring_outer, r_in + 0.05, 1.35))
    r_peak = float(np.clip(ring_peak, r_in, r_out))
    sig = max(0.05, ring_sigma)

    ring = (r >= r_in) & (r <= r_out)
    if int(ring.sum()) < 8:
        ring = (r >= max(0.18, r_in * 0.55)) & (r <= min(1.12, r_out + 0.06))
    if int(ring.sum()) < 6:
        return None

    gw = np.exp(-((r - r_peak) ** 2) / (2.0 * sig * sig))
    gw = np.where(ring, gw, 0.0)

    ang = np.arctan2(yy - cy, xx - cx)
    sector = (np.floor((ang + np.pi) / (2.0 * np.pi / 6.0))).astype(np.int32) % 6

    return {
        "u": u, "v": v, "patch": patch, "ring": ring, "weights": gw,
        "sector": sector, "n_ring": float(max(int(ring.sum()), 1)),
    }


def _filter_depth_samples(vals, weights, min_mm=80.0, max_mm=900.0, min_count=3):
    """去掉无效/空洞深度，避免 0~10mm 进入环带或融合。"""
    if vals is None or vals.size < 1:
        return None, None
    v = np.asarray(vals, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel() if weights is not None else np.ones_like(v)
    if w.size != v.size:
        w = np.ones_like(v)
    ok = (v >= float(min_mm)) & (v <= float(max_mm)) & np.isfinite(v)
    if int(ok.sum()) < int(min_count):
        return None, None
    return v[ok], w[ok]


def _sector_medians_from_ring(patch, ring, sector, weights, min_sector_pix=2, min_mm=80.0, max_mm=900.0):
    """六扇区各自加权中位数 → 抗某一方向孔洞漏光。"""
    medians = []
    for s in range(6):
        m = ring & (sector == s)
        if int(m.sum()) < min_sector_pix:
            continue
        vals, w = _filter_depth_samples(
            patch[m], weights[m], min_mm=min_mm, max_mm=max_mm, min_count=2,
        )
        if vals is None:
            continue
        d = _weighted_median(vals, w)
        if d is not None and np.isfinite(d) and min_mm <= float(d) <= max_mm:
            medians.append(float(d))
    return medians


def _hollow_ring_depth_estimate(depth, box, grasp_u, grasp_v, p, prior_mm=None):
    """空心六角螺母：环带 + 六扇区 + 平面峰；失败返回 (u,v,None,meta)。"""
    pack = _extract_hollow_ring_patch(depth, box, grasp_u, grasp_v, p)
    meta = {
        "u": grasp_u, "v": grasp_v, "n_valid": 0, "fill": 0.0,
        "estimate": None, "mode": "hollow_ring", "rejected": None,
        "confidence": 0.0,
    }
    if pack is None:
        meta["rejected"] = "no_ring"
        u = float(grasp_u if grasp_u is not None else (box[0] + box[2]) * 0.5)
        v = float(grasp_v if grasp_v is not None else (box[1] + box[3]) * 0.5)
        return u, v, None, meta

    patch = pack["patch"]
    ring = pack["ring"]
    weights = pack["weights"]
    sector = pack["sector"]
    vals = patch[ring].astype(np.float32)
    wts = weights[ring].astype(np.float32)
    min_v = float(p.get("min_valid_mm", 80.0))
    max_v = float(p.get("max_valid_mm", 900.0))
    vals, wts = _filter_depth_samples(vals, wts, min_mm=min_v, max_mm=max_v, min_count=4)
    if vals is None:
        meta["rejected"] = "ring_invalid_depth"
        return pack["u"], pack["v"], None, meta
    meta["u"], meta["v"] = pack["u"], pack["v"]
    meta["n_ring"] = int(pack["n_ring"])

    use_sectors = bool(p.get("ring_use_sectors", True))
    sector_medians = (
        _sector_medians_from_ring(
            patch, ring, sector, weights, min_mm=min_v, max_mm=max_v,
        )
        if use_sectors else None
    )

    band = float(p.get("cluster_band_mm", 12.0))
    if use_sectors and sector_medians:
        band = min(band, float(p.get("sector_band_mm", 14.0)))

    est, conf = _ring_plane_depth_mm(
        vals,
        wts,
        band_mm=band,
        prior_mm=prior_mm,
        min_pixels=int(p.get("min_pix", 10)),
        sector_spread_max_mm=float(p.get("sector_spread_max_mm", 22.0)),
        sector_medians=sector_medians,
    )
    meta["n_valid"] = int(vals.size)
    meta["fill"] = float(vals.size) / max(pack["n_ring"], 1.0)
    meta["confidence"] = float(conf)
    meta["n_sectors"] = len(sector_medians or [])
    if est is not None:
        meta["estimate"] = float(est)
    else:
        meta["rejected"] = "ring_plane"
    return pack["u"], pack["v"], est, meta


def estimate_box_depth(
    depth,
    box,
    prior_mm=None,
    freeze_below_mm=150.0,
    grasp_u=None,
    grasp_v=None,
    roi_mode=None,
    outlier_mm=20.0,
    eye=None,
    min_valid_mm=None,
    max_valid_mm=None,
    min_fill_ratio=None,
    min_pixels=None,
    return_meta=False,
):
    """
    框内深度 (mm)。默认 nut_surface：孔核精修 + 顶面/桌面双层。

    roi_mode:
      - nut_surface / surface: 金属顶面（推荐）
      - hollow_ring / annulus / hex_ring: 旧椭圆环带
      - foreground / weighted / median / trimmed: 兼容旧逻辑

    不再因 prior≤freeze 停止采样（硬冻结会 UV 更新、深度卡死 → 抓歪）。
    prior 仅用于拒绝「突然飙远」的飞点像素。
    """
    p = _depth_eye_params(eye)
    if roi_mode is None or str(roi_mode).strip() == "":
        roi_mode = p["roi_mode"]
    roi_mode = str(roi_mode).strip().lower()
    outlier_mm = float(outlier_mm if outlier_mm is not None else p["outlier_mm"])
    min_valid = float(min_valid_mm if min_valid_mm is not None else p["min_valid_mm"])
    max_valid = float(max_valid_mm if max_valid_mm is not None else p["max_valid_mm"])
    min_fill = float(min_fill_ratio if min_fill_ratio is not None else p["min_fill"])
    min_pix = int(min_pixels if min_pixels is not None else p["min_pix"])

    use_surface = roi_mode in ("nut_surface", "surface", "top_surface", "nut_top")
    if use_surface:
        u, v, estimate, surf_meta = estimate_nut_surface_depth(
            depth, box, grasp_u, grasp_v, eye=eye, prior_mm=prior_mm,
        )
        meta = dict(surf_meta)
        meta["mode"] = roi_mode
        if estimate is not None and np.isfinite(estimate):
            estimate = float(estimate)
            if estimate < min_valid or estimate > max_valid:
                estimate = None
                meta["rejected"] = "surface_out_of_range"
                meta["layer"] = "rejected"
        if estimate is not None and np.isfinite(estimate):
            meta["estimate"] = float(estimate)
            return (u, v, float(estimate), meta) if return_meta else (u, v, float(estimate))
        # 顶面失败 → 回退 hollow_ring（仍禁整框桌面）
        meta["rejected"] = surf_meta.get("rejected") or "surface_fail_to_ring"
        u, v, estimate, ring_meta = _hollow_ring_depth_estimate(
            depth, box, grasp_u, grasp_v, p, prior_mm=prior_mm,
        )
        if estimate is not None and np.isfinite(estimate):
            estimate = float(estimate)
            if min_valid <= estimate <= max_valid:
                meta.update({
                    "estimate": estimate,
                    "confidence": float(ring_meta.get("confidence", 0.35)),
                    "n_valid": ring_meta.get("n_valid", 0),
                    "fill": ring_meta.get("fill", 0.0),
                    "n_sectors": ring_meta.get("n_sectors", 0),
                    "layer": "top_ring_fallback",
                    "fallback": "hollow_ring",
                })
                return (u, v, estimate, meta) if return_meta else (u, v, estimate)
        return (u, v, None, meta) if return_meta else (u, v, None)

    use_ring = roi_mode in (
        "hollow_ring", "annulus", "hex_ring", "ring", "hollow", "nut_ring",
    )
    if use_ring:
        u, v, estimate, ring_meta = _hollow_ring_depth_estimate(
            depth, box, grasp_u, grasp_v, p, prior_mm=prior_mm,
        )
        meta = {
            "u": u, "v": v, "n_valid": ring_meta.get("n_valid", 0),
            "fill": ring_meta.get("fill", 0.0), "estimate": None,
            "mode": roi_mode, "rejected": ring_meta.get("rejected"),
            "confidence": float(ring_meta.get("confidence", 0.0)),
            "n_sectors": ring_meta.get("n_sectors", 0),
        }
        if estimate is not None and np.isfinite(estimate):
            estimate = float(estimate)
            if estimate < min_valid or estimate > max_valid:
                estimate = None
                meta["rejected"] = "ring_out_of_range"
        if estimate is not None and np.isfinite(estimate):
            estimate = float(estimate)
            if prior_mm is not None and float(prior_mm) > 0:
                prior_f = float(prior_mm)
                up_tol = max(90.0, prior_f * 0.40)
                if estimate > prior_f + up_tol:
                    estimate = 0.15 * estimate + 0.85 * prior_f
                    meta["rejected"] = "soft_far_blend"
            meta["estimate"] = estimate
            return (u, v, estimate, meta) if return_meta else (u, v, estimate)
        # 环失败：带中心孔遮罩的 foreground（禁止整框含孔）
        meta["rejected"] = ring_meta.get("rejected") or "ring_fail_masked_fg"
        u, v, vals_all, weights_all, n_total = _extract_depth_roi_hollow_ring(
            depth, box, grasp_u, grasp_v,
            margin_frac=min(p["margin_frac"], 0.12),
            hole_inner_frac=max(p["hole_inner_frac"], 0.40),
            ring_outer_frac=p["ring_outer_frac"],
            ring_peak_frac=p["ring_peak_frac"],
            ring_weight_sigma=p["ring_weight_sigma"],
        )
    else:
        u, v, vals_all, weights_all, n_total = _extract_depth_roi(
            depth, box, grasp_u, grasp_v, margin_frac=p["margin_frac"],
        )
    meta = {
        "u": u, "v": v, "n_valid": 0, "fill": 0.0,
        "estimate": None, "mode": roi_mode, "rejected": None,
        "confidence": 0.0,
    }
    if n_total < 1 or vals_all.size < 1:
        return (u, v, None, meta) if return_meta else (u, v, None)

    # 环模式：只保留权重>0 的环带像素（中心孔已置零）
    if use_ring:
        on_ring = weights_all > 1e-8
        vals_all = vals_all[on_ring]
        weights_all = weights_all[on_ring]
        n_total = float(max(vals_all.size, 1))

    ok = (vals_all >= min_valid) & (vals_all <= max_valid)
    # prior：丢掉明显远于记忆的背景像素（勿整段停采样）
    if prior_mm is not None and float(prior_mm) > 0:
        prior_f = float(prior_mm)
        far_cut = prior_f * 1.40 + 80.0
        # 近距记忆时仍允许小幅回退变远，但挡住墙面跳变
        if prior_f <= float(freeze_below_mm) + 40.0:
            far_cut = min(far_cut, prior_f + max(70.0, prior_f * 0.35))
        ok &= vals_all <= far_cut

    vals = vals_all[ok]
    weights = weights_all[ok]
    fill = float(vals.size) / float(max(n_total, 1))
    meta["n_valid"] = int(vals.size)
    meta["fill"] = fill

    if vals.size < max(3, min_pix) or fill < min_fill:
        meta["rejected"] = "sparse"
        return (u, v, None, meta) if return_meta else (u, v, None)

    if roi_mode == "median":
        med = float(np.median(vals))
        band = float(max(5.0, outlier_mm))
        keep = np.abs(vals - med) <= band
        if int(keep.sum()) >= 3:
            vals = vals[keep]
        estimate = float(np.median(vals))
    elif roi_mode == "trimmed":
        order = np.argsort(vals)
        vals_s = vals[order]
        trim = max(1, int(vals_s.size * 0.10))
        if vals_s.size > trim * 2 + 2:
            vals_s = vals_s[trim:-trim]
        estimate = float(np.mean(vals_s))
    elif roi_mode == "weighted":
        order = np.argsort(vals)
        vals_s, w_s = vals[order], weights[order]
        keep_n = max(3, int(vals_s.size * 0.75))
        vals_s, w_s = vals_s[:keep_n], w_s[:keep_n]
        w_s = w_s / (w_s.sum() + 1e-9)
        estimate = float(np.sum(vals_s * w_s))
    else:
        # foreground / 环失败遮罩回退
        if use_ring and weights is not None and weights.size == vals.size:
            est, conf = _ring_plane_depth_mm(
                vals, weights,
                band_mm=min(p["cluster_band_mm"], 14.0),
                prior_mm=prior_mm,
                min_pixels=max(6, min_pix // 2),
            )
            estimate = est
            meta["confidence"] = float(conf)
        else:
            estimate = _foreground_cluster_depth_mm(
                vals,
                weights,
                band_mm=p["cluster_band_mm"],
                min_mass=p["cluster_min_mass"],
                near_percentile=p["near_percentile"],
                min_pixels=max(6, min_pix // 2),
            )
            meta["confidence"] = 0.55 if estimate is not None else 0.0

    if estimate is None or not np.isfinite(estimate):
        meta["rejected"] = "cluster"
        return (u, v, None, meta) if return_meta else (u, v, None)

    estimate = float(estimate)
    # 相对 prior 的软融合：仅抑制飙远，不阻断近距重测
    if prior_mm is not None and float(prior_mm) > 0:
        prior_f = float(prior_mm)
        up_tol = max(90.0, prior_f * 0.40)
        if estimate > prior_f + up_tol:
            estimate = 0.15 * estimate + 0.85 * prior_f
            meta["rejected"] = "soft_far_blend"

    meta["estimate"] = estimate
    if "confidence" not in meta or meta["confidence"] <= 0:
        meta["confidence"] = 0.6 if estimate is not None else 0.0
    return (u, v, estimate, meta) if return_meta else (u, v, estimate)


# 兼容旧名
box_center_and_depth = estimate_box_depth


def measure_detection_uvd(
    depth_image,
    detection,
    eye,
    prior_mm=None,
    freeze_below_mm=150.0,
    return_meta=False,
    R_world_cam=None,
    image_up_uv=None,
):
    """抓取像素 + ROI 深度 → (u, v, depth_mm[, meta])。

    hand_top 时传入 R_world_cam（或 image_up_uv）按世界上方选边。
    nut_surface 默认用深度精修环心作抓取 UV。
    """
    if depth_image is None or detection is None:
        if return_meta:
            return None, None, None, {"rejected": "no_input"}
        return None, None, None
    box = detection[:4]
    mode = eye.get("grasp_point_mode", "bbox_center")
    top_frac = float(eye.get("grasp_point_top_frac", 0.35))
    side = eye.get("grasp_point_side", "right")
    roi_mode = eye.get("depth_roi_mode", "nut_surface")
    outlier_mm = float(eye.get("depth_outlier_mm", 20.0))
    freeze = float(
        freeze_below_mm
        if freeze_below_mm is not None
        else eye.get("depth_freeze_below_mm", 150.0)
    )
    img_up = image_up_uv
    if img_up is None and R_world_cam is not None:
        img_up = world_up_in_image_uv(
            R_world_cam,
            eye.get("pinch_world_up", [0.0, 0.0, 1.0]),
        )
    u_grasp, v_grasp = grasp_pixel_from_box(
        box,
        mode,
        top_frac,
        side=side,
        image_up_uv=img_up,
    )
    u_d, v_d = grasp_depth_sample_pixel(box, u_grasp, v_grasp, mode, eye)
    # 始终取 meta：nut_surface 的 center_uv 才是环心
    u_est, v_est, depth_mm, meta = estimate_box_depth(
        depth_image,
        box,
        prior_mm=prior_mm,
        freeze_below_mm=freeze,
        grasp_u=u_d,
        grasp_v=v_d,
        roi_mode=roi_mode,
        outlier_mm=outlier_mm,
        eye=eye,
        return_meta=True,
    )
    if meta is None:
        meta = {}

    # 精修环心 → 对准/锁物 UV（只信深度）
    use_ring = bool(eye.get("grasp_use_ring_center", True))
    roi_l = str(roi_mode or "").strip().lower()
    ring_roi = roi_l in (
        "nut_surface", "surface", "top_surface", "nut_top",
        "hollow_ring", "annulus", "hex_ring", "ring", "hollow", "nut_ring",
    )
    if use_ring and ring_roi:
        cu = cv = None
        cuv = meta.get("center_uv")
        if isinstance(cuv, (tuple, list)) and len(cuv) >= 2:
            cu, cv = float(cuv[0]), float(cuv[1])
        elif meta.get("u") is not None:
            cu, cv = float(meta["u"]), float(meta["v"])
        elif u_est is not None:
            cu, cv = float(u_est), float(v_est)

        if cu is not None and np.isfinite(cu) and np.isfinite(cv):
            x1, y1, x2, y2 = map(float, box[:4])
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            max_px = float(eye.get("grasp_ring_center_max_shift_px", 0.0))
            if max_px <= 1.0:
                max_px = 0.35 * min(bw, bh)
            du = float(cu) - float(u_grasp)
            dv = float(cv) - float(v_grasp)
            if math.hypot(du, dv) <= max_px:
                u_grasp, v_grasp = float(cu), float(cv)
                meta["grasp_uv_source"] = "ring_center"
            else:
                meta["grasp_uv_source"] = "bbox_center_shift_capped"
                meta["ring_center_shift_px"] = float(math.hypot(du, dv))
        else:
            meta["grasp_uv_source"] = "bbox_center"
    else:
        meta["grasp_uv_source"] = str(mode or "bbox_center")

    meta["grasp_u"] = float(u_grasp)
    meta["grasp_v"] = float(v_grasp)

    if return_meta:
        return u_grasp, v_grasp, depth_mm, meta
    return u_grasp, v_grasp, depth_mm


def depth_acceptable_for_action(depth_mm, eye, soft_prior_mm=None):
    """
    对准/夹取前是否接受该深度。

    注意：正常工作距常在 12–16cm，切勿用 depth_freeze/retreat_min=170
    把有效近距拒掉（会导致对准中止或误采背景墙）。
    只拒：空洞、异常贴脸噪声、过远背景、相对上次可靠值飙远。
    另：相对可靠值「远一层」(约螺母高度 12–40mm) 视为孔/桌面误采，拒掉并重测。
    返回 (ok: bool, reason: str)。
    """
    if depth_mm is None or not np.isfinite(depth_mm):
        return False, "无有效深度（可能过近/空洞）"
    d = float(depth_mm)
    min_mm = float(eye.get("depth_action_min_mm", eye.get("depth_min_valid_mm", 80.0)))
    max_mm = float(eye.get("depth_action_max_mm", 300.0))
    if d < min_mm:
        return False, f"深度 {d/10:.1f}cm 异常过近（<{min_mm/10:.0f}cm）"
    if d > max_mm:
        return False, f"深度 {d/10:.1f}cm 过远疑似背景（需≤{max_mm/10:.0f}cm）"
    if soft_prior_mm is not None and np.isfinite(soft_prior_mm):
        prior = float(soft_prior_mm)
        if min_mm <= prior <= max_mm:
            delta = d - prior
            # 顶面↔孔底/桌面：常见跳 ~15–30mm（小螺母高度量级）
            layer_lo = float(eye.get("depth_layer_reject_min_mm", 12.0))
            layer_hi = float(eye.get("depth_layer_reject_max_mm", 42.0))
            if layer_lo <= delta <= layer_hi:
                return False, (
                    f"深度 {d/10:.1f}cm 相对上次可靠 {prior/10:.1f}cm "
                    f"远一层(+{delta:.0f}mm)，疑似孔/桌面"
                )
            max_up = float(
                eye.get(
                    "depth_action_max_up_jump_mm",
                    eye.get("depth_max_up_jump_mm", 18.0),
                )
            )
            if d > prior + max_up:
                return False, (
                    f"深度 {d/10:.1f}cm 相对上次可靠 "
                    f"{prior/10:.1f}cm 飙远>{max_up:.0f}mm（疑似背景）"
                )
    return True, "ok"


def sanitize_action_depth_mm(depth_mm, eye, soft_prior_mm=None):
    """
    规划用深度：合格则返回 float，否则 (None, reason)。
    不做「假 fallback 220」——缺深度应中止，而不是当成远处物体猛冲。
    """
    ok, why = depth_acceptable_for_action(depth_mm, eye, soft_prior_mm=soft_prior_mm)
    if not ok:
        return None, why
    return float(depth_mm), "ok"


def consensus_depth_mm(samples, tol_mm=25.0, near_percentile=35.0):
    """
    多帧深度共识：极差超 tol 则失败。
    通过时取近端分位（默认 P35），避免偶发偏远帧把中位数拉向桌面。
    """
    vals = [
        float(x) for x in samples
        if x is not None and np.isfinite(x) and float(x) > 1.0
    ]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    arr = np.asarray(vals, dtype=np.float64)
    if float(arr.max() - arr.min()) > float(tol_mm):
        return None
    pct = float(np.clip(near_percentile, 15.0, 50.0))
    return float(np.percentile(arr, pct))


class AlignFusionBuffer:
    """对准稳定段多帧 (u,v,depth) 融合；自动丢弃过远/飞点深度。"""

    def __init__(
        self,
        max_samples=16,
        depth_spread_mm=80.0,
        max_depth_mm=300.0,
        min_depth_mm=80.0,
    ):
        self.max_samples = int(max(3, max_samples))
        self.depth_spread_mm = float(depth_spread_mm)
        self.max_depth_mm = float(max_depth_mm) if max_depth_mm else None
        self.min_depth_mm = float(min_depth_mm)
        self._samples = []

    def reset(self):
        self._samples = []

    def add(self, u, v, depth_mm):
        if u is None or v is None:
            return False
        if depth_mm is None or not np.isfinite(depth_mm):
            return False
        d = float(depth_mm)
        if d < self.min_depth_mm:
            return False
        if self.max_depth_mm is not None and d > self.max_depth_mm:
            return False
        self._samples.append((float(u), float(v), d))
        if len(self._samples) > self.max_samples:
            self._samples.pop(0)
        return True

    def ready(self):
        return len(self._samples) >= 1

    def __len__(self):
        return len(self._samples)

    def median(self):
        if not self._samples:
            return None, None, None
        arr = np.asarray(self._samples, dtype=np.float64)
        u = float(np.median(arr[:, 0]))
        v = float(np.median(arr[:, 1]))
        depths = arr[:, 2]
        valid = depths[
            (depths >= self.min_depth_mm)
            & (self.max_depth_mm is None or depths <= self.max_depth_mm)
        ]
        if valid.size < 1:
            return None, None, None
        depths = valid
        # 融合深度：近端簇 + 近分位，压制孔/桌面远层混入
        d_near = float(np.percentile(depths, 30.0))
        near = depths[depths <= d_near + self.depth_spread_mm]
        if near.size >= max(2, (len(depths) + 1) // 2):
            d = float(np.percentile(near, 35.0))
        else:
            d = float(np.percentile(depths, 25.0))
        # 若仍呈双峰（近端与远端差 ≥ 一层），强制近簇
        d_far = float(np.percentile(depths, 75.0))
        if (d_far - d_near) >= max(12.0, self.depth_spread_mm * 0.55):
            near2 = depths[depths <= d_near + max(8.0, self.depth_spread_mm * 0.4)]
            if near2.size >= 2:
                d = float(np.percentile(near2, 40.0))
        if self.max_depth_mm is not None and d > self.max_depth_mm:
            d = float(np.percentile(depths, 25.0))
        return u, v, d


def _size_class_lookup_keys(size_info=None, size_class=None):
    """归一化尺寸档 → yaml by_size 查找键列表。"""
    cls = None
    if isinstance(size_info, dict):
        cls = size_info.get("class")
    if cls is None and size_class is not None:
        cls = str(size_class).strip().lower() or None
    if isinstance(cls, str):
        cls = cls.strip().lower()
    if cls in ("big", "large"):
        return ["big", "large"]
    if cls:
        return [str(cls)]
    return []


def resolve_bool_by_size(
    eye,
    base_key,
    by_size_key,
    default=False,
    size_info=None,
    size_class=None,
):
    """bool 配置：优先 {base}_by_size.{class}，否则 base_key。"""
    eye = eye or {}
    base = bool(eye.get(base_key, default))
    by = eye.get(by_size_key) or {}
    if not isinstance(by, dict) or not by:
        return base
    for k in _size_class_lookup_keys(size_info=size_info, size_class=size_class):
        if k in by and by[k] is not None:
            return bool(by[k])
    return base


def resolve_pinch_level_finger_plane(eye=None, size_info=None, size_class=None) -> bool:
    """
    夹取是否强制指平面∥水平（EE Z∥重力）。
    优先 pinch_level_finger_plane_by_size；否则 pinch_level_finger_plane。
    """
    return resolve_bool_by_size(
        eye,
        "pinch_level_finger_plane",
        "pinch_level_finger_plane_by_size",
        default=False,
        size_info=size_info,
        size_class=size_class,
    )


def resolve_place_level_finger_plane(eye=None, size_info=None, size_class=None) -> bool:
    """
    放下是否强制指平面∥水平。
    优先 place_level_finger_plane_by_size；否则 place_level_finger_plane。
    """
    return resolve_bool_by_size(
        eye,
        "place_level_finger_plane",
        "place_level_finger_plane_by_size",
        default=True,
        size_info=size_info,
        size_class=size_class,
    )


def resolve_pinch_hand_short_m(eye=None, size_info=None) -> float:
    """
    实指短于 MJ 的法兰再降量 (m)。手长与螺母尺寸无关，三档应同值。
    优先 pinch_hand_short_by_size；否则 pinch_hand_short_m。
    高度差请调 pinch_grasp_below_*，勿用 hand_short 按尺寸「凑」。
    """
    eye = eye or {}
    base = float(eye.get("pinch_hand_short_m", 0.0) or 0.0)
    by = eye.get("pinch_hand_short_by_size") or {}
    if not isinstance(by, dict) or not by:
        return float(np.clip(base, 0.0, 0.08))
    keys = _size_class_lookup_keys(size_info=size_info)
    for k in keys:
        if k in by and by[k] is not None:
            try:
                return float(np.clip(float(by[k]), 0.0, 0.08))
            except (TypeError, ValueError):
                continue
    return float(np.clip(base, 0.0, 0.08))


def resolve_pinch_standoff_m(eye=None, size_info=None) -> float:
    """
    黄–青 standoff (m)：优先 pinch_standoff_by_size.{small,medium,big}，
    否则 pinch_standoff_m（默认 0.100）。大号约 0.150。
    """
    eye = eye or {}
    base = float(eye.get("pinch_standoff_m", 0.100))
    by = eye.get("pinch_standoff_by_size") or {}
    if not isinstance(by, dict) or not by:
        return float(np.clip(base, 0.0, 0.30))
    keys = _size_class_lookup_keys(size_info=size_info)
    for k in keys:
        if k in by and by[k] is not None:
            try:
                return float(np.clip(float(by[k]), 0.0, 0.30))
            except (TypeError, ValueError):
                continue
    return float(np.clip(base, 0.0, 0.30))


def resolve_pinch_below_surface_frac(eye=None, size_info=None) -> float:
    """
    顶面下偏 frac（×外径 AF）：优先 pinch_grasp_below_surface_frac_by_size，
    否则 pinch_grasp_below_surface_frac。小 0.7 / 中 0.6 / 大 0.5。
    """
    eye = eye or {}
    base = float(eye.get("pinch_grasp_below_surface_frac", 0.0) or 0.0)
    by = eye.get("pinch_grasp_below_surface_frac_by_size") or {}
    if not isinstance(by, dict) or not by:
        return float(np.clip(base, 0.0, 2.0))
    keys = _size_class_lookup_keys(size_info=size_info)
    for k in keys:
        if k in by and by[k] is not None:
            try:
                return float(np.clip(float(by[k]), 0.0, 2.0))
            except (TypeError, ValueError):
                continue
    return float(np.clip(base, 0.0, 2.0))


def resolve_pregrasp_depth_mm(depth_mm, eye, size_info=None):
    """
    预抓目标深度 (mm)。
    fixed: pinch_pregrasp_depth_mm
    adaptive: 随物体深度减去 standoff/余量，且不超过 fixed 上限。
    """
    fixed = float(eye.get("pinch_pregrasp_depth_mm", 200))
    mode = str(eye.get("pregrasp_depth_mode", "adaptive")).strip().lower()
    if mode != "adaptive" or depth_mm is None or not np.isfinite(depth_mm):
        return fixed
    # 动作深度上限：防止假远深把预抓行程拉到十几厘米
    action_max = float(eye.get("depth_action_max_mm", 320.0))
    depth_mm = min(float(depth_mm), action_max)
    standoff = resolve_pinch_standoff_m(eye, size_info) * 1000.0
    margin = float(eye.get("pregrasp_margin_mm", 80))
    safety = float(eye.get("pregrasp_safety_mm", 40))
    min_d = float(eye.get("pregrasp_min_depth_mm", 140))
    adaptive = float(depth_mm) - standoff - margin - safety
    return float(np.clip(adaptive, min_d, fixed))


class DepthTracker:
    """
    跨帧深度：EMA + 拒绝飙远；近距软保持（不阻断采样）。

    旧「硬冻结」会让 estimate 停采 → UV 更新、深度卡死 → 3D 锁歪。
    现在每帧仍采样；低置信或新值不合理时沿用记忆。
    """

    def __init__(
        self,
        freeze_below_mm=150.0,
        smooth=0.4,
        max_up_jump_mm=90.0,
        max_down_jump_mm=140.0,
        min_confidence=0.30,
        min_valid_mm=80.0,
        max_valid_mm=900.0,
    ):
        self.freeze_below_mm = float(freeze_below_mm)
        self.smooth = float(np.clip(smooth, 0.05, 0.95))
        self.max_up_jump_mm = float(max_up_jump_mm)
        self.max_down_jump_mm = float(max_down_jump_mm)
        self.min_confidence = float(min_confidence)
        self.min_valid_mm = float(min_valid_mm)
        self.max_valid_mm = float(max_valid_mm)
        self.mm = None
        self.frozen = False  # UI：近距软保持标记

    def reset(self):
        self.mm = None
        self.frozen = False

    @property
    def value(self):
        return self.mm

    @property
    def memory_mm(self):
        return self.mm

    def update(self, raw_mm, confidence=1.0):
        conf = float(confidence) if confidence is not None else 1.0
        if raw_mm is None or not np.isfinite(raw_mm):
            self.frozen = self.mm is not None and (
                self.mm < self.freeze_below_mm + 5.0
            )
            return self.mm, self.mm is not None

        raw_mm = float(raw_mm)
        if raw_mm < self.min_valid_mm or raw_mm > self.max_valid_mm:
            self.frozen = self.mm is not None
            return self.mm, self.mm is not None

        if self.mm is None:
            self.mm = raw_mm
            self.frozen = raw_mm < self.freeze_below_mm
            return self.mm, False

        up_lim = max(self.max_up_jump_mm, self.mm * 0.35)
        # 近距时更严：挡住回退后采到的背景墙
        if self.mm < self.freeze_below_mm + 50.0:
            up_lim = min(up_lim, max(55.0, self.max_up_jump_mm * 0.7))

        if raw_mm > self.mm + up_lim:
            self.frozen = True
            return self.mm, True

        if raw_mm < self.mm - self.max_down_jump_mm:
            # 突然过近多为空洞/噪声，沿用
            self.frozen = True
            return self.mm, True

        # 低置信时减小 EMA 权重，避免单帧拉动
        alpha = self.smooth * float(np.clip(conf, 0.35, 1.0))
        self.mm = alpha * raw_mm + (1.0 - alpha) * self.mm
        self.frozen = self.mm < self.freeze_below_mm
        return self.mm, False


class RingDepthBuffer:
    """
    环带深度多帧共识（类似 RealSense temporal filter 的思路，但只在 ROI 标量上）。

    连续 N 帧中位数 + 极差门限，抑制单帧孔洞漏采导致的跳变。
    """

    def __init__(
        self,
        maxlen=7,
        spread_tol_mm=18.0,
        min_samples=3,
        min_confidence=0.30,
    ):
        self.maxlen = int(max(3, maxlen))
        self.spread_tol_mm = float(spread_tol_mm)
        self.min_samples = int(max(1, min_samples))
        self.min_confidence = float(min_confidence)
        self._samples = []

    def reset(self):
        self._samples = []

    def push(self, depth_mm, confidence=1.0):
        if depth_mm is None or not np.isfinite(depth_mm):
            return self.value()
        conf = float(confidence) if confidence is not None else 1.0
        if conf < self.min_confidence:
            return self.value()
        self._samples.append((float(depth_mm), conf))
        if len(self._samples) > self.maxlen:
            self._samples.pop(0)
        return self.value()

    def value(self):
        if not self._samples:
            return None
        vals = np.asarray([s[0] for s in self._samples], dtype=np.float64)
        if vals.size < self.min_samples:
            return float(np.median(vals))
        spread = float(vals.max() - vals.min())
        if spread <= self.spread_tol_mm:
            return float(np.median(vals))
        # 极差过大：去掉最远 1 个再取中位数（常为偶发桌面）
        order = np.argsort(vals)
        trimmed = vals[order[:-1]] if vals.size >= 4 else vals
        return float(np.median(trimmed))


class SizeClassTracker:
    """
    尺寸定档抗干扰：raw 中值/EMA + 换档滞回 + 连续投票。

    不单看单帧 raw；换档须跨过阈值±hysteresis，且连续 vote_frames 帧同意。
    """

    def __init__(
        self,
        detection_cfg=None,
        maxlen=7,
        ema=0.35,
        jump_reject_mm=8.0,
        hysteresis_mm=2.0,
        vote_frames=3,
        min_samples=3,
    ):
        self.detection_cfg = detection_cfg
        self.maxlen = int(max(3, maxlen))
        self.ema = float(np.clip(ema, 0.05, 0.95))
        self.jump_reject_mm = float(jump_reject_mm)
        self.hysteresis_mm = float(max(0.0, hysteresis_mm))
        self.vote_frames = int(max(1, vote_frames))
        self.min_samples = int(max(1, min_samples))
        self._raws = []
        self._ema_raw = None
        self._stable_class = None
        self._pending_class = None
        self._pending_n = 0
        self._last_info = None

    def reset(self):
        self._raws = []
        self._ema_raw = None
        self._stable_class = None
        self._pending_class = None
        self._pending_n = 0
        self._last_info = None

    def configure(self, detection_cfg):
        self.detection_cfg = detection_cfg

    @property
    def smooth_raw_mm(self):
        return self._ema_raw

    @property
    def stable_class(self):
        return self._stable_class

    def _smooth_raw(self, raw_mm, ambiguous=False):
        if raw_mm is None or not np.isfinite(raw_mm):
            return self._ema_raw
        r = float(raw_mm)
        if self._ema_raw is not None and abs(r - self._ema_raw) > self.jump_reject_mm:
            # 单帧尖刺：仍记入短窗，但不立刻拉动 EMA
            self._raws.append(r)
            if len(self._raws) > self.maxlen:
                self._raws.pop(0)
            if len(self._raws) >= self.min_samples:
                med = float(np.median(self._raws[-self.min_samples :]))
                if abs(med - self._ema_raw) <= self.jump_reject_mm * 0.85:
                    alpha = self.ema * 0.45
                    self._ema_raw = alpha * med + (1.0 - alpha) * self._ema_raw
            return self._ema_raw

        self._raws.append(r)
        if len(self._raws) > self.maxlen:
            self._raws.pop(0)
        med = float(np.median(self._raws)) if self._raws else r
        alpha = self.ema * (0.55 if ambiguous else 1.0)
        if self._ema_raw is None:
            self._ema_raw = med
        else:
            self._ema_raw = alpha * med + (1.0 - alpha) * self._ema_raw
        return self._ema_raw

    def _classify_raw(self, raw_mm):
        return classify_nut_by_size_mm(
            float(raw_mm), None, self.detection_cfg,
        )

    def _accept_switch(self, candidate, smooth_raw):
        """滞回：相对当前档，须越过分界±hysteresis 才允许换档。"""
        if self._stable_class is None or candidate is None:
            return True
        if candidate == self._stable_class:
            return True
        cfg = _size_metric_cfg(self.detection_cfg)
        nom, _space = _nominal_hex_for_classify(cfg)
        ordered, thrs = _midpoint_thresholds(nom, cfg)
        # 找稳定档与候选在序中的位置
        keys = [k for k, _ in ordered]
        if self._stable_class not in keys or candidate not in keys:
            return True
        i_s = keys.index(self._stable_class)
        i_c = keys.index(candidate)
        # 向更大档：须 ≥ thr + hyst；向更小档：须 < thr - hyst
        # thr[i] 分隔 ordered[i](大) 与 ordered[i+1](小)
        L = float(smooth_raw)
        h = self.hysteresis_mm
        if i_c < i_s:
            # 候选更大（index 更小）：跨过中间各 thr 的上沿
            thr = float(thrs[i_c]) if i_c < len(thrs) else L
            return L >= thr + h
        # 候选更小
        thr = float(thrs[i_s - 1]) if i_s - 1 < len(thrs) else L
        return L < thr - h

    def update(self, size_info):
        """
        输入单帧 measure_and_classify_nut 结果，返回抗干扰后的 size_info（原地改 class）。
        无有效输入时返回上一帧稳定结果副本。
        """
        if not isinstance(size_info, dict):
            if self._last_info is not None:
                out = dict(self._last_info)
                out["size_tracked"] = True
                out["size_held"] = True
                return out
            return None

        raw = size_info.get("hex_raw_mm", size_info.get("long_raw_mm"))
        amb = bool(size_info.get("ambiguous"))
        smooth = self._smooth_raw(raw, ambiguous=amb)
        out = dict(size_info)
        if smooth is None or not np.isfinite(smooth):
            out["size_tracked"] = False
            self._last_info = out
            return out

        # 用平滑 raw 重定档（显示真值仍用单帧 calib，或再映射 smooth）
        cls_info = self._classify_raw(smooth)
        candidate = cls_info.get("class") if isinstance(cls_info, dict) else None
        true_mm, calib_meta = apply_size_calib_mm(smooth, self.detection_cfg)
        out["hex_raw_mm"] = float(smooth)
        out["long_raw_mm"] = float(smooth)
        out["hex_mm"] = float(true_mm)
        out["long_mm"] = float(true_mm)
        out["hex_cls_mm"] = float(smooth)
        out["calib"] = calib_meta
        if isinstance(cls_info, dict):
            out["ambiguous"] = bool(cls_info.get("ambiguous"))
            out["dist_mm"] = cls_info.get("dist_mm")
            out["confidence"] = cls_info.get("confidence")
            out["near_boundary"] = cls_info.get("near_boundary")
            out["thresholds_mm"] = cls_info.get("thresholds_mm")

        if candidate is None:
            candidate = size_info.get("class")

        if self._stable_class is None:
            self._stable_class = candidate
            self._pending_class = None
            self._pending_n = 0
        elif candidate == self._stable_class:
            self._pending_class = None
            self._pending_n = 0
        else:
            # 跨两档（大↔小）：邻轨串扰时滞回会把大粘成小；raw 已落在候选簇则加速换档
            force_cross = False
            cfg_sw = _size_metric_cfg(self.detection_cfg)
            nom_sw, _ = _nominal_hex_for_classify(cfg_sw)
            keys_sw = [k for k, _ in _midpoint_thresholds(nom_sw, cfg_sw)[0]]
            if (
                self._stable_class in keys_sw
                and candidate in keys_sw
                and abs(keys_sw.index(candidate) - keys_sw.index(self._stable_class)) >= 2
            ):
                c_nom = float(nom_sw.get(candidate, smooth))
                if abs(float(smooth) - c_nom) <= abs(
                    float(smooth) - float(nom_sw.get(self._stable_class, smooth))
                ) - 1.0:
                    force_cross = True
            if force_cross or self._accept_switch(candidate, smooth):
                if candidate == self._pending_class:
                    self._pending_n += 1
                else:
                    self._pending_class = candidate
                    self._pending_n = 1
                need = 1 if force_cross else self.vote_frames
                if self._pending_n >= need:
                    self._stable_class = candidate
                    self._pending_class = None
                    self._pending_n = 0
                    if force_cross:
                        # 丢掉被串扰污染的 EMA，跟当前 raw
                        self._ema_raw = float(smooth)
                        self._raws = [float(smooth)]
            else:
                self._pending_class = None
                self._pending_n = 0

        out["class"] = self._stable_class
        out["class_instant"] = candidate
        out["size_tracked"] = True
        out["size_held"] = candidate != self._stable_class
        tag = "≈" if out.get("ambiguous") else ""
        hold = "·稳" if out["size_held"] else ""
        out["label"] = (
            f"{tag}{out['class']}{hold} {float(true_mm):.0f}mm"
            f"(raw{float(smooth):.0f})"
        )
        # 厚度随稳定档
        cfg = _size_metric_cfg(self.detection_cfg)
        h_nom = (cfg.get("nominal_height_mm") or {}).get(str(self._stable_class or ""))
        if h_nom is not None and np.isfinite(float(h_nom)) and float(h_nom) > 0.5:
            out["height_mm"] = float(h_nom)
            out["nominal_height_mm"] = float(h_nom)
        self._last_info = out
        return out


class MultiSizeClassTracker:
    """
    多目标尺寸跟踪：按框心最近邻一对一关联，每框独立 EMA+滞回+投票。

    旧版邻格抢轨：两大目标靠太近时会互偷 SizeClassTracker，把大号粘成小号。
    现：每帧每个轨最多认领一次；超距则新开轨。
    """

    def __init__(self, detection_cfg=None, grid_px=48, max_tracks=12, stale_frames=18, **kwargs):
        self.detection_cfg = detection_cfg
        self.grid_px = float(max(16.0, grid_px))
        self.max_tracks = int(max(2, max_tracks))
        self.stale_frames = int(max(4, stale_frames))
        self._kw = dict(kwargs)
        # tid -> {tracker, last_seen, cx, cy}
        self._tracks = {}
        self._next_id = 1
        self._frame = 0
        self._claimed = set()

    def reset(self):
        self._tracks.clear()
        self._frame = 0
        self._next_id = 1
        self._claimed = set()

    def configure(self, detection_cfg):
        self.detection_cfg = detection_cfg
        for rec in self._tracks.values():
            rec["tracker"].configure(detection_cfg)

    @staticmethod
    def _box_center(box):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    def begin_frame(self):
        self._frame += 1
        self._claimed = set()
        dead = [
            tid for tid, rec in self._tracks.items()
            if self._frame - int(rec["last_seen"]) > self.stale_frames
        ]
        for tid in dead:
            self._tracks.pop(tid, None)

    def _match_or_create(self, cx, cy, box=None):
        # 紧关联：只跟「同一物体微移」续轨；邻框绝不互抢（每帧轨一对一）
        max_dist = self.grid_px * 0.75
        if box is not None and len(box) >= 4:
            bw = abs(float(box[2]) - float(box[0]))
            bh = abs(float(box[3]) - float(box[1]))
            # 不超过本框半短边，避免贴邻的另一颗螺母抢走本轨
            max_dist = min(max_dist, 0.45 * min(bw, bh))
            max_dist = max(max_dist, 12.0)

        best_tid = None
        best_d = max_dist
        for tid, rec in self._tracks.items():
            if tid in self._claimed:
                continue
            d = math.hypot(float(cx) - float(rec["cx"]), float(cy) - float(rec["cy"]))
            if d < best_d:
                best_d = d
                best_tid = tid

        if best_tid is None:
            tid = self._next_id
            self._next_id += 1
            tr = SizeClassTracker(detection_cfg=self.detection_cfg, **self._kw)
            self._tracks[tid] = {
                "tracker": tr,
                "last_seen": self._frame,
                "cx": float(cx),
                "cy": float(cy),
            }
            self._claimed.add(tid)
            if len(self._tracks) > self.max_tracks:
                oldest = min(
                    self._tracks.items(),
                    key=lambda kv: int(kv[1]["last_seen"]),
                )
                if oldest[0] not in self._claimed:
                    self._tracks.pop(oldest[0], None)
            return tr

        rec = self._tracks[best_tid]
        rec["last_seen"] = self._frame
        rec["cx"] = float(cx)
        rec["cy"] = float(cy)
        self._claimed.add(best_tid)
        return rec["tracker"]

    def update(self, box, size_info):
        if box is None or not isinstance(size_info, dict):
            return size_info
        cx, cy = self._box_center(box)
        tr = self._match_or_create(cx, cy, box=box)
        tr.configure(self.detection_cfg)
        return tr.update(size_info)


def _servo_xy_remap_inverse(point_cam, rotate_deg=0.0, flip_x=False, flip_y=False):
    """detection_point_from_camera_pose 中 XY remap 的逆变换。"""
    out = _servo_xy_rotate(
        np.asarray(point_cam, dtype=np.float64), -float(rotate_deg)
    )
    if flip_x:
        out[0] = -out[0]
    if flip_y:
        out[1] = -out[1]
    return out


def world_point_to_image_pixel(
    world_point,
    ee_pose,
    intrinsics,
    R_ee_cam,
    t_ee_cam,
    xy_rotate_deg=90.0,
    flip_x=True,
    flip_y=False,
    depth_sign=-1.0,
):
    """
    工作/模型系 3D 点 → 当前相机画面像素（与 detection_point_in_world 互逆）。
    用于校准：锁定物体投影应落在检测框中心。
    """
    cam_pos, cam_rot = camera_pose_in_world(R_ee_cam, t_ee_cam, ee_pose)
    if cam_pos is None:
        return None
    world = np.asarray(world_point, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(world)):
        return None
    p_remapped = cam_rot.T @ (world - cam_pos)
    p_cam = _servo_xy_remap_inverse(
        p_remapped, xy_rotate_deg, flip_x, flip_y,
    )
    ds = float(depth_sign)
    if abs(ds) > 1e-9:
        p_cam = np.array(p_cam, dtype=np.float64, copy=True)
        p_cam[2] /= ds
    if p_cam[2] <= 1e-4:
        return None
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics.get("cx", 320))
    cy = float(intrinsics.get("cy", 240))
    u = cx + fx * p_cam[0] / p_cam[2]
    v = cy + fy * p_cam[1] / p_cam[2]
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return float(u), float(v)


def draw_nut_surface_depth_debug(image, depth_meta, box=None):
    """
    在彩色/深度伪彩图上叠加 nut_surface 诊断：精修环心、孔半径、顶/桌深度。
    depth_meta 来自 measure_detection_uvd / estimate_box_depth(..., return_meta=True)。
    """
    if image is None or not isinstance(depth_meta, dict):
        return image
    out = image
    cu, cv = depth_meta.get("center_uv") or (None, None)
    if cu is None or cv is None:
        cu = depth_meta.get("u")
        cv = depth_meta.get("v")
    if cu is None or cv is None:
        return out
    cu_i, cv_i = int(round(float(cu))), int(round(float(cv)))
    hole_r = depth_meta.get("hole_r")
    if hole_r is not None and float(hole_r) > 1.0:
        r_draw = float(hole_r)
        # 诊断圆勿超出框半轴，避免小框被虚大半径盖住
        if box is not None and len(box) >= 4:
            half = 0.5 * min(
                abs(float(box[2]) - float(box[0])),
                abs(float(box[3]) - float(box[1])),
            )
            if half > 2.0:
                r_draw = min(r_draw, half * 0.92)
        cv2.circle(out, (cu_i, cv_i), int(round(r_draw)), (180, 80, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(
        out, (cu_i, cv_i), (255, 0, 255), markerType=cv2.MARKER_CROSS,
        markerSize=14, thickness=2, line_type=cv2.LINE_AA,
    )
    top = depth_meta.get("top_mm") or depth_meta.get("estimate")
    table = depth_meta.get("table_mm")
    layer = depth_meta.get("layer") or ""
    conf = float(depth_meta.get("confidence") or 0.0)
    y0 = cv_i + 16
    if box is not None and len(box) >= 4:
        y0 = max(y0, int(box[1]) + 14)
    label = f"surf {layer} conf={conf:.2f}"
    if top is not None:
        label += f" top={float(top)/10:.1f}cm"
    if table is not None:
        label += f" tbl={float(table)/10:.1f}cm"
    rej = depth_meta.get("rejected")
    if rej and layer == "rejected":
        label += f" [{rej}]"
    cv2.putText(
        out, label, (cu_i + 8, y0),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 80), 1, cv2.LINE_AA,
    )
    return out


def draw_model_object_overlay(
    image,
    u_proj,
    v_proj,
    u_det=None,
    v_det=None,
):
    """相机画面：上表面锁定点反投影（未含下偏）。应对准通孔/检测环心。"""
    h, w = image.shape[:2]
    pu = int(round(float(u_proj)))
    pv = int(round(float(v_proj)))
    in_view = 0 <= pu < w and 0 <= pv < h
    if in_view:
        cv2.circle(image, (pu, pv), 14, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(
            image, (pu, pv), (255, 0, 255), cv2.MARKER_CROSS, 22, 2,
        )
    label_y = max(18, pv - 18) if in_view else 36
    label_x = max(8, pu - 36) if in_view else 8
    cv2.putText(
        image, "紫=上表面锁点", (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA,
    )
    if u_det is not None and v_det is not None:
        iu = int(round(float(u_det)))
        iv = int(round(float(v_det)))
        if in_view:
            cv2.line(image, (iu, iv), (pu, pv), (255, 0, 255), 1, cv2.LINE_AA)
        err = math.hypot(float(u_proj) - float(u_det), float(v_proj) - float(v_det))
        cv2.putText(
            image, f"校准偏差 {err:.0f}px",
            (8, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 255), 1, cv2.LINE_AA,
        )
    cv2.putText(
        image, "紫十字应与通孔中心重合（上表面；青球下偏会透视偏移）",
        (8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA,
    )


def hand_axis_image_point(intrinsics, R_ee_cam, t_ee_cam, depth_m):
    """
    手轴（EE 原点沿 +Z）在深度 depth_m 处投影到图像的像素。
    t_ee_cam = 相机原点在末端系中的位置（默认取自 MJCF arm_right_camera）。
    """
    depth_m = max(float(depth_m), 0.08)
    point_ee = np.array([0.0, 0.0, depth_m], dtype=np.float64)
    point_cam = R_ee_cam.T @ (point_ee - t_ee_cam)
    if point_cam[2] <= 1e-4:
        return None
    u = intrinsics["cx"] + intrinsics["fx"] * point_cam[0] / point_cam[2]
    v = intrinsics["cy"] + intrinsics["fy"] * point_cam[1] / point_cam[2]
    return int(round(u)), int(round(v))


def draw_axis_overlay(image, intrinsics, R_ee_cam, t_ee_cam, depth_mm=None):
    """仅标相机光轴（画面中心）。去掉旧的手轴/示教箭头残留。"""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 28, 2)
    cv2.putText(
        image, "camera optical", (cx + 12, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
    )


def draw_servo_direction_hint(
    image,
    u,
    v,
    depth_mm,
    intrinsics,
    R_ee_cam,
    xy_rotate_deg=90.0,
    flip_x=True,
    flip_y=False,
    gain=0.45,
):
    """
    标出对准伺服方向（便于核对上下左右）：
    - 黄线：画面十字 → 检测目标（目标相对十字偏哪边）
    - 蓝箭头：预测臂/末端平移方向（把目标拉向十字，仅对准示意）
    - 文字：画面偏心 + 工作系 Δxyz(mm)
    OpenCV 图像：+u 右、+v 下；手偏移 camera 系 +Y 为画面上方。
    """
    if u is None or v is None or not np.isfinite(float(u)) or not np.isfinite(float(v)):
        return
    h, w = image.shape[:2]
    cx = float(intrinsics.get("cx", w * 0.5))
    cy = float(intrinsics.get("cy", h * 0.5))
    du = float(u) - cx
    dv = float(v) - cy
    iu, iv = int(round(float(u))), int(round(float(v)))
    icx, icy = int(round(cx)), int(round(cy))

    cv2.arrowedLine(
        image, (icx, icy), (iu, iv), (0, 220, 255), 2, tipLength=0.25,
    )
    cv2.putText(
        image, "黄:十字→目标", (max(8, icx - 28), max(14, icy - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA,
    )
    img_parts = []
    if abs(du) > 4:
        img_parts.append(f"目标偏{'右' if du > 0 else '左'}{abs(du):.0f}px")
    if abs(dv) > 4:
        img_parts.append(f"偏{'下' if dv > 0 else '上'}{abs(dv):.0f}px")
    if not img_parts:
        img_parts.append("已居中")

    delta_ee = center_only_correction_in_ee(
        float(u), float(v), depth_mm, intrinsics, R_ee_cam, gain,
        xy_rotate_deg=xy_rotate_deg, flip_x=flip_x, flip_y=flip_y,
    )
    world_txt = "Δtool—"
    if delta_ee is not None and np.all(np.isfinite(delta_ee)):
        wx, wy, wz = (float(delta_ee[i]) * 1000.0 for i in range(3))
        world_txt = f"Δ工作 {wx:+.0f},{wy:+.0f},{wz:+.0f} mm"
        # 工作系增量 → 相机系，再投影到图像平面（方向示意）
        try:
            delta_cam = R_ee_cam.T @ delta_ee
            z = 0.25
            if depth_mm is not None and np.isfinite(depth_mm) and depth_mm > 1.0:
                z = float(np.clip(depth_mm / 1000.0, 0.08, 0.60))
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            pu = int(np.clip(iu + delta_cam[0] / z * fx, 8, w - 8))
            pv = int(np.clip(iv + delta_cam[1] / z * fy, 8, h - 8))
            cv2.arrowedLine(
                image, (iu, iv), (pu, pv), (255, 200, 80), 2, tipLength=0.35,
            )
            cv2.putText(
                image, "蓝:臂将动", (max(8, pu - 8), max(14, pv - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 200, 255), 1, cv2.LINE_AA,
            )
        except (TypeError, ValueError, IndexError):
            pass

    line1 = " · ".join(img_parts)
    y0 = min(h - 8, iv + 22)
    cv2.putText(
        image, line1, (max(8, iu - 40), y0),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image, world_txt, (max(8, iu - 40), y0 + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 200, 255), 1, cv2.LINE_AA,
    )


def is_grasp_aligned(u, v, depth_mm, intrinsics, eye_cfg, R_ee_cam=None, t_ee_cam=None):
    """对准：靠近画面中心；深度有效且不超过目标+容差（允许比目标距离更近）。"""
    if depth_mm is None or depth_mm <= 0:
        return False
    cx = int(round(intrinsics.get("cx", 320)))
    cy = int(round(intrinsics.get("cy", 240)))
    pixel_tol = eye_cfg.get("pixel_tolerance_px", 40)
    dist_tol = eye_cfg["distance_tolerance_mm"]
    target = eye_cfg["target_distance_mm"]
    min_mm = eye_cfg.get("min_distance_mm", 50)
    # 近了也算距离达标；只有太远才不合格。近距空洞由采样修复。
    distance_ok = (depth_mm >= min_mm) and (depth_mm <= target + dist_tol)
    return (
        abs(u - cx) <= pixel_tol
        and abs(v - cy) <= pixel_tol
        and distance_ok
    )


def select_arm(y_robot):
    return "left" if y_robot >= 0.0 else "right"


def within_workspace(point, workspace):
    for axis, key in zip(point, ("x", "y", "z")):
        lower, upper = workspace[key]
        if point[axis] < lower or point[axis] > upper:
            return False
    return True


def draw_grasp_point_marker(image, u, v, mode="bbox_center"):
    """在画面上标抓取/对准像素（青十字）。"""
    if image is None or u is None or v is None:
        return
    if not np.isfinite(float(u)) or not np.isfinite(float(v)):
        return
    iu = int(round(float(u)))
    iv = int(round(float(v)))
    h, w = image.shape[:2]
    if not (0 <= iu < w and 0 <= iv < h):
        return
    cv2.drawMarker(
        image, (iu, iv), (255, 220, 0), cv2.MARKER_CROSS, 14, 2, line_type=cv2.LINE_AA,
    )
    label = {
        "hand_top": "抓:世界上边",
        "hand_upper": "抓:世界上边",
        "world_top": "抓:世界上边",
        "world_up": "抓:世界上边",
        "side_top": "抓:手上边",
        "top_face": "抓:顶面",
        "top": "抓:顶面",
    }.get(str(mode or "").strip().lower(), "抓:中心")
    cv2.putText(
        image, label, (min(iu + 8, w - 80), max(iv - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 0), 1, cv2.LINE_AA,
    )


def track_class_display_name(name):
    """画面框标签用英文档名（OpenCV 默认字体不支持中文）。"""
    raw = str(name or "").strip()
    key = normalize_track_color(raw) or raw.lower()
    # 放置类：中文「框」等映射为可显示英文
    if is_place_like_label(raw) or is_place_like_label(key):
        return {
            "框": "bin",
            "螺母框": "bin",
            "料框": "bin",
            "筐": "basket",
            "筐子": "basket",
            "盘": "plate",
            "盆": "bowl",
            "盒": "box",
            "篮子": "basket",
            "kuang": "basket",
        }.get(raw, {
            "plate": "plate",
            "basket": "basket",
            "bin": "bin",
            "bowl": "bowl",
            "tray": "tray",
            "box": "box",
            "kuang": "basket",
            "frame": "bin",
        }.get(key, "bin"))
    return {
        "big": "big",
        "medium": "medium",
        "small": "small",
        "nut": "nut",
        "red": "red",
        "green": "green",
        "blue": "blue",
    }.get(key, key or "?")


def draw_detections(
    image,
    detections,
    names,
    depth=None,
    class_overrides=None,
    size_infos=None,
):
    """
    画检测框，格式与 YOLO 一致：`类别 置信度  [对边xxmm]  [深度cm]`。
    class_overrides / size_infos：与 detections 等长；深度定档时覆盖类别并显示对边。
    """
    if detections is None:
        return
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        override = None
        if class_overrides is not None and i < len(class_overrides):
            override = class_overrides[i]
        if override:
            name = track_class_display_name(override)
        else:
            logical = logical_color_from_detection(names, class_id)
            raw = detection_class_name(names, class_id)
            name = track_class_display_name(logical or raw)
        label = f"{name} {float(confidence):.2f}"
        info = None
        if size_infos is not None and i < len(size_infos):
            info = size_infos[i]
        if isinstance(info, dict):
            hex_mm = info.get("hex_mm", info.get("long_mm"))
            if hex_mm is not None:
                label += f"  {float(hex_mm):.0f}mm"
                if info.get("ambiguous"):
                    label += "?"
        if depth is not None:
            _, _, depth_mm = box_center_and_depth(depth, (x1, y1, x2, y2))
            if depth_mm is not None:
                label += f"  {depth_mm / 10:.1f}cm"
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        color = CLASS_BGR.get(str(name).lower(), CLASS_BGR.get(str(name), (0, 0, 255)))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        cv2.putText(
            image,
            label,
            (x1, max(32, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            3,
            cv2.LINE_AA,
        )


def detection_class_name(names, class_id):
    cid = int(class_id)
    if isinstance(names, dict):
        return names.get(cid, str(cid))
    if names is not None and 0 <= cid < len(names):
        return names[cid]
    return str(cid)


def normalize_track_color(name):
    """映射到标准追踪键（big/medium/small 或 red/green/blue）；无法识别则返回小写原名。"""
    raw = str(name).strip().lower()
    for key, aliases in TRACK_COLOR_ALIASES.items():
        if raw in {a.lower() for a in aliases} or raw == key:
            return key
    raw2 = str(name).strip()
    for key, aliases in TRACK_COLOR_ALIASES.items():
        if raw2 in aliases:
            return key
    return raw or None


def apply_model_color_correction(color):
    """把模型输出色纠正为实物逻辑色（仅旧 three_color 的 red↔blue）。"""
    if color is None:
        return None
    if MODEL_SWAP_RED_BLUE:
        if color == "red":
            return "blue"
        if color == "blue":
            return "red"
    return color


def logical_color_from_detection(names, class_id):
    """检测框 → 纠正后的追踪类别（供勾选过滤与显示）。"""
    return apply_model_color_correction(
        normalize_track_color(detection_class_name(names, class_id))
    )


def filter_detections_by_colors(detections, names, allowed_colors):
    """只保留勾选颜色对应的检测框（按纠正后的逻辑色）。"""
    if detections is None or len(detections) == 0:
        return []
    allowed = {str(c).strip().lower() for c in (allowed_colors or []) if c}
    if not allowed:
        return []
    kept = []
    for det in detections:
        color = logical_color_from_detection(names, det[5])
        if color is not None and color in allowed:
            kept.append(det)
    return kept


def best_detection(detections, names=None, allowed_colors=None):
    """置信度最高的一框；可按颜色过滤。"""
    if detections is None or len(detections) == 0:
        return None
    if allowed_colors is not None:
        detections = filter_detections_by_colors(detections, names, allowed_colors)
    if detections is None or len(detections) == 0:
        return None
    return max(detections, key=lambda item: item[4])


def estimate_bbox_size_mm(box, depth_mm, intrinsics, box_shrink=0.12, size_scale=1.0):
    """
    检测框像素宽高 + 深度 → 物理宽高(mm)（原始针孔，未做非线性校准）。
    针孔：W_mm = du * Z_mm / fx（框可内缩减轻 YOLO 膨胀）。
    返回 dict: w_mm, h_mm, long_mm, short_mm, du, dv, depth_mm；失败 None。
    """
    if box is None or depth_mm is None:
        return None
    d = float(depth_mm)
    if not np.isfinite(d) or d <= 1.0:
        return None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    du = abs(x2 - x1)
    dv = abs(y2 - y1)
    if du < 2.0 or dv < 2.0:
        return None
    shrink = float(np.clip(box_shrink, 0.0, 0.45))
    du_eff = du * (1.0 - 2.0 * shrink)
    dv_eff = dv * (1.0 - 2.0 * shrink)
    fx = float(intrinsics.get("fx", 520.0))
    fy = float(intrinsics.get("fy", 520.0))
    if fx < 1.0 or fy < 1.0:
        return None
    scale = float(size_scale) if size_scale is not None else 1.0
    if not np.isfinite(scale) or scale <= 0.01:
        scale = 1.0
    w_mm = du_eff * d / fx * scale
    h_mm = dv_eff * d / fy * scale
    long_mm = max(w_mm, h_mm)
    short_mm = min(w_mm, h_mm)
    return {
        "w_mm": float(w_mm),
        "h_mm": float(h_mm),
        "long_mm": float(long_mm),
        "short_mm": float(short_mm),
        "du": float(du_eff),
        "dv": float(dv_eff),
        "depth_mm": float(d),
        "size_scale": float(scale),
    }


def _size_metric_cfg(detection_cfg):
    if not isinstance(detection_cfg, dict):
        return {}
    raw = detection_cfg.get("size_metric")
    return raw if isinstance(raw, dict) else {}


def _parse_size_calib_points(cfg):
    """yaml size_calib_points: [[measured, true], ...] → (meas[], true[]) 按 measured 升序。"""
    raw = cfg.get("size_calib_points") if isinstance(cfg, dict) else None
    if not raw:
        return None, None
    pts = []
    for item in raw:
        try:
            m, t = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(m) and np.isfinite(t) and m > 1.0 and t > 1.0:
            pts.append((m, t))
    if len(pts) < 2:
        return None, None
    pts.sort(key=lambda p: p[0])
    # 相同 measured 取平均
    merged = {}
    for m, t in pts:
        merged.setdefault(round(m, 3), []).append(t)
    ms = sorted(merged.keys())
    ts = [float(np.mean(merged[m])) for m in ms]
    return np.asarray(ms, dtype=np.float64), np.asarray(ts, dtype=np.float64)


def fit_size_calib_poly(meas, true, degree=None):
    """
    拟合 true ≈ poly(measured)。
    点数 n→默认阶 min(2, n-1)：3 点二次，精确过校点（默认 80→70 / 60→50 / 50→41）。
    返回高次→低次系数（np.polyval 格式），失败 None。
    """
    meas = np.asarray(meas, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    if meas.size < 2 or meas.size != true.size:
        return None
    if degree is None:
        degree = int(min(2, meas.size - 1))
    degree = int(np.clip(degree, 1, meas.size - 1))
    try:
        coef = np.polyfit(meas, true, degree)
    except Exception:
        return None
    return np.asarray(coef, dtype=np.float64)


def apply_size_calib_mm(measured_mm, detection_cfg=None):
    """
    针孔原始 mm → 显示用「真值」mm。
    默认 piecewise + 端点比例外推（勿用 poly：在中/小区斜率过平，易算混）。
    返回 (corrected_mm, meta_dict)。
    """
    m = float(measured_mm)
    cfg = _size_metric_cfg(detection_cfg)
    if not cfg and isinstance(detection_cfg, dict) and (
        "size_calib_points" in detection_cfg or "size_scale" in detection_cfg
    ):
        cfg = detection_cfg

    ms, ts = _parse_size_calib_points(cfg)
    meta = {"mode": "none", "raw_mm": m}
    if ms is not None:
        mode = str(cfg.get("size_calib_mode", "piecewise")).strip().lower()
        if mode in ("poly", "quadratic"):
            deg = cfg.get("size_calib_degree")
            coef = fit_size_calib_poly(ms, ts, degree=None if deg is None else int(deg))
            if coef is not None:
                out = float(np.polyval(coef, m))
                meta["mode"] = "poly"
                meta["poly_coef"] = [float(c) for c in coef]
                meta["points"] = list(zip(ms.tolist(), ts.tolist()))
                return max(out, 1.0), meta

        # piecewise：区间内线性插值；区间外按最近端点 true/meas 比例缩放（避免二次外推炸）
        lo, hi = float(ms[0]), float(ms[-1])
        if m <= lo:
            out = float(m * (ts[0] / ms[0]))
        elif m >= hi:
            out = float(m * (ts[-1] / ms[-1]))
        else:
            out = float(np.interp(m, ms, ts))
        meta["mode"] = "piecewise"
        meta["points"] = list(zip(ms.tolist(), ts.tolist()))
        return max(out, 1.0), meta

    scale = float(cfg.get("size_scale", 1.0)) if cfg else 1.0
    if not np.isfinite(scale) or scale <= 0.01:
        scale = 1.0
    meta["mode"] = "scale"
    meta["size_scale"] = scale
    return max(m * scale, 1.0), meta


_SIZE_CLASS_ORDER = ("big", "medium", "small")
_SIZE_CLASS_RANK = {"big": 2, "medium": 1, "small": 0}


def _default_nominal_hex_mm(cfg=None):
    cfg = cfg or {}
    nom = cfg.get("nominal_hex_mm") or cfg.get("nominal_long_mm") or {
        "big": 70.0, "medium": 50.0, "small": 41.0,
    }
    return {str(k).strip().lower(): float(v) for k, v in nom.items()}


def _nominal_hex_for_classify(cfg):
    """
    定档标称对边（mm）与工作空间。
    默认 calibrated：先 piecewise 映射到真值 70/50/41，再用中点边界定档。
    classify_space: calibrated | raw
    """
    cfg = cfg or {}
    true_nom = _default_nominal_hex_mm(cfg)
    space = str(cfg.get("classify_space", "calibrated")).strip().lower()
    if space in ("calibrated", "true", "nominal"):
        return true_nom, "calibrated"

    raw_nom = cfg.get("nominal_raw_mm")
    if isinstance(raw_nom, dict) and raw_nom:
        return {str(k).lower(): float(v) for k, v in raw_nom.items()}, "raw"

    ms, ts = _parse_size_calib_points(cfg)
    if ms is None:
        return true_nom, "calibrated"
    out = {}
    for key, t0 in true_nom.items():
        j = int(np.argmin(np.abs(ts - float(t0))))
        out[key] = float(ms[j])
    return out, "raw"


def _ordered_nominals(nom):
    """按对边从大到小: [(class, center_mm), ...]。"""
    ordered = []
    for key in _SIZE_CLASS_ORDER:
        if key in nom:
            ordered.append((key, float(nom[key])))
    ordered.sort(key=lambda t: t[1], reverse=True)
    return ordered


def _midpoint_thresholds(nom, cfg=None):
    """
    相邻档分界阈值（降序）。L >= thr[i] → ordered[i]。
    默认取标称中点；可用 classify_thresholds_mm: [big_med, med_small] 覆盖
    （例 calibrated 下 [60, 43.5]）。
    """
    ordered = _ordered_nominals(nom)
    thrs = []
    for i in range(len(ordered) - 1):
        thrs.append(0.5 * (ordered[i][1] + ordered[i + 1][1]))
    cfg = cfg or {}
    override = cfg.get("classify_thresholds_mm")
    if override is not None:
        try:
            ov = [float(x) for x in override]
            if len(ov) >= len(thrs):
                thrs = ov[: len(thrs)]
            elif len(ov) > 0:
                for i in range(min(len(ov), len(thrs))):
                    thrs[i] = ov[i]
        except (TypeError, ValueError):
            pass
    return ordered, thrs


def _assign_by_thresholds(L, ordered, thrs):
    for i, thr in enumerate(thrs):
        if L >= thr:
            return ordered[i][0]
    return ordered[-1][0] if ordered else "medium"


def _hex_feature_mm(w_mm, h_mm, mode="min"):
    """
    从 AABB 宽高提取六边形「对边」特征。
    正六边形平视：min(w,h)≈对边 AF，max/min≈1~1.155；mean/max 会随旋转漂。
    """
    w = float(w_mm)
    h = float(h_mm)
    mn, mx = min(w, h), max(w, h)
    mode = str(mode or "min").strip().lower()
    if mode in ("max", "long"):
        return mx
    if mode in ("mean", "avg", "average"):
        return 0.5 * (w + h)
    if mode in ("geom", "geomean", "sqrt"):
        return float(math.sqrt(max(w * h, 1e-6)))
    # 默认 min = across-flats
    return mn


def classify_nut_by_size_mm(
    long_mm,
    short_mm=None,
    detection_cfg=None,
    yolo_cls=None,
    yolo_conf=None,
):
    """
    六边形对边定档（重建版）。

    1) 中点边界硬分档（不再用易抖的最近邻）
    2) 距边界越近 conf 越低；可用 YOLO 弱先验破平局
    3) 过远 / 贴边 → ambiguous（仍给最近合理档，不瞎拆档）
    """
    del short_mm
    cfg = _size_metric_cfg(detection_cfg)
    nom, space = _nominal_hex_for_classify(cfg)
    L = float(long_mm)
    ordered, thrs = _midpoint_thresholds(nom, cfg)
    best_key = _assign_by_thresholds(L, ordered, thrs)

    center = float(nom.get(best_key, L))
    dist_center = abs(L - center)
    dist_edge = min((abs(L - e) for e in thrs), default=1e9)

    half_gaps = [0.5 * abs(ordered[i][1] - ordered[i + 1][1])
                 for i in range(len(ordered) - 1)]
    typical_half = float(np.median(half_gaps)) if half_gaps else 8.0

    margin = float(cfg.get("boundary_margin_mm", max(2.5, 0.35 * typical_half)))
    reject_mm = float(cfg.get("reject_mm", max(12.0, 1.2 * typical_half)))
    if space == "raw":
        reject_mm = float(cfg.get("reject_mm_raw", reject_mm))

    near_boundary = bool(dist_edge <= margin)
    far_from_center = bool(dist_center > reject_mm)

    # 仅中|小缝：刚过阈值但仍明显更靠近 small 中心时才下调（防框胀小→中）
    # 不对大|中做偏小，避免 raw≈50 被拽成 medium
    prefer_smaller = bool(cfg.get("prefer_smaller_on_boundary", True))
    if prefer_smaller and near_boundary and len(ordered) >= 2:
        for i, thr in enumerate(thrs):
            if abs(L - float(thr)) > margin:
                continue
            larger_k, larger_c = ordered[i]
            smaller_k, smaller_c = ordered[i + 1]
            if larger_k != "medium" or smaller_k != "small":
                continue
            mid = 0.5 * (float(smaller_c) + float(larger_c))
            if best_key == larger_k and L < mid:
                if abs(L - float(smaller_c)) + 0.35 < abs(L - float(larger_c)):
                    best_key = smaller_k
                    center = float(smaller_c)
                    dist_center = abs(L - center)
            break

    # 高斯隶属 + 贴边 YOLO 弱先验
    scores = {}
    for key, c0 in nom.items():
        d = abs(L - float(c0))
        scores[key] = float(math.exp(-0.5 * (d / max(typical_half, 1.0)) ** 2))

    yolo_key = None
    if yolo_cls:
        yolo_key = normalize_track_color(yolo_cls)
        if yolo_key not in nom:
            yolo_key = None
    yolo_w = float(cfg.get("yolo_prior_weight", 0.40))
    yc = float(yolo_conf) if yolo_conf is not None and np.isfinite(float(yolo_conf)) else 0.55
    yc = float(np.clip(yc, 0.0, 1.0))
    used_yolo = False
    if yolo_key is not None and yolo_w > 0.0 and near_boundary:
        yolo_center = float(nom[yolo_key])
        # 仅当 YOLO 标称离测值不太远时才破平局，避免 metric≈41 却被 YOLO=big 拽走
        if abs(L - yolo_center) <= abs(L - center) + max(4.0, margin):
            scores[yolo_key] = float(scores.get(yolo_key, 0.0) + yolo_w * yc)
            used_yolo = True
            if abs(L - yolo_center) <= max(reject_mm * 0.85, typical_half):
                best_key = yolo_key
                center = yolo_center
                dist_center = abs(L - center)

    scored = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if used_yolo and scored and near_boundary:
        # 得分最高与阈值档不一致时，仅贴边且 YOLO 合理才采纳
        if scored[0][0] != best_key:
            yolo_center = float(nom.get(scored[0][0], L))
            if abs(L - yolo_center) <= abs(L - float(nom.get(best_key, L))) + max(4.0, margin):
                best_key = scored[0][0]
                center = float(nom.get(best_key, L))
                dist_center = abs(L - center)

    ambiguous = bool(near_boundary or far_from_center)
    conf = float(np.clip(dist_edge / max(typical_half, 1e-3), 0.0, 1.0))
    if far_from_center:
        conf *= 0.5
    if used_yolo and yolo_key == best_key:
        conf = float(min(1.0, conf + 0.15))

    candidates = [
        (k, abs(L - float(nom[k])), float(nom[k]))
        for k in _SIZE_CLASS_ORDER if k in nom
    ]
    candidates.sort(key=lambda t: t[1])

    return {
        "class": best_key,
        "dist_mm": float(dist_center),
        "dist_edge_mm": float(dist_edge) if dist_edge < 1e8 else None,
        "confidence": float(conf),
        "ambiguous": bool(ambiguous),
        "near_boundary": bool(near_boundary),
        "long_mm": L,
        "hex_mm": L,
        "short_mm": None,
        "classify_space": space,
        "nominal_used": {k: float(v) for k, v in nom.items()},
        "thresholds_mm": [float(t) for t in thrs],
        "candidates": candidates[:3],
        "scores": {k: float(v) for k, v in scored},
        "yolo_prior": yolo_key if used_yolo else None,
    }


def measure_and_classify_nut(
    box,
    depth_mm,
    intrinsics,
    detection_cfg=None,
    yolo_cls=None,
    yolo_conf=None,
):
    """
    框+深度 → 对边(mm) → 校准到真值 → 中点边界定档。
    YOLO 尺寸类仅作可选贴边先验（single.pt 的 nut 不参与）；最终档以深度为准。
    """
    cfg = _size_metric_cfg(detection_cfg)
    if not bool(cfg.get("enabled", True)):
        return None
    ms, _ts = _parse_size_calib_points(cfg)
    pre_scale = 1.0 if ms is not None else float(cfg.get("size_scale", 1.0))
    shrink = float(cfg.get("box_shrink", 0.12))
    size = estimate_bbox_size_mm(
        box,
        depth_mm,
        intrinsics,
        box_shrink=shrink,
        size_scale=pre_scale,
    )
    if size is None:
        return None

    w_raw = float(size["w_mm"])
    h_raw = float(size["h_mm"])
    aspect = max(w_raw, h_raw) / max(min(w_raw, h_raw), 1e-3)
    # 正六边形 AABB 理论 max/min ≤ 2/√3≈1.155；过大多半侧视/框歪
    aspect_max = float(cfg.get("aspect_max", 1.35))
    mode = str(cfg.get("hex_size_mode", "min")).strip().lower()
    if aspect > aspect_max and mode in ("min", "af", "flat", "across_flats"):
        # 侧视时 min 会偏小，改用几何平均更稳
        hex_raw = _hex_feature_mm(w_raw, h_raw, "geom")
        aspect_mode = "geom_fallback"
    else:
        hex_raw = _hex_feature_mm(w_raw, h_raw, mode)
        aspect_mode = mode

    hex_disp, calib_meta = apply_size_calib_mm(hex_raw, cfg)
    space = str(cfg.get("classify_space", "calibrated")).strip().lower()
    hex_for_cls = (
        hex_disp if space in ("calibrated", "true", "nominal") else hex_raw
    )

    size["hex_raw_mm"] = float(hex_raw)
    size["long_raw_mm"] = float(hex_raw)
    size["short_raw_mm"] = float(min(w_raw, h_raw))
    size["aspect"] = float(aspect)
    size["hex_feature"] = aspect_mode
    size["hex_mm"] = float(hex_disp)
    size["long_mm"] = float(hex_disp)
    scale_r = hex_disp / hex_raw if hex_raw > 1e-3 else 1.0
    size["short_mm"] = float(min(w_raw, h_raw) * scale_r)
    size["w_mm"] = float(w_raw * scale_r)
    size["h_mm"] = float(h_raw * scale_r)
    size["calib"] = calib_meta

    cls = classify_nut_by_size_mm(
        hex_for_cls,
        None,
        detection_cfg,
        yolo_cls=yolo_cls,
        yolo_conf=yolo_conf,
    )
    out = {**size, **cls}
    out["hex_mm"] = float(hex_disp)
    out["long_mm"] = float(hex_disp)
    out["hex_cls_mm"] = float(hex_for_cls)
    # 侧面捏取用标称厚度（AF 定档后查表；与对边无关）
    h_nom = (cfg.get("nominal_height_mm") or {}).get(str(out.get("class") or ""))
    if h_nom is not None and np.isfinite(float(h_nom)) and float(h_nom) > 0.5:
        out["height_mm"] = float(h_nom)
        out["nominal_height_mm"] = float(h_nom)
    if aspect > aspect_max:
        out["ambiguous"] = True
        out["aspect_warn"] = True
    tag = "≈" if out.get("ambiguous") else ""
    out["label"] = (
        f"{tag}{out.get('class')} {hex_disp:.0f}mm"
        f"(raw{hex_raw:.0f})"
    )
    return out


def _relabel_size_info(info, cls, relative=False):
    cls = str(cls).strip().lower()
    info["class"] = cls
    if relative:
        info["classify_relative"] = True
    raw = info.get("hex_raw_mm", info.get("hex_cls_mm", 0.0))
    hex_mm = info.get("hex_mm", raw)
    tag = "≈" if info.get("ambiguous") else ""
    info["label"] = f"{tag}{cls} {float(hex_mm):.0f}mm(raw{float(raw):.0f})"


def refine_size_classes_relative(size_infos, detection_cfg=None):
    """
    可选：多螺母时做「尺寸序 ↔ 档序」单调性修复。

    默认关闭（classify_relative=false）：各框只按自身绝对阈值定档，互不影响。
    开启时：同档允许并存；仅当「明显更大却档更小」时才微调。
    """
    cfg = _size_metric_cfg(detection_cfg)
    out_classes = [
        (info or {}).get("class") if isinstance(info, dict) else None
        for info in (size_infos or [])
    ]
    if not bool(cfg.get("classify_relative", False)):
        return out_classes

    indexed = []
    for i, info in enumerate(size_infos or []):
        if not isinstance(info, dict):
            continue
        meas = info.get("hex_cls_mm", info.get("hex_mm", info.get("hex_raw_mm")))
        cls = info.get("class")
        if meas is None or cls is None or not np.isfinite(float(meas)):
            continue
        if cls not in _SIZE_CLASS_RANK:
            continue
        indexed.append([i, float(meas), info])

    if len(indexed) < 2:
        return out_classes

    indexed.sort(key=lambda t: t[1], reverse=True)
    nom, _sp = _nominal_hex_for_classify(cfg)
    ordered, _thrs = _midpoint_thresholds(nom, cfg)
    half_gaps = [
        0.5 * abs(ordered[i][1] - ordered[i + 1][1])
        for i in range(len(ordered) - 1)
    ]
    # 默认至少约半档间距，避免几毫米噪声触发「相对定档」
    min_gap = float(cfg.get("relative_min_gap_mm", 6.0))
    if half_gaps:
        min_gap = max(min_gap, float(min(half_gaps)) * 0.9)

    rank_of = _SIZE_CLASS_RANK
    cls_of = {2: "big", 1: "medium", 0: "small"}

    for k in range(len(indexed) - 1):
        _i, m_hi, info_hi = indexed[k]
        _j, m_lo, info_lo = indexed[k + 1]
        if (m_hi - m_lo) < min_gap:
            continue
        r_hi = rank_of[info_hi["class"]]
        r_lo = rank_of[info_lo["class"]]
        if r_hi >= r_lo:
            continue

        # 尺寸差够大却档序反了：先各自按绝对阈值重判（通常已单调）
        abs_hi = _assign_by_thresholds(m_hi, ordered, _thrs)
        abs_lo = _assign_by_thresholds(m_lo, ordered, _thrs)
        if rank_of[abs_hi] >= rank_of[abs_lo]:
            _relabel_size_info(info_hi, abs_hi, relative=True)
            _relabel_size_info(info_lo, abs_lo, relative=True)
            info_hi["ambiguous"] = False
            info_lo["ambiguous"] = False
            continue

        # 绝对阈值仍违序（极少见）：抬高较大者、压低较小者
        _relabel_size_info(info_hi, cls_of[min(2, r_lo)], relative=True)
        _relabel_size_info(info_lo, cls_of[max(0, r_hi)], relative=True)
        if rank_of[info_hi["class"]] < rank_of[info_lo["class"]]:
            _relabel_size_info(info_hi, "big", relative=True)
            _relabel_size_info(info_lo, "small", relative=True)
        info_hi["ambiguous"] = False
        info_lo["ambiguous"] = False

    return [
        (info or {}).get("class") if isinstance(info, dict) else None
        for info in (size_infos or [])
    ]


def nut_size_classes():
    return ("big", "medium", "small")


def is_nut_like_label(name):
    """检测框是否为螺母（generic nut 或 big/medium/small）。"""
    raw = str(name or "").strip().lower()
    if not raw:
        return False
    logical = normalize_track_color(raw)
    if logical in nut_size_classes() or logical == "nut":
        return True
    return raw in TRACK_COLOR_ALIASES.get("nut", set())


def yolo_size_prior_class(names, class_id):
    """
    仅当 YOLO 本身输出 big/medium/small 时才作尺寸先验。
    single.pt 的 nut 不参与定档。
    """
    logical = logical_color_from_detection(names, class_id)
    if logical in nut_size_classes():
        return logical
    return None


def filter_nut_detections(detections, names):
    """保留所有螺母框（nut/框 或大中小），忽略界面尺寸勾选。"""
    if detections is None or len(detections) == 0:
        return []
    out = []
    for d in detections:
        label = logical_color_from_detection(names, d[5]) or detection_class_name(
            names, d[5]
        )
        if is_nut_like_label(label):
            out.append(d)
    if out:
        return out
    # 单类抓取模型（如 single.pt 类名未登记时）仍收全部框
    n_names = 0
    if isinstance(names, dict):
        n_names = len(names)
    elif names is not None:
        n_names = len(names)
    if n_names <= 1:
        return list(detections)
    return out


def is_place_like_label(label):
    """是否为放置目标类（筐/盘等）。"""
    if label is None:
        return False
    raw = str(label).strip().lower()
    if raw in {a.lower() for a in PLACE_CLASS_ALIASES}:
        return True
    raw2 = str(label).strip()
    return raw2 in PLACE_CLASS_ALIASES


def filter_place_detections(detections, names):
    """保留放置目标框（plate/basket/筐/盘…）。"""
    if detections is None or len(detections) == 0:
        return []
    out = []
    for d in detections:
        label = logical_color_from_detection(names, d[5]) or detection_class_name(
            names, d[5]
        )
        if is_place_like_label(label):
            out.append(d)
    return out


def best_place_detection(detections, names):
    """最高分放置框；无则 None。"""
    pool = filter_place_detections(detections, names)
    if not pool:
        # 若模型只有一类且类名未知，取最高分框作放置候选
        if detections is not None and len(detections) > 0:
            # 单类模型：全部框都当作放置
            n_names = 0
            if isinstance(names, dict):
                n_names = len(names)
            elif names is not None:
                n_names = len(names)
            if n_names <= 1:
                pool = list(detections)
        if not pool:
            return None
    return max(pool, key=lambda d: float(d[4]))


def refine_plate_disk_center(bgr, box_xyxy, eye=None):
    """
    在 YOLO 框内精修圆形底盘圆心（框心不够准）。

    策略（由稳到兜底）：
      1) ROI 亮区轮廓 → 圆度过滤 → minEnclosingCircle / fitEllipse
      2) Canny + HoughCircles
      3) 失败返回 None（调用方回退框心）

    返回 dict: u,v,radius_px,method 或 None。
    """
    if bgr is None or box_xyxy is None:
        return None
    eye = eye or {}
    if not bool(eye.get("plate_circle_refine", True)):
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in box_xyxy[:4]]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([x1, y1, x2, y2])):
        return None
    h, w = bgr.shape[:2]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = float(np.clip(eye.get("plate_circle_pad_frac", 0.12), 0.0, 0.35))
    rx1 = int(max(0, math.floor(x1 - pad * bw)))
    ry1 = int(max(0, math.floor(y1 - pad * bh)))
    rx2 = int(min(w, math.ceil(x2 + pad * bw)))
    ry2 = int(min(h, math.ceil(y2 + pad * bh)))
    if rx2 - rx1 < 24 or ry2 - ry1 < 24:
        return None

    roi = bgr[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fill = float(np.count_nonzero(bw_mask)) / float(bw_mask.size)
    if fill < 0.08 or fill > 0.92:
        bw_mask = cv2.bitwise_not(bw_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw_mask = cv2.morphologyEx(bw_mask, cv2.MORPH_OPEN, k, iterations=1)
    bw_mask = cv2.morphologyEx(bw_mask, cv2.MORPH_CLOSE, k, iterations=2)

    min_area = 0.12 * float((rx2 - rx1) * (ry2 - ry1))
    max_area = 0.98 * float((rx2 - rx1) * (ry2 - ry1))
    min_circ = float(np.clip(eye.get("plate_circle_min_circularity", 0.65), 0.4, 0.95))
    best = None  # (score, cx, cy, r, method)

    contours, _ = cv2.findContours(bw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours or []:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        peri = float(cv2.arcLength(cnt, True))
        if peri < 1e-3:
            continue
        circ = float(4.0 * math.pi * area / (peri * peri))
        if circ < min_circ:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        r = float(r)
        if r < 8.0:
            continue
        box_r = 0.25 * (bw + bh)
        score = circ * 2.0 - abs(r - box_r) / max(box_r, 1.0)
        method = "enclose"
        if len(cnt) >= 5:
            try:
                ell = cv2.fitEllipse(cnt)
                ecx, ecy = float(ell[0][0]), float(ell[0][1])
                ra, rb = float(ell[1][0]) * 0.5, float(ell[1][1]) * 0.5
                if min(ra, rb) > 8.0 and max(ra, rb) / max(min(ra, rb), 1e-3) < 1.35:
                    cx, cy, r = ecx, ecy, 0.5 * (ra + rb)
                    score += 0.15
                    method = "ellipse"
            except cv2.error:
                pass
        if best is None or score > best[0]:
            best = (score, float(cx), float(cy), float(r), method)

    if best is None:
        edges = cv2.Canny(gray, 60, 140)
        min_r = int(max(8, 0.18 * min(bw, bh)))
        max_r = int(max(min_r + 1, 0.62 * max(bw, bh)))
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(12, int(0.25 * min(rx2 - rx1, ry2 - ry1))),
            param1=80,
            param2=int(np.clip(eye.get("plate_hough_param2", 28), 12, 80)),
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None and len(circles) > 0:
            c = circles[0]
            rh, rw = gray.shape[:2]
            cxc, cyc = 0.5 * rw, 0.5 * rh
            order = sorted(
                c,
                key=lambda t: (
                    abs(float(t[0]) - cxc) + abs(float(t[1]) - cyc),
                    -float(t[2]),
                ),
            )
            cx, cy, r = float(order[0][0]), float(order[0][1]), float(order[0][2])
            best = (0.5, cx, cy, r, "hough")

    if best is None:
        return None
    _, cx, cy, r, method = best
    u = float(cx + rx1)
    v = float(cy + ry1)
    if not (0.0 <= u < w and 0.0 <= v < h):
        return None
    if u < x1 - 0.15 * bw or u > x2 + 0.15 * bw:
        return None
    if v < y1 - 0.15 * bh or v > y2 + 0.15 * bh:
        return None
    return {
        "u": u,
        "v": v,
        "radius_px": float(r),
        "method": str(method),
    }


def measure_place_uvd(depth_image, detection, eye=None, prior_mm=None, bgr=None):
    """
    放置目标：圆心（有彩图则精修）+ 前景深度（不用 nut_surface）。
    返回 (u, v, depth_mm)；失败 depth_mm 可为 None。

    注意：精修后必须在精修 UV 上采深度（勿框心深度 + 事后改 UV）。
    """
    if detection is None:
        return None, None, None
    box = detection[:4]
    u = float((box[0] + box[2]) * 0.5)
    v = float((box[1] + box[3]) * 0.5)
    eye = eye or {}
    if bgr is not None:
        refined = refine_plate_disk_center(bgr, box, eye=eye)
        if refined is not None:
            u = float(refined["u"])
            v = float(refined["v"])
    if depth_image is None:
        return u, v, None
    _, _, depth_mm = estimate_box_depth(
        depth_image,
        box,
        prior_mm=prior_mm,
        freeze_below_mm=0.0,
        grasp_u=u,
        grasp_v=v,
        roi_mode=str(eye.get("place_depth_roi_mode", "foreground") or "foreground"),
        outlier_mm=float(eye.get("depth_outlier_mm", 20.0)),
        eye=eye,
        return_meta=False,
    )
    return u, v, depth_mm


def resolve_place_model_path(config=None, cli_path=None):
    """
    放置模型路径：CLI --place-model > detection.place_model > DEFAULT_PLACE_MODEL。
    返回绝对 Path 或 None（显式关闭）。须 resolve，避免相对名被 YOLOv5 当 HF repo。
    """
    if cli_path is not None:
        # 空字符串 / "-" 表示关闭
        if str(cli_path).strip() in ("", "-", "none", "None"):
            return None
        p = Path(cli_path).expanduser()
        if not p.is_file():
            cand = PROJECT_ROOT / str(cli_path).strip()
            if cand.is_file():
                return cand.resolve()
        return p.resolve() if p.is_file() else p
    det = {}
    if isinstance(config, dict):
        det = config.get("detection") or {}
    raw = det.get("place_model", None)
    if raw is None:
        return (
            DEFAULT_PLACE_MODEL.resolve()
            if DEFAULT_PLACE_MODEL.is_file()
            else None
        )
    s = str(raw).strip()
    if s.lower() in ("", "-", "none", "null", "false", "0"):
        return None
    p = Path(s).expanduser()
    if not p.is_file():
        cand = PROJECT_ROOT / s
        if cand.is_file():
            return cand.resolve()
    return p.resolve() if p.is_file() else p


def resolve_basket_model_path(config=None, cli_path=None):
    """
    额外筐子检验模型：CLI --basket-model > detection.basket_model > DEFAULT_BASKET_MODEL。
    与 place_model（盘）并行画框；传 none/- 关闭。不改变放置动作所用模型。
    """
    if cli_path is not None:
        if str(cli_path).strip() in ("", "-", "none", "None"):
            return None
        p = Path(cli_path).expanduser()
        if not p.is_file():
            cand = PROJECT_ROOT / str(cli_path).strip()
            if cand.is_file():
                return cand.resolve()
        return p.resolve() if p.is_file() else p
    det = {}
    if isinstance(config, dict):
        det = config.get("detection") or {}
    raw = det.get("basket_model", None)
    if raw is None:
        return (
            DEFAULT_BASKET_MODEL.resolve()
            if DEFAULT_BASKET_MODEL.is_file()
            else None
        )
    s = str(raw).strip()
    if s.lower() in ("", "-", "none", "null", "false", "0"):
        return None
    p = Path(s).expanduser()
    if not p.is_file():
        cand = PROJECT_ROOT / s
        if cand.is_file():
            return cand.resolve()
    return p.resolve() if p.is_file() else p


def resolve_driver_model_path(config=None, cli_path=None):
    """
    电动螺丝刀 YOLO：CLI --driver-model > detection.driver_model > 刀_best(2).pt。
    传 none/- 关闭。
    """
    if cli_path is not None:
        if str(cli_path).strip() in ("", "-", "none", "None"):
            return None
        p = Path(cli_path).expanduser()
        if not p.is_file():
            cand = PROJECT_ROOT / str(cli_path).strip()
            if cand.is_file():
                return cand.resolve()
        return p.resolve() if p.is_file() else p
    det = {}
    if isinstance(config, dict):
        det = config.get("detection") or {}
    raw = det.get("driver_model", None)
    if raw is None:
        for cand in (
            PROJECT_ROOT / "刀_best(2).pt",
            PROJECT_ROOT / "刀_best.pt",
        ):
            if cand.is_file():
                return cand.resolve()
        return None
    s = str(raw).strip()
    if s.lower() in ("", "-", "none", "null", "false", "0"):
        return None
    p = Path(s).expanduser()
    if not p.is_file():
        cand = PROJECT_ROOT / s
        if cand.is_file():
            return cand.resolve()
    return p.resolve() if p.is_file() else p


def resolve_board_model_path(config=None, cli_path=None):
    """
    螺丝板 YOLO：CLI --board-model > detection.board_model > 板_best.pt。
    传 none/- 关闭（仍可用 RGB 经典检测）。
    """
    if cli_path is not None:
        if str(cli_path).strip() in ("", "-", "none", "None"):
            return None
        p = Path(cli_path).expanduser()
        if not p.is_file():
            cand = PROJECT_ROOT / str(cli_path).strip()
            if cand.is_file():
                return cand.resolve()
        return p.resolve() if p.is_file() else p
    det = {}
    if isinstance(config, dict):
        det = config.get("detection") or {}
    raw = det.get("board_model", None)
    if raw is None:
        for cand in (
            PROJECT_ROOT / "板_best.pt",
            PROJECT_ROOT / "板best.pt",
        ):
            if cand.is_file():
                return cand.resolve()
        return None
    s = str(raw).strip()
    if s.lower() in ("", "-", "none", "null", "false", "0"):
        return None
    p = Path(s).expanduser()
    if not p.is_file():
        cand = PROJECT_ROOT / s
        if cand.is_file():
            return cand.resolve()
    return p.resolve() if p.is_file() else p


def best_detection_with_size_metric(
    detections,
    names,
    allowed_colors,
    depth_image,
    intrinsics,
    detection_cfg,
    eye,
    prior_mm=None,
):
    """
    YOLO 找框 + 深度定档过滤。
    yolo_all_then_metric=true 时：先收全部螺母框（含 single.pt 的 nut），再按深度档 ∩ 勾选过滤。
    返回 (detection, size_info|None)；无目标 (None, None)。
    """
    cfg = _size_metric_cfg(detection_cfg)
    enabled = bool(cfg.get("enabled", False))
    allowed = {str(c).strip().lower() for c in (allowed_colors or []) if c}
    if detections is None or len(detections) == 0 or not allowed:
        return None, None

    if not enabled or depth_image is None:
        # 无深度定档时：nut-only 模型无法按勾选尺寸过滤，直接取最高分螺母框
        pool = filter_nut_detections(detections, names)
        if pool:
            return max(pool, key=lambda d: float(d[4])), None
        return best_detection(detections, names=names, allowed_colors=allowed), None

    use_all = bool(cfg.get("yolo_all_then_metric", True))
    if use_all:
        pool = filter_nut_detections(detections, names)
        if not pool:
            # 兼容仍用 red/green/blue 的旧权重
            pool = filter_detections_by_colors(detections, names, allowed)
    else:
        pool = filter_detections_by_colors(detections, names, allowed)
        if not pool:
            pool = filter_nut_detections(detections, names)
    if not pool:
        return None, None

    scored = []
    for det in pool:
        yolo_cls = yolo_size_prior_class(names, det[5])
        yolo_conf = float(det[4]) if len(det) > 4 else None
        _u, _v, d_mm = measure_detection_uvd(
            depth_image, det, eye, prior_mm=prior_mm, freeze_below_mm=0.0,
        )
        info = measure_and_classify_nut(
            det[:4],
            d_mm,
            intrinsics,
            detection_cfg,
            yolo_cls=yolo_cls,
            yolo_conf=yolo_conf,
        )
        if info is None:
            # 无深度：nut-only 无法定档；仅当 YOLO 本身给出尺寸且在勾选内才保留
            if yolo_cls is not None and yolo_cls in allowed:
                scored.append((det, None, float(det[4])))
            continue
        metric_cls = info.get("class")
        if metric_cls not in allowed:
            continue
        score = float(det[4])
        score *= 0.55 + 0.45 * float(info.get("confidence") or 0.5)
        if info.get("ambiguous"):
            score *= 0.85
        if yolo_cls is not None and yolo_cls == metric_cls:
            score *= 1.08
        scored.append((det, info, score))

    if not scored:
        # 深度暂不可用：不定档、不硬选其它尺寸（避免勾「小」却对准「大」）
        return None, None
    # 多目标时先做单调性修复，再按勾选过滤
    refine_size_classes_relative([t[1] for t in scored], detection_cfg)
    kept = []
    for det, info, score in scored:
        if info is not None and info.get("class") not in allowed:
            continue
        kept.append((det, info, score))
    if not kept:
        # 有螺母但定档都不在勾选内：不返回目标（显示层另画全部框）
        return None, None
    kept.sort(key=lambda t: t[2], reverse=True)
    return kept[0][0], kept[0][1]


def object_in_ee(u, v, depth_mm, intrinsics, R_ee_cam, t_ee_cam):
    """检测点从相机坐标系变到手/法兰坐标系。"""
    point_cam = pixel_depth_to_camera_point(u, v, depth_mm, intrinsics)
    if point_cam is None:
        return None
    return R_ee_cam @ point_cam + t_ee_cam


def camera_optical_center(intrinsics):
    """相机光轴在图像上的投影 = 画面主点（中心）。"""
    return int(round(intrinsics["cx"])), int(round(intrinsics["cy"]))


def _servo_xy_rotate(delta_cam, rotate_deg):
    """图像平面内旋转伺服增量（度，逆时针）。用于腕相机安装姿态与 OpenCV 轴不一致。"""
    if abs(rotate_deg) < 1e-6:
        return delta_cam
    th = math.radians(float(rotate_deg))
    c, s = math.cos(th), math.sin(th)
    x, y = float(delta_cam[0]), float(delta_cam[1])
    out = np.array(delta_cam, dtype=np.float64, copy=True)
    out[0] = c * x - s * y
    out[1] = s * x + c * y
    return out


def _servo_xy_remap(delta_cam, rotate_deg=0.0, flip_x=False, flip_y=False):
    """先可选镜像，再平面旋转。"""
    out = np.array(delta_cam, dtype=np.float64, copy=True)
    if flip_x:
        out[0] = -out[0]
    if flip_y:
        out[1] = -out[1]
    return _servo_xy_rotate(out, rotate_deg)


def camera_delta_to_ee(
    delta_cam, R_ee_cam, xy_rotate_deg=0.0, flip_x=False, flip_y=False,
):
    """相机系平移增量（与对准伺服相同的 XY remap）→ 末端系。"""
    return R_ee_cam @ _servo_xy_remap(
        delta_cam, xy_rotate_deg, flip_x, flip_y
    )


def camera_correction_in_ee(
    u, v, depth_mm, intrinsics, target_mm, R_ee_cam, t_ee_cam, gain,
    xy_rotate_deg=0.0, flip_x=False, flip_y=False,
):
    """
    视觉伺服：对准光轴并调到 target_mm。
    相机平移 t_cam ≈ point_cam - target_cam（把光轴移向目标）；再变到末端系。
    """
    point_cam = pixel_depth_to_camera_point(u, v, depth_mm, intrinsics)
    if point_cam is None:
        return None
    target_cam = np.array([0.0, 0.0, target_mm / 1000.0], dtype=np.float64)
    delta_cam = _servo_xy_remap(
        point_cam - target_cam, xy_rotate_deg, flip_x, flip_y
    )
    return (R_ee_cam @ delta_cam) * gain


def center_only_correction_in_ee(
    u, v, depth_mm, intrinsics, R_ee_cam, gain,
    xy_rotate_deg=90.0, flip_x=True, flip_y=False,
):
    """
    比例对准：相机系横向误差 × gain → 末端系平移（不改深度）。
    深度只作尺度，钳在安全区间，避免 (u/f)*z 爆炸。
    """
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    z = 0.25
    if depth_mm is not None and np.isfinite(depth_mm) and depth_mm > 1.0:
        z = float(np.clip(depth_mm / 1000.0, 0.08, 0.60))
    point_cam = np.array(
        [(u - cx) / fx * z, (v - cy) / fy * z, z], dtype=np.float64
    )
    if not np.all(np.isfinite(point_cam)):
        return None
    target_cam = np.array([0.0, 0.0, z], dtype=np.float64)
    delta_cam = _servo_xy_remap(
        point_cam - target_cam, xy_rotate_deg, flip_x, flip_y
    )
    delta_cam[2] = 0.0
    # 仅防算崩：相机系横向硬限（正常比例控制远小于此）
    lat = float(np.hypot(delta_cam[0], delta_cam[1]))
    if lat > 0.15:
        delta_cam[:2] *= 0.15 / lat
    out = (R_ee_cam @ delta_cam) * float(gain)
    if not np.all(np.isfinite(out)):
        return None
    return out


def align_depth_to_size(depth, width, height):
    """把深度对齐到彩色分辨率，不做伪彩（比 depth_colormap_view 轻很多）。"""
    if depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = np.asarray(depth)
    if depth.shape[1] != int(width) or depth.shape[0] != int(height):
        depth = cv2.resize(
            depth, (int(width), int(height)), interpolation=cv2.INTER_NEAREST,
        )
    return depth


def depth_colormap_view(depth, width, height):
    """与 demo_detect_square 相同的伪彩色深度图。返回 (彩色图, 对齐后的深度或 None)。"""
    if depth is None:
        view = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            view,
            "Waiting for depth image...",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return view, None
    depth = align_depth_to_size(depth, width, height)
    depth_view = np.clip(depth.astype(np.float32), 0, 5000)
    depth_view = (depth_view * 255 / 5000).astype(np.uint8)
    depth_view[depth <= 0] = 0
    colored = cv2.applyColorMap(depth_view, cv2.COLORMAP_JET)
    colored[depth <= 0] = 0
    return colored, depth


def load_yolo_model(model_path, device, conf):
    try:
        import torch
        from yolov5.helpers import YOLOv5
        from yolov5.utils import downloads as y5_downloads
        import yolov5.models.common as y5_common
    except ImportError as error:
        raise RuntimeError(
            "缺少 YOLOv5: pip install yolov5 'setuptools<81'"
        ) from error
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到 YOLO 模型: {path}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU 不可用，请使用 --device cpu")
    # YOLOv5 DetectMultiBackend 会对权重字符串先 list_repo_files(HuggingFace)；
    # 相对名如 plate.pt 会被当成 repo id，离线时长时间卡住。本地文件直接跳过 hub。
    _orig_hub = y5_downloads.attempt_download_from_hub

    def _skip_hub_if_local(repo_id, hf_token=None, revision=None):
        cand = Path(str(repo_id)).expanduser()
        if cand.is_file():
            return None
        return _orig_hub(repo_id, hf_token=hf_token, revision=revision)

    y5_downloads.attempt_download_from_hub = _skip_hub_if_local
    y5_common.attempt_download_from_hub = _skip_hub_if_local
    try:
        model = YOLOv5(str(path), device=device)
    finally:
        y5_downloads.attempt_download_from_hub = _orig_hub
        y5_common.attempt_download_from_hub = _orig_hub
    model.model.conf = float(conf)
    return model


def yolo_class_names_brief(model):
    """启动日志用：简短打印类名表。"""
    names = getattr(getattr(model, "model", None), "names", None)
    if names is None:
        return "?"
    if isinstance(names, dict):
        return "{" + ", ".join(f"{k}:{v}" for k, v in sorted(names.items())) + "}"
    return str(list(names))


def _ros2_topic_exists(topic_name, timeout_s=2.0):
    """粗查 topic 是否已在图上（用于复用已开的 Orbbec，避免二次 launch 抢设备）。"""
    try:
        proc = subprocess.run(
            [
                "bash",
                "-lc",
                "source /opt/ros/jazzy/setup.bash >/dev/null 2>&1 && "
                "ros2 topic list",
            ],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_s)),
        )
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return False
    if proc.returncode != 0:
        return False
    lines = {ln.strip() for ln in (proc.stdout or "").splitlines()}
    return str(topic_name).strip() in lines


def start_depth_processes(depth_topic, color_topic="/camera/color/image_raw"):
    """
    启动 Orbbec launch（若彩色话题已存在则跳过）+ ros_depth_relay。
    返回 (launch_or_None, relay, depth_file, color_file, log_path)。
    """
    relay = Path(__file__).with_name("ros_depth_relay.py")
    pid = os.getpid()
    depth_file = Path(f"/tmp/lbot_depth_{pid}.png")
    color_file = Path(f"/tmp/lbot_color_{pid}.jpg")
    log_path = Path(f"/tmp/lbot_cam_{pid}.log")
    # 清掉可能残留的空文件，避免读到旧残片
    for p in (depth_file, color_file):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    color_topic = str(color_topic or "/camera/color/image_raw")
    depth_topic = str(depth_topic or "/camera/depth/image_raw")
    reuse = _ros2_topic_exists(color_topic, timeout_s=3.0)
    launch = None
    with open(log_path, "w", encoding="utf-8") as header:
        header.write(
            f"# lbot cam log pid={pid} reuse_orbbec={reuse}\n"
            f"# depth={depth_topic} color={color_topic}\n"
        )

    if not reuse:
        launch = subprocess.Popen(
            [
                "bash",
                "-lc",
                "source /opt/ros/jazzy/setup.bash && "
                "exec ros2 launch orbbec_camera gemini2.launch.py",
            ],
            stdout=open(log_path, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
    else:
        with open(log_path, "a", encoding="utf-8") as header:
            header.write("# skip orbbec launch: color topic already present\n")

    relay_process = subprocess.Popen(
        [
            "bash",
            "-lc",
            f"source /opt/ros/jazzy/setup.bash && exec /usr/bin/python3 "
            f"'{relay}' --topic '{depth_topic}' --color-topic '{color_topic}' "
            f"--depth-output '{depth_file}' --color-output '{color_file}'",
        ],
        stdout=open(log_path, "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    return launch, relay_process, depth_file, color_file, log_path


def servo_xy_remap_matrix(rotate_deg=0.0, flip_x=False, flip_y=False):
    """与 _servo_xy_remap 相同的 3×3：先镜像再绕光轴旋转。可并入 R_ee_cam。"""
    F = np.eye(3, dtype=np.float64)
    if flip_x:
        F[0, 0] = -1.0
    if flip_y:
        F[1, 1] = -1.0
    th = math.radians(float(rotate_deg))
    c, s = math.cos(th), math.sin(th)
    rot = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rot @ F


def build_camera_on_ee(config, fold_servo_xy=True):
    """
    yaml camera_on_ee → (R_ee_cam, t_ee_cam)。

    安装姿态 = euler_rpy（SO(3)）再右乘 optical flip（行列式 −1，无法写进欧拉）。
    旧 servo_xy_rotate/flip 在 fold_servo_xy=True 时一并并入，运行时勿再 remap。
    """
    eye = config["eye_in_hand"]
    mount = eye["camera_on_ee"]
    translation = np.array(mount["translation"], dtype=np.float64)
    rotation = euler_rpy_to_matrix(*mount["euler_rpy"])
    m_fx = bool(mount.get("flip_x", False))
    m_fy = bool(mount.get("flip_y", False))
    if m_fx or m_fy:
        rotation = rotation @ servo_xy_remap_matrix(0.0, m_fx, m_fy)
    if fold_servo_xy:
        rot = float(eye.get("servo_xy_rotate_deg", 0.0) or 0.0)
        fx = bool(eye.get("servo_flip_x", False))
        fy = bool(eye.get("servo_flip_y", False))
        if abs(rot) > 1e-6 or fx or fy:
            rotation = rotation @ servo_xy_remap_matrix(rot, fx, fy)
    return rotation, translation


def camera_on_ee_from_mjcf(mjcf_path):
    """
    从 workstation.mjcf 读取右腕相机相对法兰(tool0/R8)的平移。
    返回 (optical_in_parent, euler_rpy, optical_offset_in_body)。
    optical = arm_right_camera 本体位置 + 双目 <camera> 中点偏移。
    """
    import xml.etree.ElementTree as ET

    path = Path(mjcf_path)
    root = ET.parse(path).getroot()
    cam_body = None
    for body in root.iter("body"):
        if body.get("name") == "arm_right_camera":
            cam_body = body
            break
    if cam_body is None:
        raise FileNotFoundError(f"{path} 中没有 arm_right_camera")

    body_pos = np.array(
        [float(x) for x in cam_body.get("pos", "0 0 0").split()],
        dtype=np.float64,
    )
    optical_in_body = np.zeros(3, dtype=np.float64)
    cam_nodes = list(cam_body.findall("camera"))
    if cam_nodes:
        offsets = [
            np.array([float(x) for x in node.get("pos", "0 0 0").split()], dtype=np.float64)
            for node in cam_nodes
        ]
        optical_in_body = np.mean(offsets, axis=0)
    optical_in_parent = body_pos + optical_in_body
    euler_rpy = [0.0, 0.0, 0.0]
    body_euler = cam_body.get("euler")
    if body_euler:
        parts = [float(x) for x in body_euler.split()]
        if len(parts) >= 3:
            euler_rpy = parts[:3]
    return optical_in_parent, euler_rpy, optical_in_body


def camera_body_pos_for_optical(t_ee_cam, optical_in_body, euler_rpy=None):
    """
    yaml 光心 t_ee_cam（tool0 系）→ arm_right_camera 的 body pos（父系=tool0）。
    optical = body_pos + R(euler) @ optical_in_body
    """
    t = np.asarray(t_ee_cam, dtype=np.float64).reshape(3)
    opt = np.asarray(optical_in_body, dtype=np.float64).reshape(3)
    if euler_rpy is None:
        R = np.eye(3, dtype=np.float64)
    else:
        e = np.asarray(euler_rpy, dtype=np.float64).reshape(3)
        R = euler_rpy_to_matrix(float(e[0]), float(e[1]), float(e[2]))
    return t - R @ opt


def apply_camera_on_ee_to_mj_model(
    model,
    t_ee_cam,
    optical_in_body,
    euler_rpy=None,
    body_name="arm_right_camera",
):
    """
    把 MuJoCo 中腕相机 body 挪到与 yaml camera_on_ee 光心一致。
    返回 (old_body_pos, new_body_pos)；失败返回 None。
    """
    try:
        import mujoco
    except ImportError:
        return None
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    new_pos = camera_body_pos_for_optical(t_ee_cam, optical_in_body, euler_rpy=euler_rpy)
    old = np.asarray(model.body_pos[bid], dtype=np.float64).copy()
    model.body_pos[bid] = new_pos
    if euler_rpy is not None:
        e = np.asarray(euler_rpy, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(e)) > 1e-9:
            quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_euler2Quat(quat, e, "xyz")
            model.body_quat[bid] = quat
    return old, new_pos.copy()


def write_mjcf_camera_body_pos(mjcf_path, body_pos, body_name="arm_right_camera"):
    """写回 MJCF 里 arm_right_camera 的 pos（使文件与 yaml 光心一致）。"""
    import xml.etree.ElementTree as ET

    path = Path(mjcf_path)
    tree = ET.parse(path)
    root = tree.getroot()
    cam_body = None
    for body in root.iter("body"):
        if body.get("name") == body_name:
            cam_body = body
            break
    if cam_body is None:
        raise FileNotFoundError(f"{path} 中没有 {body_name}")
    p = np.asarray(body_pos, dtype=np.float64).reshape(3)
    cam_body.set("pos", f"{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}")
    try:
        tree.write(str(path), encoding="utf-8", xml_declaration=True)
    except PermissionError:
        path.chmod(path.stat().st_mode | 0o200)
        tree.write(str(path), encoding="utf-8", xml_declaration=True)
    return p


def apply_mjcf_camera_to_config(config, mjcf_path):
    """用模型相机支架覆盖 config 中的 camera_on_ee。"""
    translation, euler_rpy, _optical_in_body = camera_on_ee_from_mjcf(mjcf_path)
    config.setdefault("eye_in_hand", {}).setdefault("camera_on_ee", {})
    config["eye_in_hand"]["camera_on_ee"]["translation"] = [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    ]
    config["eye_in_hand"]["camera_on_ee"]["euler_rpy"] = list(euler_rpy)
    return build_camera_on_ee(config)


def save_camera_on_ee_translation(config_path, translation, prefer_yaml=None):
    """写回 camera_on_ee.translation；prefer_yaml=True 时下次启动不再被 MJCF 覆盖。"""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("需要 PyYAML: pip install pyyaml") from exc
    new_t = [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    ]
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    eye = config.setdefault("eye_in_hand", {})
    mount = eye.setdefault("camera_on_ee", {})
    old_t = mount.get("translation")
    changed = True
    if old_t is not None and len(old_t) == 3:
        if max(abs(float(a) - float(b)) for a, b in zip(old_t, new_t)) < 1e-9:
            changed = False
    if changed:
        mount["translation"] = new_t
    if prefer_yaml is not None:
        eye["camera_on_ee_prefer_yaml"] = bool(prefer_yaml)
        changed = True
    if not changed:
        return
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def stop_process(process):
    if process is None:
        return
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
    except Exception:
        pass
    # 关闭 Popen 自带的 stdout 文件（若曾 open() 传入）
    try:
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()
    except Exception:
        pass


def solve_translation_from_teach(robot_xyz, camera_point, rotation):
    """已知末端在方块上的位姿 + 相机测得的 3D 点，反解 camera_to_robot 平移 t。"""
    robot_xyz = np.asarray(robot_xyz, dtype=np.float64)
    camera_point = np.asarray(camera_point, dtype=np.float64)
    return robot_xyz - rotation @ camera_point


def save_config_translation(config_path, translation):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("需要 PyYAML: pip install pyyaml") from exc
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["camera_to_robot"]["translation"] = [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    ]
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
