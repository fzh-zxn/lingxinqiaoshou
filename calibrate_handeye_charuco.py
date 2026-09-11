#!/usr/bin/env python3
"""
眼在手上 · ChArUco 手眼标定（方法 A / Tsai AX=XB）

板固定桌面，右臂多姿态采集：法兰位姿(基座系) + 板在相机系位姿 → 解 camera_on_ee。

用法：
  # 先退出 demo_pick_square（或确保相机话题可复用）
  cd lbot_pick_demo
  python calibrate_handeye_charuco.py

  # 仅用已采样本重算
  python calibrate_handeye_charuco.py --solve-only

  # 确认结果后写回 yaml（只改 camera_on_ee，不动 plate_bias）
  python calibrate_handeye_charuco.py --solve-only --write-yaml

按键（采集模式）：
  s / 空格  记录当前样本（板需检出足够角点）
  c         清除全部样本
  q / Esc   结束采集并求解
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from lbot.lbot_robot import LbotArm, LbotRobot
from lbot_grasp_utils import (
    build_camera_on_ee,
    euler_rpy_to_matrix,
    pose_to_matrix,
    rotation_matrix_to_rpy,
    servo_xy_remap_matrix,
    start_depth_processes,
    stop_process,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "grasp_config.yaml"
DEFAULT_OUT = ROOT / "calib_handeye"

# 已验证板参数
CHARUCO_SQUARES = (14, 9)
CHARUCO_SQUARE_M = 0.020
CHARUCO_MARKER_M = 0.015
CHARUCO_DICT = cv2.aruco.DICT_5X5_100
MIN_CORNERS = 40
MIN_SAMPLES = 8


# ---------------------------------------------------------------------------
# Hand-eye solvers (OpenCV5 无 calibrateHandEye，自实现 Tsai / Park)
# ---------------------------------------------------------------------------

def _skew(v):
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def _rot_log(R):
    """SO(3) → so(3) 向量（旋转向量）。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_th = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = math.acos(cos_th)
    if th < 1e-8:
        return np.zeros(3, dtype=np.float64)
    w = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    return w * (th / (2.0 * math.sin(th)))


def _motion_pairs(Rg, tg, Rc, tc):
    """
    Eye-in-hand 相邻运动对（使 A X = X B，X=cam→ee）：
      A = inv(G_i) @ G_{i+1}
      B = C_i @ inv(C_{i+1})
    其中 G=法兰相对基座，C=标定板相对相机（OpenCV）。
    """
    n = len(Rg)
    As, Bs = [], []
    for i in range(n - 1):
        Gi = np.eye(4)
        Gi[:3, :3], Gi[:3, 3] = Rg[i], tg[i]
        Gj = np.eye(4)
        Gj[:3, :3], Gj[:3, 3] = Rg[i + 1], tg[i + 1]
        A = np.linalg.inv(Gi) @ Gj

        Ci = np.eye(4)
        Ci[:3, :3], Ci[:3, 3] = Rc[i], tc[i]
        Cj = np.eye(4)
        Cj[:3, :3], Cj[:3, 3] = Rc[i + 1], tc[i + 1]
        B = Ci @ np.linalg.inv(Cj)

        # 跳过几乎不动的样本对
        if np.linalg.norm(_rot_log(A[:3, :3])) < math.radians(3.0):
            continue
        if np.linalg.norm(A[:3, 3]) < 0.005 and np.linalg.norm(_rot_log(A[:3, :3])) < math.radians(5.0):
            continue
        As.append(A)
        Bs.append(B)
    return As, Bs


def solve_handeye_tsai(Rg, tg, Rc, tc):
    """Tsai-Lenz：先旋转后平移。返回 R_cam2ee, t_cam2ee（p_ee = R @ p_cam + t）。"""
    As, Bs = _motion_pairs(Rg, tg, Rc, tc)
    if len(As) < 3:
        raise RuntimeError(f"有效运动对过少 ({len(As)})，请多采、姿态差大一些")

    # 旋转：对每个运动 alpha, beta：skew(alpha+beta) * p = beta - alpha
    # R = I + 2/(1+||p||^2) * (skew(p) + skew(p)^2) 的 Tsai 参数化
    C_list, d_list = [], []
    for A, B in zip(As, Bs):
        a = _rot_log(A[:3, :3])
        b = _rot_log(B[:3, :3])
        an, bn = np.linalg.norm(a), np.linalg.norm(b)
        if an < 1e-8 or bn < 1e-8:
            continue
        a = a / an * 2.0 * math.sin(an / 2.0)
        b = b / bn * 2.0 * math.sin(bn / 2.0)
        C_list.append(_skew(a + b))
        d_list.append(b - a)
    if len(C_list) < 3:
        raise RuntimeError("Tsai 旋转方程不足")
    C = np.vstack(C_list)
    d = np.concatenate(d_list)
    p, *_ = np.linalg.lstsq(C, d, rcond=None)
    pn2 = float(np.dot(p, p))
    R = np.eye(3) + (2.0 / (1.0 + pn2)) * (_skew(p) + _skew(p) @ _skew(p))

    # 平移：(I - Ra) t = ta - R tb
    E_list, f_list = [], []
    for A, B in zip(As, Bs):
        Ra, ta = A[:3, :3], A[:3, 3]
        tb = B[:3, 3]
        E_list.append(np.eye(3) - Ra)
        f_list.append(ta - R @ tb)
    E = np.vstack(E_list)
    f = np.concatenate(f_list)
    t, *_ = np.linalg.lstsq(E, f, rcond=None)
    return R, t.reshape(3)


def solve_handeye_park(Rg, tg, Rc, tc):
    """Park-Martin 旋转最小二乘 + 平移最小二乘。"""
    As, Bs = _motion_pairs(Rg, tg, Rc, tc)
    if len(As) < 3:
        raise RuntimeError(f"有效运动对过少 ({len(As)})")

    M = np.zeros((3, 3), dtype=np.float64)
    for A, B in zip(As, Bs):
        M += _rot_log(B[:3, :3]).reshape(3, 1) @ _rot_log(A[:3, :3]).reshape(1, 3)
    U, _, Vt = np.linalg.svd(M)
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T

    E_list, f_list = [], []
    for A, B in zip(As, Bs):
        Ra, ta = A[:3, :3], A[:3, 3]
        tb = B[:3, 3]
        E_list.append(np.eye(3) - Ra)
        f_list.append(ta - R @ tb)
    E = np.vstack(E_list)
    f = np.concatenate(f_list)
    t, *_ = np.linalg.lstsq(E, f, rcond=None)
    return R, t.reshape(3)


def solve_handeye_t_given_R(Rg, tg, Rc, tc, R):
    """
    旋转已知（CAD / 手调 yaml）时，只解平移。
    用 AX=XB 的平移式：(I - Ra) t = ta - R tb
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    As, Bs = _motion_pairs(Rg, tg, Rc, tc)
    if len(As) < 2:
        raise RuntimeError(f"有效运动对过少 ({len(As)})，无法只解平移")
    E_list, f_list = [], []
    for A, B in zip(As, Bs):
        Ra, ta = A[:3, :3], A[:3, 3]
        tb = B[:3, 3]
        E_list.append(np.eye(3) - Ra)
        f_list.append(ta - R @ tb)
    E = np.vstack(E_list)
    f = np.concatenate(f_list)
    t, *_ = np.linalg.lstsq(E, f, rcond=None)
    return R, t.reshape(3)


def handeye_board_positions(Rg, tg, Rc, tc, R, t):
    """各样本把板原点投到基座系：p = tg + Rg @ (R @ tc + t)。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    ps = []
    for Rgi, tgi, tci in zip(Rg, tg, tc):
        Rgi = np.asarray(Rgi, dtype=np.float64).reshape(3, 3)
        tgi = np.asarray(tgi, dtype=np.float64).reshape(3)
        tci = np.asarray(tci, dtype=np.float64).reshape(3)
        ps.append(tgi + Rgi @ (R @ tci + t))
    return np.asarray(ps, dtype=np.float64)


def handeye_board_stats(Rg, tg, Rc, tc, R, t):
    """板原点在基座系的 mean 与 std(mm)。固定板时 std 越小越好。"""
    ps = handeye_board_positions(Rg, tg, Rc, tc, R, t)
    mean = ps.mean(axis=0)
    std_mm = ps.std(axis=0) * 1000.0
    rms_mm = float(np.sqrt(np.mean(np.sum((ps - mean) ** 2, axis=1)))) * 1000.0
    return mean, std_mm, rms_mm


def solve_handeye_t_board(Rg, tg, Rc, tc, R):
    """
    固定 R：用「板在基座系应重合」线性解 t（及板原点）。
    Rg_i @ t - p = -(tg_i + Rg_i @ R @ tc_i)
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    A_rows, b_rows = [], []
    for Rgi, tgi, tci in zip(Rg, tg, tc):
        Rgi = np.asarray(Rgi, dtype=np.float64).reshape(3, 3)
        tgi = np.asarray(tgi, dtype=np.float64).reshape(3)
        tci = np.asarray(tci, dtype=np.float64).reshape(3)
        A_rows.append(np.hstack([Rgi, -np.eye(3)]))
        b_rows.append(-(tgi + Rgi @ R @ tci))
    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return R, x[:3].reshape(3)


def _rodrigues(w):
    w = np.asarray(w, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = w / th
    K = _skew(k)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def solve_handeye_refine(
    Rg, tg, Rc, tc, R0, t0, fix_R=False, iters=50, board_weight=1.0, ax_weight=0.25,
):
    """
    非线性精修：默认同时拟合
      · 板在基座系重合（主目标，工程上最直观）
      · AX=XB 平移残差（次要）
    fix_R=True 时只动平移（假装姿态已知）。
    """
    R0 = np.asarray(R0, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t0, dtype=np.float64).reshape(3).copy()
    w = np.zeros(3, dtype=np.float64)
    As, Bs = _motion_pairs(Rg, tg, Rc, tc)
    n = len(Rg)

    def pack_R():
        return R0 if fix_R else (R0 @ _rodrigues(w))

    for _ in range(int(iters)):
        R = pack_R()
        ps = handeye_board_positions(Rg, tg, Rc, tc, R, t)
        pbar = ps.mean(axis=0)
        r_list, j_rows = [], []

        # board residuals
        for i in range(n):
            ri = (ps[i] - pbar) * (1000.0 * float(board_weight))
            r_list.append(ri)
            # numerical J w.r.t params
            if fix_R:
                J = np.zeros((3, 3), dtype=np.float64)
                for k in range(3):
                    tp = t.copy()
                    tp[k] += 1e-6
                    psp = handeye_board_positions(Rg, tg, Rc, tc, R, tp)
                    rip = (psp[i] - psp.mean(axis=0)) * (1000.0 * float(board_weight))
                    J[:, k] = (rip - ri) / 1e-6
            else:
                J = np.zeros((3, 6), dtype=np.float64)
                for k in range(3):
                    wp = w.copy()
                    wp[k] += 1e-6
                    Rp = R0 @ _rodrigues(wp)
                    psp = handeye_board_positions(Rg, tg, Rc, tc, Rp, t)
                    rip = (psp[i] - psp.mean(axis=0)) * (1000.0 * float(board_weight))
                    J[:, k] = (rip - ri) / 1e-6
                for k in range(3):
                    tp = t.copy()
                    tp[k] += 1e-6
                    psp = handeye_board_positions(Rg, tg, Rc, tc, R, tp)
                    rip = (psp[i] - psp.mean(axis=0)) * (1000.0 * float(board_weight))
                    J[:, 3 + k] = (rip - ri) / 1e-6
            j_rows.append(J)

        # AX=XB translation residuals
        for A, B in zip(As, Bs):
            Ra, ta = A[:3, :3], A[:3, 3]
            tb = B[:3, 3]
            ri = ((np.eye(3) - Ra) @ t - (ta - R @ tb)) * (1000.0 * float(ax_weight))
            r_list.append(ri)
            if fix_R:
                J = (np.eye(3) - Ra) * (1000.0 * float(ax_weight))
            else:
                J = np.zeros((3, 6), dtype=np.float64)
                for k in range(3):
                    wp = w.copy()
                    wp[k] += 1e-6
                    Rp = R0 @ _rodrigues(wp)
                    rip = ((np.eye(3) - Ra) @ t - (ta - Rp @ tb)) * (
                        1000.0 * float(ax_weight)
                    )
                    J[:, k] = (rip - ri) / 1e-6
                J[:, 3:] = (np.eye(3) - Ra) * (1000.0 * float(ax_weight))
            j_rows.append(J)

        r = np.concatenate(r_list)
        J = np.vstack(j_rows)
        dx, *_ = np.linalg.lstsq(J, -r, rcond=None)
        if fix_R:
            t = t + dx.reshape(3)
        else:
            w = w + dx[:3]
            t = t + dx[3:]
        if float(np.linalg.norm(dx)) < 1e-9:
            break

    return pack_R(), t.reshape(3)


def handeye_reproj_stats(Rg, tg, Rc, tc, R, t):
    """
    一致性：对每对 i,j 比较 A X 与 X B 的旋转角/平移差。
    """
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t
    As, Bs = _motion_pairs(Rg, tg, Rc, tc)
    ang, trans = [], []
    for A, B in zip(As, Bs):
        LHS = A @ X
        RHS = X @ B
        dR = LHS[:3, :3].T @ RHS[:3, :3]
        ang.append(math.degrees(np.linalg.norm(_rot_log(dR))))
        trans.append(float(np.linalg.norm(LHS[:3, 3] - RHS[:3, 3])) * 1000.0)
    if not ang:
        return 0.0, 0.0
    return float(np.mean(ang)), float(np.mean(trans))


# ---------------------------------------------------------------------------
# Vision / robot helpers
# ---------------------------------------------------------------------------

def load_config(path: Path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        CHARUCO_SQUARES, CHARUCO_SQUARE_M, CHARUCO_MARKER_M, dictionary,
    )
    return cv2.aruco.CharucoDetector(board), board


def camera_K_dist(cfg):
    intr = cfg["camera_intrinsics"]
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    # 出厂内参未存畸变 → 零
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist, (int(intr["width"]), int(intr["height"]))


def estimate_board_pose(detector, board, frame, K, dist, min_corners=MIN_CORNERS):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    char_corners, char_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    n_c = 0 if char_ids is None else int(len(char_ids))
    n_m = 0 if marker_ids is None else int(len(marker_ids))
    if char_ids is None or n_c < min_corners:
        return None, n_c, n_m, marker_corners, marker_ids, char_corners, char_ids
    obj, img = board.matchImagePoints(char_corners, char_ids)
    if obj is None or len(obj) < min_corners:
        return None, n_c, n_m, marker_corners, marker_ids, char_corners, char_ids
    ok, rvec, tvec = cv2.solvePnP(
        obj.astype(np.float32), img.astype(np.float32), K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, n_c, n_m, marker_corners, marker_ids, char_corners, char_ids
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return (R, t), n_c, n_m, marker_corners, marker_ids, char_corners, char_ids


def read_ros_color(color_file: Path):
    if not color_file.is_file() or color_file.stat().st_size < 100:
        return None
    return cv2.imread(str(color_file), cv2.IMREAD_COLOR)


def flange_pose_matrix(robot) -> np.ndarray | None:
    pose = robot.get_cartesian_pose(LbotArm.RIGHT_ARM)
    if pose is None:
        return None
    pos, eul = pose
    return pose_to_matrix(pos, eul)


def r_t_to_yaml_mount(R_full, t, keep_flip_y=True, keep_flip_x=False, ref_rpy=None):
    """
    求解得到的 R_ee_cam（OpenCV 光心系→法兰）拆成 yaml：
      R_full = R_euler @ F(flip)
    保留现有 flip_x/y（与运行时 build_camera_on_ee 一致）。
    """
    F = servo_xy_remap_matrix(0.0, keep_flip_x, keep_flip_y)
    # R_full = R_euler @ F  →  R_euler = R_full @ F^{-1} = R_full @ F
    R_euler = R_full @ F
    # 保证接近旋转
    U, _, Vt = np.linalg.svd(R_euler)
    R_euler = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    rpy = rotation_matrix_to_rpy(R_euler)
    if ref_rpy is not None:
        # 选与旧值接近的等价角（简单：直接用 unwrap 差）
        ref = np.asarray(ref_rpy, dtype=np.float64)
        cand = np.asarray(rpy, dtype=np.float64)
        for i in range(3):
            while cand[i] - ref[i] > math.pi:
                cand[i] -= 2 * math.pi
            while cand[i] - ref[i] < -math.pi:
                cand[i] += 2 * math.pi
        rpy = tuple(cand.tolist())
    return {
        "translation": [float(t[0]), float(t[1]), float(t[2])],
        "euler_rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
        "flip_x": bool(keep_flip_x),
        "flip_y": bool(keep_flip_y),
    }


def write_camera_on_ee(config_path: Path, mount: dict):
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    eye = cfg.setdefault("eye_in_hand", {})
    old = dict(eye.get("camera_on_ee") or {})
    eye["camera_on_ee"] = {
        "translation": list(mount["translation"]),
        "euler_rpy": list(mount["euler_rpy"]),
        "flip_x": bool(mount.get("flip_x", old.get("flip_x", False))),
        "flip_y": bool(mount.get("flip_y", old.get("flip_y", True))),
    }
    eye["camera_on_ee_prefer_yaml"] = True
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return old, eye["camera_on_ee"]


# ---------------------------------------------------------------------------
# Capture / solve
# ---------------------------------------------------------------------------

def save_samples(out_dir: Path, samples: list, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = {
        "Rg": np.stack([s["Rg"] for s in samples]),
        "tg": np.stack([s["tg"] for s in samples]),
        "Rc": np.stack([s["Rc"] for s in samples]),
        "tc": np.stack([s["tc"] for s in samples]),
        "n_corners": np.array([s["n_corners"] for s in samples]),
    }
    np.savez(out_dir / "samples.npz", **npz)
    (out_dir / "samples_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def load_samples(out_dir: Path):
    data = np.load(out_dir / "samples.npz")
    return data["Rg"], data["tg"], data["Rc"], data["tc"]


def enumerate_handeye_solutions(cfg, Rg, tg, Rc, tc):
    """
    多种求解器对比，返回 list[dict]。
    评分优先看「板在基座系重合 RMS」(固定板物理意义最直接)，其次 AX=XB 平移。
    """
    Rg = [np.asarray(r) for r in Rg]
    tg = [np.asarray(t) for t in tg]
    Rc = [np.asarray(r) for r in Rc]
    tc = [np.asarray(t) for t in tc]
    R_yaml, t_yaml = build_camera_on_ee(cfg, fold_servo_xy=True)

    candidates = []

    def _add(name, R, t):
        ang, mm = handeye_reproj_stats(Rg, tg, Rc, tc, R, t)
        _mean, std_mm, rms = handeye_board_stats(Rg, tg, Rc, tc, R, t)
        candidates.append(
            {
                "name": name,
                "R": np.asarray(R, dtype=np.float64),
                "t": np.asarray(t, dtype=np.float64).reshape(3),
                "ang": float(ang),
                "mm": float(mm),
                "board_rms_mm": float(rms),
                "board_std_mm": np.asarray(std_mm, dtype=np.float64),
            }
        )

    for name, fn in (("tsai", solve_handeye_tsai), ("park", solve_handeye_park)):
        try:
            R, t = fn(Rg, tg, Rc, tc)
            _add(name, R, t)
        except Exception as exc:
            candidates.append({"name": name, "error": str(exc)})

    # 固定 yaml 旋转，只解平移（CAD/手调姿态可信时用）
    try:
        R, t = solve_handeye_t_given_R(Rg, tg, Rc, tc, R_yaml)
        _add("fixR_yaml+t_motion", R, t)
    except Exception as exc:
        candidates.append({"name": "fixR_yaml+t_motion", "error": str(exc)})

    try:
        R, t = solve_handeye_t_board(Rg, tg, Rc, tc, R_yaml)
        _add("fixR_yaml+t_board", R, t)
    except Exception as exc:
        candidates.append({"name": "fixR_yaml+t_board", "error": str(exc)})

    # Park 初值 + 板系一致性精修（推荐）
    park = next((c for c in candidates if c.get("name") == "park" and "R" in c), None)
    if park is not None:
        try:
            R, t = solve_handeye_refine(
                Rg, tg, Rc, tc, park["R"], park["t"], fix_R=False,
            )
            _add("park+board_refine", R, t)
        except Exception as exc:
            candidates.append({"name": "park+board_refine", "error": str(exc)})
        try:
            R, t = solve_handeye_refine(
                Rg, tg, Rc, tc, park["R"], park["t"], fix_R=True,
            )
            _add("parkR+t_refine", R, t)
        except Exception as exc:
            candidates.append({"name": "parkR+t_refine", "error": str(exc)})

    # yaml 初值 + 只精修 t
    try:
        R, t = solve_handeye_refine(
            Rg, tg, Rc, tc, R_yaml, t_yaml, fix_R=True,
        )
        _add("yamlR+t_refine", R, t)
    except Exception as exc:
        candidates.append({"name": "yamlR+t_refine", "error": str(exc)})

    ok = [c for c in candidates if "R" in c]
    if not ok:
        return candidates, None

    # 综合分：板 RMS 为主，AX 平移为辅；固定 yamlR 若板 RMS 很差则降权
    def score(c):
        rms = float(c["board_rms_mm"])
        mm = float(c["mm"])
        ang = float(c["ang"])
        # 固定 yaml 旋转但板散很大 → 说明姿态已知假设不成立，惩罚
        pen = 0.0
        if c["name"].startswith("fixR_yaml") or c["name"].startswith("yamlR"):
            if rms > 20.0:
                pen += 100.0 + rms
        return (rms + 0.35 * mm + 0.15 * ang) + pen

    best = min(ok, key=score)
    return candidates, best


def run_solve(cfg, out_dir: Path, write_yaml: bool, config_path: Path):
    Rg, tg, Rc, tc = load_samples(out_dir)
    print(f"载入样本 {len(Rg)} 组")

    candidates, best = enumerate_handeye_solutions(cfg, Rg, tg, Rc, tc)
    for c in candidates:
        if "error" in c:
            print(f"[{c['name']}] 失败: {c['error']}")
            continue
        print(
            f"[{c['name']}] AX=XB {c['ang']:.2f}°/{c['mm']:.1f}mm  "
            f"板RMS={c['board_rms_mm']:.1f}mm  "
            f"板std_mm={np.round(c['board_std_mm'], 1).tolist()}  "
            f"t={np.round(c['t'], 4).tolist()}"
        )

    if best is None:
        raise SystemExit("所有求解器失败")

    R, t, ang, mm = best["R"], best["t"], best["ang"], best["mm"]
    print(
        f"\n选用: {best['name']}  (板RMS={best['board_rms_mm']:.1f}mm, "
        f"AX平移={mm:.1f}mm)"
    )
    if str(best["name"]).startswith(("fixR_yaml", "yamlR")):
        print("提示: 固定了 yaml 旋转只解平移；若板RMS仍大，说明姿态假设不成立。")

    mount_old = cfg["eye_in_hand"]["camera_on_ee"]
    R_old, t_old = build_camera_on_ee(cfg, fold_servo_xy=True)
    print(f"当前 yaml(折叠后) t={np.round(t_old, 4).tolist()}")
    print(f"新解             t={np.round(t, 4).tolist()}")
    print(f"Δt(mm)={np.round((t - t_old) * 1000, 2).tolist()}")

    mount_new = r_t_to_yaml_mount(
        R, t,
        keep_flip_y=bool(mount_old.get("flip_y", True)),
        keep_flip_x=bool(mount_old.get("flip_x", False)),
        ref_rpy=mount_old.get("euler_rpy"),
    )

    # 校验：yaml 重建应接近 R,t
    cfg_try = yaml.safe_load(yaml.dump(cfg))
    cfg_try["eye_in_hand"]["camera_on_ee"] = mount_new
    R_chk, t_chk = build_camera_on_ee(cfg_try, fold_servo_xy=True)
    print(f"写回后折叠校验 Δt(mm)={np.round((t_chk - t)*1000, 2).tolist()}")

    snippet = {
        "camera_on_ee": mount_new,
        "camera_on_ee_prefer_yaml": True,
        "_meta": {
            "method": best["name"],
            "pair_rot_err_deg": ang,
            "pair_trans_err_mm": mm,
            "board_rms_mm": float(best["board_rms_mm"]),
            "n_samples": int(len(Rg)),
            "note": "勿改 plate_bias_*；写回前请实机验参",
        },
    }
    out_yaml = out_dir / "camera_on_ee_snippet.yaml"
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(snippet, f, allow_unicode=True, sort_keys=False)
    print(f"\n已写建议片段: {out_yaml}")
    print("建议 camera_on_ee:")
    print(yaml.dump({"camera_on_ee": mount_new}, allow_unicode=True, sort_keys=False))

    if write_yaml:
        old, new = write_camera_on_ee(config_path, mount_new)
        print(f"已写回 {config_path}")
        print(f"  旧 translation={old.get('translation')}")
        print(f"  新 translation={new.get('translation')}")
        print("  plate_bias 未改动")
    else:
        print("未写回 grasp_config（加 --write-yaml 才写入）")

    return mount_new


def run_capture(args):
    cfg = load_config(args.config)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    K, dist, _wh = camera_K_dist(cfg)
    detector, board = make_board()

    host = str(cfg.get("robot", {}).get("host", "192.168.10.21"))
    print(f"连接机器人 {host} …")
    robot = LbotRobot(host)
    if not robot.connect(timeout=10.0):
        raise RuntimeError("机器人连接失败")

    print("启动/复用 Orbbec 彩色 …")
    launch, relay, _df, color_file, log_path = start_depth_processes(
        args.depth_topic
    )
    print(f"相机日志: {log_path}")

    # 等首帧
    t0 = time.time()
    frame0 = None
    while time.time() - t0 < args.startup_wait:
        frame0 = read_ros_color(color_file)
        if frame0 is not None:
            break
        time.sleep(0.05)
    if frame0 is None:
        stop_process(relay)
        stop_process(launch)
        robot.disconnect()
        raise RuntimeError("无彩色画面：请先关 demo_pick_square 或检查相机")

    print(f"彩色就绪 {frame0.shape[1]}x{frame0.shape[0]}")
    print("板固定桌面；移动右臂到不同姿态，角点够了按 s 记录。")
    print(f"目标 ≥{MIN_SAMPLES} 组，建议 12～15；姿态差尽量大（绕板转/高低/侧倾）")

    samples = []
    window = "HandEye ChArUco"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = read_ros_color(color_file)
            if frame is None:
                time.sleep(0.02)
                continue

            pose_est, n_c, n_m, mcorn, mids, ccorn, cids = estimate_board_pose(
                detector, board, frame, K, dist, min_corners=args.min_corners,
            )
            Tg = flange_pose_matrix(robot)

            vis = frame.copy()
            if mids is not None and mcorn is not None:
                cv2.aruco.drawDetectedMarkers(vis, mcorn, mids)
            if cids is not None and ccorn is not None:
                for p in np.asarray(ccorn).reshape(-1, 2):
                    cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)

            ok = pose_est is not None and Tg is not None
            color = (0, 220, 0) if ok else (0, 140, 255)
            lines = [
                f"samples={len(samples)}  corners={n_c} markers={n_m}  "
                f"{'READY' if ok else 'wait'}",
                "s=record  c=clear  q=solve&quit",
            ]
            if Tg is not None:
                lines.append(
                    f"flange xyz=[{Tg[0,3]:.3f},{Tg[1,3]:.3f},{Tg[2,3]:.3f}]"
                )
            y = 28
            for t in lines:
                cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
                y += 28

            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                samples.clear()
                print("已清空样本")
            if key in (ord("s"), ord(" ")):
                if not ok:
                    print(f"未记录：corners={n_c}（需>={args.min_corners}）或无法兰位姿")
                    continue
                Rc, tc = pose_est
                Rg, tg = Tg[:3, :3].copy(), Tg[:3, 3].copy()
                # 与上一帧太近则拒
                if samples:
                    d = float(np.linalg.norm(tg - samples[-1]["tg"]))
                    dR = math.degrees(
                        np.linalg.norm(_rot_log(Rg.T @ samples[-1]["Rg"]))
                    )
                    if d < 0.02 and dR < 5.0:
                        print(f"与上一样本太近 (Δt={d*1000:.0f}mm ΔR={dR:.1f}°)，换姿态再采")
                        continue
                idx = len(samples) + 1
                img_path = out_dir / f"pose_{idx:02d}.png"
                cv2.imwrite(str(img_path), frame)
                samples.append(
                    {
                        "Rg": Rg,
                        "tg": tg,
                        "Rc": Rc,
                        "tc": tc,
                        "n_corners": n_c,
                        "image": img_path.name,
                    }
                )
                print(
                    f"#{len(samples)} 记录 corners={n_c}  "
                    f"flange={np.round(tg,3).tolist()}  "
                    f"board_z={tc[2]:.3f}m"
                )
    finally:
        cv2.destroyAllWindows()
        stop_process(relay)
        stop_process(launch)
        try:
            robot.disconnect()
        except Exception:
            pass

    if len(samples) < MIN_SAMPLES:
        print(f"样本不足 ({len(samples)}<{MIN_SAMPLES})，已保存部分数据供 --solve-only 检查")
        if samples:
            save_samples(
                out_dir, samples,
                {"time": datetime.now().isoformat(), "n": len(samples), "incomplete": True},
            )
        return

    save_samples(
        out_dir, samples,
        {
            "time": datetime.now().isoformat(),
            "n": len(samples),
            "board": "CC300-20-15",
            "squares": list(CHARUCO_SQUARES),
            "config": str(args.config),
        },
    )
    print(f"样本已存 {out_dir}/samples.npz")
    run_solve(cfg, out_dir, write_yaml=args.write_yaml, config_path=args.config)


def parse_args():
    p = argparse.ArgumentParser(description="ChArUco 眼在手上手眼标定")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--depth-topic", default="/camera/depth/image_raw")
    p.add_argument("--startup-wait", type=float, default=30.0)
    p.add_argument("--min-corners", type=int, default=MIN_CORNERS)
    p.add_argument(
        "--solve-only",
        action="store_true",
        help="不采图，仅用 out-dir/samples.npz 求解",
    )
    p.add_argument(
        "--write-yaml",
        action="store_true",
        help="把结果写入 grasp_config.yaml 的 camera_on_ee（不动 plate_bias）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.solve_only:
        cfg = load_config(args.config)
        if not (args.out_dir / "samples.npz").is_file():
            raise SystemExit(f"缺少 {args.out_dir}/samples.npz，请先采集")
        run_solve(cfg, args.out_dir, write_yaml=args.write_yaml, config_path=args.config)
    else:
        run_capture(args)


if __name__ == "__main__":
    main()
