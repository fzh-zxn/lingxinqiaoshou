"""MuJoCo 捏取规划（⑤⑥）与 robot IK 转换。"""
from __future__ import annotations

import math
import time

import numpy as np

from lbot.lbot_robot import LbotArm
from lbot_grasp_utils import (
    apply_pinch_grasp_bias,
    build_pinch_frame_at_object,
    clamp_joints_to_soft_limits,
    ee_target_align_rotation_only,
    ee_target_from_pinch,
    euler_rpy_to_matrix,
    joints_soft_limit_violation,
    matrix_to_rpy_near,
    object_up_direction_world,
    pinch_dict_to_matrix,
    pinch_frame_orient_diag,
    pinch_grasp_down_m,
    pinch_standoff_target,
    pose_to_matrix,
    resolve_pinch_standoff_m,
    resolve_pinch_hand_short_m,
    resolve_pinch_level_finger_plane,
    resolve_pregrasp_depth_mm,
    resolve_soft_joint_limits_rad,
    rotation_geodesic_deg,
    rotation_matrix_to_rpy,
    soft_limit_inward_seed,
)
from lbot_mj_hand import compute_o6_pinch_kinematics
from lbot_mj_scene import mj_object_pose, object_point_in_mj
from lbot_logutil import eye_debug

ARM = LbotArm.RIGHT_ARM

DEFAULT_GRASP_CLOSE = [150, 0, 255, 140, 140, 255]
DEFAULT_GRASP_STABILIZE = [150, 0, 140, 140, 140, 140]


def pinch_grasp_mode(eye=None, controller=None) -> str:
    """power5（默认）| pinch（旧三指）。"""
    eye = eye or {}
    if not eye and controller is not None:
        eye = getattr(controller, "eye", None) or {}
    mode = str(eye.get("pinch_grasp_mode", "power5") or "power5").strip().lower()
    if mode in ("pinch", "tri", "three", "3"):
        return "pinch"
    return "power5"


def _size_class_keys(size_info=None):
    cls = None
    if isinstance(size_info, dict):
        cls = size_info.get("class")
    if cls in ("big", "large"):
        return ["big", "large"]
    if cls:
        return [str(cls)]
    return []


def _preset_by_size(presets, key, size_info=None, fallback=None):
    by = presets.get(key) or {}
    for k in _size_class_keys(size_info):
        if k in by and by[k] is not None:
            return [int(np.clip(int(v), 0, 255)) for v in list(by[k])[:6]]
    if fallback is not None:
        return [int(np.clip(int(v), 0, 255)) for v in list(fallback)[:6]]
    return None


def _fk_try_seeds(robot, joints, log=None, pose_fallback=None, soft_limits_rad=None):
    """
    FK 重试：只用「真实关节读数」，禁止人为扰动关节再当当前位姿
    （扰动 FK 会算出假 rpy，⑥ 会带错姿态）。
    若读数略超软限位导致厂商 FK 直接失败，则向内钳 0.3° 再试（不改真实姿态语义）。
    返回 (fk_or_None, joints_used, tag)。
    """
    def _log(msg):
        if log:
            log(msg)

    if robot is None:
        return None, joints, "no_robot"

    candidates = []
    if joints is not None and len(joints) >= 7:
        jj = [float(j) for j in joints[:7]]
        if all(np.isfinite(jj)):
            candidates.append(("seed0", jj))

    # 多次重读实机关节（旧版：读失败就再读，不是改角度）
    for ti in range(5):
        raw = robot.get_joint_positions(ARM)
        if raw and len(raw) >= 7:
            jj = [float(j) for j in raw[:7]]
            if all(np.isfinite(jj)):
                candidates.append((f"reread{ti}", jj))
        time.sleep(0.04)

    seen = set()
    for tag, sd in candidates:
        key = tuple(round(float(x), 5) for x in sd)
        if key in seen:
            continue
        seen.add(key)
        fk = robot.compute_forward_kinematics(ARM, sd)
        if fk is not None:
            if tag != "seed0":
                _log(f"MJ→robot：FK 重读成功 tag={tag}")
            return fk, sd, tag
        # 贴软限位外壁时厂商 FK 常直接失败（如 J7≥91.15°）；向内钳再试
        if soft_limits_rad is not None:
            lo, hi = soft_limits_rad
            margin = math.radians(0.35)
            clamped = [
                float(np.clip(float(a), float(lo[i]) + margin, float(hi[i]) - margin))
                for i, a in enumerate(sd[:7])
            ]
            ckey = tuple(round(float(x), 5) for x in clamped)
            if ckey not in seen:
                seen.add(ckey)
                fk2 = robot.compute_forward_kinematics(ARM, clamped)
                if fk2 is not None:
                    _log(
                        f"MJ→robot：FK 软限位内钳成功 tag={tag} "
                        f"（读数略超硬限）"
                    )
                    return fk2, clamped, f"{tag}_clamp"

    _log(f"MJ→robot：FK 均失败（试了 {len(seen)} 组真实关节），改用笛卡尔位姿")
    if pose_fallback is not None:
        return pose_fallback, joints, "pose_fallback"
    try:
        live = robot.get_cartesian_pose(ARM)
    except Exception:
        live = None
    if live is not None:
        return live, joints, "get_cartesian_pose"
    return None, joints, "fail"


def _robot_tool0_matrix(robot, joints, pose_fallback=None, log=None, soft_limits_rad=None):
    """
    实机 tool0 齐次矩阵 + ref_rpy。
    FK 失败时多种子重试，再回退笛卡尔位姿。
    """
    if robot is None:
        return None, None
    fk, _j_used, tag = _fk_try_seeds(
        robot, joints, log=log, pose_fallback=pose_fallback,
        soft_limits_rad=soft_limits_rad,
    )
    if fk is None:
        return None, None
    pos, euler = fk
    ref_rpy = (float(euler.x), float(euler.y), float(euler.z))
    if log and tag not in ("seed0",):
        log(f"MJ→robot：tool0 来源={tag}")
    return pose_to_matrix(pos, euler), ref_rpy


def grasp_close_preset(controller, size_info=None, eye=None) -> list:
    """
    按尺寸档选主抓闭合 L6。
    power5: pinch_close_by_size（H2=0，中+无名闭，食小开）
    pinch:  pinch_close_by_size_pinch（旧拇+食+中）
    """
    presets = getattr(controller, "hand_presets", None) or {}
    mode = pinch_grasp_mode(eye=eye, controller=controller)
    if mode == "pinch":
        hit = _preset_by_size(
            presets, "pinch_close_by_size_pinch", size_info=size_info,
        )
        if hit is not None:
            return hit
    hit = _preset_by_size(presets, "pinch_close_by_size", size_info=size_info)
    if hit is not None:
        return hit
    return list(
        presets.get("grasp")
        or presets.get("pinch_close")
        or DEFAULT_GRASP_CLOSE
    )


def grasp_stabilize_preset(controller, size_info=None, eye=None) -> list | None:
    """
    ⑧b 食指+小指辅助固定 L6（power5）。pinch 模式返回 None。
    yaml: pinch_stabilize_by_size；缺档则在主抓上把 H3/H6 收到中/无名 curl+20。
    """
    mode = pinch_grasp_mode(eye=eye, controller=controller)
    if mode != "power5":
        return None
    presets = getattr(controller, "hand_presets", None) or {}
    primary = grasp_close_preset(controller, size_info=size_info, eye=eye)
    hit = _preset_by_size(
        presets, "pinch_stabilize_by_size", size_info=size_info,
    )
    if hit is not None:
        return hit
    # 回退：主抓基础上合食/小（略松）
    out = list(primary)
    curl = int(out[3])
    soft = int(np.clip(curl, 0, 255))  # 与中/无名同 curl，勿再 +20 留缝
    out[2] = soft
    out[5] = soft
    return out


def grasp_close_sep_target_mm(size_info=None, eye=None) -> float | None:
    """闭合目标 tip 间距(mm)≈外径×ratio（侧捏略小于 AF）。"""
    eye = eye or {}
    ratio = float(eye.get("pinch_close_sep_af_ratio", 0.72))
    af = None
    if isinstance(size_info, dict):
        af = size_info.get("hex_mm") or size_info.get("long_mm")
        if af is None:
            nom = {"small": 41.0, "medium": 50.0, "big": 70.0, "large": 70.0}
            af = nom.get(str(size_info.get("class") or "").lower())
    if af is None:
        return None
    return float(af) * ratio


def mj_from_robot_transform(
    robot, mj_model, mj_data, mj_joints, mj_hand_ranges, joints, hand_cmd, sync_fn,
    robot_pose=None,
    soft_limits_rad=None,
):
    """同关节：robot tool0 → MuJoCo tool0。"""
    if robot is None or mj_model is None:
        return None
    T_r, _ref = _robot_tool0_matrix(
        robot, joints, pose_fallback=robot_pose, soft_limits_rad=soft_limits_rad,
    )
    if T_r is None:
        return None
    if not sync_fn(joints, hand_cmd):
        return None
    mj_pose = mj_object_pose(mj_model, mj_data, "site", "arm_right_tool0")
    if mj_pose is None:
        return None
    p_m, R_m = mj_pose
    T_m = np.eye(4, dtype=np.float64)
    T_m[:3, :3] = R_m
    T_m[:3, 3] = p_m
    return T_m @ np.linalg.inv(T_r)


def tool0_matrices_from_poses(robot_pose, mj_tool0_pose):
    """实机 (pos,euler) + MJ (pos,R) → (T_r, T_m)；失败返回 (None, None)。"""
    if robot_pose is None or mj_tool0_pose is None:
        return None, None
    pos_r, eul_r = robot_pose
    p_m, R_m = mj_tool0_pose
    if hasattr(pos_r, "x"):
        T_r = pose_to_matrix(pos_r, eul_r)
    else:
        T_r = np.eye(4, dtype=np.float64)
        T_r[:3, 3] = np.asarray(pos_r, dtype=np.float64).reshape(3)
        if hasattr(eul_r, "x"):
            T_r[:3, :3] = euler_rpy_to_matrix(eul_r.x, eul_r.y, eul_r.z)
        else:
            T_r[:3, :3] = euler_rpy_to_matrix(
                float(eul_r[0]), float(eul_r[1]), float(eul_r[2])
            )
    T_m = np.eye(4, dtype=np.float64)
    T_m[:3, :3] = np.asarray(R_m, dtype=np.float64).reshape(3, 3)
    T_m[:3, 3] = np.asarray(p_m, dtype=np.float64).reshape(3)
    return T_r, T_m


def point_robot_to_mj(p_robot, T_r, T_m):
    """
    同一物理点：实机工作系 → MuJoCo 世界系。
    用当前 tool0 对齐得到的 T_mr = T_m @ inv(T_r)。
    """
    if T_r is None or T_m is None or p_robot is None:
        return None
    T_mr = T_m @ np.linalg.inv(T_r)
    p = np.asarray(p_robot, dtype=np.float64).reshape(3)
    return (T_mr[:3, :3] @ p + T_mr[:3, 3]).astype(np.float64)


def point_mj_to_robot(p_mj, T_r, T_m):
    """同一物理点：MuJoCo 世界系 → 实机工作系（T_rm = T_r @ inv(T_m)）。"""
    if T_r is None or T_m is None or p_mj is None:
        return None
    T_rm = T_r @ np.linalg.inv(T_m)
    p = np.asarray(p_mj, dtype=np.float64).reshape(3)
    return (T_rm[:3, :3] @ p + T_rm[:3, 3]).astype(np.float64)


def plan_mj_to_robot_ik(
    robot, plan, joints, hand_cmd, sync_fn, mj_model, mj_data, mj_joints, mj_hand_ranges,
    log=None,
    robot_pose=None,
    eye=None,
    config=None,
):
    """
    把 MJ 系 T_ee_des 转成实机 tool0 目标。

    关键：FK 失败时绝不能用「笛卡尔 pose + 关节同步的 MJ」拼 T_mr
    （两套状态不一致 → 目标乱飞）。改为 tool0 系位移迁移并强制保持姿态。
    """
    detail = eye_debug(eye) if eye is not None else True

    def _log(msg):
        if not log:
            return
        s = str(msg)
        if detail or ("⚠" in s) or ("失败" in s) or ("为空" in s) or ("缺少" in s) or ("无法" in s):
            log(msg)

    if plan is None:
        _log("MJ→robot：规划为空")
        return None
    if robot is None:
        _log("MJ→robot：未连接机械臂")
        return None
    T_ee_mj = plan.get("T_ee_des")
    if T_ee_mj is None:
        _log("MJ→robot：规划缺少 T_ee_des")
        return None
    T_ee_mj = np.asarray(T_ee_mj, dtype=np.float64).reshape(4, 4)
    align_hand = plan.get("grasp_close_hand") or hand_cmd
    align_mode = str(plan.get("align_mode") or "auto")

    # 实机当前：FK 多种子；失败才用笛卡尔（仅当前位置，不做 T_mr）
    soft = resolve_soft_joint_limits_rad(config) if config is not None else None
    fk, joints_fk, pose_src = _fk_try_seeds(
        robot, joints, log=_log, pose_fallback=robot_pose, soft_limits_rad=soft,
    )
    if fk is None:
        deg = [round(math.degrees(float(j)), 1) for j in (joints or [])[:7]]
        _log(f"MJ→robot：无法取得实机 tool0 joints°={deg}")
        return None
    pos_r, eul_r = fk
    # 若重读关节成功，用该组同步 MJ，避免 joints 过期
    joints_sync = joints_fk if joints_fk is not None else joints
    fk_ok = pose_src not in ("pose_fallback", "get_cartesian_pose", "fail")
    if not fk_ok:
        _log(
            f"MJ→robot：⚠ FK 多种子失败，改用 {pose_src}（仅当前位置，强制仅平移）"
        )
    p_r = np.array([pos_r.x, pos_r.y, pos_r.z], dtype=np.float64)
    r_r = (float(eul_r.x), float(eul_r.y), float(eul_r.z))
    R_r = euler_rpy_to_matrix(r_r[0], r_r[1], r_r[2])
    jdeg = [round(math.degrees(float(j)), 1) for j in (joints_sync or [])[:7]]
    _log(
        f"MJ→robot：实机当前({pose_src}) "
        f"xyz=[{p_r[0]:.4f},{p_r[1]:.4f},{p_r[2]:.4f}] "
        f"rpy°=[{math.degrees(r_r[0]):.1f},{math.degrees(r_r[1]):.1f},{math.degrees(r_r[2]):.1f}] "
        f"joints°={jdeg}"
    )

    if not sync_fn(joints_sync, align_hand):
        _log("MJ→robot：MuJoCo 同步失败")
        return None
    mj_pose = mj_object_pose(mj_model, mj_data, "site", "arm_right_tool0")
    if mj_pose is None:
        _log("MJ→robot：MuJoCo tool0 位姿不可用")
        return None
    p_m, R_m = mj_pose
    p_m = np.asarray(p_m, dtype=np.float64).reshape(3)
    R_m = np.asarray(R_m, dtype=np.float64).reshape(3, 3)
    p_des_m = T_ee_mj[:3, 3].copy()
    delta_m = p_des_m - p_m
    _log(
        f"MJ→robot：MJ tool0 当前[{p_m[0]:.4f},{p_m[1]:.4f},{p_m[2]:.4f}] "
        f"→目标[{p_des_m[0]:.4f},{p_des_m[1]:.4f},{p_des_m[2]:.4f}] "
        f"Δxyz_mm=[{delta_m[0]*1000:.1f},{delta_m[1]*1000:.1f},{delta_m[2]*1000:.1f}] "
        f"|Δ|={float(np.linalg.norm(delta_m))*1000:.1f}mm "
        f"plan_mode={align_mode}"
    )

    # tool0 系位移：MJ 与实机共用同一相对增量（不依赖不可靠的 T_mr）
    delta_tool0 = R_m.T @ delta_m
    delta_r = R_r @ delta_tool0
    target_pos_translate = p_r + delta_r
    _log(
        f"MJ→robot：tool0系Δ[{delta_tool0[0]*1000:.1f},{delta_tool0[1]*1000:.1f},{delta_tool0[2]*1000:.1f}]mm "
        f"→实机世界Δ[{delta_r[0]*1000:.1f},{delta_r[1]*1000:.1f},{delta_r[2]*1000:.1f}]mm "
        f"仅平移目标[{target_pos_translate[0]:.4f},{target_pos_translate[1]:.4f},{target_pos_translate[2]:.4f}]"
    )

    am = str(align_mode or "").strip().lower()
    # rotation_only：实机只改姿态，位置始终用当前 FK（避免规划瞬间旧 xyz）
    if am in ("rotation_only", "rot_align", "orient", "orientation"):
        R_des_m = T_ee_mj[:3, :3]
        R_des_r = R_des_m @ R_m.T @ R_r
        T_tmp = np.eye(4, dtype=np.float64)
        T_tmp[:3, :3] = R_des_r
        T_tmp[:3, 3] = p_r
        _pos_ign, target_rpy = matrix_to_rpy_near(T_tmp, ref_rpy=r_r)
        geo = rotation_geodesic_deg(r_r, target_rpy)
        _log(
            f"MJ→robot：rotation_only 保持实机当前位置，仅转腕 "
            f"rpy°=[{math.degrees(target_rpy[0]):.1f},{math.degrees(target_rpy[1]):.1f},"
            f"{math.degrees(target_rpy[2]):.1f}] 测地Δ={geo:.1f}°"
        )
        return {
            "target_pos": p_r.copy(),
            "target_rpy": target_rpy,
            "method": "rotation_only_live_xyz",
            "fk_ok": fk_ok,
            "pose_src": pose_src,
        }

    # 仅「纯平移」模式才砍掉姿态；full_6dof / auto 保留转腕
    force_translate = (
        not fk_ok
        or am in ("translate", "translate_tcp", "pos_only", "position")
        or am.endswith("→translate_tcp")
    )
    if force_translate:
        why = "FK失败" if not fk_ok else f"模式={align_mode}"
        _log(f"MJ→robot：强制仅平移（{why}），姿态保持当前 rpy")
        return {
            "target_pos": target_pos_translate,
            "target_rpy": r_r,
            "method": "tool0_delta_translate",
            "fk_ok": fk_ok,
            "pose_src": pose_src,
        }

    # full_6dof 且 FK 成功：可用齐次标定；仍打印一致性与对照仅平移目标
    T_r = pose_to_matrix(pos_r, eul_r)
    T_m = np.eye(4, dtype=np.float64)
    T_m[:3, :3] = R_m
    T_m[:3, 3] = p_m
    T_mr = T_m @ np.linalg.inv(T_r)
    T_rm = np.linalg.inv(T_mr)
    T_ee_robot = T_rm @ T_ee_mj
    target_pos, target_rpy = matrix_to_rpy_near(T_ee_robot, ref_rpy=r_r)
    d_pos = target_pos - target_pos_translate
    geo = rotation_geodesic_deg(r_r, target_rpy)
    _log(
        f"MJ→robot：全姿态目标 xyz=[{target_pos[0]:.4f},{target_pos[1]:.4f},{target_pos[2]:.4f}] "
        f"rpy°=[{math.degrees(target_rpy[0]):.1f},{math.degrees(target_rpy[1]):.1f},"
        f"{math.degrees(target_rpy[2]):.1f}] 测地Δ={geo:.1f}° "
        f"相对仅平移偏差[{d_pos[0]*1000:.1f},{d_pos[1]*1000:.1f},{d_pos[2]*1000:.1f}]mm"
    )
    # 全姿态与仅平移位置差过大 → 标定可疑，降级
    if float(np.linalg.norm(d_pos)) > 0.025:
        _log(
            f"MJ→robot：⚠ 全姿态与仅平移目标相差 {float(np.linalg.norm(d_pos))*1000:.0f}mm "
            f"> 25mm，降级仅平移防乱跳"
        )
        return {
            "target_pos": target_pos_translate,
            "target_rpy": r_r,
            "method": "tool0_delta_translate_fallback",
            "fk_ok": True,
            "pose_src": pose_src,
        }
    return {
        "target_pos": target_pos,
        "target_rpy": target_rpy,
        "method": "T_rm_full",
        "fk_ok": True,
        "pose_src": pose_src,
    }


def _robot_ik_for_mj_tool0_des(
    robot,
    mj_model,
    mj_data,
    joints,
    hand_cmd,
    sync_fn,
    T_m_des,
    max_jump_rad=1.8,
    soft_limits_rad=None,
    config=None,
):
    """MuJoCo tool0 目标 4x4 → 实机关节 IK。"""
    from lbot.lbot_robot import LbotEuler, LbotPosition

    if robot is None or not sync_fn(joints, hand_cmd):
        return None
    # 笛卡尔回退：FK 贴硬限失败时仍能建 T_mr（否则 Plan⑥ 第 1 点就挂）
    pose_fb = None
    try:
        pose_fb = robot.get_cartesian_pose(ARM)
    except Exception:
        pose_fb = None
    T_mr = mj_from_robot_transform(
        robot, mj_model, mj_data, None, None, joints, hand_cmd, sync_fn,
        robot_pose=pose_fb,
        soft_limits_rad=soft_limits_rad,
    )
    if T_mr is None:
        return None
    T_m_des = np.asarray(T_m_des, dtype=np.float64).reshape(4, 4)
    T_r_des = np.linalg.inv(T_mr) @ T_m_des
    fk = robot.compute_forward_kinematics(ARM, joints)
    if fk is None and soft_limits_rad is not None:
        lo, hi = soft_limits_rad
        margin = math.radians(0.35)
        clamped = [
            float(np.clip(float(a), float(lo[i]) + margin, float(hi[i]) - margin))
            for i, a in enumerate(joints[:7])
        ]
        fk = robot.compute_forward_kinematics(ARM, clamped)
    if fk is None and pose_fb is not None:
        fk = pose_fb
    if fk is None:
        return None
    _pos, euler = fk
    target_pos, target_rpy = matrix_to_rpy_near(
        T_r_des, ref_rpy=(float(euler.x), float(euler.y), float(euler.z)),
    )

    if soft_limits_rad is None and config is not None:
        soft_limits_rad = resolve_soft_joint_limits_rad(config)
    eps = 0.0
    reseed = True
    if config is not None:
        robot_cfg = config.get("robot") or {}
        eps = math.radians(float(robot_cfg.get("soft_limit_eps_deg", 1.0)))
        reseed = bool(robot_cfg.get("ik_limit_reseed", True))

    seeds = [list(joints)]
    if soft_limits_rad is not None and reseed:
        lo, hi = soft_limits_rad
        seeds.append(soft_limit_inward_seed(joints, lo, hi, pull_rad=math.radians(5.0)))
        s2 = list(joints)
        s2[1] = float(np.clip(float(s2[1]) - math.radians(10.0), lo[1], hi[1]))
        seeds.append(s2)

    for sd in seeds:
        ik = robot.compute_inverse_kinematics(
            ARM,
            LbotPosition(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
            LbotEuler(float(target_rpy[0]), float(target_rpy[1]), float(target_rpy[2])),
            sd,
        )
        if not ik:
            continue
        jump = max(abs(float(a) - float(b)) for a, b in zip(ik, joints))
        if jump > float(max_jump_rad):
            continue
        if soft_limits_rad is not None:
            lo, hi = soft_limits_rad
            if joints_soft_limit_violation(ik, lo, hi, eps_rad=eps) is not None:
                seed2 = soft_limit_inward_seed(ik, lo, hi, pull_rad=math.radians(8.0))
                ik2 = robot.compute_inverse_kinematics(
                    ARM,
                    LbotPosition(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
                    LbotEuler(float(target_rpy[0]), float(target_rpy[1]), float(target_rpy[2])),
                    seed2,
                )
                if ik2 is None:
                    continue
                jump2 = max(abs(float(a) - float(b)) for a, b in zip(ik2, joints))
                if jump2 > float(max_jump_rad):
                    continue
                if joints_soft_limit_violation(ik2, lo, hi, eps_rad=eps) is not None:
                    continue
                ik = ik2
            return clamp_joints_to_soft_limits(ik, lo, hi)
        return list(ik)
    return None


def mj_tool0_delta_to_robot_ik(
    robot, mj_model, mj_data, joints, hand_cmd, sync_fn, delta_tool0,
    max_jump_rad=1.8,
    soft_limits_rad=None,
    config=None,
):
    """
    在 MuJoCo tool0 坐标系下平移 delta → 转 robot IK 关节。
    预览 ③④ 用 MJ 几何驱动；跳变上限与实机一致（默认 1.8 rad）。
    """
    if robot is None or not sync_fn(joints, hand_cmd):
        return None
    mj_pose = mj_object_pose(mj_model, mj_data, "site", "arm_right_tool0")
    if mj_pose is None:
        return None
    p_m, R_m = mj_pose
    delta = np.asarray(delta_tool0, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(delta)) or float(np.linalg.norm(delta)) < 1e-6:
        return list(joints)
    T_m_des = np.eye(4, dtype=np.float64)
    T_m_des[:3, :3] = R_m
    T_m_des[:3, 3] = p_m + R_m @ delta
    return _robot_ik_for_mj_tool0_des(
        robot, mj_model, mj_data, joints, hand_cmd, sync_fn, T_m_des,
        max_jump_rad=max_jump_rad,
        soft_limits_rad=soft_limits_rad,
        config=config,
    )


def mj_tool0_pose_to_robot_ik(
    robot,
    mj_model,
    mj_data,
    joints,
    hand_cmd,
    sync_fn,
    target_pos_mj,
    target_rot_mj,
    max_jump_rad=1.8,
    soft_limits_rad=None,
    config=None,
):
    """MuJoCo 世界系 tool0 目标位姿 → 实机关节 IK（预览 ⑥ 用）。"""
    T_m_des = np.eye(4, dtype=np.float64)
    T_m_des[:3, :3] = np.asarray(target_rot_mj, dtype=np.float64).reshape(3, 3)
    T_m_des[:3, 3] = np.asarray(target_pos_mj, dtype=np.float64).reshape(3)
    return _robot_ik_for_mj_tool0_des(
        robot, mj_model, mj_data, joints, hand_cmd, sync_fn, T_m_des,
        max_jump_rad=max_jump_rad,
        soft_limits_rad=soft_limits_rad,
        config=config,
    )


def compute_pinch_plan_mj(
    mj_model,
    mj_data,
    u,
    v,
    depth_mm,
    joints,
    hand_cmd,
    controller,
    locked_object_mj,
    mj_cam_optical_in_body,
    config,
    sync_fn,
    log=print,
    already_at_pregrasp=False,
    locked_object_robot=None,
    robot_pose=None,
    size_info=None,
):
    """
    全程 MuJoCo 世界系规划 ⑤⑥（⑥ 闭合手 TCP 6DOF 对齐到 standoff 点）。

    物体点优先：实机锁定点 locked_object_robot（yaml 外参）经 tool0 对齐映射进 MJ；
    避免 MJ 相机射线与手调 t_ee_cam 两套点打架。
    """
    if not sync_fn(joints, hand_cmd):
        return None

    grasp_close = grasp_close_preset(
        controller, size_info=size_info, eye=config.get("eye_in_hand"),
    )

    obj_world = None
    obj_src = None
    # ① 实机点 → MJ（与验参/⑦ 同一物理点）
    if locked_object_robot is not None:
        pose_r = robot_pose
        if pose_r is None and getattr(controller, "get_pose", None):
            pose_r = controller.get_pose()
        tool0_mj = mj_object_pose(mj_model, mj_data, "site", "arm_right_tool0")
        T_r, T_m = tool0_matrices_from_poses(pose_r, tool0_mj)
        mapped = point_robot_to_mj(locked_object_robot, T_r, T_m)
        if mapped is not None:
            obj_world = mapped
            obj_src = "robot→MJ"
    # ② 已锁定的 MJ 点（若上面已由实机映射写入，通常一致）
    if obj_world is None and locked_object_mj is not None:
        obj_world = np.asarray(locked_object_mj, dtype=np.float64).copy()
        obj_src = "locked_mj"
    # ③ 回退：MJ 相机射线（无实机位姿时）
    if obj_world is None:
        obj_world = object_point_in_mj(
            mj_model, mj_data, u, v, depth_mm,
            controller.intrinsics, mj_cam_optical_in_body,
            **controller._vision_ray_params(),
        )
        obj_src = "mj_cam_ray"
        # 未走锁定时补侧向/下偏修正（锁定时已在 lock 修过）
        eye = config["eye_in_hand"]
        if obj_world is not None:
            tool0_pose0 = mj_object_pose(mj_model, mj_data, "site", "arm_right_tool0")
            if tool0_pose0 is not None:
                ee0, R0 = tool0_pose0
                R_cam = R0 @ np.asarray(controller.R_ee_cam, dtype=np.float64).reshape(3, 3)
                up0 = object_up_direction_world(
                    R_cam,
                    grasp_point_mode=eye.get("grasp_point_mode", "bbox_center"),
                    grasp_point_side=eye.get("grasp_point_side", "right"),
                    xy_rotate_deg=float(eye.get("servo_xy_rotate_deg", 90.0)),
                    flip_x=bool(eye.get("servo_flip_x", True)),
                    flip_y=bool(eye.get("servo_flip_y", False)),
                    world_up_fallback=eye.get("pinch_world_up", [0.0, 0.0, 1.0]),
                )
                obj_world = apply_pinch_grasp_bias(
                    obj_world, ee0, up0, eye, log=log, size_info=size_info,
                    detection=config.get("detection") if isinstance(config, dict) else None,
                )
    if obj_world is None:
        log("规划失败：无法计算物体 3D 点")
        return None
    log(f"⑤ 规划物体来源={obj_src}")

    eye = config["eye_in_hand"]

    kin_kw = {
        "tip_frac": float(config["eye_in_hand"].get("pinch_pad_tip_frac", 0.08)),
        "tip_mode": str(config["eye_in_hand"].get("pinch_tip_mode", "tip")),
        "grasp_mode": pinch_grasp_mode(eye=config.get("eye_in_hand")),
    }
    gmode = kin_kw["grasp_mode"]
    log(
        f"⑥ 接触点={kin_kw['tip_mode']}（tip=指尖 / pad_plane=指腹） "
        f"frac={kin_kw['tip_frac']} grasp={gmode}"
    )
    pinch_info_open = compute_o6_pinch_kinematics(mj_model, mj_data, **kin_kw)
    if pinch_info_open is None:
        log("规划失败：MuJoCo 三指 frame 不可用")
        return None

    if not sync_fn(joints, grasp_close):
        log("规划失败：无法同步闭合手姿到 MuJoCo")
        return None

    tool0_pose = mj_object_pose(mj_model, mj_data, "site", "arm_right_tool0")
    if tool0_pose is None:
        log("规划失败：无 tool0 位姿")
        return None
    tool0_pos, tool0_rot = tool0_pose
    ee_pos = tool0_pos.copy()
    ref_rpy = rotation_matrix_to_rpy(tool0_rot)
    T_w_ee_cur = np.eye(4, dtype=np.float64)
    T_w_ee_cur[:3, :3] = tool0_rot
    T_w_ee_cur[:3, 3] = tool0_pos

    pinch_info = compute_o6_pinch_kinematics(mj_model, mj_data, **kin_kw)
    if pinch_info is None:
        log("规划失败：闭合手三指 frame 不可用")
        return None

    open_mid = np.asarray(pinch_info_open["pinch_mid_m"], dtype=np.float64)
    close_mid = np.asarray(pinch_info["pinch_mid_m"], dtype=np.float64)
    shift = close_mid - open_mid
    shift_mm = float(np.linalg.norm(shift)) * 1000.0
    if shift_mm > 0.3:
        to_obj_open = obj_world - open_mid
        to_obj_close = obj_world - close_mid
        d_open = float(np.linalg.norm(to_obj_open))
        d_close = float(np.linalg.norm(to_obj_close))
        toward = "靠近" if d_close < d_open - 0.001 else (
            "远离" if d_close > d_open + 0.001 else "持平"
        )
        # 设计：⑥⑦ 规划闭合 TCP → ⑧ 合手后 mid 落在黄/青；接近时张手 mid 偏开是预期
        log(
            f"⑥ 规划用闭合 TCP（合手后 mid 对黄/青；张→闭 mid {shift_mm:.1f}mm "
            f"Δxyz_mm=[{shift[0]*1000:.1f},{shift[1]*1000:.1f},{shift[2]*1000:.1f}]，"
            f"相对物体{toward}：张{d_open*1000:.0f}→闭{d_close*1000:.0f}mm）"
        )

    eye = config["eye_in_hand"]
    standoff_m = resolve_pinch_standoff_m(eye, size_info)
    use_hand_z = bool(eye.get("pinch_standoff_use_hand_z", False))
    cls_s = (
        str(size_info.get("class"))
        if isinstance(size_info, dict) and size_info.get("class")
        else "—"
    )
    log(f"⑤ standoff={standoff_m*1000:.0f}mm（档={cls_s}）")
    if already_at_pregrasp:
        ee_sim = ee_pos.copy()
    else:
        if controller.pregrasp_apply_hand_offset():
            delta_off = controller.pick_hand_offset_delta_ee()
        else:
            delta_off = np.zeros(3, dtype=np.float64)
        delta_pre, _advance_pre = controller.pick_advance_delta_ee(
            depth_mm,
            resolve_pregrasp_depth_mm(depth_mm, eye, size_info=size_info),
        )
        delta_sum = np.asarray(delta_off, dtype=np.float64) + np.asarray(delta_pre, dtype=np.float64)
        ee_sim = ee_pos + tool0_rot @ delta_sum

    log(f"⑤ 物体(模型系) [{obj_world[0]:.3f}, {obj_world[1]:.3f}, {obj_world[2]:.3f}] m")

    # 闭合 TCP：刚体变换把「闭合 mid」送到黄/青；⑥⑦ 执行时手仍张开，⑧ 合手后对准
    T_w_pinch_cur = pinch_dict_to_matrix(pinch_info)
    x_hint = np.asarray(pinch_info.get("x_close", [1, 0, 0]), dtype=np.float64)

    level_fp = resolve_pinch_level_finger_plane(eye, size_info=size_info)
    world_up_g = np.asarray(
        eye.get("pinch_world_up", [0.0, 0.0, 1.0]), dtype=np.float64
    ).reshape(3)
    obj_target = pinch_standoff_target(
        obj_world,
        ee_sim,
        standoff_m,
        pinch_info=pinch_info,
        use_hand_z=use_hand_z,
        world_up=world_up_g,
        level_finger_plane=level_fp,
    )
    if obj_target is None:
        log("规划失败：无法计算 standoff 点")
        return None

    # 诊断：黄球相对几何接近轴 / 重力
    geo = ee_sim - obj_world
    gn = float(np.linalg.norm(geo))
    if gn > 1e-6:
        along = float(np.dot(obj_target - obj_world, geo / gn))
        lateral = float(np.linalg.norm((obj_target - obj_world) - (geo / gn) * along))
        if level_fp:
            up_n = float(np.linalg.norm(world_up_g))
            up_u = world_up_g / up_n if up_n > 1e-9 else world_up_g
            along_up = float(np.dot(obj_target - obj_world, up_u))
            log(
                f"黄球 standoff: 沿重力 {along_up*1000:.0f}mm "
                f"(相对几何接近轴 along={along*1000:.0f} 侧向={lateral*1000:.0f}mm；"
                f"level 模式黄球在物体正上，⑦竖直下落)"
            )
        else:
            log(
                f"黄球 standoff: 沿接近轴 {along*1000:.0f}mm / 侧向偏 {lateral*1000:.0f}mm "
                f"(期望≈{standoff_m*1000:.0f}mm，{'手系Z' if use_hand_z else '几何ee−obj'})"
            )

    z_ref = pinch_info.get("z_approach")
    # 物体「上」= hand_top 等抓取语义上边（非世界竖直左右偏）
    R_world_cam = tool0_rot @ np.asarray(controller.R_ee_cam, dtype=np.float64).reshape(3, 3)
    object_up = object_up_direction_world(
        R_world_cam,
        grasp_point_mode=eye.get("grasp_point_mode", "bbox_center"),
        grasp_point_side=eye.get("grasp_point_side", "right"),
        xy_rotate_deg=float(eye.get("servo_xy_rotate_deg", 90.0)),
        flip_x=bool(eye.get("servo_flip_x", True)),
        flip_y=bool(eye.get("servo_flip_y", False)),
        world_up_fallback=eye.get("pinch_world_up", [0.0, 0.0, 1.0]),
    )
    z_plane = str(eye.get("pinch_des_z_tilt_plane", "world_up")).strip().lower()
    prefer_hx = bool(eye.get("pinch_prefer_horizontal_x", True))
    T_w_pinch_des = build_pinch_frame_at_object(
        obj_target,
        ee_sim,
        x_close_hint=x_hint,
        world_up=world_up_g,
        object_up=object_up,
        z_tilt_deg=0.0 if level_fp else float(eye.get("pinch_des_z_tilt_deg", 0.0)),
        z_tilt_ref=z_ref,
        z_tilt_plane=z_plane,
        prefer_horizontal_x=prefer_hx,
        level_finger_plane=level_fp,
    )
    if T_w_pinch_des is None:
        log("规划失败：无法建立目标捏取 frame")
        return None
    # 倾角符号：相对重力上仰/俯（与倾角平面一致）；object_up 平面时改用物体上边
    tilt_up_diag = (
        object_up if z_plane in ("object_up", "toward_up") else world_up_g
    )
    ang_z, ang_x, signed_z = pinch_frame_orient_diag(
        T_w_pinch_des, ee_sim, obj_target,
        world_up=tilt_up_diag,
    )
    if level_fp:
        log(
            f"⑥ 目标捏取姿态：三指平面∥世界水平（Z∥重力），"
            f"相对几何接近轴 {ang_z:.1f}°，X倾角 {ang_x:.1f}°（水平闭合轴）"
        )
    else:
        tilt_want = float(eye.get("pinch_des_z_tilt_deg", 0.0))
        if z_plane in ("object_up", "toward_up"):
            plane_note = "object_up(hand_top易左右歪)"
        else:
            plane_note = f"{z_plane}(重力俯仰轴)"
        x_note = "水平X" if prefer_hx else "闭合hint"
        log(
            f"⑥ 目标捏取姿态：Z相对接近轴 {ang_z:.1f}°"
            f"（就近±{tilt_want:.0f}° → 取 {signed_z:+.1f}°，plane={plane_note}），"
            f"X倾角 {ang_x:.1f}°（{x_note}）"
        )

    # auto：优先全姿态对齐三指（可转腕）；过大才降级
    # full_6dof：强制全姿态（分步执行）—— TCP 落到黄球 + 手指方向对齐
    # rotation_only：仅原地转腕，法兰位置不变（TCP 不到黄球，关③后易「歪」）
    # translate_tcp：始终只平移（手指方向不改）
    align_mode = str(eye.get("pinch_align_mode", "full_6dof")).strip().lower()
    max_d_rpy = float(eye.get("pinch_align_max_d_rpy_deg", 45))
    max_travel = float(eye.get("pinch_align_max_travel_m", 0.20))

    T_full = ee_target_from_pinch(T_w_pinch_des, T_w_pinch_cur, T_w_ee_cur)
    pos_full, rpy_full = matrix_to_rpy_near(T_full, ref_rpy=ref_rpy)
    d_rpy_full = [
        abs(math.degrees(
            (float(rpy_full[i]) - float(ref_rpy[i]) + math.pi) % (2 * math.pi) - math.pi
        ))
        for i in range(3)
    ]
    travel_full = float(np.linalg.norm(pos_full - ee_pos))
    geo_deg = rotation_geodesic_deg(ref_rpy, rpy_full)

    force_translate = align_mode in ("translate", "translate_tcp", "pos_only", "position")
    want_rot_only = align_mode in ("rotation_only", "rot_align", "orient", "orientation")
    want_full = align_mode in ("full", "full_6dof", "6dof")
    want_auto = align_mode in ("auto", "")
    mode_tag = align_mode

    if want_rot_only:
        # 仅转姿态；法兰不动。关手偏移后残差大，仅作对照
        T_w_ee_des = ee_target_align_rotation_only(
            T_w_pinch_des, T_w_pinch_cur, T_w_ee_cur,
        )
        mode_tag = "rotation_only"
        log(
            "⑥ rotation_only：法兰不移，仅转腕；"
            "TCP 可能仍离黄球较远（关③时建议改 full_6dof）"
        )
    elif force_translate:
        delta_tcp = obj_target - close_mid
        T_w_ee_des = np.asarray(T_w_ee_cur, dtype=np.float64).copy()
        T_w_ee_des[:3, 3] = ee_pos + delta_tcp
        mode_tag = "translate_tcp"
    elif want_full:
        T_w_ee_des = T_full
        mode_tag = "full_6dof"
        if max(d_rpy_full) > max_d_rpy or travel_full > max_travel or geo_deg > max_d_rpy * 1.2:
            log(
                f"⑥ full_6dof 姿态/行程偏大 Δrpy°=[{d_rpy_full[0]:.0f},{d_rpy_full[1]:.0f},{d_rpy_full[2]:.0f}] "
                f"测地{geo_deg:.0f}° 行程{travel_full*1000:.0f}mm —— 仍执行全姿态（分步转腕）"
            )
    elif want_auto:
        if (
            max(d_rpy_full) > max_d_rpy
            or travel_full > max_travel
            or geo_deg > max_d_rpy * 1.2
        ):
            # 过大时降级为「平移到黄球」（保留位置对齐，避免只转腕变歪）
            log(
                f"⑥ 全姿态过大 Δrpy°=[{d_rpy_full[0]:.0f},{d_rpy_full[1]:.0f},{d_rpy_full[2]:.0f}] "
                f"测地{geo_deg:.0f}° 行程{travel_full*1000:.0f}mm → 降级 translate_tcp"
                f"（阈值 {max_d_rpy:.0f}° / {max_travel*1000:.0f}mm）"
            )
            delta_tcp = obj_target - close_mid
            T_w_ee_des = np.asarray(T_w_ee_cur, dtype=np.float64).copy()
            T_w_ee_des[:3, 3] = ee_pos + delta_tcp
            mode_tag = "auto→translate_tcp"
        else:
            T_w_ee_des = T_full
            mode_tag = "full_6dof"
    else:
        T_w_ee_des = T_full
        mode_tag = align_mode or "full_6dof"
    target_pos, target_rpy = matrix_to_rpy_near(T_w_ee_des, ref_rpy=ref_rpy)
    tcp_err_mm = float(np.linalg.norm(close_mid - obj_target)) * 1000.0
    travel_mm = float(np.linalg.norm(target_pos - ee_pos)) * 1000.0
    # 刚体变换后闭合 TCP 应落到黄球；残差应≈0（数值误差）
    T_rel = np.linalg.inv(T_w_pinch_cur) @ T_w_ee_cur
    pinch_at_des = (T_w_ee_des @ np.linalg.inv(T_rel))[:3, 3]
    residual_mm = float(np.linalg.norm(pinch_at_des - obj_target)) * 1000.0
    d_rpy = []
    for i in range(3):
        d = (float(target_rpy[i]) - float(ref_rpy[i]) + math.pi) % (2 * math.pi) - math.pi
        d_rpy.append(math.degrees(d))

    # ── 张手指尖顶面净空（⑥⑦ 张手；闭合 mid@青 ≠ 张手指尖高度）──
    # 水平捏时拇指尖常比 mid 高 ~8–12mm；仅 mid 下偏 0.2×AF(~10mm) 时拇指会擦顶。
    if level_fp:
        wu = world_up_g / max(float(np.linalg.norm(world_up_g)), 1e-12)
        down_m, down_src = pinch_grasp_down_m(
            eye,
            size_info=size_info,
            detection=config.get("detection") if isinstance(config, dict) else None,
            with_source=True,
        )
        margin = float(eye.get("pinch_open_tip_top_margin_m", 0.015))
        T_ee_inv = np.linalg.inv(T_w_ee_cur)
        T_to6 = T_w_ee_des @ T_ee_inv
        adv7 = np.asarray(obj_world, dtype=np.float64).reshape(3) - np.asarray(
            obj_target, dtype=np.float64
        ).reshape(3)

        def _xf_pt(p):
            h = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
            return (T_to6 @ h)[:3] + adv7

        open_tips = []
        tip_labs = []
        for key, lab in (
            ("thumb_tip_m", "拇"),
            ("index_tip_m", "食"),
            ("middle_tip_m", "中"),
            ("ring_tip_m", "无"),
            ("pinky_tip_m", "小"),
        ):
            raw = pinch_info_open.get(key)
            if raw is None:
                continue
            open_tips.append(
                _xf_pt(np.asarray(raw, dtype=np.float64).reshape(3))
            )
            tip_labs.append(lab)
        if open_tips:
            # 顶面 ≈ 青球沿 +up 回退已下偏量（锁物时的 world_down）
            top = np.asarray(obj_world, dtype=np.float64).reshape(3) + wu * float(down_m)
            tip_below = [float(np.dot(top - t, wu)) for t in open_tips]
            worst_i = int(np.argmin(tip_below))
            worst_clear = tip_below[worst_i]
            open_mid_7 = _xf_pt(open_mid)
            mid_below = float(np.dot(top - open_mid_7, wu))
            log(
                f"  张手指尖@⑦相对顶面: "
                + " ".join(f"{lab}{c*1000:.0f}" for lab, c in zip(tip_labs, tip_below))
                + f"mm（mid顶下{mid_below*1000:.0f}；下偏{down_m*1000:.1f}/{down_src}）"
            )
            if worst_clear < margin:
                extra = float(margin - worst_clear)
                # 上限：避免为保张手指尖净空把侧中心青球再拽深一截（日志曾 14mm）
                max_extra = float(eye.get("pinch_open_tip_clear_max_extra_m", 0.005))
                if extra > max_extra:
                    log(
                        f"  ⇒ 张手{tip_labs[worst_i]}尖顶下{worst_clear*1000:.1f}mm"
                        f"（<{margin*1000:.0f}mm），需降 {extra*1000:.1f}mm 但封顶 "
                        f"{max_extra*1000:.0f}mm（防毁侧中心；可调 pinch_open_tip_clear_max_extra_m）"
                    )
                    extra = max_extra
                shift = -wu * extra
                obj_world = np.asarray(obj_world, dtype=np.float64).reshape(3) + shift
                obj_target = np.asarray(obj_target, dtype=np.float64).reshape(3) + shift
                T_w_ee_des = np.asarray(T_w_ee_des, dtype=np.float64).copy()
                T_w_ee_des[:3, 3] = T_w_ee_des[:3, 3] + shift
                target_pos = target_pos + shift
                log(
                    f"  ⇒ 张手{tip_labs[worst_i]}尖仅在顶下{worst_clear*1000:.1f}mm"
                    f"（<{margin*1000:.0f}mm），青/黄/tool0 再降 {extra*1000:.1f}mm 保净空"
                )
                # 刷新残差口径
                pinch_at_des = pinch_at_des + shift
                residual_mm = float(np.linalg.norm(pinch_at_des - obj_target)) * 1000.0

    # 实指比 MJ 手模短：只降法兰，青/黄不动 → MJ mid 会偏深，实 mid≈青
    short_m = resolve_pinch_hand_short_m(eye, size_info)
    if short_m > 1e-6:
        wu_s = np.asarray(
            eye.get("pinch_world_up", [0.0, 0.0, 1.0]), dtype=np.float64
        ).reshape(3)
        wn = float(np.linalg.norm(wu_s))
        wu_s = wu_s / wn if wn > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        shift_s = -wu_s * short_m
        T_w_ee_des = np.asarray(T_w_ee_des, dtype=np.float64).copy()
        T_w_ee_des[:3, 3] = T_w_ee_des[:3, 3] + shift_s
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3) + shift_s
        cls_s = ""
        if isinstance(size_info, dict) and size_info.get("class"):
            cls_s = f"/{size_info.get('class')}"
        log(
            f"  ⇒ 实指短补偿 pinch_hand_short{cls_s}={short_m*1000:.0f}mm："
            f"tool0 再降（青/黄不动；MJ mid 会深于青约同量）"
        )

    tcp_label = "外接圆心" if gmode == "power5" else "三指质心"
    log(
        f"⑥ {tcp_label} TCP 对齐 tool0 xyz=[{target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}] "
        f"rpy°=[{math.degrees(target_rpy[0]):.1f},{math.degrees(target_rpy[1]):.1f},"
        f"{math.degrees(target_rpy[2]):.1f}] 法兰行程 {travel_mm:.0f}mm "
        f"闭合TCP→黄球(当前) {tcp_err_mm:.1f}mm 变换后残差 {residual_mm:.1f}mm "
        f"Δrpy°=[{d_rpy[0]:+.1f},{d_rpy[1]:+.1f},{d_rpy[2]:+.1f}] "
        f"模式={mode_tag}/{gmode}"
    )
    thumb = np.asarray(pinch_info.get("thumb_tip_m"), dtype=np.float64).reshape(3)
    middle = np.asarray(pinch_info.get("middle_tip_m"), dtype=np.float64).reshape(3)
    index_raw = pinch_info.get("index_tip_m")
    ring_raw = pinch_info.get("ring_tip_m")
    index = (
        np.asarray(index_raw, dtype=np.float64).reshape(3)
        if index_raw is not None else None
    )
    ring = (
        np.asarray(ring_raw, dtype=np.float64).reshape(3)
        if ring_raw is not None else None
    )
    # 当前闭合 tip → 变换到目标 tool0 后（预览里应对准黄球附近）
    T_ee_cur_inv = np.linalg.inv(T_w_ee_cur)
    T_move = T_w_ee_des @ T_ee_cur_inv

    def _xf(p):
        h = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
        return (T_move @ h)[:3]

    thumb_d = _xf(thumb)
    middle_d = _xf(middle)
    index_d = _xf(index) if index is not None else None
    ring_d = _xf(ring) if ring is not None else None
    mid_d = _xf(close_mid)
    circ_r = float(pinch_info.get("circumradius_mm", 0.0) or 0.0)
    if gmode == "power5" and ring_d is not None:
        log(
            f"  闭合主抓 tip@目标(MJ) 拇[{thumb_d[0]:.3f},{thumb_d[1]:.3f},{thumb_d[2]:.3f}] "
            f"中[{middle_d[0]:.3f},{middle_d[1]:.3f},{middle_d[2]:.3f}] "
            f"无[{ring_d[0]:.3f},{ring_d[1]:.3f},{ring_d[2]:.3f}] "
            f"circum R={circ_r:.1f}mm sep={float(pinch_info.get('pinch_sep_mm', 0)):.1f}mm "
            f"mode={pinch_info.get('tcp_origin_mode', '')}"
        )
    else:
        idx_s = (
            f"食[{index_d[0]:.3f},{index_d[1]:.3f},{index_d[2]:.3f}] "
            if index_d is not None else ""
        )
        log(
            f"  闭合三指 tip@目标(MJ) 拇[{thumb_d[0]:.3f},{thumb_d[1]:.3f},{thumb_d[2]:.3f}] "
            f"{idx_s}"
            f"中[{middle_d[0]:.3f},{middle_d[1]:.3f},{middle_d[2]:.3f}] "
            f"sep={float(pinch_info.get('pinch_sep_mm', 0)):.1f}mm"
        )
    # level：沿重力分解（黄在青正上）；勿用斜向 ee−obj，否则会假「侧向 20mm」
    if level_fp:
        wu = world_up_g / max(float(np.linalg.norm(world_up_g)), 1e-12)
        d_mid = mid_d - obj_world
        along = float(np.dot(d_mid, wu))
        lat_n = float(np.linalg.norm(d_mid - along * wu))
        log(
            f"  闭合mid@目标→青球 along↑={along*1000:.1f}mm 侧向={lat_n*1000:.1f}mm "
            f"(期望 along≈{standoff_m*1000:.0f} / 侧向≈0；重力轴)"
        )
    else:
        geo_u = (ee_sim - obj_world) / max(float(np.linalg.norm(ee_sim - obj_world)), 1e-9)
        d_mid = mid_d - obj_world
        along = float(np.dot(d_mid, geo_u))
        lat_v = d_mid - along * geo_u
        lat_n = float(np.linalg.norm(lat_v))
        log(
            f"  闭合mid@目标→青球 along={along*1000:.1f}mm 侧向={lat_n*1000:.1f}mm "
            f"(期望 along≈{standoff_m*1000:.0f} / 侧向≈0；侧向大=预览会歪)"
        )

    off_cam = eye.get("hand_offset_in_camera_m", [0.06, -0.09, 0.0])
    stab = grasp_stabilize_preset(
        controller, size_info=size_info, eye=eye,
    )
    plan = {
        "obj_world": obj_world,
        "obj_target": obj_target,
        "ee_sim": ee_sim,
        "target_pos": target_pos,
        "target_rpy": target_rpy,
        "T_pinch_des": T_w_pinch_des,
        "T_ee_des": T_w_ee_des,
        "align_mode": mode_tag,
        "grasp_mode": gmode,
        "grasp_center_m": close_mid.tolist(),
        "open_center_m": open_mid.tolist(),
        "grasp_close_hand": list(grasp_close),
        "grasp_stabilize_hand": list(stab) if stab is not None else None,
        "depth_mm": float(depth_mm),
        "hand_offset_cam": list(off_cam),
        "thumb_tip_m": thumb_d.tolist(),
        "index_tip_m": index_d.tolist() if index_d is not None else None,
        "middle_tip_m": middle_d.tolist(),
        "ring_tip_m": ring_d.tolist() if ring_d is not None else None,
        "circumradius_mm": circ_r if gmode == "power5" else None,
        "tcp_origin_mode": pinch_info.get("tcp_origin_mode") if gmode == "power5" else None,
    }
    return plan
