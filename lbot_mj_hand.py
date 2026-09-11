"""MuJoCo O6 右手：L6 指令同步 + 指尖/捏取 kinematics（与 workstation.mjcf 一致）。"""
from __future__ import annotations

import math

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

from lbot_grasp_utils import (
    O6_L6_MASTER_JOINTS,
    O6_L6_MIMIC_JOINTS,
    build_grasp_task_frame,
    build_grasp_task_frame_power,
    l6_value_to_joint_angle_mj,
    rotation_matrix_to_rpy,
)

# distal body 系下指腹中心（mesh 不可用时的 fallback；pad_plane 会缓存覆盖）
THUMB_PAD_LOCAL = np.array([-0.010, 0.0, 0.022], dtype=np.float64)
MIDDLE_PAD_LOCAL = np.array([-0.002, 0.0, 0.018], dtype=np.float64)
INDEX_PAD_LOCAL = np.array([-0.002, 0.0, 0.018], dtype=np.float64)
RING_PAD_LOCAL = np.array([-0.002, 0.0, 0.018], dtype=np.float64)
PINKY_PAD_LOCAL = np.array([-0.002, 0.0, 0.018], dtype=np.float64)

PALM_BODY = "hand_right_rh_hand_base_link"
THUMB_DISTAL_BODY = "hand_right_rh_thumb_distal"
INDEX_DISTAL_BODY = "hand_right_rh_index_distal"
MIDDLE_DISTAL_BODY = "hand_right_rh_middle_distal"
RING_DISTAL_BODY = "hand_right_rh_ring_distal"
PINKY_DISTAL_BODY = "hand_right_rh_pinky_distal"
TOOL0_SITE = "arm_right_tool0"
# workstation.mjcf / recipe.yaml 默认（双人形掌心相对；单臂工位可能需 config 覆盖）
DEFAULT_HAND_MOUNT_RPY = (3.1416, 0.0, 1.5708)

# 实机 API 关节角 → MuJoCo qpos 符号（+1 同向，-1 取反）。
# 实测右臂控制器与 workstation.mjcf 在 J1/J2/J3/J7 正方向相反
# （掌心朝向左右镜像）；仅影响模型显示/规划用 FK，不改实机指令。
DEFAULT_ARM_JOINT_SIGNS = (-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)

# body_name -> (centroid_local 3, normal_local 3)；一次标定，避免每帧抖动
_PAD_PLANE_CACHE = {}
# body_name -> fingertip_local 3（远端尖端，非指腹）
_TIP_POINT_CACHE = {}


def clear_pad_plane_cache():
    _PAD_PLANE_CACHE.clear()
    _TIP_POINT_CACHE.clear()


def _quat_wxyz_to_mat(quat):
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _mj_body_mesh_verts_local(model, body_name):
    """收集 body 下 mesh 顶点（body 局部系；与姿态无关，可稳定缓存）。"""
    if mujoco is None:
        return None
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    chunks = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != bid:
            continue
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        dataid = int(model.geom_dataid[gid])
        if dataid < 0:
            continue
        vadr = int(model.mesh_vertadr[dataid])
        nvert = int(model.mesh_vertnum[dataid])
        if nvert <= 0:
            continue
        verts = np.asarray(
            model.mesh_vert[vadr : vadr + nvert], dtype=np.float64,
        ).reshape(-1, 3)
        scale = np.asarray(model.mesh_scale[dataid], dtype=np.float64).reshape(3)
        verts = verts * scale
        R_g = _quat_wxyz_to_mat(model.geom_quat[gid])
        p_g = np.asarray(model.geom_pos[gid], dtype=np.float64).reshape(3)
        chunks.append((R_g @ verts.T).T + p_g)
    if not chunks:
        return None
    return np.vstack(chunks)


def _fit_pad_plane_local(verts_local, tip_frac=0.22, pad_axis_prefer=None):
    """
    取指尖一段顶点拟小平面：质心 + 法向（body 局部）。
    tip_frac: 沿最长主轴顶端比例；pad_axis_prefer 偏指腹一侧。
    """
    v = np.asarray(verts_local, dtype=np.float64).reshape(-1, 3)
    if v.shape[0] < 8:
        return None
    mean0 = v.mean(axis=0)
    _, _, vt = np.linalg.svd(v - mean0, full_matrices=False)
    tip_axis = vt[0]
    # 远端：投影大的一侧应更远离 body 原点（关节）
    proj = (v - mean0) @ tip_axis
    hi = v[proj >= float(np.quantile(proj, 0.75))]
    lo = v[proj <= float(np.quantile(proj, 0.25))]
    if float(np.mean(np.linalg.norm(hi, axis=1))) < float(
        np.mean(np.linalg.norm(lo, axis=1))
    ):
        tip_axis = -tip_axis
        proj = -proj
    thr = float(np.quantile(proj, max(0.0, 1.0 - float(tip_frac))))
    tip = v[proj >= thr]
    if tip.shape[0] < 6:
        tip = v[np.argsort(proj)[-max(6, v.shape[0] // 5) :]]
    if pad_axis_prefer is not None and tip.shape[0] >= 8:
        pref = np.asarray(pad_axis_prefer, dtype=np.float64).reshape(3)
        pn = float(np.linalg.norm(pref))
        if pn > 1e-9:
            pref = pref / pn
            tip_mean = tip.mean(axis=0)
            s = (tip - tip_mean) @ pref
            tip2 = tip[s >= float(np.median(s))]
            if tip2.shape[0] >= 4:
                tip = tip2
    c = tip.mean(axis=0)
    _, _, vt2 = np.linalg.svd(tip - c, full_matrices=False)
    n = vt2[-1]
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    if pad_axis_prefer is not None:
        pref = np.asarray(pad_axis_prefer, dtype=np.float64).reshape(3)
        if float(np.dot(n, pref)) < 0.0:
            n = -n
    thick = float(np.std((tip - c) @ n))
    c_pad = c + n * min(max(thick * 0.35, 0.0005), 0.004)
    return c_pad.copy(), n.copy()


def resolve_pad_local(model, body_name, fallback_local, tip_frac=0.22):
    """返回 distal body 系指腹质心；优先缓存的 pad 平面。"""
    cached = _PAD_PLANE_CACHE.get(body_name)
    if cached is not None:
        return np.asarray(cached[0], dtype=np.float64)
    verts = _mj_body_mesh_verts_local(model, body_name)
    fb = np.asarray(fallback_local, dtype=np.float64).reshape(3)
    prefer = None
    xy = fb.copy()
    xy[2] = 0.0
    if float(np.linalg.norm(xy)) > 1e-6:
        prefer = xy / np.linalg.norm(xy)
    fitted = None
    if verts is not None:
        fitted = _fit_pad_plane_local(verts, tip_frac=tip_frac, pad_axis_prefer=prefer)
    if fitted is None:
        _PAD_PLANE_CACHE[body_name] = (fb.copy(), np.array([0.0, 0.0, 1.0]))
        return fb.copy()
    c, n = fitted
    _PAD_PLANE_CACHE[body_name] = (c, n)
    return c.copy()


def _fit_fingertip_local(verts_local, tip_frac=0.08):
    """
    取远端一小撮顶点的质心作为指尖（不做指腹法向偏移）。
    tip_frac 越小越靠尖端。
    """
    v = np.asarray(verts_local, dtype=np.float64).reshape(-1, 3)
    if v.shape[0] < 8:
        return None
    mean0 = v.mean(axis=0)
    _, _, vt = np.linalg.svd(v - mean0, full_matrices=False)
    tip_axis = vt[0]
    proj = (v - mean0) @ tip_axis
    hi = v[proj >= float(np.quantile(proj, 0.75))]
    lo = v[proj <= float(np.quantile(proj, 0.25))]
    if float(np.mean(np.linalg.norm(hi, axis=1))) < float(
        np.mean(np.linalg.norm(lo, axis=1))
    ):
        tip_axis = -tip_axis
        proj = -proj
    frac = float(np.clip(tip_frac, 0.03, 0.25))
    thr = float(np.quantile(proj, max(0.0, 1.0 - frac)))
    tip = v[proj >= thr]
    if tip.shape[0] < 4:
        tip = v[np.argsort(proj)[-max(4, v.shape[0] // 8) :]]
    c = tip.mean(axis=0)
    tip_proj = (tip - c) @ tip_axis
    c = c + tip_axis * float(np.clip(np.max(tip_proj) * 0.55, 0.0, 0.006))
    return c.copy()


def resolve_tip_local(model, body_name, fallback_local, tip_frac=0.08):
    """返回 distal body 系指尖点（非指腹）。"""
    cached = _TIP_POINT_CACHE.get(body_name)
    if cached is not None:
        return np.asarray(cached, dtype=np.float64)
    verts = _mj_body_mesh_verts_local(model, body_name)
    fb = np.asarray(fallback_local, dtype=np.float64).reshape(3)
    fb_tip = fb.copy()
    fb_n = float(np.linalg.norm(fb_tip))
    if fb_n > 1e-6:
        fb_tip = fb_tip * 1.35
    fitted = None
    if verts is not None:
        fitted = _fit_fingertip_local(verts, tip_frac=tip_frac)
    if fitted is None:
        _TIP_POINT_CACHE[body_name] = fb_tip.copy()
        return fb_tip.copy()
    _TIP_POINT_CACHE[body_name] = fitted.copy()
    return fitted.copy()


def resolve_arm_joint_signs(config_eye, side="right"):
    """从 eye_in_hand.model_arm_joint_signs[_left] 读取 7 元符号表。"""
    raw = None
    if isinstance(config_eye, dict):
        if str(side).lower() == "left":
            raw = config_eye.get("model_arm_joint_signs_left")
            if raw is None:
                # 无单独配置时：右臂符号再翻 J4、J6
                base = resolve_arm_joint_signs(config_eye, side="right")
                out = list(base)
                out[3] = -out[3]
                out[5] = -out[5]
                return out
        else:
            raw = config_eye.get("model_arm_joint_signs")
    if raw is None:
        return list(DEFAULT_ARM_JOINT_SIGNS)
    signs = [float(s) for s in raw]
    if len(signs) != 7:
        return list(DEFAULT_ARM_JOINT_SIGNS)
    out = []
    for s in signs:
        if s >= 0:
            out.append(1.0)
        else:
            out.append(-1.0)
    return out


def robot_joints_to_mj_qpos(joints, signs):
    """实机关节角 → MuJoCo 臂 qpos（配合符号表）。"""
    js = [float(j) for j in list(joints)[:7]]
    while len(js) < 7:
        js.append(0.0)
    sg = list(signs) if signs is not None else list(DEFAULT_ARM_JOINT_SIGNS)
    while len(sg) < 7:
        sg.append(1.0)
    return [js[i] * sg[i] for i in range(7)]


def resolve_hand_l6_cmd(model_hand, robot_hand, default_open):
    """模型显示用 L6：预览手指令 > 实机缓存 > 默认张开。"""
    if model_hand is not None:
        return list(model_hand)
    if robot_hand is not None:
        return list(robot_hand)
    return list(default_open)


def apply_o6_hand_to_mjcf(
    mj_data, mj_joints, mj_hand_ranges, hand_cmd, mirror_qpos=False, side="right",
):
    """L6 六通道 → 主关节 + mimic 从动（仅 MuJoCo 显示）。side=right|left。"""
    if hand_cmd is None or len(hand_cmd) < 6:
        return

    def _side_name(name):
        if side == "left":
            return str(name).replace("hand_right_rh_", "hand_left_lh_")
        return str(name)

    master_angles = {}
    for ch, jname0 in O6_L6_MASTER_JOINTS:
        jname = _side_name(jname0)
        if jname not in mj_joints or jname not in mj_hand_ranges:
            continue
        lo, hi = mj_hand_ranges[jname]
        ang = l6_value_to_joint_angle_mj(lo, hi, hand_cmd[ch], mirror_qpos=mirror_qpos)
        mj_data.qpos[mj_joints[jname]] = ang
        master_angles[jname] = ang
    for dep0, master0, ratio in O6_L6_MIMIC_JOINTS:
        dep = _side_name(dep0)
        master = _side_name(master0)
        if dep not in mj_joints or master not in master_angles:
            continue
        lo, hi = mj_hand_ranges.get(dep, (0.0, 1.5))
        ang = float(np.clip(master_angles[master] * ratio, lo, hi))
        mj_data.qpos[mj_joints[dep]] = ang


def apply_hand_mount_rpy(model, rpy):
    """覆盖 MuJoCo 手 base 相对法兰的安装角（与实机不一致时调 grasp_config）。"""
    if mujoco is None or rpy is None:
        return False
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PALM_BODY)
    if bid < 0:
        return False
    euler = np.asarray(rpy, dtype=np.float64).reshape(3)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, euler, "xyz")
    model.body_quat[bid] = quat
    return True


def mj_body_pose(model, data, body_name):
    if mujoco is None:
        return None
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def mj_site_pos(model, data, site_name):
    if mujoco is None:
        return None
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        return None
    return data.site_xpos[sid].copy()


def mj_body_tip(model, data, body_name, tip_local):
    pose = mj_body_pose(model, data, body_name)
    if pose is None:
        return None
    pos, rot = pose
    return pos + rot @ np.asarray(tip_local, dtype=np.float64)


def mj_body_pad_world(model, data, body_name, fallback_local, tip_frac=0.22):
    pose = mj_body_pose(model, data, body_name)
    if pose is None:
        return None
    pos, rot = pose
    local = resolve_pad_local(model, body_name, fallback_local, tip_frac=tip_frac)
    return pos + rot @ local


def _finger_contact(model, data, body, fallback_local, tip_frac, mode):
    """mode: tip | fixed | pad_plane。"""
    if mode in ("tip", "fingertip", "distal_tip"):
        local = resolve_tip_local(model, body, fallback_local, tip_frac=tip_frac)
        return mj_body_tip(model, data, body, local)
    if mode == "fixed":
        return mj_body_tip(model, data, body, fallback_local)
    return mj_body_pad_world(
        model, data, body, fallback_local, tip_frac=tip_frac,
    )


def _pad_normal_world(model, data, body_name):
    cached = _PAD_PLANE_CACHE.get(body_name)
    if cached is None:
        return None
    pose = mj_body_pose(model, data, body_name)
    if pose is None:
        return None
    return (pose[1] @ cached[1]).tolist()


def compute_o6_pinch_kinematics(
    model, data, tip_frac=0.22, tip_mode="pad_plane", grasp_mode="pinch",
):
    """
    指尖/指腹接触点 + 夹取 task frame。
    tip_mode: tip | pad_plane | fixed
    grasp_mode: pinch(拇/食/中·质心) | power5(拇/中/无名·外接圆心)
    """
    palm = mj_body_pose(model, data, PALM_BODY)
    mode = str(tip_mode or "pad_plane").strip().lower()
    gmode = str(grasp_mode or "pinch").strip().lower()
    use_power = gmode in ("power5", "power", "five", "5", "envelope")
    use_tip = mode in ("tip", "fingertip", "distal_tip")
    frac = float(tip_frac)
    if use_tip and frac > 0.15:
        frac = 0.08
    thumb = _finger_contact(
        model, data, THUMB_DISTAL_BODY, THUMB_PAD_LOCAL, frac, mode,
    )
    index = _finger_contact(
        model, data, INDEX_DISTAL_BODY, INDEX_PAD_LOCAL, frac, mode,
    )
    middle = _finger_contact(
        model, data, MIDDLE_DISTAL_BODY, MIDDLE_PAD_LOCAL, frac, mode,
    )
    ring = _finger_contact(
        model, data, RING_DISTAL_BODY, RING_PAD_LOCAL, frac, mode,
    )
    pinky = _finger_contact(
        model, data, PINKY_DISTAL_BODY, PINKY_PAD_LOCAL, frac, mode,
    )
    if palm is None or thumb is None or middle is None:
        return None
    if use_power:
        if ring is None:
            return None
    elif index is None:
        return None
    _palm_pos, palm_rot = palm
    if use_power:
        center = (thumb + middle + ring) / 3.0
    else:
        center = (thumb + index + middle) / 3.0
    tool0 = mj_site_pos(model, data, TOOL0_SITE)
    approach_hint = tool0 - center if tool0 is not None else None
    if use_power:
        frame = build_grasp_task_frame_power(
            thumb, middle, ring, approach_hint=approach_hint, palm_rot=palm_rot,
        )
    else:
        frame = build_grasp_task_frame(
            thumb, index, middle, approach_hint=approach_hint, palm_rot=palm_rot,
        )
    if frame is None:
        return None
    rpy = rotation_matrix_to_rpy(frame["R_world"])
    if use_tip:
        mode_out = "tip"
    elif mode == "fixed":
        mode_out = "fixed"
    else:
        mode_out = "pad_plane"
    out = {
        "thumb_tip_m": thumb.tolist(),
        "index_tip_m": index.tolist() if index is not None else None,
        "middle_tip_m": middle.tolist(),
        "ring_tip_m": ring.tolist() if ring is not None else None,
        "pinky_tip_m": pinky.tolist() if pinky is not None else None,
        "pinch_mid_m": frame["origin_m"],
        "pinch_sep_mm": float(frame["pinch_sep_m"]) * 1000.0,
        "x_close": frame["x_close"],
        "y_axis": frame["y_axis"],
        "z_approach": frame["z_approach"],
        "R_world": frame["R_world"],
        "virtual_ee_rpy_rad": rpy,
        "virtual_ee_rpy_deg": [math.degrees(v) for v in rpy],
        "thumb_pad_normal": (
            None if use_tip or mode == "fixed"
            else _pad_normal_world(model, data, THUMB_DISTAL_BODY)
        ),
        "index_pad_normal": (
            None if use_tip or mode == "fixed" or index is None
            else _pad_normal_world(model, data, INDEX_DISTAL_BODY)
        ),
        "middle_pad_normal": (
            None if use_tip or mode == "fixed"
            else _pad_normal_world(model, data, MIDDLE_DISTAL_BODY)
        ),
        "tip_mode": mode_out,
        "grasp_mode": "power5" if use_power else "pinch",
    }
    if use_power:
        out["circumradius_mm"] = float(frame.get("circumradius_m", 0.0)) * 1000.0
        out["inradius_mm"] = float(frame.get("inradius_m", 0.0)) * 1000.0
        out["incenter_m"] = frame.get("incenter_m")
        out["circumcenter_m"] = frame.get("circumcenter_m", frame["origin_m"])
        out["tcp_origin_mode"] = frame.get("tcp_origin_mode", "circumcenter")
    return out


def collect_hand_joint_maps(mj_model):
    """加载 MJCF 后收集手关节 qpos 索引与量程。"""
    mj_joints = {}
    mj_hand_ranges = {}
    for joint_id in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            mj_joints[name] = mj_model.jnt_qposadr[joint_id]
    for _ch, jname in O6_L6_MASTER_JOINTS:
        for name in (
            jname,
            str(jname).replace("hand_right_rh_", "hand_left_lh_"),
        ):
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                mj_hand_ranges[name] = (
                    float(mj_model.jnt_range[jid, 0]),
                    float(mj_model.jnt_range[jid, 1]),
                )
    for dep, _master, _ratio in O6_L6_MIMIC_JOINTS:
        for name in (
            dep,
            str(dep).replace("hand_right_rh_", "hand_left_lh_"),
        ):
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                mj_hand_ranges[name] = (
                    float(mj_model.jnt_range[jid, 0]),
                    float(mj_model.jnt_range[jid, 1]),
                )
    return mj_joints, mj_hand_ranges


def hand_l6_master_angles(hand_cmd, mj_hand_ranges, mirror_qpos=False):
    """L6 六通道 → 主关节角（rad），便于与实机/面板预设对照。"""
    if hand_cmd is None or len(hand_cmd) < 6:
        return {}
    out = {}
    for ch, jname in O6_L6_MASTER_JOINTS:
        if jname not in mj_hand_ranges:
            continue
        lo, hi = mj_hand_ranges[jname]
        out[jname] = l6_value_to_joint_angle_mj(
            lo, hi, hand_cmd[ch], mirror_qpos=mirror_qpos,
        )
    return out


def summarize_hand_l6_geometry(
    model,
    data,
    joints,
    hand_cmd,
    mj_joints,
    mj_hand_ranges,
    arm_signs,
    kin_kw,
    mirror_qpos=False,
):
    """
    给定臂关节 + L6，返回三指 mid/tips/sep 与主关节角（MJ FK）。
    调用方需已持 mj lock；本函数会写 qpos 并 mj_forward。
    """
    if model is None or data is None or hand_cmd is None:
        return None
    mj_q = robot_joints_to_mj_qpos(joints, arm_signs)
    for index, value in enumerate(mj_q, start=1):
        name = f"arm_right_R{index}_Joint"
        if name in mj_joints:
            data.qpos[mj_joints[name]] = value
    apply_o6_hand_to_mjcf(
        data, mj_joints, mj_hand_ranges, hand_cmd, mirror_qpos=mirror_qpos,
    )
    mujoco.mj_forward(model, data)
    kin = compute_o6_pinch_kinematics(model, data, **(kin_kw or {}))
    if kin is None:
        return None
    return {
        "hand_cmd": [int(v) for v in hand_cmd[:6]],
        "master_angles": hand_l6_master_angles(
            hand_cmd, mj_hand_ranges, mirror_qpos=mirror_qpos,
        ),
        "pinch_mid_m": np.asarray(kin["pinch_mid_m"], dtype=np.float64).reshape(3),
        "pinch_sep_mm": float(kin.get("pinch_sep_mm", 0.0)),
        "thumb_tip_m": np.asarray(kin["thumb_tip_m"], dtype=np.float64).reshape(3),
        "index_tip_m": (
            np.asarray(kin["index_tip_m"], dtype=np.float64).reshape(3)
            if kin.get("index_tip_m") is not None else None
        ),
        "middle_tip_m": np.asarray(kin["middle_tip_m"], dtype=np.float64).reshape(3),
        "ring_tip_m": (
            np.asarray(kin["ring_tip_m"], dtype=np.float64).reshape(3)
            if kin.get("ring_tip_m") is not None else None
        ),
        "inradius_mm": float(kin.get("inradius_mm") or 0.0),
        "circumradius_mm": float(kin.get("circumradius_mm") or 0.0),
        "tcp_origin_mode": str(kin.get("tcp_origin_mode") or ""),
        "grasp_mode": str(kin.get("grasp_mode") or ""),
        "z_approach": np.asarray(kin["z_approach"], dtype=np.float64).reshape(3),
        "x_close": np.asarray(kin["x_close"], dtype=np.float64).reshape(3),
    }


def log_hand_preset_mj_compare(
    model,
    data,
    mj_joints,
    mj_hand_ranges,
    pinch_open,
    pinch_close,
    joints,
    arm_signs,
    kin_kw,
    mirror_qpos=False,
    log=print,
):
    """启动/调试：MJ 下 open vs close 预设几何（与实机 pinch_open/close 应对齐）。"""
    pinch_open = list(pinch_open)
    pinch_close = list(pinch_close)
    with_summary = []
    for label, cmd in (("open", pinch_open), ("close", pinch_close)):
        s = summarize_hand_l6_geometry(
            model, data, joints, cmd, mj_joints, mj_hand_ranges,
            arm_signs, kin_kw, mirror_qpos=mirror_qpos,
        )
        if s is None:
            log(f"  O6-{label}: MJ 几何不可用")
            continue
        with_summary.append((label, s))
        mid = s["pinch_mid_m"]
        ang = s["master_angles"]
        ang_s = ", ".join(
            f"{k.split('_')[-1]}={math.degrees(v):.0f}°"
            for k, v in sorted(ang.items())
        )
        extra = ""
        if s.get("grasp_mode") == "power5":
            extra = (
                f" R={s.get('circumradius_mm', s.get('inradius_mm', 0)):.1f}mm"
                f" mode={s.get('tcp_origin_mode', '')}"
            )
        log(
            f"  O6-{label} L6={s['hand_cmd']} sep={s['pinch_sep_mm']:.1f}mm "
            f"mid=[{mid[0]:.3f},{mid[1]:.3f},{mid[2]:.3f}]{extra}"
        )
        if ang_s:
            log(f"    MJ主关节°: {ang_s}")

    if len(with_summary) == 2:
        _lo, s_o = with_summary[0]
        _lc, s_c = with_summary[1]
        d_mid = s_c["pinch_mid_m"] - s_o["pinch_mid_m"]
        d_sep = s_c["pinch_sep_mm"] - s_o["pinch_sep_mm"]
        mid_mm = float(np.linalg.norm(d_mid)) * 1000.0
        # 分解到闭合/接近轴：便于区分「运动学漂移」与到位误差
        along_c = along_a = lat = None
        try:
            from lbot_grasp_utils import (
                build_grasp_task_frame,
                build_grasp_task_frame_power,
            )

            if s_c.get("grasp_mode") == "power5" and s_c.get("ring_tip_m") is not None:
                fr = build_grasp_task_frame_power(
                    s_c["thumb_tip_m"],
                    s_c["middle_tip_m"],
                    s_c["ring_tip_m"],
                    approach_hint=s_c.get("z_approach"),
                )
            elif s_c.get("index_tip_m") is not None:
                fr = build_grasp_task_frame(
                    s_c["thumb_tip_m"],
                    s_c["index_tip_m"],
                    s_c["middle_tip_m"],
                    approach_hint=s_c.get("z_approach"),
                )
            else:
                fr = None
            if fr is not None:
                R = np.asarray(fr["R_world"], dtype=np.float64).reshape(3, 3)
                along_c = float(np.dot(d_mid, R[:, 0])) * 1000.0
                lat = float(np.dot(d_mid, R[:, 1])) * 1000.0
                along_a = float(np.dot(d_mid, R[:, 2])) * 1000.0
        except Exception:
            pass
        axis_s = ""
        if along_c is not None:
            axis_s = (
                f" 闭合轴={along_c:+.1f} 接近轴={along_a:+.1f} 侧向={lat:+.1f}mm"
            )
        log(
            f"  O6 张→闭(MJ·运动学漂移≠到位误差): mid|Δ|={mid_mm:.1f}mm "
            f"sep {s_o['pinch_sep_mm']:.0f}→{s_c['pinch_sep_mm']:.0f}mm "
            f"(Δ{d_sep:.1f}){axis_s}"
        )
        note = "规划用闭合 TCP 已补偿"
        if float(s_o["pinch_sep_mm"]) < 120.0:
            note += "；张手 sep<120 易⑦蹭螺母侧沿（勿为压 midΔ 过度收拢）"
        elif mid_mm > 30.0:
            note += "；|Δ|≳30mm 多为张手够开时的 tip 漂移（正常）"
        log(
            f"    Δxyz_mm=[{d_mid[0]*1000:.1f},{d_mid[1]*1000:.1f},{d_mid[2]*1000:.1f}]"
            f"；{note}"
        )
    mirror_note = "mirror_qpos=开" if mirror_qpos else "mirror_qpos=关"
    log(f"  （{mirror_note}；实机 ⑧ 应下发 close L6，无读回则只能对照预设）")
