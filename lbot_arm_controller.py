"""右臂控制器：视觉伺服、夹取流程、手/臂运动。

日志默认安静（见 lbot_logutil / yaml: verbose_logs、pinch_debug）。
夹取主路径：execute_pick_pinch（①张手→⑪松手）。
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from lbot.lbot_robot import LbotArm, LbotEuler, LbotPosition
from lbot_logutil import dbg, eye_debug, eye_verbose, move_print, vrb
from lbot_grasp_utils import (
    DEFAULT_CONFIG,
    AlignFusionBuffer,
    apply_mjcf_camera_to_config,
    build_pinch_frame_at_object,
    build_camera_on_ee,
    camera_delta_to_ee,
    camera_pose_in_world,
    center_only_correction_in_ee,
    clamp_joints_to_soft_limits,
    detection_point_in_world,
    ee_target_from_pinch,
    approach_unit_toward_object,
    pinch_standoff_target,
    euler_rpy_to_matrix,
    is_grasp_aligned,
    joints_soft_limit_violation,
    sanitize_action_depth_mm,
    lerp_pose_rpy,
    matrix_to_rpy_near,
    object_up_direction_world,
    pinch_dict_to_matrix,
    pixel_depth_to_camera_point,
    pose_to_matrix,
    resolve_pinch_standoff_m,
    resolve_pinch_hand_short_m,
    resolve_pinch_level_finger_plane,
    resolve_place_level_finger_plane,
    resolve_pregrasp_depth_mm,
    resolve_soft_joint_limits_rad,
    rotation_geodesic_deg,
    rotation_matrix_to_rpy,
    save_camera_on_ee_translation,
    soft_limit_inward_seed,
)

ARM = LbotArm.RIGHT_ARM
# 张开：H1=200 半开 pitch，H2=0 拇指不侧摆；四指全开
DEFAULT_OPEN_PRESET = [200, 0, 255, 255, 255, 255]
DEFAULT_GRASP_PRESET = [150, 0, 255, 140, 140, 255]
DEFAULT_PINCH_OPEN = [200, 0, 255, 255, 255, 255]
DEFAULT_PINCH_CLOSE = [150, 0, 255, 140, 140, 255]
DEFAULT_PINCH_STABILIZE = [150, 0, 140, 140, 140, 140]

class RightArmController:
    """单臂运动 + O6 手；默认右臂，可用 set_control_arm 切左/右。"""

    def __init__(self, config, robot=None, dry_run=False, config_path=DEFAULT_CONFIG, arm=None):
        self.config = config
        self.config_path = config_path
        self.robot = robot
        self.dry_run = dry_run
        self.arm = arm if arm is not None else ARM
        self.eye = config["eye_in_hand"]
        self.grasp_size_info = None
        self.intrinsics_base = dict(config["camera_intrinsics"])
        self.intrinsics = dict(self.intrinsics_base)
        self.hand_presets = self._hand_presets_for_arm(self.arm)
        # servo_xy_* 若仍非零会并入 R_ee_cam；运行时射线/伺服不再二次 remap
        self.R_ee_cam, self.t_ee_cam = build_camera_on_ee(config, fold_servo_xy=True)
        self.xy_rotate_deg = 0.0
        self.servo_flip_x = False
        self.servo_flip_y = False
        self.speed = config["robot"]["speed"]
        self.accel = config["robot"]["accel"]
        self.retreat_offset = config["robot"]["retreat_offset_m"]
        self.default_home_joints = list(
            config["robot"].get(
                "default_home_joints",
                [0.699, -0.22, -0.04, -0.86, -1.38, 0.032, 0.880],
            )
        )
        self.stable_required = config["detection"]["stable_frames"]
        self.stable_count = 0
        # 未知：勿默认 pinch_open，否则启动后第一次①会跳过实张（后几次本就正常，不改）
        self.hand_cmd = None
        # 运动线程写入、UI 只读：busy 时模型仍能跟实机，且主线程不抢 TCP
        self._live_lock = threading.Lock()
        self.live_joints = None
        self.live_pose = None
        # 运动等待可被 UI「停止」打断
        self._motion_abort = False
        # 连续比例对准中（视觉线程需保持 YOLO 刷新，避免重复伺服同一像素误差）
        self._aligning = False
        self._vision_det_seq = 0
        self._ee_trace_stop = None
        self._ee_trace_thread = None
        self._ee_trace_phase = ""

    def _hand_presets_for_arm(self, arm):
        hp_all = self.config.get("hand_presets") or {}
        key = "left" if arm == LbotArm.LEFT_ARM else "right"
        hp = hp_all.get(key) or hp_all.get("right") or {}
        hp = dict(hp)
        if "pinch_open" not in hp:
            hp["pinch_open"] = list(hp.get("open", DEFAULT_PINCH_OPEN))
        if "pinch_close" not in hp:
            hp["pinch_close"] = list(hp.get("grasp", DEFAULT_PINCH_CLOSE))
        return hp

    def set_control_arm(self, arm, log=None):
        """切换当前控制臂（左/右）。抓取/相机流水线默认仍用右臂。"""
        arm = LbotArm.LEFT_ARM if int(arm) == int(LbotArm.LEFT_ARM) else LbotArm.RIGHT_ARM
        self.arm = arm
        self.hand_presets = self._hand_presets_for_arm(arm)
        self.hand_cmd = None
        robot_cfg = self.config.get("robot") or {}
        if arm == LbotArm.LEFT_ARM:
            # 左臂默认以笛卡尔示教位为准；关节表仅作回退
            home = robot_cfg.get("default_home_joints_left")
            if home is None:
                home = robot_cfg.get("default_home_joints")
        else:
            home = robot_cfg.get("default_home_joints")
        if home is not None:
            self.default_home_joints = list(home)
        if log is not None:
            log(f"控制臂已切换为{self.control_arm_label()}")
        return self.arm

    def control_arm_label(self):
        return "左臂" if self.arm == LbotArm.LEFT_ARM else "右臂"

    def set_ee_trace_phase(self, phase):
        """抓取阶段标签，写入 0.1s 末端轨迹日志。"""
        self._ee_trace_phase = str(phase or "")

    def start_ee_trace(self, log=print, period_s=0.1):
        """抓取全程每 period_s 打印一次 tool0 笛卡尔坐标。"""
        self.stop_ee_trace()
        period_s = float(max(period_s, 0.05))
        stop_ev = threading.Event()
        self._ee_trace_stop = stop_ev
        self._ee_trace_phase = "start"
        t0 = time.time()
        last_xyz = [None]

        def _worker():
            n = 0
            while not stop_ev.wait(period_s):
                pose = self.get_pose()
                if pose is None:
                    log(f"[ee @{(time.time()-t0):5.1f}s] phase={self._ee_trace_phase} 无位姿")
                    continue
                pos, eul = pose
                xyz = (float(pos.x), float(pos.y), float(pos.z))
                dmm = ""
                if last_xyz[0] is not None:
                    dx = (xyz[0] - last_xyz[0][0]) * 1000.0
                    dy = (xyz[1] - last_xyz[0][1]) * 1000.0
                    dz = (xyz[2] - last_xyz[0][2]) * 1000.0
                    dmm = f" Δmm=[{dx:+.1f},{dy:+.1f},{dz:+.1f}]"
                last_xyz[0] = xyz
                n += 1
                log(
                    f"[ee @{time.time()-t0:5.1f}s #{n:03d}] "
                    f"{self._ee_trace_phase} "
                    f"xyz=[{xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f}] "
                    f"rpy°=[{math.degrees(eul.x):.1f},"
                    f"{math.degrees(eul.y):.1f},{math.degrees(eul.z):.1f}]"
                    f"{dmm}"
                )

        th = threading.Thread(target=_worker, daemon=True, name="ee_trace")
        self._ee_trace_thread = th
        th.start()
        log(f"已开启末端轨迹日志（每 {period_s*1000:.0f}ms）")

    def stop_ee_trace(self):
        ev = self._ee_trace_stop
        th = self._ee_trace_thread
        self._ee_trace_stop = None
        self._ee_trace_thread = None
        if ev is not None:
            ev.set()
        if th is not None and th.is_alive():
            th.join(timeout=0.5)
        self._ee_trace_phase = ""

    def request_motion_abort(self):
        self._motion_abort = True

    def clear_motion_abort(self):
        self._motion_abort = False

    def halt_all_motion(self, log_prefix="[stop]"):
        """
        立刻打断在途轨迹：当前臂保持关节 + 双臂急停脉冲。
        UI「停止」用；不替代 clear_errors，急停后立即恢复以便后续可动。
        """
        self._motion_abort = True
        if not self.robot:
            return
        try:
            self._hold_current_motion(log_prefix)
        except Exception as exc:
            print(f"{log_prefix} hold失败: {exc}", flush=True)
        # 双臂都刹一下，避免另一臂还在走
        for arm in (LbotArm.LEFT_ARM, LbotArm.RIGHT_ARM):
            try:
                self.robot.emergency_stop(arm, True)
            except Exception:
                pass
        time.sleep(0.06)
        for arm in (LbotArm.LEFT_ARM, LbotArm.RIGHT_ARM):
            try:
                self.robot.emergency_stop(arm, False)
            except Exception:
                pass
        # 左手若在拧：回抓握停电批（不阻塞太久）
        try:
            if int(self.arm) == int(LbotArm.LEFT_ARM):
                hp = self.hand_presets or {}
                restore = list(hp.get("grasp") or hp.get("pinch_close") or [])
                if len(restore) >= 6:
                    self.robot.l6_set_position(
                        self.arm, [int(np.clip(int(v), 0, 255)) for v in restore[:6]]
                    )
                    self.hand_cmd = list(restore[:6])
        except Exception as exc:
            print(f"{log_prefix} 停手失败(忽略): {exc}", flush=True)
        try:
            self.refresh_live_state()
        except Exception:
            pass
        print(f"{log_prefix} 已请求停止一切运动", flush=True)

    def _fallback_joints(self):
        return list(self.default_home_joints)

    @staticmethod
    def _joints_look_valid(joints):
        if not joints or len(joints) != 7:
            return False
        return not all(abs(float(j)) < 1e-4 for j in joints)

    def refresh_live_state(self):
        """从实机刷新关节/位姿缓存（供 MuJoCo 与 UI；宜在运动线程调用）。"""
        if not self.robot:
            return
        raw = self.robot.get_joint_positions(self.arm)
        if raw and len(raw) == 7:
            with self._live_lock:
                self.live_joints = [float(j) for j in raw]
        try:
            pose = self.robot.get_cartesian_pose(self.arm)
        except Exception:
            pose = None
        if pose is not None:
            with self._live_lock:
                self.live_pose = pose

    def get_live_joints(self):
        with self._live_lock:
            if self.live_joints is not None:
                return list(self.live_joints)
        return self.get_joints()

    def get_live_pose(self):
        with self._live_lock:
            if self.live_pose is not None:
                return self.live_pose
        return self.get_pose()

    def get_joints(self, allow_fallback=True):
        """读实机关节。allow_fallback=False 时读失败返回 None（禁止用默认原位冒充当前角）。"""
        if not self.robot:
            return self._fallback_joints() if allow_fallback else None
        raw = self.robot.get_joint_positions(self.arm)
        if raw and self._joints_look_valid(raw):
            joints = [float(j) for j in raw]
            with self._live_lock:
                self.live_joints = list(joints)
            return joints
        return self._fallback_joints() if allow_fallback else None

    def read_joints_fresh(self, tries=5, pause_s=0.04):
        """多次尝试读实机关节；失败返回 None（绝不回落默认原位）。"""
        if not self.robot:
            return None
        for _ in range(max(1, int(tries))):
            raw = self.robot.get_joint_positions(self.arm)
            if raw and self._joints_look_valid(raw):
                joints = [float(j) for j in raw]
                with self._live_lock:
                    self.live_joints = list(joints)
                return joints
            time.sleep(pause_s)
        return None

    def capture_home_state(self, log=print):
        """抓取开始前记录关节+位姿；读失败时用 default_home_joints。"""
        joints = self.read_joints_fresh(tries=5)
        if joints is None:
            joints = self._fallback_joints()
            log(
                "警告：未能读到有效原位关节，使用默认原位 "
                + ", ".join(f"{math.degrees(j):.1f}°" for j in joints)
            )

        home_pose = None
        pose = self.get_pose()
        if pose is not None:
            pos, eul = pose
            home_pose = (
                LbotPosition(float(pos.x), float(pos.y), float(pos.z)),
                LbotEuler(float(eul.x), float(eul.y), float(eul.z)),
            )
        log(
            "记录原位关节: "
            + ", ".join(f"{math.degrees(j):.1f}°" for j in joints)
        )
        return joints, home_pose

    def home_pose_from_joints(self, joints):
        """面板/目标关节 → 笛卡尔原位（FK）；失败返回 None。"""
        if joints is None or self.robot is None:
            return None
        try:
            fk = self.robot.compute_forward_kinematics(self.arm, list(joints))
        except Exception:
            fk = None
        if fk is None:
            return None
        pos, eul = fk
        return (
            LbotPosition(float(pos.x), float(pos.y), float(pos.z)),
            LbotEuler(float(eul.x), float(eul.y), float(eul.z)),
        )

    def _joints_near(self, target, tol_rad=0.07):
        cur = self.read_joints_fresh(tries=2, pause_s=0.02)
        if cur is None or target is None:
            return False
        err = max(
            abs(float(a) - float(b))
            for a, b in zip(cur[:7], target[:7])
        )
        return err <= float(tol_rad)

    def restore_home(
        self,
        home_joints,
        home_pose=None,
        log=print,
        carry_safe=False,
        prepend_world_delta=None,
        skip_settle=False,
    ):
        """
        回到抓取前/面板关节。
        carry_safe=True（持物回程）：禁止直接关节 PTP（会弧线扎桌）。
        顺序：提离(可选) → 回程高度 → 原位正上方 → 直线降到原位 → 关节精对齐。
        prepend_world_delta：⑨ 空间系提离量，并入连续回程（与 ⑩ 一次发令）。
        """
        if not skip_settle:
            time.sleep(float(self.eye.get("align_settle_s", 0.12)))
        j_speed = float(self.eye.get("home_joint_speed", self.speed))
        j_accel = float(self.eye.get("home_joint_accel", self.accel))

        if carry_safe:
            if home_pose is None and home_joints is not None:
                home_pose = self.home_pose_from_joints(home_joints)
                if home_pose is not None:
                    log("⑩ 原位笛卡尔由面板关节 FK 得到")
                else:
                    log("⑩ 警告：原位 FK 失败，将尽量先抬高再关节回程")
            pose = self.get_pose()
            above_m = float(self.eye.get("pinch_return_above_home_m", 0.10))
            # 允许 1cm；旧 clip 下限 3cm 会把左臂「抬很高」
            above_m = float(np.clip(above_m, 0.005, 0.25))
            step_m = float(self.eye.get("pinch_align_step_m", 0.045))
            use_chain = bool(self.eye.get("pinch_return_continuous", True))

            if home_pose is not None and pose is not None:
                pos, eul = pose
                hp, he = home_pose
                cur = np.array(
                    [float(pos.x), float(pos.y), float(pos.z)], dtype=np.float64
                )
                home_p = np.array(
                    [float(hp.x), float(hp.y), float(hp.z)], dtype=np.float64
                )
                # 平行面：取较高侧 + 微小抬升（默认 1cm），禁止目标上方再叠高
                cruise_z = max(float(cur[2]), float(home_p[2]) + above_m)
                rpy_keep = (float(eul.x), float(eul.y), float(eul.z))
                rpy_home = (float(he.x), float(he.y), float(he.z))

                if use_chain:
                    keyframes = []
                    lift = prepend_world_delta
                    if lift is not None:
                        lift = np.asarray(lift, dtype=np.float64).reshape(3)
                        if float(np.linalg.norm(lift)) > 1e-4:
                            cur = cur + lift
                            keyframes.append((cur.copy(), rpy_keep))
                            log(
                                f"⑨ 提离 +{lift[2]*1000:.0f}mm上 "
                                f"{-lift[1]*1000:.0f}mm右 {lift[0]*1000:.0f}mm前"
                                f"（并入连续回程）"
                            )
                    if cruise_z - cur[2] > 0.005:
                        cur = np.array(
                            [cur[0], cur[1], cruise_z], dtype=np.float64
                        )
                        keyframes.append((cur.copy(), rpy_keep))
                    # 腕差大时先在平行面转成原位腕，再横移（避免斜腕直线卡死）
                    geo_pre = rotation_geodesic_deg(rpy_keep, rpy_home)
                    if geo_pre > 28.0:
                        keyframes.append((cur.copy(), rpy_home))
                        log(
                            f"⑩ 平行面先转腕 {geo_pre:.0f}°→原位腕，再横移"
                        )
                        rpy_lat = rpy_home
                    else:
                        rpy_lat = rpy_keep
                    above = np.array(
                        [home_p[0], home_p[1], cruise_z], dtype=np.float64
                    )
                    if float(np.linalg.norm(above - cur)) > 0.008:
                        keyframes.append((above.copy(), rpy_lat if geo_pre <= 28.0 else rpy_home))
                        cur = above.copy()
                    if abs(cruise_z - float(home_p[2])) > 0.005:
                        keyframes.append((home_p.copy(), rpy_home))
                    if keyframes:
                        log(
                            f"⑩ 连续回程 {len(keyframes)} 关键帧 "
                            f"（平行面+{above_m*1000:.0f}mm → 下降，中间不刹停）"
                        )
                        geo_oneshot = float(
                            self.eye.get("pinch_return_geo_oneshot_deg", 28.0)
                        )
                        self.move_pose_chain_continuous(
                            keyframes,
                            speed=j_speed,
                            accel=j_accel,
                            log=log,
                            max_step_m=step_m,
                            geo_oneshot_deg=geo_oneshot,
                            log_prefix="⑩",
                        )
                else:
                    if prepend_world_delta is not None:
                        lift = np.asarray(
                            prepend_world_delta, dtype=np.float64
                        ).reshape(3)
                        if float(np.linalg.norm(lift)) > 1e-4:
                            cur = cur + lift
                            self.move_to_pose_stepped(
                                cur, rpy_keep,
                                speed=j_speed, accel=j_accel, log=log,
                                max_step_m=step_m, log_prefix="⑨",
                            )
                    if cruise_z - cur[2] > 0.005:
                        log(
                            f"⑩ 先竖直到平行面 z={cruise_z:.3f} "
                            f"（+{(cruise_z - cur[2]) * 1000:.0f}mm）"
                        )
                        self.move_to_pose_stepped(
                            np.array([cur[0], cur[1], cruise_z], dtype=np.float64),
                            rpy_keep,
                            speed=j_speed, accel=j_accel, log=log,
                            max_step_m=step_m, log_prefix="⑩抬高",
                        )
                    above = np.array(
                        [home_p[0], home_p[1], cruise_z], dtype=np.float64
                    )
                    pose2 = self.get_pose()
                    if pose2 is not None:
                        p2, _ = pose2
                        cur2 = np.array(
                            [float(p2.x), float(p2.y), float(p2.z)],
                            dtype=np.float64,
                        )
                    else:
                        cur2 = np.array(
                            [cur[0], cur[1], cruise_z], dtype=np.float64
                        )
                    if float(np.linalg.norm(above - cur2)) > 0.008:
                        log(
                            f"⑩ 平行面横移到目标上方 "
                            f"xyz=[{above[0]:.3f},{above[1]:.3f},{above[2]:.3f}]"
                        )
                        self.move_to_pose_stepped(
                            above, rpy_home,
                            speed=j_speed, accel=j_accel, log=log,
                            max_step_m=step_m, log_prefix="⑩上方",
                        )
                    if abs(cruise_z - float(home_p[2])) > 0.005:
                        log(
                            f"⑩ 直线降到原位 z={home_p[2]:.3f} "
                            f"（Δz={(cruise_z - home_p[2]) * 1000:.0f}mm）"
                        )
                        self.move_to_pose_stepped(
                            home_p, rpy_home,
                            speed=j_speed * 0.85, accel=j_accel * 0.85, log=log,
                            max_step_m=min(step_m, 0.03), log_prefix="⑩下降",
                        )
            elif pose is not None:
                pos, eul = pose
                z_up = float(pos.z) + above_m
                log(
                    f"⑩ 无原位位姿：先抬 {above_m * 1000:.0f}mm 再关节 "
                    f"（z→{z_up:.3f}）"
                )
                self.move_to_pose_stepped(
                    np.array([pos.x, pos.y, z_up], dtype=np.float64),
                    (float(eul.x), float(eul.y), float(eul.z)),
                    speed=j_speed, accel=j_accel, log=log,
                    max_step_m=step_m, log_prefix="⑩抬高",
                )
            else:
                log("⑩ 警告：无当前位姿，直接关节回程（可能蹭桌）")

        skip_joint_tol = float(
            self.eye.get("pinch_return_joint_skip_rad", 0.07)
        )
        if carry_safe and self._joints_near(home_joints, skip_joint_tol):
            # 笛卡尔仍差很多时不能跳过（慢速直线卡死后易误判「关节已近」）
            pose_now = self.get_pose()
            pose_far = False
            if home_pose is not None and pose_now is not None:
                pn, _ = pose_now
                hp, _ = home_pose
                pose_far = float(np.linalg.norm(
                    np.array([pn.x, pn.y, pn.z]) - np.array([hp.x, hp.y, hp.z])
                )) > 0.020
            if not pose_far:
                log("⑩ 关节已接近原位，跳过精贴合")
                self.refresh_live_state()
                return True
            log("⑩ 关节看似近但笛卡尔仍远，继续关节贴合")

        ok = self.execute_joint_motion(
            home_joints, speed=j_speed, accel=j_accel, block=True
        )
        self.refresh_live_state()
        if ok:
            return True
        if home_pose is None:
            log("回原点失败")
            return False
        pos, eul = home_pose
        log("关节回原失败，改笛卡尔回原")
        ok2 = self.execute_pose_motion(
            (pos.x, pos.y, pos.z),
            (eul.x, eul.y, eul.z),
            speed=j_speed,
            accel=j_accel,
            block=True,
            linear=False,
        )
        self.refresh_live_state()
        return ok2

    def get_pose(self):
        if not self.robot:
            return None
        pose = self.robot.get_cartesian_pose(self.arm)
        if pose is not None:
            with self._live_lock:
                self.live_pose = pose
        return pose

    def is_ready(self, u, v, depth_mm):
        return is_grasp_aligned(u, v, depth_mm, self.intrinsics, self.eye)

    def update_stability(self, u, v, depth_mm):
        if not self.is_ready(u, v, depth_mm):
            self.stable_count = 0
            return False
        self.stable_count += 1
        return self.stable_count >= self.stable_required

    def reset_stability(self):
        self.stable_count = 0

    def move_to_position(
        self,
        position,
        euler,
        linear=False,
        block=True,
        force=False,
        speed=None,
        accel=None,
        use_follow=False,
    ):
        if not self.robot:
            return False
        if self.dry_run and not force:
            return True
        speed = float(self.speed if speed is None else speed)
        accel = float(self.accel if accel is None else accel)
        if use_follow and not block:
            return self.robot.pose_follow(self.arm, position, euler)
        for val in (
            position.x, position.y, position.z, euler.x, euler.y, euler.z, speed, accel
        ):
            if not np.isfinite(val):
                print("[move] 目标非有限，拒绝发送", flush=True)
                return False
        if linear:
            # 禁止 block=True 卡死在控制器里（极端腕姿时直线规划可能永不返回）
            dist_m = 0.0
            pose0 = self.get_pose()
            if pose0 is not None:
                p0, _ = pose0
                dist_m = math.sqrt(
                    (p0.x - position.x) ** 2
                    + (p0.y - position.y) ** 2
                    + (p0.z - position.z) ** 2
                )
            ok = self.robot.linear_move_to_pose(
                self.arm, position, euler, speed, accel, False
            )
            if not ok:
                print(
                    f"[move] 直线失败: {self.robot.get_last_error()}",
                    flush=True,
                )
                return False
            if not block:
                self.refresh_live_state()
                return True
            wait_s = self._estimate_linear_wait_s(dist_m, speed)
            tol, tol_loose = self._linear_arrive_tol_m(dist_m)
            reached = self._wait_pose_near(
                position, tol_m=tol, timeout_s=wait_s, poll_s=0.05,
                loose_tol_m=tol_loose,
            )
            self.refresh_live_state()
            if self._motion_abort:
                print("[move] 直线等待已中止", flush=True)
                self._hold_current_motion("[move]")
                return False
            if not reached:
                err = self._pose_dist_m(position)
                print(
                    f"[move] 直线超时未到位"
                    f"{'' if err is None else f' 距目标 {err*1000:.0f}mm'} "
                    f"（最长 {wait_s:.0f}s），打断",
                    flush=True,
                )
                self._hold_current_motion("[move]")
                return False
            return True
        joints = self.read_joints_fresh(tries=3)
        if joints is None:
            print("[move] IK 种子关节读失败，拒绝", flush=True)
            return False
        max_jump = float(self.eye.get("ik_max_jump_rad", 1.8))
        ik = self._ik_at_pose(
            (float(position.x), float(position.y), float(position.z)),
            (float(euler.x), float(euler.y), float(euler.z)),
            joints,
            max_jump_rad=max_jump,
            log_prefix="[move]",
        )
        if ik is None:
            print("[move] IK 失败或越软限位/跳变过大，拒绝", flush=True)
            return False
        # 关节空间也走非阻塞+轮询，避免控制器 block 卡死
        return self.execute_joint_motion(
            ik, speed=speed, accel=accel, block=block,
        )

    def jog_xyz(self, dx, dy, dz, in_tool=False):
        """点动平移：默认工作系；in_tool=True 时沿工具系。force 以支持 dry-run 下调参。"""
        pose = self.get_pose()
        if pose is None:
            return False
        position, euler = pose
        delta = np.array([dx, dy, dz], dtype=np.float64)
        if in_tool:
            R = euler_rpy_to_matrix(euler.x, euler.y, euler.z)
            delta = R @ delta
        return self.move_to_position(
            LbotPosition(position.x + delta[0], position.y + delta[1], position.z + delta[2]),
            euler, linear=True, block=True, force=True,
        )

    def jog_rpy(self, d_roll, d_pitch, d_yaw):
        """点动姿态：在当前工作系下增量 roll/pitch/yaw（弧度）。"""
        pose = self.get_pose()
        if pose is None:
            return False
        position, euler = pose
        new_euler = LbotEuler(
            euler.x + d_roll, euler.y + d_pitch, euler.z + d_yaw
        )
        return self.move_to_position(
            position, new_euler, linear=False, block=True, force=True,
        )

    def execute_joint_motion(self, joints, speed=None, accel=None, block=True, timeout_s=None,
                             tol_rad=None):
        """关节空间运动；阻塞时用非阻塞发送+轮询，便于 MuJoCo 跟随。"""
        if not self.robot:
            return False
        # 不在此 clear_motion_abort：否则「停止」后多段路径会继续走
        if self._motion_abort:
            return False
        speed = self.speed if speed is None else float(speed)
        accel = self.accel if accel is None else float(accel)
        speed, accel = self._cap_arm_speed_accel(speed, accel)
        target = [float(j) for j in joints]
        if tol_rad is None:
            if self._is_left_arm():
                tol_rad = float(
                    (self.config.get("robot") or {}).get(
                        "left_joint_arrive_tol_rad", 0.008
                    )
                )
            else:
                tol_rad = 0.06
        tol_rad = float(tol_rad)
        seed = self.read_joints_fresh(tries=2, pause_s=0.02)
        if timeout_s is None:
            # 短超时：按跨度估；上限压到十几秒，禁止再等 30–55s
            if seed is not None:
                span = max(abs(float(a) - float(b)) for a, b in zip(target, seed))
            else:
                span = 1.0
            if self._is_left_arm():
                timeout_s = float(np.clip(span / max(speed, 0.04) + 2.0, 3.0, 16.0))
            else:
                timeout_s = float(np.clip(span / max(speed, 0.05) + 2.5, 3.0, 18.0))
        ok = self.robot.move_to_joint_target(
            self.arm, list(target), speed, accel, False
        )
        if not ok:
            move_print(
                self.eye,
                f"[move] 关节运动: FAILED {self.robot.get_last_error()}",
                important=True,
            )
            return False
        move_print(self.eye, "[move] 关节运动: SUCCESS")
        if not block:
            self.refresh_live_state()
            return True
        if seed is not None:
            span_deg = math.degrees(
                max(abs(float(a) - float(b)) for a, b in zip(target, seed))
            )
            move_print(
                self.eye,
                f"[move] 等待到位（跨度约 {span_deg:.0f}°，最长 {timeout_s:.0f}s，"
                f"speed={speed:.2f}，容差{math.degrees(tol_rad):.1f}°）…",
            )
        reached = self._wait_joints_near(
            target, tol=float(tol_rad), timeout_s=float(timeout_s), poll_s=0.10,
        )
        self.refresh_live_state()
        if self._motion_abort:
            move_print(self.eye, "[move] 关节等待已中止", important=True)
            return False
        if not reached:
            cur = self.read_joints_fresh(tries=2, pause_s=0.05)
            err_deg = None
            if cur is not None:
                err_deg = math.degrees(
                    max(abs(float(a) - float(b)) for a, b in zip(cur, target))
                )
            move_print(
                self.eye,
                f"[move] 关节跟踪超时"
                + (f"（仍差 {err_deg:.1f}°）" if err_deg is not None else "")
                + " — 臂可能还在慢速走，点「停止/解锁」后可「读取当前」",
                important=True,
            )
            return False
        move_print(self.eye, "[move] 关节已到位")
        return True

    def execute_pose_motion(
        self, xyz, rpy_rad, speed=None, accel=None, block=True, linear=False,
        arrive_tol_m=None,
    ):
        """笛卡尔运动（同 demo_move 位姿控制）。"""
        if not self.robot:
            return False
        if self._motion_abort:
            return False
        speed = self.speed if speed is None else float(speed)
        accel = self.accel if accel is None else float(accel)
        speed, accel = self._cap_arm_speed_accel(speed, accel)
        position = LbotPosition(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        euler = LbotEuler(float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2]))
        dist0 = self._pose_dist_m(position)
        if linear:
            ok = self.robot.linear_move_to_pose(
                self.arm, position, euler, speed, accel, False if block else False
            )
        else:
            ok = self.robot.move_to_pose_target(
                self.arm, position, euler, speed, accel, False if block else False
            )
        if not ok:
            move_print(
                self.eye,
                f"[move] 位姿运动({'直线' if linear else '点到点'}): FAILED "
                f"{self.robot.get_last_error()}",
                important=True,
            )
            return False
        move_print(
            self.eye,
            f"[move] 位姿运动({'直线' if linear else '点到点'}): SUCCESS",
        )
        if block:
            if arrive_tol_m is None:
                if int(self.arm) == int(LbotArm.LEFT_ARM):
                    arrive_tol_m = float(
                        (self.config.get("robot") or {}).get(
                            "left_arrive_tol_m", 0.0004
                        )
                    )
                else:
                    arrive_tol_m = 0.012
            if int(self.arm) == int(LbotArm.LEFT_ARM):
                # 严容差（≤1mm）时 loose 绝不能抬到 1.5mm，否则 0.4mm 永远等不到
                if float(arrive_tol_m) <= 0.001:
                    loose = max(
                        float(arrive_tol_m) * 1.25,
                        float(arrive_tol_m) + 0.00015,
                    )
                else:
                    loose = max(
                        float(arrive_tol_m) * 2.0,
                        float(arrive_tol_m) + 0.0005,
                        0.0015,
                    )
                    loose = float(min(loose, 0.006))
            else:
                loose = max(float(arrive_tol_m) * 2.0, float(arrive_tol_m) + 0.004)
            wait_s = self._estimate_linear_wait_s(
                float(dist0 if dist0 is not None else 0.05), speed,
            )
            # 终姿亚毫米：多给一点沉降时间
            if (
                int(self.arm) == int(LbotArm.LEFT_ARM)
                and float(arrive_tol_m) <= 0.001
            ):
                wait_s = float(max(wait_s, 3.5))
            reached = self._wait_pose_near(
                position, tol_m=float(arrive_tol_m), timeout_s=wait_s, poll_s=0.06,
                loose_tol_m=loose,
            )
            if self._motion_abort:
                move_print(self.eye, "[move] 位姿等待已中止", important=True)
                self._hold_current_motion("[move]")
                self.refresh_live_state()
                return False
            if not reached:
                err_left = self._pose_dist_m(position)
                # 左臂近距超时勿 hold：hold 会打断残余跟踪，下一轮精贴更差
                if (
                    int(self.arm) == int(LbotArm.LEFT_ARM)
                    and err_left is not None
                    and float(err_left) <= 0.003
                ):
                    move_print(
                        self.eye,
                        f"[move] 位姿近距未严到位（{err_left*1000:.1f}mm），"
                        f"不打断，由终姿一次修正",
                        important=True,
                    )
                    self.refresh_live_state()
                    return False
                move_print(self.eye, "[move] 位姿跟踪超时", important=True)
                self._hold_current_motion("[move]")
                self.refresh_live_state()
                return False
        self.refresh_live_state()
        return True

    def _is_left_arm(self):
        return int(self.arm) == int(LbotArm.LEFT_ARM)

    def _left_motion_cfg(self):
        robot = self.config.get("robot") or {}
        return {
            "above_m": float(robot.get("left_return_above_m", 0.01)),
            "clear_m": float(robot.get("left_pose_parallel_clearance_m", 0.01)),
            "speed_max": float(robot.get("left_speed_max", 0.05)),
            "accel_max": float(robot.get("left_accel_max", 0.12)),
            "arrive_tol_m": float(robot.get("left_arrive_tol_m", 0.0004)),
            "joint_arrive_tol_rad": float(
                robot.get("left_joint_arrive_tol_rad", 0.008)
            ),
        }

    def _left_pose_err_m(self, xyz):
        pose = self.get_pose()
        if pose is None:
            return None
        p, _ = pose
        tgt = np.asarray(xyz, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(
            np.array([p.x, p.y, p.z], dtype=np.float64) - tgt
        ))

    def _left_final_speed(self, dist_m, speed_cap=None):
        """
        左臂终姿一次直线的速度（仅左臂）。
        距离越短越慢，避免多指令叠发造成顿挫。
        """
        robot = self.config.get("robot") or {}
        v_cruise = float(robot.get("left_near_speed", 0.018))
        v_slow = float(robot.get("left_finetune_speed", 0.008))
        d = float(max(0.0, dist_m))
        if d >= 0.025:
            v = max(v_cruise, 0.020)
        elif d >= 0.010:
            v = v_cruise
        elif d >= 0.003:
            v = 0.5 * (v_cruise + v_slow)
        else:
            v = v_slow
        if speed_cap is not None:
            v = min(v, float(speed_cap))
        return float(np.clip(v, 0.004, 0.035))

    def _left_finetune_pose(self, xyz, rpy_rad, speed, accel, arrive_tol, log=print):
        """
        左臂终姿：最多「一次主直线 + 可选一次微修正」。

        故意不做多轮 sleep/hold 刷指令——那是顿挫的根因。
        右臂 / 抓放任务不走此函数。
        """
        arrive_tol = float(arrive_tol)
        tgt = np.array(
            [float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=np.float64
        )
        rpy = (float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2]))

        def _measure():
            pose = self.get_pose()
            if pose is None:
                return None, None
            p, eul = pose
            err = float(np.linalg.norm(
                np.array([p.x, p.y, p.z], dtype=np.float64) - tgt
            ))
            geo = rotation_geodesic_deg(
                (float(eul.x), float(eul.y), float(eul.z)), rpy,
            )
            return err, geo

        err0, geo0 = _measure()
        if err0 is None:
            return False
        if err0 <= arrive_tol and (geo0 is None or geo0 < 3.0):
            log(
                f"左臂终姿已到位 "
                f"{err0*1000:.2f}mm≤{arrive_tol*1000:.2f}mm"
            )
            return True
        if self._motion_abort:
            return False

        v = self._left_final_speed(err0, speed_cap=speed)
        a = float(np.clip(max(0.025, v * 3.5), 0.025, min(float(accel), 0.08)))
        log(
            f"左臂终姿一次直线 "
            f"{err0*1000:.1f}mm→≤{arrive_tol*1000:.2f}mm v={v:.3f}"
        )
        self.execute_pose_motion(
            tgt.tolist(), rpy,
            speed=v, accel=a,
            block=True, linear=True,
            arrive_tol_m=arrive_tol,
        )
        if self._motion_abort:
            return False

        err1, geo1 = _measure()
        if err1 is None:
            return False
        if err1 <= arrive_tol and (geo1 is None or geo1 < 3.0):
            log(f"左臂终姿到位 {err1*1000:.2f}mm")
            return True

        # 仅当卡在严容差外一点点时，再发一条更慢的修正（仍禁止循环刷）
        if arrive_tol < err1 <= 0.0020:
            v2 = float(np.clip(
                min(v, float((self.config.get("robot") or {}).get(
                    "left_finetune_speed", 0.008
                ))),
                0.004, 0.012,
            ))
            a2 = float(np.clip(v2 * 3.0, 0.02, 0.05))
            log(
                f"左臂终姿微修正 "
                f"{err1*1000:.2f}mm→≤{arrive_tol*1000:.2f}mm v={v2:.3f}"
            )
            self.execute_pose_motion(
                tgt.tolist(), rpy,
                speed=v2, accel=a2,
                block=True, linear=True,
                arrive_tol_m=arrive_tol,
            )
            err1, geo1 = _measure()
            if err1 is None:
                return False
            if err1 <= arrive_tol and (geo1 is None or geo1 < 3.0):
                log(f"左臂终姿到位 {err1*1000:.2f}mm")
                return True

        log(
            f"左臂终姿未达严容差："
            f"{err1*1000:.2f}mm（目标≤{arrive_tol*1000:.2f}mm）"
        )
        return False

    def _left_lift_traverse_descend(
        self,
        xyz,
        rpy_rad,
        speed=None,
        accel=None,
        log=print,
        lift_m=None,
        arrive_tol_m=None,
        final_joints=None,
        precise=True,
    ):
        """
        左臂安全路径（软限位不改）：
        - 短距标定挪动：慢直线 + 精贴（不抬降，少触发钳位 IK）
        - 中大行程：抬 → 高位到位（IK 无解/被拒钳位则改笛卡尔）→ 降 → 精贴
        """
        if not self.robot:
            return False
        del precise
        speed, accel = self._cap_arm_speed_accel(speed, accel)
        robot_cfg = self.config.get("robot") or {}
        left_cfg = self._left_motion_cfg()
        if lift_m is None:
            lift_m = float(robot_cfg.get("left_return_above_m", 0.01))
        lift_m = float(np.clip(lift_m, 0.005, 0.02))
        if arrive_tol_m is None:
            arrive_tol_m = float(left_cfg["arrive_tol_m"])
        arrive_tol_m = float(arrive_tol_m)
        j_tol = float(left_cfg["joint_arrive_tol_rad"])
        tgt = np.array(
            [float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=np.float64
        )
        rpy = (float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2]))
        pose0 = self.get_pose()
        if pose0 is None:
            log("左臂抬-横-降失败：无当前位姿")
            return False
        p0, e0 = pose0
        cur = np.array([float(p0.x), float(p0.y), float(p0.z)], dtype=np.float64)
        rpy_keep = (float(e0.x), float(e0.y), float(e0.z))
        xy = float(np.hypot(tgt[0] - cur[0], tgt[1] - cur[1]))
        dz = float(abs(tgt[2] - cur[2]))
        geo0 = rotation_geodesic_deg(rpy_keep, rpy)
        nudge_xy = float(robot_cfg.get("left_nudge_xy_m", 0.028))
        nudge_dz = float(robot_cfg.get("left_nudge_dz_m", 0.018))
        nudge_geo = float(robot_cfg.get("left_nudge_geo_deg", 12.0))

        # 已贴目标 / 短距示教：不抬降，一次终姿直线（禁止 mid+多轮精贴）
        if (
            (xy < 0.012 and dz < 0.010 and geo0 < 8.0)
            or (xy <= nudge_xy and dz <= nudge_dz and geo0 <= nudge_geo)
        ):
            tag = "已近" if xy < 0.012 and dz < 0.010 else "短距"
            log(
                f"左臂{tag} XY={xy*1000:.0f}mm ΔZ={dz*1000:.0f}mm "
                f"姿态{geo0:.0f}°：终姿一次直线（不抬降）"
            )
            return self._left_finetune_pose(
                tgt, rpy, speed, accel, arrive_tol_m, log=log,
            )

        cruise_z = max(float(cur[2]) + lift_m, float(tgt[2]) + lift_m)
        above = np.array([tgt[0], tgt[1], cruise_z], dtype=np.float64)
        log(
            f"左臂抬→横→降：+{lift_m*1000:.0f}mm → 上方z={cruise_z:.3f} "
            f"→ 目标z={tgt[2]:.3f} speed≤{speed:.3f} "
            f"tol={arrive_tol_m*1000:.2f}mm"
        )

        if cruise_z - float(cur[2]) > 0.003:
            ok_lift = self.execute_pose_motion(
                (float(cur[0]), float(cur[1]), cruise_z),
                rpy_keep,
                speed=speed, accel=accel, block=True, linear=True,
                arrive_tol_m=max(0.004, min(0.008, arrive_tol_m * 10)),
            )
            if not ok_lift:
                try:
                    ik_lift = self.ik_to_pose(
                        [float(cur[0]), float(cur[1]), cruise_z],
                        list(rpy_keep), max_jump_rad=1.2,
                    )
                except Exception:
                    ik_lift = None
                if ik_lift is None:
                    log("左臂抬高：IK 不可用（可能软限位），继续尝试高位笛卡尔")
                elif not self.execute_joint_motion(
                    ik_lift, speed=speed, accel=accel, block=True, tol_rad=j_tol,
                ):
                    log("左臂抬高失败")
                    return False

        geo = rotation_geodesic_deg(rpy_keep, rpy)
        rpy_via = rpy if geo > 8.0 else rpy_keep
        moved_above = False
        try:
            ik_above = self.ik_to_pose(
                above.tolist(), list(rpy_via), max_jump_rad=3.5,
            )
        except Exception as exc:
            log(f"左臂上方 IK 异常: {exc}")
            ik_above = None
        if ik_above is not None:
            if self.execute_joint_motion(
                ik_above, speed=speed, accel=accel, block=True, tol_rad=j_tol,
            ):
                moved_above = True
            else:
                pose_chk = self.get_pose()
                if pose_chk is not None:
                    pc, _ = pose_chk
                    xy_left = float(np.hypot(above[0] - pc.x, above[1] - pc.y))
                    z_left = float(abs(above[2] - pc.z))
                    if xy_left < 0.03 and z_left < 0.035:
                        log(
                            f"左臂高位未严到位，已在上方附近"
                            f"（XY差{xy_left*1000:.0f}mm），继续下降"
                        )
                        moved_above = True
        if not moved_above:
            log("左臂高位：改笛卡尔横移（IK 无解或拒钳位）")
            ok_ab = self.execute_pose_motion(
                above.tolist(), rpy_via,
                speed=speed, accel=accel, block=True, linear=False,
                arrive_tol_m=max(0.008, arrive_tol_m * 15),
            )
            if not ok_ab:
                ok_ab = self.execute_pose_motion(
                    above.tolist(), rpy,
                    speed=max(0.025, speed * 0.6), accel=accel,
                    block=True, linear=True,
                    arrive_tol_m=max(0.008, arrive_tol_m * 15),
                )
            if not ok_ab:
                pose_chk = self.get_pose()
                if pose_chk is None:
                    log("左臂高位横移失败")
                    return False
                pc, _ = pose_chk
                xy_left = float(np.hypot(above[0] - pc.x, above[1] - pc.y))
                if xy_left > 0.04:
                    log("左臂高位横移失败")
                    return False
                log(
                    f"左臂高位大致到达（XY差{xy_left*1000:.0f}mm），继续下降"
                )

        if geo > 8.0:
            try:
                ik_above2 = self.ik_to_pose(
                    above.tolist(), list(rpy), max_jump_rad=1.5,
                )
            except Exception:
                ik_above2 = None
            if ik_above2 is not None:
                self.execute_joint_motion(
                    ik_above2, speed=speed, accel=accel, block=True, tol_rad=j_tol,
                )
            else:
                self.execute_pose_motion(
                    above.tolist(), rpy,
                    speed=max(0.02, speed * 0.5), accel=accel,
                    block=True, linear=True,
                    arrive_tol_m=max(0.006, arrive_tol_m * 12),
                )

        # 高位到终姿：一次终姿直线，不再「下降 mid_tol + 多轮精贴」
        log(f"左臂下降→目标（一次终姿直线，抬高 {lift_m*1000:.0f}mm）")
        if final_joints is not None:
            # 仅作兜底：先试直线；失败才用示教关节（仍只各一次）
            ok = self._left_finetune_pose(
                tgt, rpy, speed, accel, arrive_tol_m, log=log,
            )
            if ok:
                return True
            try:
                ik_final = self.ik_to_pose(
                    tgt.tolist(), list(rpy), max_jump_rad=1.2,
                )
            except Exception:
                ik_final = None
            if ik_final is not None:
                log("左臂终姿直线未到位，改终姿关节一次…")
                self.execute_joint_motion(
                    ik_final, speed=speed, accel=accel, block=True, tol_rad=j_tol,
                )
            else:
                log("左臂终姿直线未到位，试示教终姿关节一次…")
                self.execute_joint_motion(
                    final_joints, speed=speed, accel=accel, block=True, tol_rad=j_tol,
                )
            err = self._left_pose_err_m(tgt)
            ok = err is not None and err <= arrive_tol_m
            log(
                f"左臂终姿关节后："
                f"{'到位' if ok else '未达'} "
                f"{0.0 if err is None else err*1000:.2f}mm"
            )
            return bool(ok)
        return self._left_finetune_pose(
            tgt, rpy, speed, accel, arrive_tol_m, log=log,
        )

    def _cap_arm_speed_accel(self, speed=None, accel=None):
        """左臂强制限速，减小拧螺丝漂移。"""
        sp = self.speed if speed is None else float(speed)
        ac = self.accel if accel is None else float(accel)
        if self._is_left_arm():
            lim = self._left_motion_cfg()
            sp = min(sp, float(lim["speed_max"]))
            ac = min(ac, float(lim["accel_max"]))
        return float(sp), float(ac)

    def execute_pose_motion_safe(
        self,
        xyz,
        rpy_rad,
        speed=None,
        accel=None,
        log=print,
        force_direct=False,
        above_m=None,
        precise=True,
        wait_each=False,
    ):
        """
        防撞桌笛卡尔：平行面横移（仅比原高度/目标高 1cm）→ 再落到目标。
        左臂全程严容差 + 精贴合（precise 保留兼容，默认 True）。
        wait_each：安全链每段都等到位（手眼采样用，避免「还在飞就记」）。
        """
        if not self.robot:
            return False
        if self._motion_abort:
            return False
        del precise
        speed, accel = self._cap_arm_speed_accel(speed, accel)
        target = np.array(
            [float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=np.float64
        )
        rpy_tgt = (
            float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2]),
        )
        is_left = self._is_left_arm()
        robot_cfg = self.config.get("robot") or {}
        left_cfg = self._left_motion_cfg() if is_left else None
        if is_left:
            use_safe = bool(robot_cfg.get("left_pose_move_safe_path", True))
            default_clear = float(left_cfg["clear_m"])
            arrive_tol = float(left_cfg["arrive_tol_m"])
        else:
            use_safe = bool(self.eye.get("pose_move_safe_path", True))
            default_clear = float(self.eye.get("pose_return_above_m", 0.01))
            arrive_tol = 0.010
        if force_direct:
            use_safe = False
        pose = self.get_pose()
        if pose is None:
            log("安全位姿运动失败：读不到当前位姿")
            return False
        pos, eul = pose
        cur = np.array(
            [float(pos.x), float(pos.y), float(pos.z)], dtype=np.float64
        )
        rpy_keep = (float(eul.x), float(eul.y), float(eul.z))
        travel = float(np.linalg.norm(target - cur))
        xy_travel = float(np.hypot(target[0] - cur[0], target[1] - cur[1]))
        geo = rotation_geodesic_deg(rpy_keep, rpy_tgt)

        # 近零位大位移：笛卡尔抬-横-降会秒失败，先 IK 关节直达再精贴
        cur_j = self.read_joints_fresh(tries=2, pause_s=0.02)
        near_folded = (
            cur_j is not None
            and max(abs(float(j)) for j in cur_j[:7]) < math.radians(10.0)
        )
        if near_folded and travel >= 0.05:
            log(
                f"近零位→目标 |Δ|={travel*1000:.0f}mm：关节直达（跳过笛卡尔安全链）"
            )
            try:
                ik = self.ik_to_pose(
                    target.tolist(), list(rpy_tgt), max_jump_rad=6.0,
                )
            except Exception as exc:
                log(f"近零位 IK 异常: {exc}")
                ik = None
            if ik is None:
                log("近零位 IK 失败，回退安全路径…")
            else:
                j_tol = 0.06
                if is_left and left_cfg is not None:
                    j_tol = float(left_cfg.get("joint_arrive_tol_rad", 0.008))
                ok_j = self.execute_joint_motion(
                    ik, speed=speed, accel=accel, block=True, tol_rad=j_tol,
                )
                if ok_j and is_left:
                    return self._left_finetune_pose(
                        target, rpy_tgt, speed, accel, arrive_tol, log=log,
                    )
                return bool(ok_j)
        if travel < 0.012 and abs(float(target[2] - cur[2])) < 0.008 and geo < 8.0:
            if is_left:
                log(
                    f"左臂面板近距 |Δ|={travel*1000:.0f}mm：终姿一次直线"
                )
                return self._left_finetune_pose(
                    target, rpy_tgt, speed, accel, arrive_tol, log=log,
                )
            return self.execute_pose_motion(
                target.tolist(), rpy_tgt, speed=speed, accel=accel,
                block=True, linear=True, arrive_tol_m=arrive_tol,
            )
        if not use_safe:
            if is_left:
                return self._left_finetune_pose(
                    target, rpy_tgt, speed, accel, arrive_tol, log=log,
                )
            return self.execute_pose_motion(
                target.tolist(), rpy_tgt, speed=speed, accel=accel,
                block=True, linear=False, arrive_tol_m=arrive_tol,
            )

        def _left_joint_safe():
            """左臂：抬1cm → 高位横移到目标上方 → 降1cm + 精贴合。"""
            log(
                f"左臂安全路径 |Δ|={travel*1000:.0f}mm 姿态={geo:.0f}° "
                f"speed≤{speed:.3f}"
            )
            return self._left_lift_traverse_descend(
                target, rpy_tgt,
                speed=speed, accel=accel, log=log,
                lift_m=float(left_cfg["above_m"]) if left_cfg else 0.01,
                arrive_tol_m=arrive_tol,
                precise=True,
            )

        # 拧螺丝位↔默认位等：姿态/位移一大，笛卡尔慢速直线几乎必卡
        prefer_joint = bool(robot_cfg.get("left_pose_prefer_joint", True))
        joint_travel = float(robot_cfg.get("left_pose_joint_travel_m", 0.028))
        joint_geo = float(robot_cfg.get("left_pose_joint_geo_deg", 12.0))
        if is_left and prefer_joint and (travel >= joint_travel or geo >= joint_geo):
            ok_j = _left_joint_safe()
            if ok_j:
                return True
            err_left = self._left_pose_err_m(target)
            # 未达 0.4mm 也绝不回退整段抬降；仅报告
            log(
                f"左臂路径未达严容差"
                f"{'' if err_left is None else f'（仍差 {err_left*1000:.2f}mm）'}，"
                f"不再回退笛卡尔抬降"
            )
            return False

        if is_left:
            return _left_joint_safe()

        clear = float(above_m if above_m is not None else default_clear)
        # 全部按「原高度抬 1cm」量级；禁止再抬到 3–8cm
        clear = float(np.clip(clear, 0.0, 0.02))
        if clear < 1e-6:
            clear = 0.01
        # 必须相对当前与目标都抬 clear，否则同高横移会扫到螺丝
        cruise_z = max(float(cur[2]) + clear, float(target[2]) + clear)
        step_m = float(self.eye.get("pinch_align_step_m", 0.035))
        if is_left:
            step_m = min(step_m, 0.025)

        keyframes = []
        if cruise_z - float(cur[2]) > 0.003:
            keyframes.append(
                (np.array([cur[0], cur[1], cruise_z], dtype=np.float64), rpy_keep)
            )
            cur_h = np.array([cur[0], cur[1], cruise_z], dtype=np.float64)
        else:
            cur_h = cur.copy()
            cur_h[2] = cruise_z
        if geo > 28.0:
            keyframes.append((cur_h.copy(), rpy_tgt))
            log(f"安全位姿：平行面先转腕 {geo:.0f}°")
            rpy_lat = rpy_tgt
        else:
            rpy_lat = rpy_keep if geo < 8.0 else rpy_tgt
        above_xy = np.array(
            [target[0], target[1], cruise_z], dtype=np.float64
        )
        if xy_travel > 0.006 or float(np.linalg.norm(above_xy - cur_h)) > 0.006:
            keyframes.append((above_xy.copy(), rpy_lat))
        if abs(cruise_z - float(target[2])) > 0.003:
            keyframes.append((target.copy(), rpy_tgt))
        elif geo >= 0.5:
            keyframes.append((target.copy(), rpy_tgt))

        if not keyframes:
            return self.execute_pose_motion(
                target.tolist(), rpy_tgt, speed=speed, accel=accel,
                block=True, linear=True, arrive_tol_m=arrive_tol,
            )

        log(
            f"安全位姿（平行面+{clear*1000:.0f}mm）："
            f"z={cruise_z:.3f} XY={xy_travel*1000:.0f}mm → z={target[2]:.3f}"
            f" speed≤{speed:.3f} tol={arrive_tol*1000:.1f}mm"
        )
        ok = self.move_pose_chain_continuous(
            keyframes,
            speed=speed,
            accel=accel,
            log=log,
            max_step_m=step_m,
            geo_oneshot_deg=float(
                self.eye.get("pinch_return_geo_oneshot_deg", 28.0)
            ),
            log_prefix="安全位姿",
            wait_each=bool(wait_each),
        )
        if not ok:
            log("安全位姿笛卡尔失败")
            return False
        pose2 = self.get_pose()
        if pose2 is not None:
            p2, _ = pose2
            err = float(np.linalg.norm(
                np.array([p2.x, p2.y, p2.z], dtype=np.float64) - target
            ))
            # 必须真到 tol 内；旧逻辑 *2（右臂≈20mm）会「还差一截就返回成功」
            if err <= float(arrive_tol):
                return True
            log(f"安全位姿：末段微调（还差 {err*1000:.1f}mm）")
            ok_f = self.execute_pose_motion(
                target.tolist(), rpy_tgt,
                speed=max(0.03, speed * 0.6),
                accel=max(0.08, accel * 0.6),
                block=True, linear=True, arrive_tol_m=arrive_tol,
            )
            return bool(ok_f)
        return True

    def execute_joint_path_smooth(
        self,
        waypoints,
        speed=None,
        accel=None,
        log=print,
        should_stop=None,
        log_prefix="关节路径",
        blend_deg=15.0,
    ):
        """
        多点关节路径连续执行：中间点未完全刹停就换下一目标，末点再等到位。
        用于 MJ-IK 位姿插值路径，避免「每段 block→停顿再起步」的难看停顿。
        """
        stop = should_stop or (lambda: False)
        wps = []
        for w in waypoints or []:
            if w is None:
                continue
            jj = [float(x) for x in list(w)[:7]]
            if wps:
                dmax = max(abs(a - b) for a, b in zip(jj, wps[-1]))
                if dmax < math.radians(0.8):
                    wps[-1] = jj
                    continue
            wps.append(jj)
        if not wps:
            log(f"{log_prefix}：无路点")
            return False
        blend_tol = math.radians(float(max(blend_deg, 4.0)))
        n = len(wps)
        log(
            f"{log_prefix}：平滑关节 {n} 点"
            f"（中间提前换路≤{float(blend_deg):.0f}°，末点刹停）"
        )
        for i, jtgt in enumerate(wps):
            if stop():
                self._hold_current_motion(f"[{log_prefix}]")
                return False
            is_last = i == n - 1
            if is_last:
                ok = self.execute_joint_motion(
                    jtgt, speed=speed, accel=accel, block=True,
                )
                return bool(ok)
            ok = self.execute_joint_motion(
                jtgt, speed=speed, accel=accel, block=False,
            )
            if not ok:
                log(f"{log_prefix}：第 {i + 1}/{n} 点下发失败")
                return False
            # 估超时：到本点跨度；到不了也换下一点（连续 retarget）
            seed = self.read_joints_fresh(tries=1, pause_s=0.0)
            if seed is not None:
                span = max(abs(float(a) - float(b)) for a, b in zip(jtgt, seed))
            else:
                span = 0.5
            # 中间点等太久会「一顿一顿」；超时偏短、靠 blend 提前换路
            wait_s = float(
                np.clip(span / max(float(speed or self.speed), 0.08) + 0.8, 1.2, 6.0)
            )
            self._wait_joints_near(
                jtgt, tol=blend_tol, timeout_s=wait_s, poll_s=0.04,
            )
            if stop():
                self._hold_current_motion(f"[{log_prefix}]")
                return False
        return True

    def move_to_pose_stepped(
        self,
        target_pos,
        target_rpy,
        speed=None,
        accel=None,
        log=print,
        should_stop=None,
        max_step_deg=8.0,
        max_step_m=0.025,
        log_prefix="⑥",
    ):
        """
        夹取路径默认跟模型预览一致：笛卡尔位姿插值（直线优先），
        避免「终姿一步关节 PTP」绕弧导致实机与绿线不一致。

        pinch_path_mode:
          - cart / cartesian（默认）：分步直线/插值
          - joint_oneshot：旧行为（终姿 IK 一步关节）
        """
        stop = should_stop or (lambda: False)
        pfx = str(log_prefix or "⑥")
        pose = self.get_pose()
        if pose is None:
            log(f"{pfx} 到位失败：无当前位姿")
            return False
        pos0, eul0 = pose
        p0 = np.array([pos0.x, pos0.y, pos0.z], dtype=np.float64)
        r0 = (float(eul0.x), float(eul0.y), float(eul0.z))
        p1 = np.asarray(target_pos, dtype=np.float64).reshape(3)
        r1 = (float(target_rpy[0]), float(target_rpy[1]), float(target_rpy[2]))
        dxyz = p1 - p0
        travel = float(np.linalg.norm(dxyz))
        geo = rotation_geodesic_deg(r0, r1)
        j0 = self.read_joints_fresh(tries=2, pause_s=0.02)
        j0deg = (
            [round(math.degrees(float(j)), 1) for j in j0[:7]] if j0 is not None else None
        )
        log(
            f"{pfx} |Δ|={travel*1000:.0f}mm 姿态={geo:.1f}°"
        )
        vrb(
            log, self.eye,
            f"{pfx} 运动前 当前xyz=[{p0[0]:.4f},{p0[1]:.4f},{p0[2]:.4f}] "
            f"rpy°=[{math.degrees(r0[0]):.1f},{math.degrees(r0[1]):.1f},{math.degrees(r0[2]):.1f}]",
        )
        vrb(
            log, self.eye,
            f"{pfx} 运动前 目标xyz=[{p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f}] "
            f"rpy°=[{math.degrees(r1[0]):.1f},{math.degrees(r1[1]):.1f},{math.degrees(r1[2]):.1f}] "
            f"joints°={j0deg}",
        )
        if stop():
            return False

        r_use = r1 if geo >= 0.5 else r0
        path_mode = str(self.eye.get("pinch_path_mode", "cart")).strip().lower()
        force_step = bool(self.eye.get("pinch_force_stepped", True))
        prefer_joint = bool(self.eye.get("pinch_step_prefer_joint", False))
        # 默认尽量一整段直线，避免分步每段都等到位→停顿再起步
        continuous = bool(self.eye.get("pinch_path_continuous", True))
        # 姿态过大时整段直线易「位置到了、腕没转到」→ mid 侧向歪；默认 12°
        geo_oneshot = float(self.eye.get("pinch_cart_oneshot_geo_deg", 12.0))
        # cart_mj_polish：⑥⑦ 后面还有一次 MJ 精修，这里优先一整段，
        # 勿因 ~22° 拆成多段「中间不等待」连发（上次会先卡后冲，很难看）
        defer_mj_polish = path_mode in (
            "cart_mj_polish", "mj_ik_joint", "mj_joint",
            "preview_joint", "preview_ik",
        )
        if defer_mj_polish:
            geo_oneshot = max(
                geo_oneshot,
                float(self.eye.get("pinch_cart_bulk_geo_deg", 28.0)),
            )
        arrival_tol = float(self.eye.get("pinch_arrival_tol_m", self.eye.get("linear_tol_m", 0.012)))
        arrival_loose = float(
            self.eye.get("pinch_arrival_tol_loose_m", self.eye.get("linear_tol_loose_m", 0.020))
        )
        arrival_deg = float(self.eye.get("pinch_arrival_tol_deg", 3.5))
        oneshot = path_mode in ("joint_oneshot", "oneshot", "joint") and not force_step

        def _pose_err_m():
            pose_a = self.get_pose()
            if pose_a is None:
                return None, None
            pa, ea = pose_a
            err = math.sqrt(
                (pa.x - p1[0]) ** 2
                + (pa.y - p1[1]) ** 2
                + (pa.z - p1[2]) ** 2
            )
            return err, (pa, ea)

        def _finish_check(tag=""):
            err, got = _pose_err_m()
            if err is None:
                return False
            pa, ea = got
            rpy_drift = rotation_geodesic_deg(r_use, (ea.x, ea.y, ea.z))
            ok_pos = err <= arrival_loose
            ok_att = rpy_drift <= arrival_deg
            ok_arr = ok_pos and ok_att
            vrb(
                log, self.eye,
                f"{pfx} {tag}实际xyz=[{pa.x:.4f},{pa.y:.4f},{pa.z:.4f}] "
                f"距目标 {err*1000:.1f}mm 姿态漂移 {rpy_drift:.1f}° "
                f"{'到位' if ok_arr else '未到位'}"
                f"（容差≤{arrival_loose*1000:.0f}mm / ≤{arrival_deg:.1f}°）",
            )
            if ok_pos and not ok_att:
                vrb(
                    log, self.eye,
                    f"{pfx} 位置已近但姿态差 {rpy_drift:.1f}° "
                    f"（> {arrival_deg:.1f}°）→ mid 易侧向偏",
                )
            return ok_arr

        def _linear_correct():
            """到位偏大时用直线补到目标（姿态用 r_use）。"""
            if stop():
                return False
            err0, _ = _pose_err_m()
            if err0 is None or err0 <= arrival_tol:
                return True
            vrb(log, self.eye, f"{pfx} 到位偏差 {err0*1000:.0f}mm → 直线补到目标")
            ok = self.move_to_position(
                LbotPosition(float(p1[0]), float(p1[1]), float(p1[2])),
                LbotEuler(float(r_use[0]), float(r_use[1]), float(r_use[2])),
                linear=True, block=True, force=True,
                speed=speed, accel=accel,
            )
            self.refresh_live_state()
            return bool(ok)

        def _orient_correct():
            """位置大致到位后原地转腕，消掉测地姿态残差。"""
            if stop():
                return False
            err, got = _pose_err_m()
            if got is None:
                return False
            _pa, ea = got
            drift = rotation_geodesic_deg(r_use, (ea.x, ea.y, ea.z))
            if drift <= arrival_deg:
                return True
            vrb(
                log, self.eye,
                f"{pfx} 姿态漂移 {drift:.1f}° → 原地转腕补正"
                f"（目标 rpy°="
                f"[{math.degrees(r_use[0]):.1f},{math.degrees(r_use[1]):.1f},"
                f"{math.degrees(r_use[2]):.1f}]）",
            )
            # 关节空间更易收敛姿态；位置锁在规划终点
            ok = self.move_to_position(
                LbotPosition(float(p1[0]), float(p1[1]), float(p1[2])),
                LbotEuler(float(r_use[0]), float(r_use[1]), float(r_use[2])),
                linear=False, block=True, force=True,
                speed=float(np.clip(float(speed) * 0.65, 0.08, 0.22)),
                accel=float(np.clip(float(accel) * 0.65, 0.12, 0.45)),
            )
            self.refresh_live_state()
            return bool(ok)

        def _arrive_or_correct(tag_prefix=""):
            if _finish_check(tag_prefix):
                return True
            _linear_correct()
            if _finish_check("位置补正后 "):
                return True
            # 闭合前会 MJ 精修时：不必中途再原地转腕（多一次停顿/拧腕）
            if defer_mj_polish:
                err, _got = _pose_err_m()
                ok_pos = err is not None and err <= arrival_loose
                if ok_pos:
                    log(
                        f"{pfx} 位置已近，姿态留给闭合前 MJ 精修"
                        "（跳过中途转腕）"
                    )
                return bool(ok_pos)
            _orient_correct()
            return _finish_check("转腕补正后 ")

        # 旧路径：一步关节（轨迹绕弧，与模型预览不一致）
        if oneshot and j0 is not None:
            ik1 = self._ik_at_pose(
                p1, r_use, j0,
                max_jump_rad=self.ik_max_jump_rad(),
                log_prefix=f"[{pfx}一步]",
            )
            if ik1 is not None:
                log(
                    f"{pfx} joint_oneshot → 一步关节 "
                    f"（{travel*1000:.0f}mm / {geo:.1f}°）"
                )
                ok = self.execute_joint_motion(
                    ik1, speed=speed, accel=accel, block=True,
                )
                if ok:
                    return _arrive_or_correct("一步后 ")
                log(f"{pfx} 终姿越软限位或跳变过大 → 改笛卡尔分步")

        # 笛卡尔：优先一整段连续直线（姿态≤geo_oneshot）；大姿态才插值分步
        step_m = max(float(max_step_m), 0.008)
        step_deg = max(float(max_step_deg), 1.0)
        n_rot = int(math.ceil(geo / step_deg)) if geo > 1.0 else 1
        n_pos = int(math.ceil(travel / step_m)) if travel > 1e-4 else 1
        n_steps = max(n_rot, n_pos, 1)
        n_steps = int(np.clip(n_steps, 1, 12))

        use_whole_linear = (
            continuous
            and (geo <= geo_oneshot)
            and (not prefer_joint)
            and travel > 1e-4
        )
        if use_whole_linear or n_steps <= 1:
            use_lin = (geo <= geo_oneshot) and (not prefer_joint)
            tag = "笛卡尔直线一步" if use_lin else "单段关节"
            log(
                f"{pfx} {tag} "
                f"（{travel*1000:.0f}mm / 姿态{geo:.1f}°"
                f"{'，连续' if continuous and use_lin else ''}）"
            )
            ok = self.move_to_position(
                LbotPosition(float(p1[0]), float(p1[1]), float(p1[2])),
                LbotEuler(float(r_use[0]), float(r_use[1]), float(r_use[2])),
                linear=use_lin, block=True, force=True,
                speed=speed, accel=accel,
            )
            if not ok:
                # 用户点停止/中止：禁止再改分步连发（否则会突然大幅插值）
                if stop():
                    log(f"{pfx} 直线已中止，不再分步")
                    self._hold_current_motion(f"[{pfx}]")
                    return False
                if use_whole_linear and n_steps > 1:
                    log(f"{pfx} 直线失败，改分步插值")
                else:
                    return False
            else:
                self.refresh_live_state()
                return _arrive_or_correct("直线后 " if use_lin else "单段后 ")

        log(
            f"{pfx} 笛卡尔分步 {n_steps} 段（行程 {travel*1000:.0f}mm，姿态 {geo:.1f}°，"
            f"每步≤{step_deg:.0f}°/{step_m*1000:.0f}mm；中间段不等待到位）"
        )
        done_i = 0
        for i in range(1, n_steps + 1):
            if stop():
                return False
            t = float(i) / float(n_steps)
            pi, ri = lerp_pose_rpy(p0, r0, p1, r1, t)
            if i == n_steps:
                ri = r_use
            seg_geo = rotation_geodesic_deg(
                r0 if i == 1 else lerp_pose_rpy(p0, r0, p1, r1, float(i - 1) / float(n_steps))[1],
                ri,
            )
            use_lin = (seg_geo < 8.0) and (not prefer_joint)
            is_last = i == n_steps
            # 中间段发令即走、不卡等到位，末段再阻塞 → 看起来连续
            log(
                f"{pfx} 分步 {i}/{n_steps} → xyz=[{pi[0]:.4f},{pi[1]:.4f},{pi[2]:.4f}] "
                f"rpy°=[{math.degrees(ri[0]):.1f},{math.degrees(ri[1]):.1f},"
                f"{math.degrees(ri[2]):.1f}] ({'直线' if use_lin else '关节'}"
                f"{'/连续' if not is_last else ''})"
            )
            ok = self.move_to_position(
                LbotPosition(float(pi[0]), float(pi[1]), float(pi[2])),
                LbotEuler(float(ri[0]), float(ri[1]), float(ri[2])),
                linear=use_lin, block=is_last, force=True,
                speed=speed, accel=accel,
            )
            if not ok:
                if stop():
                    log(f"{pfx} 分步已中止")
                    self._hold_current_motion(f"[{pfx}]")
                    return False
                log(f"{pfx} 分步第 {i}/{n_steps} 失败，尝试直线到终姿…")
                pose_cur = self.get_pose()
                if pose_cur is not None:
                    _pc, eul_c = pose_cur
                    ok_tr = self.move_to_position(
                        LbotPosition(float(p1[0]), float(p1[1]), float(p1[2])),
                        LbotEuler(float(eul_c.x), float(eul_c.y), float(eul_c.z)),
                        linear=True, block=True, force=True,
                        speed=speed, accel=accel,
                    )
                    if stop():
                        self._hold_current_motion(f"[{pfx}]")
                        return False
                    if ok_tr:
                        self.move_to_position(
                            LbotPosition(float(p1[0]), float(p1[1]), float(p1[2])),
                            LbotEuler(float(r_use[0]), float(r_use[1]), float(r_use[2])),
                            linear=False, block=True, force=True,
                            speed=speed, accel=accel,
                        )
                        log(f"{pfx} 降级直线到终姿")
                        return _arrive_or_correct("降级后 ")
                if done_i >= max(1, n_steps // 2):
                    log(f"{pfx} 已完成 {done_i}/{n_steps} 段，尝试直线补正")
                    self._hold_current_motion(f"[{pfx}]")
                    return _arrive_or_correct("部分+补正 ")
                self._hold_current_motion(f"[{pfx}]")
                return False
            done_i = i
            self.refresh_live_state()
            if not is_last:
                # 给控制器一点切换时间，避免指令叠压；但不等「到位」
                time.sleep(0.05)

        return _arrive_or_correct("分步后 ")

    def move_pose_chain_continuous(
        self,
        keyframes,
        speed=None,
        accel=None,
        log=print,
        should_stop=None,
        max_step_m=0.045,
        max_step_deg=8.0,
        geo_oneshot_deg=28.0,
        log_prefix="⑩",
        wait_each=False,
    ):
        """
        多关键帧笛卡尔链。
        默认：中间点发令即走，仅末点阻塞（回程流畅）。
        wait_each=True：每点都等到位（手眼采样等需要「真停稳」时用）。
        """
        stop = should_stop or (lambda: False)
        if not keyframes:
            return True
        pfx = str(log_prefix or "链")
        speed = float(speed if speed is not None else self.speed)
        accel = float(accel if accel is not None else self.accel)
        geo_oneshot = float(geo_oneshot_deg)
        step_m = max(float(max_step_m), 0.008)
        step_deg = max(float(max_step_deg), 1.0)

        pose = self.get_pose()
        if pose is None:
            log(f"{pfx} 连续路径失败：无当前位姿")
            return False
        pos0, eul0 = pose
        p_prev = np.array([pos0.x, pos0.y, pos0.z], dtype=np.float64)
        r_prev = (float(eul0.x), float(eul0.y), float(eul0.z))

        commands = []
        for p_tgt, r_tgt in keyframes:
            p1 = np.asarray(p_tgt, dtype=np.float64).reshape(3)
            r1 = (float(r_tgt[0]), float(r_tgt[1]), float(r_tgt[2]))
            travel = float(np.linalg.norm(p1 - p_prev))
            geo = rotation_geodesic_deg(r_prev, r1)
            if travel < 1e-4 and geo < 0.5:
                p_prev, r_prev = p1, r1
                continue

            n_rot = int(math.ceil(geo / step_deg)) if geo > 1.0 else 1
            n_pos = int(math.ceil(travel / step_m)) if travel > 1e-4 else 1
            n_steps = int(np.clip(max(n_rot, n_pos, 1), 1, 12))
            use_whole = geo <= geo_oneshot and travel > 1e-4

            if use_whole or n_steps <= 1:
                use_lin = geo <= geo_oneshot
                commands.append((p1, r1, use_lin))
            else:
                for i in range(1, n_steps + 1):
                    t = float(i) / float(n_steps)
                    pi, ri = lerp_pose_rpy(p_prev, r_prev, p1, r1, t)
                    if i == n_steps:
                        ri = r1
                    prev_r = (
                        r_prev
                        if i == 1
                        else lerp_pose_rpy(
                            p_prev, r_prev, p1, r1, float(i - 1) / float(n_steps)
                        )[1]
                    )
                    seg_geo = rotation_geodesic_deg(prev_r, ri)
                    commands.append((pi, ri, seg_geo < 8.0))
            p_prev, r_prev = p1, r1

        if not commands:
            return True

        mode = "每点等待" if wait_each else "仅末点等待"
        log(f"{pfx} 连续 {len(commands)} 点（{mode}）")
        for i, (pi, ri, use_lin) in enumerate(commands):
            if stop():
                return False
            is_last = i == len(commands) - 1
            block = bool(wait_each or is_last)
            ok = self.move_to_position(
                LbotPosition(float(pi[0]), float(pi[1]), float(pi[2])),
                LbotEuler(float(ri[0]), float(ri[1]), float(ri[2])),
                linear=use_lin, block=block, force=True,
                speed=speed, accel=accel,
            )
            if not ok and block:
                return False
            if not block:
                time.sleep(0.03)
        self.refresh_live_state()
        return True

    def _wait_joints_near(self, target, tol=0.06, timeout_s=30.0, poll_s=0.10):
        """
        关节到位等待。保持可中止；超时短。
        不做「近距卡住提前 False」——那会误杀后续下降/精贴。
        """
        t0 = time.time()
        last_log = 0.0
        last_err = None
        accept = max(float(tol), math.radians(1.2))
        while time.time() - t0 < timeout_s:
            if self._motion_abort:
                return False
            self.refresh_live_state()
            cur = self.read_joints_fresh(tries=1, pause_s=0.0)
            if cur is not None:
                err = max(abs(float(a) - float(b)) for a, b in zip(cur, target))
                if err <= accept:
                    return True
                last_err = err
                now = time.time()
                if now - last_log >= 2.5:
                    print(
                        f"[move] 到位中，还差约 {math.degrees(err):.0f}° "
                        f"({now - t0:.0f}/{timeout_s:.0f}s)",
                        flush=True,
                    )
                    last_log = now
            time.sleep(poll_s)
        if last_err is not None and last_err <= math.radians(2.5):
            print(
                f"[move] 关节等待超时但仅差 {math.degrees(last_err):.1f}°，视为到位",
                flush=True,
            )
            return True
        return False

    def _pose_dist_m(self, target_pos, pose=None):
        if pose is None:
            pose = self.get_pose()
        if pose is None:
            return None
        p, _ = pose
        return math.sqrt(
            (p.x - target_pos.x) ** 2
            + (p.y - target_pos.y) ** 2
            + (p.z - target_pos.z) ** 2
        )

    def _estimate_linear_wait_s(self, dist_m, speed):
        """直线等待：按距离估，上限短（解决以前动不动等 30s+）。"""
        dist_m = float(max(0.0, dist_m))
        if self._is_left_arm():
            # 勿把慢速抬到 0.02：亚毫米慢走时等不够会假超时，再刷精贴
            speed = max(float(speed), 0.0025)
            margin = float(self.eye.get("linear_wait_margin", 3.5))
            max_s = float(self.eye.get("left_linear_wait_max_s",
                                       self.eye.get("linear_wait_max_s", 8.0)))
            min_s = float(self.eye.get("left_linear_wait_min_s", 2.0))
            base = dist_m / speed * margin + 1.2
            if dist_m < 0.015 and speed < 0.015:
                min_s = max(min_s, 3.5)
                max_s = max(max_s, 9.0)
            return float(np.clip(base, min_s, min(max_s, 12.0)))
        # 用真实下发速度估时；勿把慢速抬到 0.18 把等待估短
        speed = max(float(speed), 0.05)
        margin = float(self.eye.get("linear_wait_margin", 3.5))
        max_s = float(self.eye.get("linear_wait_max_s", 12.0))
        min_s = float(self.eye.get("linear_wait_min_s", 2.0))
        base = dist_m / speed * margin + 1.2
        if dist_m >= 0.03:
            min_s = max(min_s, 3.0)
        if dist_m < 0.02:
            return float(np.clip(base, min_s, min(5.0, max_s)))
        if dist_m < 0.08:
            return float(np.clip(base, min_s, min(8.0, max_s)))
        return float(np.clip(base, min_s, max_s))

    def _linear_arrive_tol_m(self, dist_m):
        """
        直线到位容差（右臂对准/靠近也走这里）。

        短行程：loose 不能太大，否则 10mm 回退「瞬间到位」空转。
        中长行程：loose 也不能固定 14mm——否则 50~80mm 靠近还差 12mm
        就被「稳定视为到位」，工作距几乎不动，对准会飙深/空转。
        """
        dist_m = float(max(0.0, dist_m))
        tol = float(self.eye.get("linear_tol_m", 0.018))
        loose = float(self.eye.get("linear_tol_loose_m", 0.032))
        if dist_m <= 1e-6:
            return float(tol), float(loose)
        # 严容差：行程的 ~20%，夹在 [2.5mm, yaml tol]
        tol = float(min(tol, max(0.0025, dist_m * 0.20)))
        # 松容差：最多剩行程 8%，且不超过 yaml loose / 6mm
        frac_loose = max(0.0035, dist_m * 0.08)
        loose = float(min(loose, max(tol * 1.25, frac_loose), 0.006))
        return tol, loose

    def _hold_current_motion(self, log_prefix="[move]"):
        """打断在途轨迹并保持当前关节，避免重发直线/I K 叠加重规划导致跳变。"""
        if not self.robot:
            return False
        joints = self.read_joints_fresh(tries=4, pause_s=0.04)
        if joints is None:
            time.sleep(0.12)
            return False
        ok = self.robot.move_to_joint_target(self.arm, list(joints), 0.05, 0.10, False)
        time.sleep(0.18)
        self.refresh_live_state()
        if ok:
            print(f"{log_prefix} 已保持当前关节，打断在途运动", flush=True)
        return bool(ok)

    def _wait_pose_near(
        self,
        target_pos,
        tol_m=0.012,
        timeout_s=25.0,
        poll_s=0.10,
        loose_tol_m=None,
    ):
        """
        笛卡尔到位等待（恢复简洁逻辑）：
        - err≤tol → 成功
        - 稳定且 err≤loose → 成功（本段结束）
        - 超时且 err≤loose → 成功
        - 可被 stop 中止
        不再「近距卡住提前打断」：那会在 ~1mm 处 hold，把 0.4mm 精贴打崩。
        """
        loose_tol_m = float(
            loose_tol_m if loose_tol_m is not None
            else self.eye.get("linear_tol_loose_m", 0.032)
        )
        is_left = self._is_left_arm()
        stall_eps = 0.0005 if is_left else 0.0015
        stall_ok_n = 6 if is_left else 5
        t0 = time.time()
        last_log = 0.0
        last_err = None
        stall_n = 0
        while time.time() - t0 < timeout_s:
            if self._motion_abort:
                return False
            self.refresh_live_state()
            err = self._pose_dist_m(target_pos)
            if err is not None:
                if err <= tol_m:
                    return True
                if last_err is not None and abs(last_err - err) < stall_eps:
                    stall_n += 1
                else:
                    stall_n = 0
                if stall_n >= stall_ok_n and err <= loose_tol_m:
                    print(
                        f"[move] 距目标 {err*1000:.0f}mm 已稳定，视为到位",
                        flush=True,
                    )
                    return True
                last_err = err
                now = time.time()
                if now - last_log >= 2.5:
                    print(
                        f"[move] 直线到位中，还差 {err*1000:.0f}mm "
                        f"({now - t0:.0f}/{timeout_s:.0f}s)",
                        flush=True,
                    )
                    last_log = now
            time.sleep(poll_s)
        if last_err is not None and last_err <= loose_tol_m:
            print(
                f"[move] 直线跟踪超时但距目标 {last_err*1000:.0f}mm≤"
                f"{loose_tol_m*1000:.0f}mm，视为到位",
                flush=True,
            )
            return True
        if last_err is not None:
            print(
                f"[move] 直线跟踪超时，仍差 {last_err*1000:.0f}mm",
                flush=True,
            )
        return False

    def frame_info(self):
        """工具/工作系摘要，用于界面说明角度原点。"""
        if not self.robot:
            return "未连接：位姿相对控制器当前工作系；工具系默认=法兰"
        tool = None
        try:
            tool = self.robot.get_current_tool_frame(self.arm)
        except Exception:
            tool = None
        if tool:
            name, pos, eul = tool
            return (
                f"工具系 '{name}' 相对法兰: "
                f"xyz=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f}) "
                f"rpy°=({math.degrees(eul.x):.1f},{math.degrees(eul.y):.1f},{math.degrees(eul.z):.1f})"
            )
        return "工具系: 默认法兰；位姿/rpy 均相对当前工作系（工作系默认=臂基座）"

    def execute_joint_motion_safe(
        self,
        joints,
        speed=None,
        accel=None,
        log=print,
        force_direct=False,
    ):
        """
        面板/回原位关节运动：大跨度时不用单段关节 PTP（末端易先甩远再收回）。
        默认：抬高 → 目标正上方 → 下降 → 关节精贴合（同持物回程）。
        """
        if not self.robot:
            return False
        target = [float(j) for j in list(joints)[:7]]
        if len(target) < 7:
            log("安全关节运动失败：目标关节不足 7")
            return False
        speed = self.speed if speed is None else float(speed)
        accel = self.accel if accel is None else float(accel)
        speed, accel = self._cap_arm_speed_accel(speed, accel)
        is_left = int(self.arm) == int(LbotArm.LEFT_ARM)
        robot_cfg = self.config.get("robot") or {}
        if is_left and not force_direct:
            use_safe = bool(robot_cfg.get("left_joint_move_safe_path", True))
            span_deg_thr = float(
                robot_cfg.get(
                    "left_joint_move_safe_span_deg",
                    self.eye.get("joint_move_safe_span_deg", 12.0),
                )
            )
        else:
            use_safe = bool(self.eye.get("joint_move_safe_path", True)) and not force_direct
            span_deg_thr = float(self.eye.get("joint_move_safe_span_deg", 35.0))
        cur = self.read_joints_fresh(tries=2, pause_s=0.02)
        span_deg = 0.0
        if cur is not None:
            span_deg = math.degrees(
                max(abs(float(a) - float(b)) for a, b in zip(cur[:7], target))
            )
        # 近零位 / 大跨度：笛卡尔「抬-横-降」从折叠位几乎必失败，直接关节 PTP
        near_folded = False
        if cur is not None:
            near_folded = max(abs(float(j)) for j in cur[:7]) < math.radians(10.0)
        large_reconfig = span_deg >= float(
            self.eye.get("joint_direct_span_deg", 40.0)
        )
        if not force_direct and (near_folded or large_reconfig):
            why = "近零位" if near_folded else f"大跨度{span_deg:.0f}°"
            log(f"关节直达（{why}，跳过笛卡尔安全链）speed≤{speed:.3f}")
            return self.execute_joint_motion(
                target, speed=speed, accel=accel, block=True,
            )
        if not use_safe or span_deg < span_deg_thr:
            if span_deg > 0.5:
                log(f"关节直达（跨度 {span_deg:.0f}° < {span_deg_thr:.0f}°）")
            return self.execute_joint_motion(
                target, speed=speed, accel=accel, block=True,
            )

        # 左臂：抬1cm → 高位到目标上方 → 降1cm（禁止抬完直接关节冲终姿横扫）
        skip_cart = bool(robot_cfg.get("left_joint_skip_cart_chain", True))
        if is_left and skip_cart:
            home_pose = self.home_pose_from_joints(target)
            if home_pose is None:
                log(
                    f"左臂安全关节（跨度 {span_deg:.0f}°）："
                    f"FK 失败，直接关节 speed≤{speed:.3f}"
                )
                ok = self.execute_joint_motion(
                    target, speed=speed, accel=accel, block=True,
                )
                self.refresh_live_state()
                return bool(ok)
            hp, he = home_pose
            lift_m = float(np.clip(
                float(robot_cfg.get("left_return_above_m", 0.01)), 0.005, 0.02,
            ))
            return self._left_lift_traverse_descend(
                (float(hp.x), float(hp.y), float(hp.z)),
                (float(he.x), float(he.y), float(he.z)),
                speed=speed, accel=accel, log=log,
                lift_m=lift_m,
                arrive_tol_m=float(robot_cfg.get("left_arrive_tol_m", 0.0004)),
                final_joints=target,
                precise=True,
            )

        home_pose = self.home_pose_from_joints(target)
        # 右臂等：抬高 → 横移 → 下降 → 关节精贴合
        prev_above = None
        if is_left:
            prev_above = self.eye.get("pinch_return_above_home_m")
            left_above = float(robot_cfg.get("left_return_above_m", 0.01))
            self.eye["pinch_return_above_home_m"] = float(
                np.clip(left_above, 0.005, 0.02)
            )
        log(
            f"安全关节路径（跨度 {span_deg:.0f}°）："
            f"平行面+{float(self.eye.get('pinch_return_above_home_m', 0.01))*1000:.0f}mm"
            f"→横移→下降→关节贴合 speed≤{speed:.3f}"
        )
        prev_hs = self.eye.get("home_joint_speed")
        prev_ha = self.eye.get("home_joint_accel")
        self.eye["home_joint_speed"] = speed
        self.eye["home_joint_accel"] = accel
        try:
            return self.restore_home(
                target,
                home_pose=home_pose,
                log=log,
                carry_safe=True,
                skip_settle=True,
            )
        finally:
            if prev_above is None:
                self.eye.pop("pinch_return_above_home_m", None)
            else:
                self.eye["pinch_return_above_home_m"] = prev_above
            if prev_hs is None:
                self.eye.pop("home_joint_speed", None)
            else:
                self.eye["home_joint_speed"] = prev_hs
            if prev_ha is None:
                self.eye.pop("home_joint_accel", None)
            else:
                self.eye["home_joint_accel"] = prev_ha

    def go_home_joints(self):
        """回到配置中的 default_home_joints（右臂初始位）。"""
        if not self.robot:
            return False
        home = self._fallback_joints()
        ok = self.execute_joint_motion_safe(home, log=print)
        print(f"[move] 关节原位: {'SUCCESS' if ok else 'FAILED'}")
        return ok

    def go_demo_pose(self):
        """与 demo_move.py 右臂演示位一致: xyz=(0.3,-0.3,-0.3) rpy=(0,-90°,0)。"""
        if not self.robot:
            return False
        position = LbotPosition(0.3, -0.3, -0.3)
        euler = LbotEuler(0.0, -math.pi / 2, 0.0)
        ok = self.robot.move_to_pose_target(
            self.arm, position, euler, self.speed, self.accel, True
        )
        print(f"[move] 演示位姿: {'SUCCESS' if ok else 'FAILED'} "
              f"{'' if ok else self.robot.get_last_error()}")
        return ok

    def restore_camera_mount(self, mjcf_path=None):
        """从 MJCF 右腕相机支架恢复外参（清除手调标定，下次启动也跟模型）。"""
        from pathlib import Path

        if mjcf_path is None:
            return False
        path = Path(mjcf_path)
        if not path.is_file():
            return False
        self.R_ee_cam, self.t_ee_cam = apply_mjcf_camera_to_config(self.config, path)
        self.eye["camera_on_ee_prefer_yaml"] = False
        if self.config_path is not None:
            save_camera_on_ee_translation(
                self.config_path, self.t_ee_cam, prefer_yaml=False,
            )
        return True

    def camera_mount_summary(self):
        """当前外参摘要（供 UI）。"""
        t = np.round(np.asarray(self.t_ee_cam, dtype=np.float64), 4).tolist()
        eul = list(
            self.eye.get("camera_on_ee", {}).get("euler_rpy", [0.0, 0.0, 0.0])
        )
        prefer = bool(self.eye.get("camera_on_ee_prefer_yaml", False))
        return t, eul, prefer

    def _world_bias_axes(self):
        """与放下/抓取同口径：前=place_slot_forward，右=前×world_up，上=world_up。"""
        wu = np.asarray(
            self.eye.get("pinch_world_up", [0.0, 0.0, 1.0]), dtype=np.float64
        ).reshape(3)
        wn = float(np.linalg.norm(wu))
        wu = wu / wn if wn > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        fwd = np.asarray(
            self.eye.get("place_slot_forward", [1.0, 0.0, 0.0]), dtype=np.float64
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
        return wu, right, fwd

    def world_bias_to_ee_delta(self, right_m=0.0, forward_m=0.0, up_m=0.0):
        """
        世界系「右/前/上」增量 → 当前法兰姿态下的 EE 系 Δt（写入 camera_on_ee）。

        俯视对准时 EE_X/Y/Z ≠ 世界前/右/上；勿把目视偏差直接填进旧 X/Y/Z。
        返回 (delta_ee_m, delta_world_m) 或 (None, None)。
        """
        pose = self.get_pose()
        if pose is None:
            return None, None
        pos, eul = pose
        R_we = euler_rpy_to_matrix(eul.x, eul.y, eul.z)
        wu, right, fwd = self._world_bias_axes()
        dw = (
            float(right_m) * right
            + float(forward_m) * fwd
            + float(up_m) * wu
        )
        if not np.all(np.isfinite(dw)):
            return None, None
        # p_obj += R_we @ Δt  ⇒  Δt = R_weᵀ @ Δp_world
        dt = R_we.T @ dw
        return dt, dw

    def apply_camera_translation_calib(self, delta_t_m, log=print):
        """
        校准：把增量加到 camera_on_ee.translation（法兰/tool0 系）。
        推荐用 apply_camera_translation_calib_world（右/前/上），勿直接猜 EE 轴。
        """
        d = np.asarray(delta_t_m, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(d)):
            log("校准失败：位移非法")
            return False
        old_t = np.asarray(self.t_ee_cam, dtype=np.float64).reshape(3).copy()
        new_t = old_t + d
        self.t_ee_cam = new_t
        mount = self.eye.setdefault("camera_on_ee", {})
        mount["translation"] = [float(new_t[0]), float(new_t[1]), float(new_t[2])]
        self.eye["camera_on_ee_prefer_yaml"] = True
        if self.config_path is not None:
            save_camera_on_ee_translation(
                self.config_path, new_t, prefer_yaml=True,
            )
        log(
            f"外参校准 t: {[round(x, 4) for x in old_t.tolist()]} → "
            f"{[round(x, 4) for x in new_t.tolist()]} "
            f"(法兰Δt [{d[0]*1000:.1f},{d[1]*1000:.1f},{d[2]*1000:.1f}]mm)"
        )
        return True

    def apply_camera_translation_calib_world(
        self, right_m=0.0, forward_m=0.0, up_m=0.0, log=print,
    ):
        """
        目视校准（世界口径，与 place/pinch bias 相同）：
        - 锁点/紫球相对实物偏左 → 填 right>0（把锁点往右拉）
        - 偏前 → forward<0（往后）
        - 偏高 → up<0（往下）
        内部按当前法兰姿态换算到 camera_on_ee.translation。
        """
        dt, dw = self.world_bias_to_ee_delta(right_m, forward_m, up_m)
        if dt is None:
            log("外参校准失败：无当前位姿，无法把世界增量映到法兰系")
            return False
        if float(np.linalg.norm(dt)) < 1e-9:
            log("外参校准：右/前/上均为 0")
            return False
        log(
            f"外参校准(世界) 右{right_m*1000:.1f} 前{forward_m*1000:.1f} "
            f"上{up_m*1000:.1f}mm → 世界Δlock "
            f"[{dw[0]*1000:.1f},{dw[1]*1000:.1f},{dw[2]*1000:.1f}]mm "
            f"(+X前 +Y左 −Y右)"
        )
        return self.apply_camera_translation_calib(dt, log=log)

    def move_delta_base_world(
        self, delta_world, speed=None, accel=None, prefer_joint=None,
    ):
        """
        基座/工作系平移（保持姿态），用于验参把指尖移到目标。

        俯视对准后大行程直线常「SUCCESS 但 0mm 不动」→ 默认 >5cm 优先 IK 关节，
        失败再直线；小位移仍先直线。
        """
        d = np.asarray(delta_world, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(d)):
            print("[move] move_delta_base：增量非法", flush=True)
            return False
        norm = float(np.linalg.norm(d))
        if norm < 1e-5:
            return True
        max_seg = float(self.eye.get("approach_max_m", 0.28)) + 0.08
        if norm > max_seg:
            print(
                f"[move] 验参位移 {norm*1000:.0f}mm 超限（≤{max_seg*1000:.0f}mm）",
                flush=True,
            )
            return False
        pose = self.get_pose()
        if pose is None:
            return False
        pos, eul = pose
        target = LbotPosition(
            float(pos.x + d[0]), float(pos.y + d[1]), float(pos.z + d[2]),
        )
        v = float(speed if speed is not None else self.speed)
        a = float(accel if accel is not None else self.accel)
        if prefer_joint is None:
            prefer_joint = norm > 0.05

        def _joint():
            seed = self.read_joints_fresh(tries=3, pause_s=0.02)
            if seed is None:
                return self.move_to_position(
                    target, eul, linear=False, block=True, force=True,
                    speed=v, accel=a,
                )
            max_jump = float(self.eye.get("ik_max_jump_rad", 1.8))
            # 验参下探常 >8cm，放宽跳变以免软拒
            if norm > 0.08:
                max_jump = max(max_jump, 2.6)
            ik = self.ik_to_pose(
                (float(target.x), float(target.y), float(target.z)),
                (float(eul.x), float(eul.y), float(eul.z)),
                seed_joints=seed,
                max_jump_rad=max_jump,
            )
            if ik is None:
                print(
                    f"[move] 基座位移 IK 无解（max_jump={max_jump:.1f}rad），"
                    f"再试默认 move_to_position",
                    flush=True,
                )
                return self.move_to_position(
                    target, eul, linear=False, block=True, force=True,
                    speed=v, accel=a,
                )
            return self.execute_joint_motion(
                ik, speed=v, accel=a, block=True,
            )

        def _linear():
            return self.move_to_position(
                target, eul, linear=True, block=True, force=True,
                speed=v, accel=a,
            )

        if prefer_joint:
            print(
                f"[move] 基座位移 {norm*1000:.0f}mm：优先 IK 关节"
                f"（俯视大行程直线易卡死）",
                flush=True,
            )
            if _joint():
                return True
            print("[move] 关节未到位，改直线重试…", flush=True)
            return bool(_linear())
        if _linear():
            return True
        print("[move] 直线未到位，改 IK 关节重试…", flush=True)
        return bool(_joint())

    # ── 对准（pose_follow）──────────────────────────────────────────

    def _align_speed_frac(self, err_px, deadband):
        """偏心→速度比例：远快近慢，末段保底，避免平滑三次方贴近死区几乎停住。"""
        far = float(self.eye.get("align_far_err_px", 120))
        min_frac = float(self.eye.get("align_min_speed_frac", 0.28))
        eff = max(float(err_px) - float(deadband), 0.0)
        if far <= 1e-3:
            return 1.0
        t = float(np.clip(eff / far, 0.0, 1.0))
        # √t：中段仍较快；旧版 smoothstep*t 在 t≈0.15 时几乎只剩 min_frac
        shaped = math.sqrt(t)
        return float(np.clip(min_frac + (1.0 - min_frac) * shaped, min_frac, 1.0))

    def servo_center_step(
        self,
        u, v, depth_mm,
        speed=None,
        dt_s=0.02,
        start_xyz=None,
        max_travel_m=0.12,
        err_px=None,
        approach_depth_mm=None,
    ):
        """连续视觉伺服一步：梯度速度 + 位移低通；可选边对准边沿光轴靠近。"""
        pose = self.get_pose()
        if pose is None:
            return False
        position, euler = pose
        for val in (position.x, position.y, position.z, euler.x, euler.y, euler.z):
            if not np.isfinite(val):
                print("[align] 位姿非有限，拒绝", flush=True)
                return False

        cx = float(self.intrinsics.get("cx", 320))
        cy = float(self.intrinsics.get("cy", 240))
        if err_px is None:
            err_px = math.hypot(u - cx, v - cy)
        deadband = float(self.eye.get("align_servo_deadband_px", 12))
        depth_tol = float(self.eye.get("distance_tolerance_mm", 12))
        need_xy = err_px > deadband
        need_z = False
        d_err = 0.0
        if (
            approach_depth_mm is not None
            and depth_mm is not None
            and np.isfinite(depth_mm)
        ):
            d_err = float(depth_mm) - float(approach_depth_mm)
            need_z = abs(d_err) > depth_tol * 0.45
        if not need_xy and not need_z:
            self._align_delta_ema = None
            return True

        base_gain = float(self.eye.get("servo_gain", 0.28))
        soft_k = float(self.eye.get("align_gain_soft_px", 45))
        gain_floor = float(self.eye.get("align_gain_floor", 0.32))
        p_scale = float(err_px) / (float(err_px) + soft_k)
        gain = base_gain * float(np.clip(p_scale, gain_floor, 1.0))

        if need_xy:
            delta_ee = center_only_correction_in_ee(
                u, v, depth_mm, self.intrinsics, self.R_ee_cam, gain,
                xy_rotate_deg=self.xy_rotate_deg,
                flip_x=self.servo_flip_x,
                flip_y=self.servo_flip_y,
            )
            if delta_ee is None or not np.all(np.isfinite(delta_ee)):
                delta_ee = np.zeros(3, dtype=np.float64)
        else:
            delta_ee = np.zeros(3, dtype=np.float64)

        # 边对准边下降：偏心大时仍要推深度（旧下限 0.12 会像卡住）
        if need_z:
            gate = float(self.eye.get("align_approach_err_px", 140))
            z_frac = float(np.clip(1.0 - float(err_px) / max(gate, 1.0), 0.38, 1.0))
            # 深度误差越大步长越大，但单步封顶
            max_z = float(self.eye.get("align_approach_max_step_m", 0.018))
            z_soft = float(self.eye.get("align_approach_step_frac", 0.42))
            adv_m = float(np.clip((d_err / 1000.0) * z_soft * z_frac, -max_z, max_z))
            axis_sign = float(self.eye.get("approach_axis_sign", -1.0))
            delta_cam = np.array([0.0, 0.0, axis_sign * adv_m], dtype=np.float64)
            delta_ee = np.asarray(delta_ee, dtype=np.float64) + self._cam_delta_to_ee(
                delta_cam
            )

        if delta_ee is None or not np.all(np.isfinite(delta_ee)):
            return True

        ema_a = float(np.clip(self.eye.get("align_delta_ema", 0.35), 0.0, 1.0))
        if ema_a > 0.0:
            prev = getattr(self, "_align_delta_ema", None)
            if prev is None:
                self._align_delta_ema = np.asarray(delta_ee, dtype=np.float64).copy()
            else:
                self._align_delta_ema = (
                    ema_a * np.asarray(delta_ee, dtype=np.float64)
                    + (1.0 - ema_a) * prev
                )
            delta_ee = self._align_delta_ema

        norm = float(np.linalg.norm(delta_ee))
        if norm < 0.00035:
            return True

        speed = float(self.speed if speed is None else speed)
        v_mul = float(self.eye.get("align_velocity_mul", 0.55))
        v_max = float(self.eye.get("align_v_cap_max_m", 0.35))
        # 同时调深度时略降速，防冲过头
        if need_z:
            v_max = min(v_max, float(self.eye.get("align_approach_v_cap_m", 0.18)))
        v_cap = float(np.clip((0.05 + speed * 0.42) * v_mul, 0.04, v_max))
        # 速度同时受像素误差与深度误差调制
        frac_xy = self._align_speed_frac(err_px, deadband) if need_xy else 0.35
        if need_z:
            frac_z = float(np.clip(abs(d_err) / max(depth_tol * 4.0, 1.0), 0.25, 1.0))
            frac = max(frac_xy, frac_z * 0.85) if need_xy else frac_z
        else:
            frac = frac_xy
        v_now = v_cap * float(frac)
        max_d = max(v_now * float(dt_s), 0.0004)
        if norm > max_d:
            delta_ee = delta_ee * (max_d / norm)

        R = euler_rpy_to_matrix(euler.x, euler.y, euler.z)
        if not np.all(np.isfinite(R)):
            return False
        d = R @ delta_ee
        if not np.all(np.isfinite(d)):
            return True

        tx = float(position.x + d[0])
        ty = float(position.y + d[1])
        tz = float(position.z + d[2])
        if start_xyz is not None:
            travel = math.sqrt(
                (tx - start_xyz[0]) ** 2
                + (ty - start_xyz[1]) ** 2
                + (tz - start_xyz[2]) ** 2
            )
            if travel > float(max_travel_m):
                print(f"[align] 行程 {travel*1000:.0f}mm 超限，停止", flush=True)
                return False

        if not self.robot:
            return False
        return self.robot.pose_follow(
            self.arm, LbotPosition(tx, ty, tz), LbotEuler(euler.x, euler.y, euler.z)
        )

    def is_centered(self, u, v):
        cx = float(self.intrinsics.get("cx", 320))
        cy = float(self.intrinsics.get("cy", 240))
        tol = float(self.eye.get("pixel_tolerance_px", 40))
        return abs(u - cx) <= tol and abs(v - cy) <= tol

    def coarse_align_approach(
        self,
        sample_fn,
        should_stop,
        log=print,
        approach_depth_mm=None,
        speed=None,
        accel=None,
        max_rounds=None,
        chunk_m=None,
        ok_err_px=None,
        ok_depth_tol_mm=None,
    ):
        """
        对准板粗段：大步 XY+光轴（基座平移，>3cm 优先关节），
        避免 pose_follow 微步俯视「几乎不动」。
        """
        if approach_depth_mm is None:
            return True
        speed = float(self.speed if speed is None else speed)
        accel = float(self.accel if accel is None else accel)
        max_rounds = int(
            max_rounds
            if max_rounds is not None
            else self.eye.get("align_coarse_rounds", 7)
        )
        chunk_m = float(
            chunk_m
            if chunk_m is not None
            else self.eye.get("align_coarse_chunk_m", 0.05)
        )
        chunk_m = float(np.clip(chunk_m, 0.02, 0.10))
        ok_err_px = float(
            ok_err_px
            if ok_err_px is not None
            else self.eye.get("align_coarse_ok_err_px", 36)
        )
        ok_depth_tol = float(
            ok_depth_tol_mm
            if ok_depth_tol_mm is not None
            else self.eye.get("align_coarse_depth_tol_mm", 18)
        )
        gain = float(self.eye.get("align_coarse_gain", 0.55))
        cx = float(self.intrinsics.get("cx", 320))
        cy = float(self.intrinsics.get("cy", 240))
        axis_sign = float(self.eye.get("approach_axis_sign", -1.0))
        log(
            f"粗对准靠近→{float(approach_depth_mm):.0f}mm："
            f"每步≤{chunk_m*1000:.0f}mm×{max_rounds}，大行程优先关节"
        )
        for i in range(max_rounds):
            if should_stop():
                log("粗对准已停止")
                return False
            sample = sample_fn()
            if sample is None:
                log(f"粗对准 {i+1}/{max_rounds}：短暂无检测，跳过")
                time.sleep(0.08)
                continue
            u, v, depth_mm = sample
            err = math.hypot(float(u) - cx, float(v) - cy)
            d_err = 0.0
            has_d = depth_mm is not None and np.isfinite(depth_mm)
            if has_d:
                d_err = float(depth_mm) - float(approach_depth_mm)
            depth_ok = (not has_d) or abs(d_err) <= ok_depth_tol
            if err <= ok_err_px and depth_ok and has_d:
                log(
                    f"粗对准到位 偏心{err:.0f}px 深{float(depth_mm)/10:.1f}cm "
                    f"（{i+1}/{max_rounds}）"
                )
                return True
            delta_ee = np.zeros(3, dtype=np.float64)
            if err > 8.0:
                xy = center_only_correction_in_ee(
                    float(u), float(v), depth_mm, self.intrinsics, self.R_ee_cam,
                    gain,
                    xy_rotate_deg=self.xy_rotate_deg,
                    flip_x=self.servo_flip_x,
                    flip_y=self.servo_flip_y,
                )
                if xy is not None and np.all(np.isfinite(xy)):
                    delta_ee = np.asarray(xy, dtype=np.float64)
            if has_d and abs(d_err) > ok_depth_tol * 0.5:
                # 偏心大时仍推 ≥45% 深度，避免只横移不靠近
                z_frac = float(np.clip(1.0 - err / 160.0, 0.45, 1.0))
                adv_m = float(np.clip((d_err / 1000.0) * 0.85 * z_frac, -chunk_m, chunk_m))
                delta_ee = delta_ee + self._cam_delta_to_ee(
                    np.array([0.0, 0.0, axis_sign * adv_m], dtype=np.float64)
                )
            norm = float(np.linalg.norm(delta_ee))
            if norm < 0.002:
                log(f"粗对准 {i+1}/{max_rounds}：增量过小，转精修")
                return True
            if norm > chunk_m:
                delta_ee = delta_ee * (chunk_m / norm)
                norm = chunk_m
            pose = self.get_pose()
            if pose is None:
                return False
            pos, eul = pose
            world_d = euler_rpy_to_matrix(eul.x, eul.y, eul.z) @ delta_ee
            dtag = f"深{float(depth_mm)/10:.1f}→{float(approach_depth_mm)/10:.1f}cm" if has_d else "无深度"
            log(
                f"粗对准 {i+1}/{max_rounds}：偏心{err:.0f}px {dtag} "
                f"|Δ|={norm*1000:.0f}mm"
            )
            ok = self.move_delta_base_world(
                world_d,
                speed=max(0.12, float(speed)),
                accel=max(0.20, float(accel)),
                prefer_joint=(norm >= 0.028),
            )
            if not ok:
                log("粗对准一步失败，改试末端直线…")
                ok = self.move_delta_world(
                    delta_ee,
                    max(0.12, float(speed)),
                    max(0.20, float(accel)),
                )
            if not ok:
                log("粗对准移动失败，停止")
                return False
            time.sleep(0.06)
            self.refresh_live_state()
        log("粗对准结束（转精修）")
        return True

    def align_center_only(
        self,
        sample_fn,
        should_stop,
        log=print,
        speed=None,
        accel=None,
        step_period_s=0.02,
        max_step_m=None,
        depth_holder=None,
        fusion_buffer=None,
        fusion_out=None,
        approach_depth_mm=None,
        max_travel_m=None,
    ):
        """
        梯度速度对准：远快近慢 + UV/位移滤波。
        approach_depth_mm：边对准边沿光轴靠近该深度（mm），到位需中心+距离都稳。
        """
        del max_step_m, accel
        speed = float(self.speed if speed is None else speed)
        step_period_s = float(step_period_s)
        max_jump_px = float(self.eye.get("align_max_jump_px", 120))
        if max_travel_m is None:
            max_travel_m = float(self.eye.get("align_max_travel_m", 0.18))
            if approach_depth_mm is not None:
                max_travel_m = max(
                    max_travel_m,
                    float(self.eye.get("align_approach_max_travel_m", 0.35)),
                )
        max_travel_m = float(max_travel_m)
        max_steps = int(self.eye.get("max_servo_steps", 15)) * 50
        if approach_depth_mm is not None:
            max_steps = int(max_steps * 1.6)
        stable_ok = 0
        need_stable = int(self.eye.get("align_stable_frames", 3))
        settle_err_px = float(self.eye.get("align_settle_err_px", 24))
        depth_tol = float(self.eye.get("distance_tolerance_mm", 12))
        if approach_depth_mm is not None:
            settle_err_px = min(
                settle_err_px,
                float(self.eye.get("align_approach_settle_err_px", 14)),
            )
            depth_tol = float(
                self.eye.get("align_approach_depth_tol_mm", depth_tol)
            )
        max_lost = int(self.eye.get("align_lost_max_frames", 30))
        jump_confirm = int(max(1, self.eye.get("align_jump_confirm_frames", 4)))
        jump_confirm_tol = float(self.eye.get("align_jump_confirm_tol_px", 55))
        last_uv = None
        last_good_depth = None
        pending_uv = None
        pending_n = 0
        lost_frames = 0
        log_every = max(1, int(0.30 / max(step_period_s, 0.01)))
        self._align_delta_ema = None

        start_pose = self.get_pose()
        if start_pose is None:
            log("对准失败：读不到起点位姿")
            return False
        sp, _se = start_pose
        start_xyz = (float(sp.x), float(sp.y), float(sp.z))
        v_mul = float(self.eye.get("align_velocity_mul", 0.55))
        v_max = float(self.eye.get("align_v_cap_max_m", 0.35))
        if approach_depth_mm is not None:
            v_max = min(v_max, float(self.eye.get("align_approach_v_cap_m", 0.18)))
        v_cap = float(np.clip((0.05 + speed * 0.42) * v_mul, 0.04, v_max))

        mode = (
            f"边对准边靠近→{float(approach_depth_mm):.0f}mm"
            if approach_depth_mm is not None else "仅中心对准"
        )
        log(
            f"梯度对准({mode}): speed={speed:.2f} → v≤{v_cap*100:.0f}cm/s "
            f"周期={step_period_s*1000:.0f}ms 行程≤{max_travel_m*1000:.0f}mm"
        )
        ema_uv = None
        ema_alpha = float(np.clip(self.eye.get("align_uv_ema", 0.40), 0.0, 1.0))
        tol = float(self.eye.get("pixel_tolerance_px", 40))
        crawl_err = float(self.eye.get("align_crawl_err_px", settle_err_px + 18))
        self._aligning = True
        try:
            for step in range(max_steps):
                if should_stop():
                    log("对准已停止")
                    return False
                t0 = time.time()
                sample = sample_fn()
                if sample is None:
                    lost_frames += 1
                    if lost_frames > max_lost:
                        log(f"对准失败：连续 {lost_frames} 帧无检测")
                        return False
                    if step % log_every == 0:
                        log(f"短暂丢框 ({lost_frames}/{max_lost})，暂停伺服")
                    elapsed = time.time() - t0
                    time.sleep(max(0.0, step_period_s - elapsed))
                    continue
                lost_frames = 0
                u, v, depth_mm = sample
                cx = float(self.intrinsics.get("cx", 320))
                cy = float(self.intrinsics.get("cy", 240))
                err = math.hypot(u - cx, v - cy)
                centered = self.is_centered(u, v)
                u_servo, v_servo = float(u), float(v)
                depth_use = depth_mm
                jump_gate = float(max_jump_px) + 0.35 * float(err)
                jump_gate = float(np.clip(jump_gate, max_jump_px, 280.0))

                if last_uv is not None:
                    jump = math.hypot(u - last_uv[0], v - last_uv[1])
                    if jump > jump_gate:
                        if not centered:
                            if (
                                pending_uv is not None
                                and math.hypot(u - pending_uv[0], v - pending_uv[1])
                                <= jump_confirm_tol
                            ):
                                pending_n += 1
                            else:
                                pending_uv = (float(u), float(v))
                                pending_n = 1
                            if pending_n >= jump_confirm:
                                last_uv = (float(u), float(v))
                                if depth_mm is not None and np.isfinite(depth_mm):
                                    last_good_depth = float(depth_mm)
                                pending_uv = None
                                pending_n = 0
                                ema_uv = None
                                if step % log_every == 0:
                                    log(
                                        f"检测跳变确认 {jump:.0f}px → 切换跟踪"
                                    )
                            else:
                                u_servo, v_servo = last_uv
                                if last_good_depth is not None:
                                    depth_use = last_good_depth
                                if step % log_every == 0:
                                    log(
                                        f"检测跳变 {jump:.0f}px，沿用上一帧 "
                                        f"（确认 {pending_n}/{jump_confirm}）"
                                    )
                        else:
                            last_uv = (float(u), float(v))
                            if depth_mm is not None and np.isfinite(depth_mm):
                                last_good_depth = float(depth_mm)
                            pending_uv = None
                            pending_n = 0
                    else:
                        last_uv = (float(u), float(v))
                        if depth_mm is not None and np.isfinite(depth_mm):
                            last_good_depth = float(depth_mm)
                        pending_uv = None
                        pending_n = 0
                else:
                    last_uv = (float(u), float(v))
                    if depth_mm is not None and np.isfinite(depth_mm):
                        last_good_depth = float(depth_mm)

                depth_mm = depth_use
                err = math.hypot(u_servo - cx, v_servo - cy)
                depth_ok = True
                if approach_depth_mm is not None:
                    if depth_mm is None or not np.isfinite(depth_mm):
                        depth_ok = False
                    else:
                        depth_ok = abs(
                            float(depth_mm) - float(approach_depth_mm)
                        ) <= depth_tol

                if err <= settle_err_px and depth_ok:
                    self._align_delta_ema = None
                    stable_ok += 1
                    if fusion_buffer is not None:
                        fusion_buffer.add(u_servo, v_servo, depth_mm)
                    if depth_mm is not None and np.isfinite(depth_mm) and depth_holder is not None:
                        depth_holder[0] = float(depth_mm)
                    if step % log_every == 0 or stable_ok >= need_stable:
                        dtag = ""
                        if approach_depth_mm is not None and depth_mm is not None:
                            dtag = f" 深{float(depth_mm)/10:.1f}cm"
                        if stable_ok >= need_stable or eye_verbose(self.eye):
                            log(
                                f"对准中 偏心 {err:.0f}px "
                                f"({stable_ok}/{need_stable}){dtag}"
                            )
                    if stable_ok >= need_stable:
                        if fusion_buffer is not None and fusion_buffer.ready():
                            fu, fv, fd = fusion_buffer.median()
                            if fd is not None and np.isfinite(fd):
                                if fusion_out is not None:
                                    fusion_out[:] = [fu, fv, fd]
                                if depth_holder is not None:
                                    depth_holder[0] = fd
                                log(
                                    f"已对准 融合 u,v,depth "
                                    f"({fu:.0f},{fv:.0f}) {fd/10:.1f}cm"
                                )
                            elif depth_holder is not None and depth_holder[0] is not None:
                                log(
                                    f"已对准黄十字 偏心 {err:.0f}px "
                                    f"（融合深度无效，沿用 {depth_holder[0]/10:.1f}cm）"
                                )
                            else:
                                log(f"已对准黄十字 偏心 {err:.0f}px（深度待确认）")
                                stable_ok = need_stable - 1
                                elapsed = time.time() - t0
                                time.sleep(max(0.0, step_period_s - elapsed))
                                continue
                        else:
                            dmsg = ""
                            if approach_depth_mm is not None and depth_mm is not None:
                                dmsg = f" 深{float(depth_mm)/10:.1f}cm"
                            log(f"已对准黄十字 偏心 {err:.0f}px{dmsg}")
                        time.sleep(float(self.eye.get("align_settle_s", 0.15)))
                        return True
                else:
                    if err > tol or (approach_depth_mm is not None and not depth_ok):
                        stable_ok = 0
                    elif err > settle_err_px:
                        stable_ok = max(0, stable_ok - 1)
                    if ema_alpha > 0.0:
                        if ema_uv is None:
                            ema_uv = (u_servo, v_servo)
                        else:
                            a = ema_alpha if err > crawl_err else max(0.15, ema_alpha * 0.55)
                            ema_uv = (
                                a * u_servo + (1.0 - a) * ema_uv[0],
                                a * v_servo + (1.0 - a) * ema_uv[1],
                            )
                        u_servo, v_servo = ema_uv
                    ok = self.servo_center_step(
                        u_servo, v_servo, depth_mm,
                        speed=speed,
                        dt_s=step_period_s,
                        start_xyz=start_xyz,
                        max_travel_m=max_travel_m,
                        err_px=err,
                        approach_depth_mm=approach_depth_mm,
                    )
                    if not ok:
                        log("对准安全限触发，已停止")
                        return False
                    if step % log_every == 0:
                        frac = self._align_speed_frac(
                            err, float(self.eye.get("align_servo_deadband_px", 12)),
                        )
                        dtag = ""
                        if (
                            approach_depth_mm is not None
                            and depth_mm is not None
                            and np.isfinite(depth_mm)
                        ):
                            dtag = (
                                f" 深{float(depth_mm)/10:.1f}→"
                                f"{float(approach_depth_mm)/10:.1f}cm"
                            )
                        vrb(
                            log, self.eye,
                            f"伺服 偏心 {err:.0f}px 速度×{frac:.2f}{dtag}",
                        )
                if step % 2 == 0:
                    self.refresh_live_state()
                elapsed = time.time() - t0
                time.sleep(max(0.0, step_period_s - elapsed))
        finally:
            self._aligning = False
            self._align_delta_ema = None
        log("对准超时")
        return False

    def settle_after_follow(self, log=print):
        """
        对准结束后打断 pose_follow。
        用当前笛卡尔位姿保持，避免 joint hold 与 follow 位姿不一致导致跳变。
        """
        if not self.robot:
            return False
        time.sleep(0.08)
        pose = self.get_pose()
        if pose is None:
            log("退出 follow：读不到位姿，改关节保持")
            return self._hold_current_motion("[align]")
        pos, euler = pose
        ok = self.robot.linear_move_to_pose(
            self.arm,
            LbotPosition(float(pos.x), float(pos.y), float(pos.z)),
            LbotEuler(float(euler.x), float(euler.y), float(euler.z)),
            0.06, 0.12, False,
        )
        time.sleep(0.22)
        self.refresh_live_state()
        if ok:
            log("已退出 pose_follow（笛卡尔保持）")
        else:
            err = self.robot.get_last_error()
            log(f"退出 follow 直线保持失败: {err}，改关节保持")
            self._hold_current_motion("[align]")
        return bool(ok)

    # ── 抓取笛卡尔段 ────────────────────────────────────────────────

    def _cam_delta_to_ee(self, delta_cam):
        return camera_delta_to_ee(
            delta_cam,
            self.R_ee_cam,
            xy_rotate_deg=self.xy_rotate_deg,
            flip_x=self.servo_flip_x,
            flip_y=self.servo_flip_y,
        )

    def pregrasp_apply_hand_offset(self):
        """③ 是否沿相机系做手偏移（false 时 ⑥ 一次对齐物体）。"""
        return bool(self.eye.get("pregrasp_apply_hand_offset", False))

    def pick_hand_offset_delta_ee(self):
        """手相对光轴的横向补偿（画面右/上，不含光轴靠近）。"""
        off = np.array(
            self.eye.get("hand_offset_in_camera_m", [0.06, -0.09, 0.0]),
            dtype=np.float64,
        )
        return self._cam_delta_to_ee(off)

    def retreat_along_optical_axis(self, retreat_m=0.01, speed=None, accel=None):
        """
        沿相机光轴远离物体（与预抓靠近方向相反）。
        阻塞：必须等本段回退实际走完再返回，避免深度确认循环空转叠指令。
        """
        return self.move_along_optical_axis(
            -float(retreat_m), speed=speed, accel=accel, label="回退",
        )

    def approach_along_optical_axis(self, advance_m=0.01, speed=None, accel=None):
        """沿相机光轴靠近物体（阻塞）。"""
        return self.move_along_optical_axis(
            float(advance_m), speed=speed, accel=accel, label="靠近",
        )

    def move_along_optical_axis(
        self, signed_m, speed=None, accel=None, label="光轴",
    ):
        """
        沿光轴移动：signed_m>0 靠近物体，<0 远离。
        阻塞到位确认，与 retreat_along_optical_axis 同口径。
        """
        signed_m = float(signed_m)
        if abs(signed_m) <= 1e-6:
            return True
        axis_sign = float(self.eye.get("approach_axis_sign", -1.0))
        # 靠近：axis_sign * (+advance)；远离：axis_sign * (−retreat) = −axis_sign * retreat
        delta_cam = np.array(
            [0.0, 0.0, axis_sign * signed_m], dtype=np.float64,
        )
        delta_ee = self._cam_delta_to_ee(delta_cam)
        dist = abs(signed_m)
        v = float(
            speed
            if speed is not None
            else np.clip(float(self.speed) * 0.45, 0.08, 0.22)
        )
        a = float(
            accel
            if accel is not None
            else np.clip(float(self.accel) * 0.45, 0.15, 0.55)
        )

        pose0 = self.get_pose()
        if pose0 is None:
            print(f"[move] {label}失败：读不到位姿", flush=True)
            return False
        p0, _e0 = pose0
        start = np.array([p0.x, p0.y, p0.z], dtype=np.float64)

        ok = self.move_delta_world(delta_ee, v, a)
        pose1 = self.get_pose()
        if pose1 is None:
            return bool(ok)
        p1, _ = pose1
        moved = float(np.linalg.norm(
            np.array([p1.x, p1.y, p1.z], dtype=np.float64) - start
        ))
        # 工作距靠近/回退必须走完大部分，否则深度几乎不动却回报成功
        need = max(0.004, dist * 0.85)
        if moved >= need:
            return True
        if ok and moved >= need * 0.92:
            return True
        wait_s = self._estimate_linear_wait_s(dist, v)
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if self._motion_abort:
                return False
            time.sleep(0.05)
            self.refresh_live_state()
            pose = self.get_pose()
            if pose is None:
                continue
            p, _ = pose
            moved = float(np.linalg.norm(
                np.array([p.x, p.y, p.z], dtype=np.float64) - start
            ))
            if moved >= need:
                return True
        print(
            f"[move] 光轴{label}未走够：目标 {dist*1000:.0f}mm，"
            f"实测 {moved*1000:.1f}mm",
            flush=True,
        )
        return moved >= need * 0.5

    def pick_advance_delta_ee(self, depth_mm, target_mm=None, max_advance_m=None):
        """沿光轴靠近到 target_mm（默认 eye_in_hand.target_distance_mm）。"""
        target_mm = float(
            target_mm if target_mm is not None else self.eye["target_distance_mm"]
        )
        tol = float(self.eye.get("distance_tolerance_mm", 18))
        max_adv = float(
            max_advance_m if max_advance_m is not None
            else self.eye.get("approach_max_m", 0.28)
        )
        axis_sign = float(self.eye.get("approach_axis_sign", -1.0))

        advance_m = 0.0
        if depth_mm is not None and np.isfinite(depth_mm):
            # 用动作深度上限卡住假远景，避免预抓冲出十几厘米
            d = float(depth_mm)
            d_cap = float(self.eye.get("depth_action_max_mm", 300.0))
            d = min(d, d_cap)
            if d > target_mm + tol:
                advance_m = min((d - target_mm) / 1000.0, max_adv)

        delta_cam = np.array([0.0, 0.0, axis_sign * advance_m], dtype=np.float64)
        return self._cam_delta_to_ee(delta_cam), advance_m

    def resolve_pregrasp_depth_mm(self, depth_mm):
        return resolve_pregrasp_depth_mm(
            depth_mm, self.eye, size_info=getattr(self, "grasp_size_info", None),
        )

    def pick_combined_hand_pregrasp_delta_ee(self, depth_mm, target_mm=None):
        """③④ 合并：手偏移 + 光轴预抓，一次相机系增量。"""
        if target_mm is None:
            target_mm = self.resolve_pregrasp_depth_mm(depth_mm)
        else:
            target_mm = float(target_mm)
        off = np.array(
            self.eye.get("hand_offset_in_camera_m", [0.06, -0.09, 0.0]),
            dtype=np.float64,
        )
        _delta_z, advance_m = self.pick_advance_delta_ee(depth_mm, target_mm)
        axis_sign = float(self.eye.get("approach_axis_sign", -1.0))
        delta_cam = off + np.array([0.0, 0.0, axis_sign * advance_m], dtype=np.float64)
        return self._cam_delta_to_ee(delta_cam), advance_m, off

    def advance_delta_toward_world_point_ee(self, target_world, step_m):
        """世界系：末端向 target 移动 step_m → 末端系 delta。"""
        pose = self.get_pose()
        if pose is None or target_world is None:
            return None
        pos, euler = pose
        ee = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        tgt = np.asarray(target_world, dtype=np.float64).reshape(3)
        vec = tgt - ee
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            return np.zeros(3, dtype=np.float64)
        move = min(float(step_m), dist)
        world_delta = vec / dist * move
        R_we = euler_rpy_to_matrix(euler.x, euler.y, euler.z)
        return R_we.T @ world_delta

    def execute_pregrasp_phase(
        self,
        sample_fn,
        stop,
        log=print,
        v_approach=0.15,
        a_approach=0.35,
        initial_depth_mm=None,
        lock_object_fn=None,
    ):
        """
        ③④ 预抓：到 ~20cm 处停住并锁定；之后仅直线靠近（见 ⑥⑦）。
        返回 (ok, u, v, depth_mm)。
        """
        merge = bool(self.eye.get("pregrasp_merge_offset_advance", True))
        closed_loop = bool(self.eye.get("pregrasp_closed_loop", False))
        settle_s = float(self.eye.get("pregrasp_settle_s", 0.15))
        off_only_first = bool(self.eye.get("pregrasp_offset_before_loop", True))

        depth_mm = initial_depth_mm
        u = v = None
        sample = sample_fn()
        if sample is not None:
            u, v, d = sample[:3]
            if depth_mm is None and d is not None:
                depth_mm = float(d)
        depth_mm, why_d = sanitize_action_depth_mm(depth_mm, self.eye)
        if depth_mm is None:
            # 无可靠深度时用保守 fallback，但仍受 action_max 限制
            fb = min(
                float(self.eye.get("approach_fallback_depth_mm", 220)),
                float(self.eye.get("depth_action_max_mm", 300)),
            )
            log(f"预抓：{why_d}，暂用保守深度 {fb/10:.0f}cm")
            depth_mm = fb
        target_mm = self.resolve_pregrasp_depth_mm(depth_mm)
        apply_off = self.pregrasp_apply_hand_offset()
        off = self.eye.get("hand_offset_in_camera_m", [0.06, -0.09, 0.0])
        log(f"③④ 预抓至 {target_mm/10:.1f}cm（20cm 远距停；其后直线靠近）")
        if apply_off:
            log(
                f"   手偏移 右 {off[0] * 1000:.0f}mm 上 {-off[1] * 1000:.0f}mm"
            )
        else:
            log("   ③ 手偏移已关闭（⑥ 一次对齐黄球）")

        if merge and not closed_loop:
            if off_only_first:
                if apply_off:
                    delta_off = self.pick_hand_offset_delta_ee()
                    if float(np.linalg.norm(delta_off)) > 1e-4:
                        if not self.move_delta_world(delta_off, v_approach, a_approach):
                            log("预抓手偏移失败")
                            return False, u, v, depth_mm
                        if stop():
                            return False, u, v, depth_mm
                        time.sleep(settle_s * 0.5)
                delta_pre, advance_pre = self.pick_advance_delta_ee(
                    depth_mm, target_mm,
                )
                log(f"   光轴至 {target_mm/10:.1f}cm（{advance_pre*1000:.0f}mm）")
                if advance_pre > 1e-4:
                    if not self.move_delta_world(delta_pre, v_approach, a_approach):
                        log("预抓靠近失败")
                        return False, u, v, depth_mm
                elif not apply_off or float(np.linalg.norm(self.pick_hand_offset_delta_ee())) <= 1e-4:
                    log("深度已够近，跳过预抓靠近")
            else:
                if apply_off:
                    delta, advance_m, _off = self.pick_combined_hand_pregrasp_delta_ee(
                        depth_mm, target_mm,
                    )
                else:
                    delta, advance_m = self.pick_advance_delta_ee(
                        depth_mm, target_mm,
                    )
                if float(np.linalg.norm(delta)) > 1e-4:
                    if not self.move_delta_world(delta, v_approach, a_approach):
                        log("预抓合并运动失败")
                        return False, u, v, depth_mm
                elif advance_m <= 1e-4:
                    log("深度已够近，跳过预抓靠近")
        else:
            if off_only_first and apply_off:
                delta_off = self.pick_hand_offset_delta_ee()
                if float(np.linalg.norm(delta_off)) > 1e-4:
                    if not self.move_delta_world(delta_off, v_approach, a_approach):
                        log("手偏移失败")
                        return False, u, v, depth_mm
                    if stop():
                        return False, u, v, depth_mm
                    time.sleep(settle_s * 0.5)

            delta_pre, advance_pre = self.pick_advance_delta_ee(
                depth_mm, target_mm,
            )
            log(f"   光轴至 {target_mm/10:.1f}cm（{advance_pre*1000:.0f}mm）")
            if advance_pre > 1e-4:
                if not self.move_delta_world(delta_pre, v_approach, a_approach):
                    log("预抓靠近失败")
                    return False, u, v, depth_mm

        if stop():
            return False, u, v, depth_mm

        sample = sample_fn()
        if sample is not None:
            u, v, d = sample[:3]
            if d is not None and np.isfinite(d):
                depth_mm = float(d)

        if lock_object_fn is not None:
            locked = lock_object_fn(u, v, depth_mm)
            if not locked:
                log("预抓后物体锁定失败")
                return False, u, v, depth_mm
        elif depth_mm is None:
            depth_mm = min(
                float(self.eye.get("approach_fallback_depth_mm", 220)),
                float(self.eye.get("depth_action_max_mm", 300)),
            )

        log(f"⑤ 物体深度 {float(depth_mm)/10:.1f}cm（预抓后锁定）")
        return True, u, v, depth_mm

    def pick_contact_advance_m(self):
        """⑦ 补回 standoff×frac，再额外再进一截（防抓空）。"""
        standoff = resolve_pinch_standoff_m(
            self.eye, getattr(self, "grasp_size_info", None),
        )
        frac = float(self.eye.get("pinch_contact_frac", 1.0))
        extra = float(self.eye.get("pinch_contact_extra_m", 0.025))
        max_adv = float(self.eye.get("approach_max_m", 0.28))
        return min(standoff * frac + max(extra, 0.0), max_adv)

    def pick_lift_offsets_m(self):
        """⑨ 提离（米）：上 / 右 / 前（空间系语义）。"""
        up = float(self.eye.get("pinch_lift_up_m", 0.05))
        right = float(self.eye.get("pinch_lift_right_m", 0.0))
        forward = float(self.eye.get("pinch_lift_forward_m", 0.0))
        return up, right, forward

    def pick_lift_delta_world(self):
        """
        ⑨ 闭合后提离：与放下/抓取侧偏同口径。
        前 = place_slot_forward，右 = cross(前, world_up)，上 = world_up。
        返回 (world_delta_xyz, (up, right, forward))。
        """
        up, right, forward = self.pick_lift_offsets_m()
        wu = np.asarray(
            self.eye.get("pinch_world_up", [0.0, 0.0, 1.0]), dtype=np.float64
        ).reshape(3)
        wn = float(np.linalg.norm(wu))
        wu = wu / wn if wn > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        fwd = np.asarray(
            self.eye.get("place_slot_forward", [1.0, 0.0, 0.0]), dtype=np.float64
        ).reshape(3)
        fn = float(np.linalg.norm(fwd))
        fwd = fwd / fn if fn > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fwd = fwd - float(np.dot(fwd, wu)) * wu
        fn = float(np.linalg.norm(fwd))
        fwd = fwd / fn if fn > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right_axis = np.cross(fwd, wu)
        rn = float(np.linalg.norm(right_axis))
        right_axis = (
            right_axis / rn if rn > 1e-9
            else np.array([0.0, -1.0, 0.0], dtype=np.float64)
        )
        world = fwd * forward + right_axis * right + wu * up
        return world, (up, right, forward)

    def pick_final_advance_m(self, depth_mm):
        """兼容旧名；夹取 ⑦ 请用 pick_contact_advance_m。"""
        return self.pick_contact_advance_m()

    def _vision_ray_params(self):
        eye = self.eye
        return {
            "xy_rotate_deg": float(self.xy_rotate_deg),
            "flip_x": bool(self.servo_flip_x),
            "flip_y": bool(self.servo_flip_y),
            "depth_sign": float(eye.get("camera_depth_sign", -1.0)),
        }

    def camera_rotation_world(self, ee_pose=None):
        """当前相机姿态 R_world_cam（安装角已在 R_ee_cam）。"""
        pose = ee_pose if ee_pose is not None else self.get_pose()
        if pose is None:
            return None
        _, cam_rot = camera_pose_in_world(self.R_ee_cam, self.t_ee_cam, pose)
        return cam_rot

    def _pinch_object_up(self, R_we):
        """物体上边方向（世界系），与 grasp_point_mode 一致。"""
        R_we = np.asarray(R_we, dtype=np.float64).reshape(3, 3)
        R_world_cam = R_we @ np.asarray(self.R_ee_cam, dtype=np.float64).reshape(3, 3)
        eye = self.eye
        return object_up_direction_world(
            R_world_cam,
            grasp_point_mode=eye.get("grasp_point_mode", "bbox_center"),
            grasp_point_side=eye.get("grasp_point_side", "right"),
            xy_rotate_deg=0.0,
            flip_x=False,
            flip_y=False,
            world_up_fallback=eye.get("pinch_world_up", [0.0, 0.0, 1.0]),
        )

    def get_pose_for_arm(self, arm):
        """读指定臂笛卡尔位姿（不切换 self.arm）。"""
        if not self.robot:
            return None
        return self.robot.get_cartesian_pose(arm)

    def object_point_in_world_arm(self, arm, u, v, depth_mm):
        """用指定臂（通常右臂相机）位姿做像素+深度 → 工作系点。"""
        pose = self.get_pose_for_arm(arm)
        if pose is None:
            return None
        kw = self._vision_ray_params()
        return detection_point_in_world(
            u, v, depth_mm, self.intrinsics, self.R_ee_cam, self.t_ee_cam, pose,
            **kw,
        )

    def object_point_in_world(self, u, v, depth_mm):
        pose = self.get_pose()
        if pose is None:
            return None
        kw = self._vision_ray_params()
        return detection_point_in_world(
            u, v, depth_mm, self.intrinsics, self.R_ee_cam, self.t_ee_cam, pose,
            **kw,
        )

    def object_point_on_plane(self, u, v, plane_point, plane_normal=None):
        """
        像素射线 ∩ 平面（过 plane_point、法向 plane_normal）。
        筐口沿近似水平时：三格用对准点定平面，避免远格套用近处 rim 深度导致前后漂。
        """
        pose = self.get_pose()
        if pose is None:
            return None
        pp = np.asarray(plane_point, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(pp)):
            return None
        n = (
            np.asarray(plane_normal, dtype=np.float64).reshape(3)
            if plane_normal is not None
            else self._place_world_up()
        )
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            return None
        n = n / nn
        kw = self._vision_ray_params()
        cam_pos, cam_rot = camera_pose_in_world(
            self.R_ee_cam, self.t_ee_cam, pose,
        )
        if cam_pos is None:
            return None
        # 1m 处一点 → 与光心构成射线（与 detection_point_in_world 同 remap/符号）
        p_far = detection_point_in_world(
            u, v, 1000.0, self.intrinsics, self.R_ee_cam, self.t_ee_cam, pose,
            **kw,
        )
        if p_far is None:
            return None
        direction = np.asarray(p_far, dtype=np.float64).reshape(3) - cam_pos
        dn = float(np.dot(direction, n))
        if abs(dn) < 1e-9:
            return None
        t = float(np.dot(pp - cam_pos, n) / dn)
        if t <= 0.05 or t > 2.5:
            return None
        out = cam_pos + t * direction
        return out if np.all(np.isfinite(out)) else None

    def ik_max_jump_rad(self):
        """与实机 PTP 相同的 IK 跳变上限（预览不得放宽，约 103°）。"""
        return float(self.eye.get("ik_max_jump_rad", 1.8))

    def soft_joint_limits_rad(self):
        """示教器软限位 (lo, hi)，单位 rad。"""
        cached = getattr(self, "_soft_limits_rad", None)
        if cached is None:
            self._soft_limits_rad = resolve_soft_joint_limits_rad(self.config)
            cached = self._soft_limits_rad
        return cached

    def soft_limit_eps_rad(self):
        robot = self.config.get("robot") or {}
        return math.radians(float(robot.get("soft_limit_eps_deg", 1.0)))

    def _accept_ik_soft_limits(self, ik, seed=None, max_jump_rad=None, log_prefix="[plan]"):
        """
        软限位验收：贴边钳位 / 内侧换种子重解（7 轴冗余）。
        返回可接受的关节 list，或 None。

        软限位数值本身不改。左臂默认拒绝任何实质钳位解：
        钳掉的关节 FK 对不上笛卡尔目标，精贴合会从错误构型起飞。
        """
        if ik is None:
            return None
        lo, hi = self.soft_joint_limits_rad()
        eps = self.soft_limit_eps_rad()
        if max_jump_rad is None:
            max_jump_rad = self.ik_max_jump_rad()
        robot_cfg = self.config.get("robot") or {}
        reject_clamp = bool(
            self._is_left_arm()
            and robot_cfg.get("left_ik_reject_soft_clamp", True)
        )
        # 钳位相对原解超过此量即视为「实质钳位」（约 0.06°）
        clamp_eps = math.radians(0.06)

        viol = joints_soft_limit_violation(ik, lo, hi, eps_rad=eps)
        if viol is None:
            # 在 eps 内：可能仍略越硬边，钳回后再交跳变检查
            clamped = clamp_joints_to_soft_limits(ik, lo, hi)
            clamp_delta = max(
                abs(float(a) - float(b)) for a, b in zip(clamped, ik)
            )
            if reject_clamp and clamp_delta > clamp_eps:
                i_max = int(np.argmax([
                    abs(float(a) - float(b)) for a, b in zip(clamped, ik)
                ]))
                print(
                    f"{log_prefix} 软限位：拒绝贴边钳位"
                    f"（J{i_max+1} {math.degrees(ik[i_max]):.2f}°→"
                    f"{math.degrees(clamped[i_max]):.2f}°），改笛卡尔",
                    flush=True,
                )
                return None
            if seed is not None:
                jump = max(abs(float(a) - float(b)) for a, b in zip(clamped, seed))
                if jump > float(max_jump_rad):
                    return None
            return clamped

        i, q, vlo, vhi = viol
        print(
            f"{log_prefix} IK 越软限位 J{i+1}="
            f"{math.degrees(q):.1f}° 不在 "
            f"[{math.degrees(vlo):.1f}°, {math.degrees(vhi):.1f}°]",
            flush=True,
        )

        if reject_clamp:
            print(
                f"{log_prefix} 软限位：拒绝钳位解（J{i+1}="
                f"{math.degrees(q):.1f}°），改笛卡尔",
                flush=True,
            )
            return None

        if seed is None:
            return None

        # 非左臂 / 未开拒绝：钳位后若跳变可接受则用钳位解
        cand = clamp_joints_to_soft_limits(ik, lo, hi)
        jump = max(abs(float(a) - float(b)) for a, b in zip(cand, seed))
        if jump <= float(max_jump_rad) and joints_soft_limit_violation(
            cand, lo, hi, eps_rad=0.0,
        ) is None:
            print(
                f"{log_prefix} 软限位：钳位接受"
                f"（J{i+1} {math.degrees(q):.1f}°→{math.degrees(cand[i]):.1f}°）",
                flush=True,
            )
            return list(cand)
        return None

    def _ik_at_pose(self, target_pos, target_rpy, seed, max_jump_rad=None, log_prefix="[plan]"):
        """带软限位恢复的 IK：多种子重解。"""
        if max_jump_rad is None:
            max_jump_rad = self.ik_max_jump_rad()
        if not self.robot or seed is None:
            return None
        pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        rpy = np.asarray(target_rpy, dtype=np.float64).reshape(3)
        target = LbotPosition(float(pos[0]), float(pos[1]), float(pos[2]))
        euler = LbotEuler(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        lo, hi = self.soft_joint_limits_rad()
        robot_cfg = self.config.get("robot") or {}
        reseed = bool(robot_cfg.get("ik_limit_reseed", True))

        seeds = [list(seed)]
        if reseed:
            seeds.append(soft_limit_inward_seed(seed, lo, hi, pull_rad=math.radians(5.0)))
            seeds.append(soft_limit_inward_seed(seed, lo, hi, pull_rad=math.radians(12.0)))
            # J2 特别容易贴 4° 上限：强制往负向再试两档
            s2 = list(seed)
            s2[1] = float(np.clip(float(s2[1]) - math.radians(10.0), lo[1], hi[1]))
            seeds.append(s2)
            s3 = list(seed)
            s3[1] = float(np.clip(0.5 * (lo[1] + hi[1]), lo[1], hi[1]))
            seeds.append(s3)

        seen = set()
        for si, sd in enumerate(seeds):
            key = tuple(round(float(x), 4) for x in sd)
            if key in seen:
                continue
            seen.add(key)
            ik = self.robot.compute_inverse_kinematics(self.arm, target, euler, sd)
            if not ik:
                continue
            jump = max(abs(float(a) - float(b)) for a, b in zip(ik, seed))
            if jump > float(max_jump_rad):
                continue
            accepted = self._accept_ik_soft_limits(
                ik, seed=seed, max_jump_rad=max_jump_rad, log_prefix=log_prefix,
            )
            if accepted is not None:
                if si > 0:
                    print(
                        f"{log_prefix} IK 软限位：第 {si+1} 个种子成功",
                        flush=True,
                    )
                return accepted
            # 硬越界：用钳位解作种子再解一次
            if reseed:
                seed2 = soft_limit_inward_seed(ik, lo, hi, pull_rad=math.radians(8.0))
                ik2 = self.robot.compute_inverse_kinematics(self.arm, target, euler, seed2)
                if not ik2:
                    continue
                jump2 = max(abs(float(a) - float(b)) for a, b in zip(ik2, seed))
                if jump2 > float(max_jump_rad):
                    continue
                accepted2 = self._accept_ik_soft_limits(
                    ik2, seed=seed, max_jump_rad=max_jump_rad, log_prefix=log_prefix,
                )
                if accepted2 is not None:
                    print(f"{log_prefix} IK 软限位：内侧重解成功", flush=True)
                    return accepted2
        print(f"{log_prefix} IK 软限位：多种子均失败", flush=True)
        return None

    def ik_after_delta_ee(self, delta_ee, seed_joints=None, max_jump_rad=None):
        """末端系平移增量 → IK 关节（仅规划/模型预览，不发运动）。"""
        if max_jump_rad is None:
            max_jump_rad = self.ik_max_jump_rad()
        if not self.robot or delta_ee is None:
            return None
        delta_ee = np.asarray(delta_ee, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(delta_ee)) or float(np.linalg.norm(delta_ee)) < 1e-6:
            seed = seed_joints if seed_joints is not None else self.read_joints_fresh(tries=3)
            return list(seed) if seed is not None else None
        joints = seed_joints if seed_joints is not None else self.read_joints_fresh(tries=3)
        if joints is None:
            return None
        fk = self.robot.compute_forward_kinematics(self.arm, joints)
        if fk is None:
            # 旧版思路：重读关节再试；仍失败则用笛卡尔位姿当 FK
            for _ in range(3):
                time.sleep(0.03)
                j2 = self.read_joints_fresh(tries=1, pause_s=0.0)
                if j2 is None:
                    continue
                fk = self.robot.compute_forward_kinematics(self.arm, j2)
                if fk is not None:
                    joints = j2
                    break
        if fk is None:
            pose = self.get_pose()
            if pose is None:
                print("[plan] ik_after_delta：FK 与 pose 均失败", flush=True)
                return None
            print("[plan] ik_after_delta：FK 失败，改用笛卡尔位姿", flush=True)
            pos, euler = pose
        else:
            pos, euler = fk
        R = euler_rpy_to_matrix(euler.x, euler.y, euler.z)
        d = R @ delta_ee
        return self._ik_at_pose(
            (float(pos.x + d[0]), float(pos.y + d[1]), float(pos.z + d[2])),
            (float(euler.x), float(euler.y), float(euler.z)),
            joints,
            max_jump_rad=max_jump_rad,
        )

    def ik_chain_delta_ee(self, deltas, seed_joints=None, max_jump_rad=None):
        """多段末端系平移增量链式 IK（用于模型预览 ③④）。"""
        if max_jump_rad is None:
            max_jump_rad = self.ik_max_jump_rad()
        if not self.robot:
            return None
        joints = seed_joints if seed_joints is not None else self.read_joints_fresh(tries=3)
        if joints is None:
            return None
        joints = list(joints)
        for delta_ee in deltas:
            nxt = self.ik_after_delta_ee(
                delta_ee, seed_joints=joints, max_jump_rad=max_jump_rad,
            )
            if nxt is None:
                return None
            joints = nxt
        return joints

    def ik_to_pose(self, target_pos, target_rpy, seed_joints=None, max_jump_rad=None):
        """目标 tool0 位姿 → IK（模型预览 ⑥）。"""
        if max_jump_rad is None:
            max_jump_rad = self.ik_max_jump_rad()
        if not self.robot:
            return None
        seed = seed_joints if seed_joints is not None else self.read_joints_fresh(tries=3)
        if seed is None:
            return None
        return self._ik_at_pose(
            target_pos, target_rpy, seed, max_jump_rad=max_jump_rad,
        )

    def plan_pinch_alignment(
        self,
        u,
        v,
        depth_mm,
        pinch_frame_fn,
        simulate_pregrasp=True,
        log=print,
        ee_pose_override=None,
        pinch_info_override=None,
    ):
        """
        仅规划夹取 ⑤⑥（不运动）。
        simulate_pregrasp=True 时按 ③手偏移 + ④预抓靠近 后的姿态估算。
        ee_pose_override / pinch_info_override：模型预览在预抓关节下规划时使用。
        """
        pregrasp_mm = float(self.eye.get("pinch_pregrasp_depth_mm", 200))
        standoff_m = resolve_pinch_standoff_m(
            self.eye, getattr(self, "grasp_size_info", None),
        )
        fallback = float(self.eye.get("approach_fallback_depth_mm", 220))
        if depth_mm is None or not np.isfinite(depth_mm):
            depth_mm = fallback

        if ee_pose_override is not None:
            pos, euler = ee_pose_override
        else:
            pose = self.get_pose()
            if pose is None:
                log("规划失败：无末端位姿（需连臂或读取 pose）")
                return None
            pos, euler = pose
        ee_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float64)

        delta_off = self.pick_hand_offset_delta_ee()
        advance_pre = 0.0
        delta_pre = np.zeros(3, dtype=np.float64)
        if simulate_pregrasp and ee_pose_override is None:
            delta_pre, advance_pre = self.pick_advance_delta_ee(depth_mm, pregrasp_mm)

        delta_sum = delta_off + delta_pre
        R_we = euler_rpy_to_matrix(euler.x, euler.y, euler.z)
        ee_sim = ee_pos + R_we @ delta_sum
        if ee_pose_override is not None:
            ee_sim = ee_pos.copy()

        obj_world = self.object_point_in_world(u, v, depth_mm)
        if obj_world is None:
            log("规划失败：无法算物体 3D 点")
            return None
        log(
            f"⑤ 物体(工作系) [{obj_world[0]:.3f}, {obj_world[1]:.3f}, {obj_world[2]:.3f}] m"
        )

        if pinch_info_override is not None:
            pinch_info = pinch_info_override
        else:
            pinch_info = pinch_frame_fn() if pinch_frame_fn else None
        if pinch_info is None:
            log("规划失败：MuJoCo 捏取 frame 不可用")
            return None
        T_w_pinch_cur = pinch_dict_to_matrix(pinch_info)
        T_w_ee_cur = pose_to_matrix(pos, euler)
        x_hint = np.asarray(pinch_info.get("x_close", [1, 0, 0]), dtype=np.float64)

        _level = resolve_pinch_level_finger_plane(
            self.eye, size_info=getattr(self, "grasp_size_info", None),
        )
        obj_target = pinch_standoff_target(
            obj_world,
            ee_sim,
            standoff_m,
            pinch_info=pinch_info,
            world_up=self.eye.get("pinch_world_up", [0.0, 0.0, 1.0]),
            level_finger_plane=_level,
        )
        if obj_target is None:
            log("规划失败：无法计算 standoff 点")
            return None

        T_w_pinch_des = build_pinch_frame_at_object(
            obj_target,
            ee_sim,
            x_close_hint=x_hint,
            world_up=self.eye.get("pinch_world_up", [0.0, 0.0, 1.0]),
            object_up=self._pinch_object_up(R_we),
            z_tilt_deg=(
                0.0 if _level else float(self.eye.get("pinch_des_z_tilt_deg", 0.0))
            ),
            z_tilt_ref=pinch_info.get("z_approach") if pinch_info else None,
            z_tilt_plane=str(self.eye.get("pinch_des_z_tilt_plane", "world_up")),
            prefer_horizontal_x=bool(self.eye.get("pinch_prefer_horizontal_x", True)),
            level_finger_plane=_level,
        )
        if T_w_pinch_des is None:
            log("规划失败：无法建立目标捏取 frame")
            return None

        if ee_pose_override is not None:
            T_w_ee_des = ee_target_from_pinch(
                T_w_pinch_des, T_w_pinch_cur, T_w_ee_cur,
            )
        else:
            T_w_ee_des = ee_target_from_pinch(
                T_w_pinch_des, T_w_pinch_cur, T_w_ee_cur,
            )
        target_pos, target_rpy = matrix_to_rpy_near(
            T_w_ee_des, ref_rpy=(euler.x, euler.y, euler.z),
        )
        log(
            f"⑥ 对齐捏取 frame → tool0 "
            f"xyz=[{target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}] "
            f"rpy°=[{math.degrees(target_rpy[0]):.1f},"
            f"{math.degrees(target_rpy[1]):.1f},{math.degrees(target_rpy[2]):.1f}]"
        )
        off_cam = self.eye.get("hand_offset_in_camera_m", [0.06, -0.09, 0.0])
        return {
            "obj_world": obj_world,
            "obj_target": obj_target,
            "ee_sim": ee_sim,
            "target_pos": target_pos,
            "target_rpy": target_rpy,
            "T_pinch_des": T_w_pinch_des,
            "depth_mm": float(depth_mm),
            "advance_pre_m": float(advance_pre),
            "delta_off_ee": delta_off,
            "delta_pre_ee": delta_pre,
            "hand_offset_cam": list(off_cam),
        }

    def move_delta_world(self, delta_ee, speed, accel):
        """
        末端系增量 → 单段直线（block 或轮询），失败不重发，避免叠加重规划跳变。
        """
        if delta_ee is None or not np.all(np.isfinite(delta_ee)):
            print("[move] move_delta：增量非法", flush=True)
            return False
        delta_ee = np.asarray(delta_ee, dtype=np.float64).reshape(3)
        max_seg = float(self.eye.get("approach_max_m", 0.28)) + 0.05
        norm = float(np.linalg.norm(delta_ee))
        if norm < 1e-5:
            return True
        if norm > max_seg:
            print(
                f"[move] 单段位移 {norm*1000:.0f}mm 超限（≤{max_seg*1000:.0f}mm），拒绝",
                flush=True,
            )
            return False

        if self._motion_abort:
            return False
        pose = self.get_pose()
        if pose is None:
            print("[move] move_delta：读不到位姿", flush=True)
            return False
        pos, eul = pose
        for val in (pos.x, pos.y, pos.z, eul.x, eul.y, eul.z):
            if not np.isfinite(val):
                print("[move] move_delta：当前位姿非法", flush=True)
                return False
        d = euler_rpy_to_matrix(eul.x, eul.y, eul.z) @ delta_ee
        if not np.all(np.isfinite(d)):
            print("[move] move_delta：目标增量非法", flush=True)
            return False
        target = LbotPosition(
            float(pos.x + d[0]), float(pos.y + d[1]), float(pos.z + d[2])
        )
        dist_m = float(np.linalg.norm(d))
        tol, tol_loose = self._linear_arrive_tol_m(dist_m)
        speed, accel = self._cap_arm_speed_accel(speed, accel)
        if self._is_left_arm():
            # 左臂禁止被 linear_min_speed 抬到 0.18（拧螺丝会冲）
            lin_speed = float(speed)
            lin_accel = float(accel)
            tol = min(tol, float(self._left_motion_cfg()["arrive_tol_m"]))
            tol_loose = max(tol * 2.0, tol + 0.003)
        else:
            lin_speed = max(
                float(speed),
                float(self.eye.get("linear_min_speed", 0.18)),
            )
            lin_accel = max(float(accel), 0.12)

        # 只发一条直线；失败仅轮询/保持，绝不重发（重发易叠加重规划导致关节跳变）
        ok = self.move_to_position(
            target, eul, linear=True, block=True, force=True,
            speed=lin_speed, accel=lin_accel,
        )
        self.refresh_live_state()
        if ok:
            return True

        err = self._pose_dist_m(target)
        if err is not None and err <= tol_loose:
            print(
                f"[move] 直线 block 未确认但距目标 {err*1000:.0f}mm≤"
                f"{tol_loose*1000:.0f}mm，视为到位",
                flush=True,
            )
            return True

        # block 早退但臂可能仍在走：只轮询，不再发第二条直线
        wait_s = self._estimate_linear_wait_s(dist_m, lin_speed)
        print(
            f"[move] 直线 block 未完成，继续等待到位（最长 {wait_s:.0f}s，不重发）…",
            flush=True,
        )
        if self._wait_pose_near(
            target, tol_m=tol, timeout_s=wait_s, poll_s=0.05,
            loose_tol_m=tol_loose,
        ):
            self.refresh_live_state()
            return True

        err = self._pose_dist_m(target)
        if err is not None and err <= tol_loose:
            print(
                f"[move] 直线未到位但距目标 {err*1000:.0f}mm≤"
                f"{tol_loose*1000:.0f}mm，视为到位",
                flush=True,
            )
            return True

        if err is not None:
            print(
                f"[move] 直线失败（仍差 {err*1000:.0f}mm），保持当前位姿，"
                "不再重发（避免关节跳变）",
                flush=True,
            )
        else:
            print("[move] 直线失败，保持当前位姿，不再重发", flush=True)
        self._hold_current_motion()
        return False

    # ── 手 ──────────────────────────────────────────────────────────

    def hand_open(self, force=False):
        """张开：先松 H1/H3–H6，再 H2 侧摆张开（与闭合相反）。"""
        target = [int(np.clip(v, 0, 255)) for v in self.hand_presets["open"]]
        if (
            not force
            and self.hand_cmd is not None
            and list(self.hand_cmd) == target
        ):
            print("[hand] 右手已张开，跳过", flush=True)
            return True
        if not self.robot:
            print("[hand] 未连接机械臂，无法张开")
            return False

        h2_from = self.hand_cmd[1] if self.hand_cmd is not None else target[1]
        step1 = list(target)
        step1[1] = h2_from

        if self.hand_cmd is None or list(self.hand_cmd) != step1:
            if step1 != target:
                if not self._hand_goto(step1, "张开①松指", allow_nudge=False, settle_s=0.35):
                    return False
            else:
                return self._hand_goto(target, "张开", allow_nudge=False)

        if list(self.hand_cmd) != target:
            if not self._hand_goto(target, "张开②拇指侧摆", allow_nudge=False, settle_s=0.0):
                return False
            if abs(h2_from - target[1]) > 3:
                self._hand_wait_channel(1, target[1], from_val=h2_from)

        return True

    def _hand_close_two_step(
        self,
        target,
        step1_label,
        step2_label,
        allow_nudge_step2=True,
        force=False,
        wake_channels=(0, 2, 3),
    ):
        """闭合/捏合：先 H2 拇指横摆到 target 的横摆值，再动其余通道到 target。"""
        target = [int(np.clip(v, 0, 255)) for v in target]
        if not self.robot:
            return False

        h2_close = int(target[1])
        if self.hand_cmd is not None:
            step1 = list(self.hand_cmd)
            h2_from = step1[1]
        else:
            step1 = [int(np.clip(v, 0, 255)) for v in self.hand_presets["open"]]
            h2_from = step1[1]
        step1[1] = h2_close

        need_step1 = force or self.hand_cmd is None or list(self.hand_cmd) != step1
        if need_step1:
            if not self._hand_goto(step1, step1_label, allow_nudge=False, settle_s=0.0):
                return False
            if abs(h2_from - h2_close) > 3:
                if not self._hand_wait_channel(1, h2_close, from_val=h2_from):
                    print("[hand] H2 等待超时，仍尝试闭合其余通道", flush=True)

        # force 时先微动手指通道再闭合，避免二次抓取「仅 H2 动、指不动」
        if force:
            wake = list(step1)
            for i in wake_channels:
                if 0 <= int(i) < 6:
                    wake[int(i)] = int(np.clip(int(wake[int(i)]) - 45, 0, 255))
            if wake != step1:
                self._hand_goto(
                    wake, f"{step2_label}·指唤醒", allow_nudge=False, settle_s=0.20,
                )

        return self._hand_goto(
            target, step2_label, allow_nudge=allow_nudge_step2, settle_s=0.40,
        )

    def hand_grasp(self):
        """闭合到 grasp 预设：先 H2 到 grasp 横摆值，再动其余通道。"""
        if not self.robot:
            print("[hand] 未连接机械臂，无法闭合")
            return False
        grasp = [int(np.clip(v, 0, 255)) for v in self.hand_presets["grasp"]]
        if self.hand_cmd is not None and list(self.hand_cmd) == grasp:
            return True
        return self._hand_close_two_step(
            grasp, "闭合①拇指侧摆", "闭合②", allow_nudge_step2=True,
        )

    def execute_screw_hand_with_z(
        self,
        hand_l6,
        dz_m,
        duration_s=2.5,
        tick_s=0.05,
        log=print,
        should_stop=None,
        label="拧螺丝",
        restore_hand=None,
        z_when="during",
        z_move_s=0.45,
        pre_press_m=0.0,
    ):
        """
        拧螺丝：下发手型保持 duration_s；Z 行程用较短 z_move_s（避免和拧拧时长叠满）。

        z_when:
          - before：先走 Z(dz_m)，再保持手型（上螺丝）
          - after：先读位 → 可选下压 pre_press_m → 保持手型 → 再抬起 dz_m（拆螺丝）
          - during：手型与 Z 同步

        上螺丝通常 dz_m<0；拆螺丝 dz_m>0（抬起量）；pre_press_m>0 表示下压幅度。
        """
        stop = should_stop or (lambda: False)
        if not self.robot:
            log(f"{label}失败：未连接机械臂")
            return False
        if self._motion_abort or stop():
            log(f"{label}：已停止")
            return False
        hand = [int(np.clip(int(v), 0, 255)) for v in list(hand_l6)[:6]]
        while len(hand) < 6:
            hand.append(0)
        # 勿把拧螺丝手型写进 grasp 预设，否则停转/抓握会串台

        z_when = str(z_when or "during").strip().lower()
        if z_when not in ("before", "after", "during"):
            z_when = "during"

        pose = self.get_pose()
        if pose is None:
            log(f"{label}：无当前位姿")
            return False
        pos, eul = pose
        x0, y0, z0 = float(pos.x), float(pos.y), float(pos.z)
        rpy = (float(eul.x), float(eul.y), float(eul.z))
        dz = float(dz_m)
        pre_press = float(max(0.0, pre_press_m))
        duration_s = float(max(duration_s, 0.5))
        z_move_s = float(max(0.15, z_move_s))
        tick_s = float(np.clip(tick_s, 0.03, 0.12))
        z_start = z0
        if z_when == "after":
            phase_txt = (
                f"先下压{pre_press*1000:.0f}mm→拧→上移{abs(dz)*1000:.0f}mm"
                if pre_press > 1e-6
                else "先拧后Z"
            )
        else:
            phase_txt = {"before": "先Z后拧", "during": "同步"}.get(z_when, z_when)
        log(
            f"{label}：手型{hand} 保持{duration_s:.1f}s，"
            f"Z行程{z_move_s:.2f}s/次（{phase_txt}）"
        )

        aborted = False
        z_last = z0
        released = False

        def _resolve_restore():
            restore = restore_hand
            if restore is None:
                restore = list(
                    (self.config.get("hand_presets") or {})
                    .get("left", {})
                    .get("grasp", [255, 0, 100, 100, 45, 50])
                )
            restore = [int(np.clip(int(v), 0, 255)) for v in list(restore)[:6]]
            while len(restore) < 6:
                restore.append(0)
            return restore

        def _follow_z(z_from, z_delta, move_s, hold_hand=False):
            nonlocal aborted, z_last, x0, y0, rpy
            move_s = float(max(0.12, move_s))
            n = max(int(round(move_s / tick_s)), 4)
            t0 = time.time()
            for i in range(1, n + 1):
                if stop() or self._motion_abort:
                    aborted = True
                    log(f"{label}：已停止")
                    return False
                alpha = float(i) / float(n)
                z = z_from + z_delta * alpha
                z_last = z
                ok = self.robot.pose_follow(
                    self.arm,
                    LbotPosition(x0, y0, z),
                    LbotEuler(rpy[0], rpy[1], rpy[2]),
                )
                if not ok:
                    log(f"{label}：Z follow 失败 {self.robot.get_last_error()}")
                    aborted = True
                    return False
                target_t = t0 + move_s * alpha
                while time.time() < target_t:
                    if stop() or self._motion_abort:
                        aborted = True
                        log(f"{label}：已停止")
                        return False
                    time.sleep(0.01)
                if hold_hand and (i == 1 or i == n // 2):
                    self._hand_goto(
                        hand, f"{label}手保持", allow_nudge=False, settle_s=0.0,
                    )
            return True

        def _hold_hand_only():
            """严格按墙钟 duration_s 保持；中途不再反复下发手型。"""
            nonlocal aborted
            log(f"{label}：开始拧/保持 {duration_s:.1f}s")
            self._hand_goto(
                hand, f"{label}手保持", allow_nudge=False, settle_s=0.0,
            )
            deadline = time.time() + duration_s
            while time.time() < deadline:
                if stop() or self._motion_abort:
                    aborted = True
                    log(f"{label}：已停止（保持中）")
                    return False
                time.sleep(0.02)
            log(f"{label}：保持结束，准备收手/上移")
            return True

        def _stop_driver_to_grasp():
            """
            停转+回抓握：只下发一次抓握手型。
            禁止 255 全开 / nudge / force 唤醒（会把手指拧来拧去）。
            """
            nonlocal released
            if released:
                return True
            restore = _resolve_restore()
            log(f"{label}：回抓握停转 {restore}")
            # 单次直达抓握；扳机通道会从按下值抬回抓握值，电批停转
            self._hand_goto(
                restore, f"{label}回抓握", allow_nudge=False, settle_s=0.35,
            )
            self.hand_cmd = list(restore)
            self.hand_presets["grasp"] = list(restore)
            self.hand_presets["pinch_close"] = list(restore)
            released = True
            return True

        def _apply_drive_hand():
            """下发拧螺丝手型：一次到位，不做两步唤醒。"""
            return self._hand_goto(
                hand, f"{label}手型", allow_nudge=False, settle_s=0.25,
            )

        def _settle_quiet():
            try:
                self.settle_after_follow(log=lambda *_a, **_k: None)
            except Exception:
                pass

        def _refresh_pose_xyrpy():
            nonlocal x0, y0, z0, rpy, z_last
            pose2 = self.get_pose()
            if pose2 is None:
                return False
            pos2, eul2 = pose2
            x0 = float(pos2.x)
            y0 = float(pos2.y)
            z0 = float(pos2.z)
            rpy = (float(eul2.x), float(eul2.y), float(eul2.z))
            z_last = z0
            log(f"{label}：读当前位置 Z={z0:.4f}")
            return True

        try:
            if z_when == "before":
                # 上螺丝：先压入 → 拧 → 立刻回抓握停转
                if abs(dz) > 1e-6:
                    if not _follow_z(z0, dz, z_move_s, hold_hand=False):
                        raise RuntimeError("z_before")
                    _settle_quiet()
                    z0 = z_last
                if not _apply_drive_hand():
                    log(f"{label}：手型下发失败")
                    return False
                if not _hold_hand_only():
                    raise RuntimeError("hold")
                _stop_driver_to_grasp()
            elif z_when == "after":
                # 拆螺丝：读位 → 下压 → 拧 → 回抓握停转 → 上移
                if not _refresh_pose_xyrpy():
                    log(f"{label}：重读位姿失败")
                    return False
                if pre_press > 1e-6:
                    log(f"{label}：先下压 {pre_press*1000:.1f}mm")
                    if not _follow_z(z0, -pre_press, z_move_s, hold_hand=False):
                        raise RuntimeError("pre_press")
                    _settle_quiet()
                    z0 = z_last
                if not _apply_drive_hand():
                    log(f"{label}：手型下发失败")
                    return False
                if not _hold_hand_only():
                    raise RuntimeError("hold")
                _stop_driver_to_grasp()
                if abs(dz) > 1e-6 and not aborted:
                    if not _refresh_pose_xyrpy():
                        z0 = z_last
                    log(f"{label}：拧完上移 {abs(dz)*1000:.1f}mm")
                    if not _follow_z(z0, abs(dz), z_move_s, hold_hand=False):
                        raise RuntimeError("z_after")
            else:
                if not _apply_drive_hand():
                    log(f"{label}：手型下发失败")
                    return False
                if abs(dz) > 1e-6:
                    if not _follow_z(z0, dz, max(duration_s, z_move_s), hold_hand=True):
                        raise RuntimeError("z_during")
                else:
                    if not _hold_hand_only():
                        raise RuntimeError("hold")
        except RuntimeError:
            pass
        finally:
            _settle_quiet()
            # 只在前面没停转时补一次；禁止再搞第二套松触发舞蹈
            if not released:
                _stop_driver_to_grasp()
            self.refresh_live_state()

        if aborted:
            log(f"{label}已停 Z≈{z_last:.4f}，已回抓握")
            return False
        log(f"{label}完成 Z {z_start:.4f}→{z_last:.4f}，已回抓握")
        return True

    def hand_open_pinch(self, force=False):
        """
        夹取张手/松手：两步（先松 H1/H3–H6，再 H2 到 pinch_open）。
        force=True 时即使软件态已是张开也强制下发。
        """
        target = [int(np.clip(v, 0, 255)) for v in self.hand_presets["pinch_open"]]
        if (
            not force
            and self.hand_cmd is not None
            and list(self.hand_cmd) == target
        ):
            return True
        if not self.robot:
            print("[hand] 未连接机械臂，无法夹取张开", flush=True)
            return False

        h2_from = (
            self.hand_cmd[1]
            if self.hand_cmd is not None
            else int(self.hand_presets.get("pinch_close", [150, 50])[1])
        )
        step1 = list(target)
        step1[1] = h2_from

        if self.hand_cmd is None or list(self.hand_cmd) != step1:
            if not self._hand_goto(
                step1, "夹取张开①松指", allow_nudge=False, settle_s=0.35,
            ):
                return False

        if list(self.hand_cmd) != target:
            if not self._hand_goto(
                target, "夹取张开②拇指侧摆", allow_nudge=False, settle_s=0.0,
            ):
                return False
            if abs(h2_from - target[1]) > 3:
                self._hand_wait_channel(1, target[1], from_val=h2_from)

        return True

    def hand_pinch_close(self, force=False, stabilize=None):
        """
        主抓闭合。power5：拇 pitch + 中 + 无名（H2=0，食小仍开）；
        pinch：旧三指。force=True 强制走完。
        stabilize：若传入 L6，主抓后再合食+小（⑧b）；None 则仅主抓。
        """
        target = [int(np.clip(v, 0, 255)) for v in self.hand_presets["pinch_close"]]
        mode = str(self.eye.get("pinch_grasp_mode", "power5") or "power5").strip().lower()
        use_power = mode not in ("pinch", "tri", "three", "3")
        if (
            not force
            and stabilize is None
            and self.hand_cmd is not None
            and list(self.hand_cmd) == target
        ):
            return True
        if not self.robot:
            print("[hand] 未连接机械臂，无法捏合", flush=True)
            return False
        wake = (0, 3, 4) if use_power else (0, 2, 3)
        label1 = "五指①拇指侧摆" if use_power else "三指①拇指侧摆"
        label2 = "五指②主抓(拇中无)" if use_power else "三指②闭合"
        ok = self._hand_close_two_step(
            target, label1, label2,
            allow_nudge_step2=True, force=bool(force), wake_channels=wake,
        )
        if not ok:
            return False
        if stabilize is None:
            return True
        return self.hand_pinch_stabilize(stabilize, force=force)

    def hand_pinch_stabilize(self, stabilize=None, force=False):
        """⑧b：在主抓基础上合食指+小指辅助固定。"""
        if stabilize is None:
            stab = self.hand_presets.get("pinch_stabilize")
            if stab is None:
                return True
            stabilize = stab
        target = [int(np.clip(v, 0, 255)) for v in list(stabilize)[:6]]
        if (
            not force
            and self.hand_cmd is not None
            and list(self.hand_cmd) == target
        ):
            return True
        if not self.robot:
            print("[hand] 未连接机械臂，无法辅助固定", flush=True)
            return False
        # 只动食/小：从当前主抓姿态直接到 stabilize（H2 已到位）
        return self._hand_goto(
            target, "五指③食小固定", allow_nudge=bool(force), settle_s=0.35,
        )

    def _hand_est_channel_move_s(self, from_val, to_val):
        """无手反馈时，按行程估算单通道运动时间。"""
        delta = abs(int(to_val) - int(from_val))
        if delta <= 3:
            return 0.0
        return float(np.clip(0.10 + delta / 255.0 * 0.45, 0.10, 1.00))

    def _hand_wait_channel(self, ch, target, from_val=None, tol=8, extra_settle_s=0.10):
        """等待指定 L6 通道到位（按行程估时，实机无位置反馈）。"""
        if from_val is None:
            if self.hand_cmd is not None:
                from_val = self.hand_cmd[ch]
            else:
                from_val = self.hand_presets["open"][ch]
        if abs(int(from_val) - int(target)) <= tol:
            return True
        wait_s = self._hand_est_channel_move_s(from_val, target) + extra_settle_s
        move_print(
            self.eye,
            f"[hand] 等待 H{ch + 1} 到位 ~{wait_s:.2f}s ({from_val}→{target})",
        )
        time.sleep(wait_s)
        return True

    def _hand_goto(self, pose, label, allow_nudge=False, settle_s=0.40):
        """下发 L6 位姿。仅重复同一目标时微抬，避免 SUCCESS 但不转。"""
        if not self.robot:
            print(f"[hand] 未连接机械臂，无法{label}")
            return False
        pose = [int(np.clip(v, 0, 255)) for v in pose]
        try:
            self.robot.l6_set_velocity(self.arm, [220] * 6)
            time.sleep(0.05)
        except Exception as error:
            print(f"[hand] 设速度失败(忽略): {error}", flush=True)

        same = self.hand_cmd is not None and list(self.hand_cmd) == pose
        # 仅「重复同一闭合」时微抬，避免 SUCCESS 但不转
        if allow_nudge and same:
            mid = [int(np.clip(p + (40 if p < 128 else -40), 0, 255)) for p in pose]
            self.robot.l6_set_position(self.arm, mid)
            time.sleep(0.22)

        ok = self.robot.l6_set_position(self.arm, pose)
        self.hand_cmd = list(pose)
        if settle_s > 0:
            time.sleep(settle_s)
        if ok:
            move_print(self.eye, f"[hand] 右手{label} {pose}: SUCCESS")
        else:
            move_print(
                self.eye,
                f"[hand] 右手{label} {pose}: FAILED {self.robot.get_last_error()}",
                important=True,
            )
        return ok

    # ── 完整抓取流程 ────────────────────────────────────────────────
    # execute_pick：旧版光轴夹取
    # execute_pick_pinch：三指捏取主路径（①…⑪）

    def execute_pick(
        self,
        sample_fn,
        log=print,
        speed=None,
        accel=None,
        should_stop=None,
        home_joints=None,
        home_pose=None,
        after_align=None,
    ):
        """
        张手 → 对准 → 退出 follow → 手偏移 → 光轴靠近 → 闭合
        → 退光轴 → 直接回面板/记录关节（不退手偏移）→ 松手
        """
        stop = should_stop or (lambda: False)
        speed = float(speed if speed is not None else self.speed)
        accel = float(accel if accel is not None else self.accel)
        v_approach = float(np.clip(
            speed * float(self.eye.get("approach_speed_scale", 0.55)), 0.08, 0.28
        ))
        a_approach = float(np.clip(accel * 0.55, 0.2, 0.7))
        fallback_depth = float(self.eye.get("approach_fallback_depth_mm", 220))
        align_speed = float(speed) * float(self.eye.get("align_speed_scale", 0.35))
        align_period = float(self.eye.get("align_step_period_s", 0.04))

        if home_joints is None:
            home_joints, home_pose = self.capture_home_state(log=log)

        depth_ref = [None]

        log("① 张手")
        if not self.hand_open(force=True):
            log("张手失败，仍继续对准（手可能已张开）")
        time.sleep(0.15)
        if stop():
            return False

        log("② 对准")
        if not self.align_center_only(
            sample_fn, stop, log=log, speed=align_speed,
            depth_holder=depth_ref, step_period_s=align_period,
        ):
            log("对准失败")
            self.settle_after_follow(log=log)
            return False
        if stop():
            self.settle_after_follow(log=log)
            return False

        # 轻量退出 pose_follow（非阻塞）；随后阻塞直线会再次确保打断
        log("②.5 退出对准跟随")
        self.settle_after_follow(log=log)
        if stop():
            return False

        depth_mm = depth_ref[0]
        sample = sample_fn()
        if after_align is not None and sample is not None:
            u, v, d = sample
            after_align(u, v, depth_ref[0] if depth_ref[0] is not None else d)
        if depth_mm is None:
            depth_mm = sample[2] if sample and sample[2] else fallback_depth
        log(f"规划深度 {float(depth_mm) / 10:.1f}cm")

        off = self.eye.get("hand_offset_in_camera_m", [0.06, -0.09, 0.0])
        if self.pregrasp_apply_hand_offset():
            log(
                f"③ 手偏移 右 {off[0] * 1000:.0f}mm 上 {-off[1] * 1000:.0f}mm（一段直线）"
            )
            delta_off = self.pick_hand_offset_delta_ee()
            if np.linalg.norm(delta_off) > 1e-4:
                log(f"   发送直线 位移 {np.linalg.norm(delta_off)*1000:.0f}mm …")
                if not self.move_delta_world(delta_off, v_approach, a_approach):
                    log("手偏移失败（已停止，未改 IK）")
                    return False
                log("手偏移完成")
            else:
                log("手偏移≈0，跳过")
        else:
            log("③ 手偏移已关闭，跳过")
        if stop():
            return False

        delta_adv, advance_m = self.pick_advance_delta_ee(depth_mm)
        log(f"④ 光轴靠近 {advance_m * 1000:.0f}mm（一段直线）")
        did_advance = advance_m > 1e-4
        if did_advance:
            log(f"   发送直线 位移 {advance_m*1000:.0f}mm …")
            if not self.move_delta_world(delta_adv, v_approach, a_approach):
                log("靠近失败（已停止，未改 IK）")
                return False
            log("光轴靠近完成")
        else:
            log("深度已够近，跳过光轴靠近")
        if stop():
            return False

        log("⑤ 闭合")
        if not self.hand_grasp():
            log("闭合失败，仍尝试退回")
        time.sleep(0.35)
        if stop():
            return False

        # 回程：只退光轴，然后立刻关节回面板目标（中间不做退手偏移）
        if did_advance:
            log(f"⑥ 退光轴 {advance_m * 1000:.0f}mm")
            if not self.move_delta_world(-delta_adv, v_approach, a_approach):
                log("退光轴失败，仍继续回面板关节")
        else:
            log("⑥ 无需退光轴")
        if stop():
            return False

        log(
            "⑦ 直接回面板关节: "
            + ", ".join(f"{math.degrees(j):.1f}°" for j in home_joints)
        )
        self.restore_home(home_joints, home_pose=home_pose, log=log)
        time.sleep(0.25)

        log("⑧ 松手")
        if not self.hand_open(force=True):
            return False
        log("抓取完成")
        return True

    def execute_pick_pinch(
        self,
        sample_fn,
        pinch_frame_fn,
        log=print,
        speed=None,
        accel=None,
        should_stop=None,
        home_joints=None,
        home_pose=None,
        after_align=None,
        lock_object_fn=None,
        get_object_world=None,
        grasp_plan_fn=None,
        tcp_diag_fn=None,
        mj_align_exec_fn=None,
        mj_pose_polish_fn=None,
        mj_contact_exec_fn=None,
        soft_prior_mm=None,
        place_sample_fn=None,
        place_invalidate_fn=None,
    ):
        stop = should_stop or (lambda: False)
        def _diag(phase, plan=None):
            if tcp_diag_fn is None:
                return
            try:
                tcp_diag_fn(phase, plan)
            except Exception as exc:
                log(f"TCP诊断[{phase}]异常: {exc}")
        speed = float(speed if speed is not None else self.speed)
        accel = float(accel if accel is not None else self.accel)
        v_approach = float(np.clip(
            speed * float(self.eye.get("approach_speed_scale", 0.55)), 0.08, 0.28
        ))
        a_approach = float(np.clip(accel * 0.55, 0.2, 0.7))
        pregrasp_mm = float(self.eye.get("pinch_pregrasp_depth_mm", 200))
        standoff_m = resolve_pinch_standoff_m(
            self.eye, getattr(self, "grasp_size_info", None),
        )
        align_speed = float(speed) * float(self.eye.get("align_speed_scale", 0.35))
        align_period = float(self.eye.get("align_step_period_s", 0.04))

        if home_joints is None:
            home_joints, home_pose = self.capture_home_state(log=log)

        if eye_verbose(self.eye):
            self.start_ee_trace(
                log=log,
                period_s=float(self.eye.get("ee_trace_period_s", 0.25)),
            )
        try:
            depth_ref = [None]
            fusion_out = [None, None, None]
            fusion_buffer = None
            if bool(self.eye.get("align_fusion_frames", True)):
                fusion_buffer = AlignFusionBuffer(
                    max_samples=int(self.eye.get("align_fusion_max_samples", 16)),
                    depth_spread_mm=float(self.eye.get("align_fusion_depth_spread_mm", 45)),
                    max_depth_mm=float(self.eye.get("depth_action_max_mm", 300)),
                    min_depth_mm=float(self.eye.get("depth_min_valid_mm", 80)),
                )

            log("① 夹取张手")
            self.set_ee_trace_phase("①张手")
            if not self.hand_open_pinch(force=True):
                log("夹取张手失败，仍继续")
            time.sleep(0.15)
            if stop():
                return False

            log("② 对准")
            self.set_ee_trace_phase("②对准")
            if not self.align_center_only(
                sample_fn, stop, log=log, speed=align_speed,
                depth_holder=depth_ref, step_period_s=align_period,
                fusion_buffer=fusion_buffer, fusion_out=fusion_out,
            ):
                log("对准失败")
                self.settle_after_follow(log=log)
                return False
            if stop():
                self.settle_after_follow(log=log)
                return False

            log("②.5 退出对准跟随")
            self.settle_after_follow(log=log)
            if stop():
                return False

            depth_mm = depth_ref[0]
            u = v = None
            if fusion_out[0] is not None:
                u, v, depth_mm = fusion_out[0], fusion_out[1], fusion_out[2]
            sample = sample_fn()
            if sample is not None and u is None:
                u, v, d = sample[:3]
                if depth_mm is None and d is not None:
                    depth_mm = d
            if sample is None and (u is None or depth_mm is None):
                log("夹取失败：无检测/深度")
                return False
            depth_mm, why_d = sanitize_action_depth_mm(
                depth_mm, self.eye, soft_prior_mm=soft_prior_mm,
            )
            if depth_mm is None:
                log(f"夹取失败：对准深度不可用（{why_d}）")
                return False
            # ② 对准后立刻锁青球（世界系）；③④ 手偏移后画面会偏，不能再靠那时的像素重锁
            if after_align is not None and u is not None and v is not None:
                locked_ok = after_align(u, v, depth_mm)
                if locked_ok is False:
                    log(
                        f"对准后锁定失败 depth={float(depth_mm)/10:.1f}cm，中止"
                    )
                    return False
                log(f"对准后已锁定物体 depth={float(depth_mm)/10:.1f}cm")
            else:
                log(f"对准深度 {float(depth_mm)/10:.1f}cm（未提供锁定回调）")

            # ③④ 预抓：Plan-once 默认跳过（锁物后直接规划⑥⑦）
            self.set_ee_trace_phase("③④预抓")
            skip_pregrasp = bool(self.eye.get("pinch_skip_pregrasp", True))
            if skip_pregrasp:
                log("③④ 跳过（Plan-once：锁物后直接规划/回放）")
            else:
                if self.pregrasp_apply_hand_offset():
                    delta_off_dbg = self.pick_hand_offset_delta_ee()
                    pose_dbg = self.get_pose()
                    if pose_dbg is not None and float(np.linalg.norm(delta_off_dbg)) > 1e-6:
                        _pd, eul_d = pose_dbg
                        R_dbg = euler_rpy_to_matrix(eul_d.x, eul_d.y, eul_d.z)
                        dw = R_dbg @ np.asarray(delta_off_dbg, dtype=np.float64)
                        log(
                            f"③ 手偏移(设计侧移，非残留)：Δee_mm="
                            f"[{delta_off_dbg[0]*1000:.1f},{delta_off_dbg[1]*1000:.1f},"
                            f"{delta_off_dbg[2]*1000:.1f}] "
                            f"世界Δmm=[{dw[0]*1000:.1f},{dw[1]*1000:.1f},{dw[2]*1000:.1f}] "
                            f"→ 随后⑥对齐三指常会侧向回撤一部分"
                        )
                else:
                    log("③ 手偏移已关闭，④ 光轴靠近后 ⑥ 一次对齐")

                ok_pre, u, v, depth_mm = self.execute_pregrasp_phase(
                    sample_fn,
                    stop,
                    log=log,
                    v_approach=v_approach,
                    a_approach=a_approach,
                    initial_depth_mm=depth_mm,
                    lock_object_fn=lock_object_fn,
                )
                if not ok_pre:
                    return False
                if stop():
                    return False

            self.refresh_live_state()

            self.set_ee_trace_phase("⑤⑥规划/对齐")
            plan = None
            if grasp_plan_fn is not None:
                try:
                    plan = grasp_plan_fn(u, v, depth_mm)
                except Exception as exc:
                    log(f"⑤⑥ 规划异常: {exc}")
                if plan is not None and plan.get("robot_ik") is None and not plan.get("plan_once"):
                    log("⑥ MJ→robot 转换失败，刷新后重试…")
                    self.refresh_live_state()
                    try:
                        plan = grasp_plan_fn(u, v, depth_mm)
                    except Exception as exc:
                        log(f"⑤⑥ 规划重试异常: {exc}")
            if plan is not None and plan.get("plan_once"):
                met = plan.get("pick_plan_metrics") or {}
                log(
                    f"⑤ Plan-once 就绪：⑥×{int(met.get('n6', 0))} "
                    f"⑦×{int(met.get('n7', 0))} 路点（回放，不重算）"
                )
            if plan is not None:
                obj_mj = np.asarray(plan["obj_world"], dtype=np.float64).reshape(3)
                obj_r = None
                if get_object_world is not None:
                    obj_r = get_object_world()
                if obj_r is not None:
                    obj_r = np.asarray(obj_r, dtype=np.float64).reshape(3)
                    dbg(
                        log, self.eye,
                        f"⑤ 物体(实机工作系) "
                        f"[{obj_r[0]:.3f}, {obj_r[1]:.3f}, {obj_r[2]:.3f}] m",
                    )
                    dbg(
                        log, self.eye,
                        f"⑤ 物体(MJ规划系) "
                        f"[{obj_mj[0]:.3f}, {obj_mj[1]:.3f}, {obj_mj[2]:.3f}] m",
                    )
                else:
                    dbg(
                        log, self.eye,
                        f"⑤ 物体(MJ规划系·无实机锁) "
                        f"[{obj_mj[0]:.3f}, {obj_mj[1]:.3f}, {obj_mj[2]:.3f}] m",
                    )
                ik = plan.get("robot_ik")
                plan_once = bool(plan.get("plan_once") and plan.get("waypoints_6"))
                if ik is None and not plan_once:
                    log("夹取失败：⑥ 无 robot IK")
                    return False
                if plan_once and ik is None:
                    # Plan-once 回放不依赖笛卡尔 robot_ik；占位供日志
                    tp = plan.get("target_pos")
                    tr = plan.get("target_rpy")
                    if tp is None:
                        tp = np.zeros(3, dtype=np.float64)
                    if tr is None:
                        tr = (0.0, 0.0, 0.0)
                    ik = {
                        "target_pos": tp,
                        "target_rpy": tr,
                        "method": "plan_once_waypoints",
                        "fk_ok": True,
                        "pose_src": "seed",
                    }
                    log("⑥ Plan-once：跳过 robot_ik，直接回放 waypoints")
                tp = ik["target_pos"]
                tr = ik["target_rpy"]
                dbg(
                    log, self.eye,
                    f"⑥ robot_ik method={ik.get('method')} fk_ok={ik.get('fk_ok')} "
                    f"pose_src={ik.get('pose_src')} align_mode={plan.get('align_mode')}",
                )
                pose_now = self.get_pose()
                # 仅平移：目标姿态必须用实时笛卡尔，禁止用 FK 种子里的假 rpy
                method = str(ik.get("method") or "")
                plan_mode = str(plan.get("align_mode") or "")
                translate_only = (
                    "translate" in method
                    or plan_mode in ("translate", "translate_tcp", "pos_only", "position")
                    or plan_mode.endswith("→translate_tcp")
                )
                rot_only = plan_mode in (
                    "rotation_only", "rot_align", "orient", "orientation",
                )
                # rotation_only：法兰必须锁「此刻」位置。规划瞬间若预抓未停稳，
                # 用 seed0 旧 xyz 会把臂拉回去（日志里常见 Δ≈−25mm「回程」）
                if pose_now is not None and rot_only and not plan_once:
                    pos_live, _eul_live = pose_now
                    tp = np.array(
                        [float(pos_live.x), float(pos_live.y), float(pos_live.z)],
                        dtype=np.float64,
                    )
                    log(
                        "⑥ rotation_only：锁定实时法兰 xyz，"
                        f"[{tp[0]:.4f},{tp[1]:.4f},{tp[2]:.4f}]（忽略规划时旧位置）"
                    )
                if pose_now is not None and translate_only and not plan_once:
                    _pn, eul_live = pose_now
                    tr = (float(eul_live.x), float(eul_live.y), float(eul_live.z))
                    log(
                        f"⑥ 仅平移：锁定实时姿态 rpy°="
                        f"[{math.degrees(tr[0]):.1f},{math.degrees(tr[1]):.1f},"
                        f"{math.degrees(tr[2]):.1f}]（忽略 IK 里可能被污染的 rpy）"
                    )
                if pose_now is not None and not plan_once:
                    pos_n, eul_n = pose_now
                    travel = math.sqrt(
                        (float(tp[0]) - pos_n.x) ** 2
                        + (float(tp[1]) - pos_n.y) ** 2
                        + (float(tp[2]) - pos_n.z) ** 2
                    )
                    dxyz = (
                        float(tp[0]) - pos_n.x,
                        float(tp[1]) - pos_n.y,
                        float(tp[2]) - pos_n.z,
                    )
                    dbg(
                        log, self.eye,
                        f"⑥ 执行校验 当前→目标 Δxyz_mm="
                        f"[{dxyz[0]*1000:.1f},{dxyz[1]*1000:.1f},{dxyz[2]*1000:.1f}] "
                        f"行程={travel*1000:.1f}mm",
                    )
                    max_align = float(self.eye.get("pinch_align_max_travel_m", 0.20))
                    if travel > max_align:
                        log(
                            f"夹取失败：⑥ 行程 {travel*1000:.0f}mm > "
                            f"{max_align*1000:.0f}mm（物体锁定/深度可能异常，拒绝乱动）"
                        )
                        return False
                    d_rpy = []
                    for a, b in zip(
                        (tr[0], tr[1], tr[2]),
                        (eul_n.x, eul_n.y, eul_n.z),
                    ):
                        d = (float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi
                        d_rpy.append(abs(math.degrees(d)))
                    max_d_rpy = float(self.eye.get("pinch_align_max_d_rpy_deg", 55))
                    geo = rotation_geodesic_deg(
                        (eul_n.x, eul_n.y, eul_n.z),
                        (tr[0], tr[1], tr[2]),
                    )
                    dbg(
                        log, self.eye,
                        f"⑥ 执行校验 Δrpy°=[{d_rpy[0]:.1f},{d_rpy[1]:.1f},{d_rpy[2]:.1f}] "
                        f"测地={geo:.1f}° 阈值={max_d_rpy:.0f}° mode={plan_mode}",
                    )
                    # 仅在「纯平移模式」或极端超限时砍姿态；auto/full 保留转腕分步执行
                    hard_cap = max(max_d_rpy * 1.5, 75.0)
                    if translate_only:
                        tr = (float(eul_n.x), float(eul_n.y), float(eul_n.z))
                    elif max(d_rpy) > hard_cap or geo > hard_cap:
                        log(
                            f"⑥ 执行降级：Δrpy°=[{d_rpy[0]:.0f},{d_rpy[1]:.0f},{d_rpy[2]:.0f}] "
                            f"测地{geo:.0f}° > 硬限{hard_cap:.0f}°，改 rotation_only 量级"
                            "（保留目标姿态分步，若仍过大再砍）"
                        )
                        if geo > hard_cap * 1.2:
                            log("⑥ 姿态过大，保持当前姿态仅平移防甩腕")
                            tr = (float(eul_n.x), float(eul_n.y), float(eul_n.z))
                            obj_r = None
                            if get_object_world is not None:
                                obj_r = get_object_world()
                            if obj_r is not None:
                                obj_r = np.asarray(obj_r, dtype=np.float64).reshape(3)
                                ee = np.array([pos_n.x, pos_n.y, pos_n.z], dtype=np.float64)
                                vec = obj_r - ee
                                dist = float(np.linalg.norm(vec))
                                if dist > 1e-6:
                                    move_m = dist - float(standoff_m)
                                    if move_m > 1e-4:
                                        move_m = min(move_m, max_align)
                                        tp = ee + vec / dist * move_m
                                    else:
                                        tp = ee
                    elif max(d_rpy) > max_d_rpy or geo > max_d_rpy * 1.2:
                        log(
                            f"⑥ 姿态偏大 Δrpy°=[{d_rpy[0]:.0f},{d_rpy[1]:.0f},{d_rpy[2]:.0f}] "
                            f"测地{geo:.0f}°，分步转腕执行（不降级仅平移）"
                        )
                elif plan_once:
                    met = plan.get("pick_plan_metrics") or {}
                    travel_mm = float(met.get("travel6_mm", 0.0))
                    max_align = float(self.eye.get("pinch_align_max_travel_m", 0.20))
                    log(
                        f"⑥ Plan-once 行程(规划) {travel_mm:.0f}mm "
                        f"路点×{int(met.get('n6', 0))}"
                    )
                    if travel_mm > max_align * 1000.0:
                        log(
                            f"夹取失败：⑥ 规划行程 {travel_mm:.0f}mm > "
                            f"{max_align*1000:.0f}mm（拒绝乱动）"
                        )
                        return False
                log(
                    f"⑥ 三指 TCP 对齐 → tool0 "
                    f"xyz=[{float(tp[0]):.3f},{float(tp[1]):.3f},{float(tp[2]):.3f}] "
                    f"rpy°=[{math.degrees(float(tr[0])):.1f},"
                    f"{math.degrees(float(tr[1])):.1f},{math.degrees(float(tr[2])):.1f}]"
                    + ("（Plan-once 回放）" if plan_once else "")
                )
                v6 = float(np.clip(v_approach * 0.95, 0.10, 0.28))
                a6 = float(np.clip(a_approach * 0.95, 0.18, 0.55))
                path_mode = str(
                    self.eye.get("pinch_path_mode", "mj_ik_smooth")
                ).strip().lower()
                use_mj_align = (
                    mj_align_exec_fn is not None
                    and path_mode not in ("cart", "cartesian")
                )
                if use_mj_align:
                    ok6 = bool(
                        mj_align_exec_fn(
                            plan,
                            speed=v6,
                            accel=a6,
                            log=log,
                            should_stop=stop,
                            target_pos=tp,
                            target_rpy=tr,
                        )
                    )
                else:
                    if path_mode not in ("cart", "cartesian"):
                        log("⑥ 无 mj_align_exec_fn，回退笛卡尔分步")
                    ok6 = self.move_to_pose_stepped(
                        tp, tr,
                        speed=v6, accel=a6, log=log, should_stop=stop,
                        max_step_deg=float(self.eye.get("pinch_align_step_deg", 8)),
                        max_step_m=float(self.eye.get("pinch_align_step_m", 0.025)),
                    )
                if not ok6:
                    if bool(self.eye.get("pinch_align_continue_on_fail", True)):
                        log(
                            "⑥ 对齐未完全到位（IK/软限位），"
                            "仍继续 ⑦ 补进 + ⑧ 闭合（防夹空）"
                        )
                    else:
                        log("三指 TCP 对齐失败")
                        return False
                _diag("⑥对齐后", plan)
                if stop():
                    return False
            else:
                # 无 MuJoCo 规划时的简化回退。有 MJ 时若 Plan-once 失败，禁止走这条：
                # pinch_mid 是 MJ 系、锁物是实机系，混算会得到米级乱目标（曾砸桌 ~1.2m）。
                if mj_align_exec_fn is not None or grasp_plan_fn is not None:
                    log(
                        "夹取失败：Plan-once/MJ 规划不可用，"
                        "拒绝「仅平移」混系回退（防砸桌）"
                    )
                    return False
                obj_world = None
                if get_object_world is not None:
                    obj_world = get_object_world()
                if obj_world is not None:
                    obj_world = np.asarray(obj_world, dtype=np.float64).reshape(3)
                    log(
                        f"⑤ 物体(工作系·锁定) "
                        f"[{obj_world[0]:.3f}, {obj_world[1]:.3f}, {obj_world[2]:.3f}] m"
                    )
                else:
                    sample = sample_fn()
                    if sample is None:
                        log("夹取失败：预抓位无检测")
                        return False
                    u, v, depth_mm = sample
                    if depth_mm is None:
                        depth_mm = pregrasp_mm
                    obj_world = self.object_point_in_world(u, v, depth_mm)
                    if obj_world is None:
                        log("夹取失败：无法算物体点")
                        return False
                    log(
                        f"⑤ 物体(工作系) "
                        f"[{obj_world[0]:.3f}, {obj_world[1]:.3f}, {obj_world[2]:.3f}] m"
                    )

                pose = self.get_pose()
                if pose is None:
                    log("夹取失败：读不到位姿")
                    return False
                pos, euler = pose
                ee_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float64)

                pinch_info = pinch_frame_fn() if pinch_frame_fn else None
                if pinch_info is None:
                    log("夹取失败：三指 frame 不可用")
                    return False
                T_w_pinch_cur = pinch_dict_to_matrix(pinch_info)
                T_w_ee_cur = pose_to_matrix(pos, euler)
                x_hint = np.asarray(pinch_info.get("x_close", [1, 0, 0]), dtype=np.float64)

                _level = resolve_pinch_level_finger_plane(
                    self.eye, size_info=getattr(self, "grasp_size_info", None),
                )
                obj_target = pinch_standoff_target(
                    obj_world,
                    ee_pos,
                    standoff_m,
                    pinch_info=pinch_info,
                    world_up=self.eye.get("pinch_world_up", [0.0, 0.0, 1.0]),
                    level_finger_plane=_level,
                )
                if obj_target is None:
                    log("夹取失败：无法计算 standoff 点")
                    return False

                T_w_pinch_des = build_pinch_frame_at_object(
                    obj_target,
                    ee_pos,
                    x_close_hint=x_hint,
                    world_up=self.eye.get("pinch_world_up", [0.0, 0.0, 1.0]),
                    object_up=self._pinch_object_up(
                        euler_rpy_to_matrix(euler.x, euler.y, euler.z)
                    ),
                    z_tilt_deg=(
                        0.0 if _level else float(self.eye.get("pinch_des_z_tilt_deg", 0.0))
                    ),
                    z_tilt_ref=pinch_info.get("z_approach") if pinch_info else None,
                    z_tilt_plane=str(self.eye.get("pinch_des_z_tilt_plane", "world_up")),
                    prefer_horizontal_x=bool(
                        self.eye.get("pinch_prefer_horizontal_x", True)
                    ),
                    level_finger_plane=_level,
                )
                if T_w_pinch_des is None:
                    log("夹取失败：无法建立目标 frame")
                    return False

                # 无 MJ 规划时：只用平移靠近黄球，避免全姿态甩腕
                # 注意：pinch_mid 与 ee/obj 必须同系；无 MJ 时 mid 应来自实机近似
                close_mid = np.asarray(
                    pinch_info.get("pinch_mid_m", ee_pos), dtype=np.float64
                ).reshape(3)
                delta_tcp = obj_target - close_mid
                target_pos = ee_pos + delta_tcp
                target_rpy = (float(euler.x), float(euler.y), float(euler.z))
                travel_fb = float(np.linalg.norm(target_pos - ee_pos))
                max_align = float(self.eye.get("pinch_align_max_travel_m", 0.20))
                log(
                    f"⑥ 三指 TCP 对齐(回退·仅平移) → tool0 "
                    f"xyz=[{target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}]"
                    f" |Δ|={travel_fb*1000:.0f}mm"
                )
                if travel_fb > max_align:
                    log(
                        f"夹取失败：⑥ 回退行程 {travel_fb*1000:.0f}mm > "
                        f"{max_align*1000:.0f}mm（拒绝乱动）"
                    )
                    return False
                if not self.move_to_pose_stepped(
                    target_pos, target_rpy,
                    speed=v_approach, accel=a_approach, log=log, should_stop=stop,
                ):
                    log("对齐运动失败")
                    return False
                if stop():
                    return False
                plan = {
                    "obj_world": obj_world,
                    "obj_target": obj_target,
                    "T_pinch_des": T_w_pinch_des,
                }

            # ⑦⑨ 补回 standoff / 退回：必须在【实机工作系】朝青球
            # 禁止用 plan["obj_world"](MJ系) 减 ee(robot系) —— 会算反/乱飞
            self.set_ee_trace_phase("⑦补回")
            final_m = self.pick_contact_advance_m()
            path7 = str(
                self.eye.get("pinch_path_mode", "mj_ik_smooth")
            ).strip().lower()
            use_mj7 = (
                mj_contact_exec_fn is not None
                and path7 not in ("cart", "cartesian")
                and final_m > 1e-4
            )
            if use_mj7:
                log(
                    f"⑦ 补回 standoff {final_m*1000:.0f}mm "
                    f"(standoff×frac + extra="
                    f"{resolve_pinch_standoff_m(self.eye, getattr(self, 'grasp_size_info', None))*1000:.0f}×"
                    f"{float(self.eye.get('pinch_contact_frac', 1.0)):.2f}+"
                    f"{float(self.eye.get('pinch_contact_extra_m', 0.025))*1000:.0f}mm)"
                )
                ok7 = bool(
                    mj_contact_exec_fn(
                        plan,
                        final_m,
                        speed=v_approach * 0.75,
                        accel=a_approach * 0.75,
                        log=log,
                        should_stop=stop,
                    )
                )
                if ok7:
                    log("⑦ 关节路点执行完成（看终验|Δ|与★C along）")
                else:
                    # ⑦ 失败时 mid 常仍高数十 cm；强行⑧ 会蹭顶/夹空。默认中止。
                    if bool(self.eye.get("pinch_contact_continue_on_fail", False)):
                        log("⑦ 补回失败，仍继续⑧闭合（pinch_contact_continue_on_fail）")
                    else:
                        log("⑦ 补回失败，中止⑧闭合（防半空合手蹭桌）")
                        return False
            elif final_m > 1e-4:
                log(
                    f"⑦ 补回 standoff {final_m*1000:.0f}mm "
                    f"(standoff×frac + extra="
                    f"{resolve_pinch_standoff_m(self.eye, getattr(self, 'grasp_size_info', None))*1000:.0f}×"
                    f"{float(self.eye.get('pinch_contact_frac', 1.0)):.2f}+"
                    f"{float(self.eye.get('pinch_contact_extra_m', 0.025))*1000:.0f}mm)"
                )
                pose_now = self.get_pose()
                if pose_now is None:
                    log("补回 standoff 跳过：无 pose")
                else:
                    pos_now, eul_now = pose_now
                    ee_pos_now = np.array(
                        [pos_now.x, pos_now.y, pos_now.z], dtype=np.float64,
                    )
                    obj_robot = None
                    src = None
                    if get_object_world is not None:
                        obj_robot = get_object_world()
                        if obj_robot is not None:
                            obj_robot = np.asarray(obj_robot, dtype=np.float64).reshape(3)
                            src = "locked_robot"
                    if obj_robot is None and plan.get("obj_target_robot") is not None:
                        obj_robot = np.asarray(
                            plan["obj_target_robot"], dtype=np.float64,
                        ).reshape(3)
                        src = "plan.obj_target_robot"
                    if obj_robot is None:
                        log(
                            "⑦ 无实机系物体点（get_object_world 空），"
                            "拒绝用 MJ 青球混算方向"
                        )
                    else:
                        toward_unit = approach_unit_toward_object(
                            ee_pos_now, obj_robot, pinch_z_away=None,
                        )
                        if toward_unit is None:
                            log("补回 standoff 跳过：无法确定靠近方向")
                        else:
                            d_vec = obj_robot - ee_pos_now
                            log(
                                f"⑦ 方向(实机系·{src}) ee[{ee_pos_now[0]:.3f},"
                                f"{ee_pos_now[1]:.3f},{ee_pos_now[2]:.3f}] → "
                                f"青球[{obj_robot[0]:.3f},{obj_robot[1]:.3f},"
                                f"{obj_robot[2]:.3f}] "
                                f"Δmm=[{d_vec[0]*1000:.0f},{d_vec[1]*1000:.0f},"
                                f"{d_vec[2]*1000:.0f}] "
                                f"unit=[{toward_unit[0]:+.3f},{toward_unit[1]:+.3f},"
                                f"{toward_unit[2]:+.3f}]"
                            )
                            target_pos = ee_pos_now + toward_unit * final_m
                            locked_rpy = (
                                float(eul_now.x),
                                float(eul_now.y),
                                float(eul_now.z),
                            )
                            step_m = float(self.eye.get("pinch_contact_step_m", 0.012))
                            log(
                                f"⑦ 目标xyz=[{target_pos[0]:.3f},{target_pos[1]:.3f},"
                                f"{target_pos[2]:.3f}] 锁定rpy°="
                                f"[{math.degrees(locked_rpy[0]):.1f},"
                                f"{math.degrees(locked_rpy[1]):.1f},"
                                f"{math.degrees(locked_rpy[2]):.1f}] "
                                f"分步≤{step_m*1000:.0f}mm（cart 回退）"
                            )
                            ok7 = self.move_to_pose_stepped(
                                target_pos,
                                locked_rpy,
                                speed=v_approach * 0.7,
                                accel=a_approach * 0.7,
                                log=log,
                                should_stop=stop,
                                max_step_m=step_m,
                                log_prefix="⑦",
                            )
                            if ok7:
                                log("⑦ 补回到位")
                            else:
                                if bool(
                                    self.eye.get("pinch_contact_continue_on_fail", False)
                                ):
                                    log(
                                        "⑦ 补回失败，仍继续⑧闭合"
                                        "（pinch_contact_continue_on_fail）"
                                    )
                                else:
                                    log("⑦ 补回失败，中止⑧闭合（防半空合手蹭桌）")
                                    return False
            _diag("⑦补回后", plan)
            # 无论⑦是否成功，都必须闭合——加长行程不能跳过⑧
            if stop():
                log("⑦后收到停止，跳过⑧闭合")
                return False

            mode = str(self.eye.get("pinch_grasp_mode", "power5") or "power5").strip().lower()
            use_power = mode not in ("pinch", "tri", "three", "3")
            if use_power:
                log("⑧a 主抓闭合（拇+中+无名，H2=0；食小仍开）")
            else:
                log("⑧ 三指闭合（拇指+食+中）")
            self.set_ee_trace_phase("⑧闭合")
            time.sleep(0.08)
            close_cmd = None
            stab_cmd = None
            if isinstance(plan, dict):
                close_cmd = plan.get("grasp_close_hand") or (
                    (plan.get("metrics") or {}).get("hand_close")
                )
                stab_cmd = plan.get("grasp_stabilize_hand") or (
                    (plan.get("metrics") or {}).get("hand_stabilize")
                )
            if close_cmd is not None:
                self.hand_presets["pinch_close"] = [
                    int(np.clip(int(v), 0, 255)) for v in list(close_cmd)[:6]
                ]
                self.hand_presets["grasp"] = list(self.hand_presets["pinch_close"])
                log(f"⑧a 主抓 L6={self.hand_presets['pinch_close']}")
            if stab_cmd is not None:
                self.hand_presets["pinch_stabilize"] = [
                    int(np.clip(int(v), 0, 255)) for v in list(stab_cmd)[:6]
                ]
            if not self.hand_pinch_close(force=True, stabilize=None):
                log("主抓闭合失败，仍尝试提离")
            else:
                log("⑧a 主抓完成")
                _diag("⑧a主抓后", plan)
                if use_power and self.hand_presets.get("pinch_stabilize"):
                    log(
                        f"⑧b 食指+小指辅助固定 L6="
                        f"{self.hand_presets['pinch_stabilize']}"
                    )
                    if not self.hand_pinch_stabilize(
                        self.hand_presets["pinch_stabilize"], force=True,
                    ):
                        log("⑧b 辅助固定失败，仍继续")
                    else:
                        log("⑧b 辅助固定完成")
            _diag("⑧闭合后", plan)
            if stop():
                return False

            world_lift, (lift_up, lift_right, lift_fwd) = self.pick_lift_delta_world()
            lift_norm = float(np.linalg.norm(world_lift))

            # ⑨ 先提离（不并入回程，便于中途放筐）
            if lift_norm > 1e-4:
                self.set_ee_trace_phase("⑨提离")
                pose0 = self.get_pose()
                if pose0 is not None:
                    pos0, eul0 = pose0
                    p0 = np.array(
                        [float(pos0.x), float(pos0.y), float(pos0.z)],
                        dtype=np.float64,
                    )
                    p_lift = p0 + np.asarray(world_lift, dtype=np.float64).reshape(3)
                    rpy0 = (float(eul0.x), float(eul0.y), float(eul0.z))
                    log(
                        f"⑨ 提离 +{lift_up*1000:.0f}mm上 "
                        f"{lift_right*1000:.0f}mm右 {lift_fwd*1000:.0f}mm前"
                    )
                    self.move_to_pose_stepped(
                        p_lift,
                        rpy0,
                        speed=v_approach * 0.75,
                        accel=a_approach * 0.75,
                        log=log,
                        should_stop=stop,
                        max_step_m=float(self.eye.get("pinch_align_step_m", 0.045)),
                        log_prefix="⑨",
                    )
                else:
                    log("⑨ 无位姿，跳过提离")
            if stop():
                return False

            # 分步放筐：默认夹取到⑨提离后持物停下，由「观察→对准2→放下」继续
            place_inline = bool(self.eye.get("place_inline", False))
            place_hold = bool(self.eye.get("place_hold_after_pick", True))
            place_on = bool(self.eye.get("place_enabled", True))
            if place_on and place_hold and not place_inline:
                log(
                    "⑨ 提离完成（持物停住）。"
                    "下一步：观察 → 对准2 → 放下"
                )
                return True

            # 可选：一次走到底（place_inline=true）
            placed = False
            if place_inline and place_on and place_sample_fn is not None:
                ok_obs = self.execute_place_observe(
                    log=log,
                    should_stop=stop,
                    place_invalidate_fn=place_invalidate_fn,
                    speed=v_approach,
                    accel=a_approach,
                    size_class=(
                        (getattr(self, "grasp_size_info", None) or {}).get("class")
                        if isinstance(getattr(self, "grasp_size_info", None), dict)
                        else getattr(self, "grasp_size_info", None)
                    ),
                )
                if stop():
                    return False
                if ok_obs:
                    placed = self.execute_place_drop(
                        place_sample_fn=place_sample_fn,
                        log=log,
                        should_stop=stop,
                        home_joints=None,
                        home_pose=None,
                        return_home=False,
                        speed=v_approach,
                        accel=a_approach,
                    )

            # ⑩ 回面板（已放筐则空手；未放则持物回再松）
            self.set_ee_trace_phase("⑩回原位")
            log(
                "⑩ 回面板（空手）" if placed
                else "⑩ 回面板（持物：原位正上方→下降→关节）"
            )
            self.restore_home(
                home_joints,
                home_pose=home_pose,
                log=log,
                carry_safe=True,
                prepend_world_delta=None,
                skip_settle=True,
            )
            if not placed:
                self.set_ee_trace_phase("⑪松手")
                log("⑪ 松手")
                if not self.hand_open_pinch(force=True):
                    log("松手失败，仍结束流程")
                time.sleep(0.08)
            log("夹取完成" + ("（已放筐）" if placed else ""))
            return True
        finally:
            self.stop_ee_trace()

    def _place_world_up(self):
        wu = np.asarray(
            self.eye.get("pinch_world_up", [0.0, 0.0, 1.0]),
            dtype=np.float64,
        ).reshape(3)
        wn = float(np.linalg.norm(wu))
        if wn < 1e-9:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return wu / wn

    def _place_level_rpy(self, eul, x_hint=None, R_pinch_in_ee=None):
        """
        放下「指平面水平」目标姿态。

        有 R_pinch_in_ee（pinch 旋转在 tool0 系）时：pinch Z∥世界上
        （与夹取 pinch_level 同语义；法兰会因手安装倾角而斜，约数十度）。

        无 R_pinch_in_ee 时：不再误把 tool0 Z∥竖直当成指平面水平
        （那是观察相机俯视姿）；保持 eul。
        """
        if R_pinch_in_ee is not None:
            return self._place_finger_plane_level_rpy(
                eul, R_pinch_in_ee, x_hint=x_hint,
            )
        return (float(eul.x), float(eul.y), float(eul.z))

    def _place_finger_plane_level_rpy(self, eul, R_pinch_in_ee, x_hint=None):
        """
        pinch Z∥世界上 → tool0 RPY。
        R_ee_des = R_pinch_des @ R_pinch_in_ee.T
        （R_pinch_in_ee = R_ee.T @ R_pinch，闭合手近似刚体）。
        """
        wu = self._place_world_up()
        R_pe = np.asarray(R_pinch_in_ee, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(R_pe)):
            return (float(eul.x), float(eul.y), float(eul.z))
        R_ee = euler_rpy_to_matrix(float(eul.x), float(eul.y), float(eul.z))
        R_pc = R_ee @ R_pe
        z = wu.copy()
        zn = float(np.linalg.norm(z))
        if zn < 1e-9:
            return (float(eul.x), float(eul.y), float(eul.z))
        z = z / zn
        if float(np.dot(z, R_pc[:, 2])) < 0.0:
            z = -z
        if x_hint is not None:
            xh = np.asarray(x_hint, dtype=np.float64).reshape(3)
        else:
            # 保留当前 pinch X 的水平投影（闭合轴航向）
            xh = R_pc[:, 0].copy()
        xh = xh - float(np.dot(xh, z)) * z
        if float(np.linalg.norm(xh)) < 1e-6:
            xh = np.cross(wu, z)
        if float(np.linalg.norm(xh)) < 1e-6:
            xh = R_pc[:, 0]
            xh = xh - float(np.dot(xh, z)) * z
        xn = float(np.linalg.norm(xh))
        if xn < 1e-6:
            return (float(eul.x), float(eul.y), float(eul.z))
        x = xh / xn
        y = np.cross(z, x)
        yn = float(np.linalg.norm(y))
        if yn < 1e-6:
            return (float(eul.x), float(eul.y), float(eul.z))
        y = y / yn
        x = np.cross(y, z)
        x = x / max(float(np.linalg.norm(x)), 1e-9)
        R_pd = np.column_stack([x, y, z])
        R_ee_des = R_pd @ R_pe.T
        return rotation_matrix_to_rpy(R_ee_des)

    def _place_flange_up_rpy(self, eul, x_hint=None):
        """
        旧逻辑：tool0 Z∥世界上（法兰竖直）。仅观察/相机俯视用，不是指平面水平。
        """
        wu = self._place_world_up()
        R = euler_rpy_to_matrix(float(eul.x), float(eul.y), float(eul.z))
        z0 = R[:, 2]
        z = wu.copy()
        zn = float(np.linalg.norm(z))
        if zn < 1e-9:
            return (float(eul.x), float(eul.y), float(eul.z))
        z = z / zn
        if float(np.dot(z, z0)) < 0.0:
            z = -z
        if x_hint is not None:
            xh = np.asarray(x_hint, dtype=np.float64).reshape(3)
            xh = xh - float(np.dot(xh, z)) * z
            if float(np.linalg.norm(xh)) < 1e-6:
                xh = R[:, 0]
        else:
            xh = R[:, 0]
        y = np.cross(z, xh)
        yn = float(np.linalg.norm(y))
        if yn < 1e-6:
            xh = R[:, 1]
            y = np.cross(z, xh)
            yn = float(np.linalg.norm(y))
        if yn < 1e-6:
            return (float(eul.x), float(eul.y), float(eul.z))
        y = y / yn
        x = np.cross(y, z)
        xn = float(np.linalg.norm(x))
        if xn < 1e-6:
            return (float(eul.x), float(eul.y), float(eul.z))
        x = x / xn
        Rn = np.column_stack([x, y, z])
        return rotation_matrix_to_rpy(Rn)

    def execute_place_observe(
        self,
        log=print,
        should_stop=None,
        place_invalidate_fn=None,
        speed=None,
        accel=None,
        size_class=None,
    ):
        """
        观察精简：抬离 → 到位(示教XYZ，夹取腕) → 转腕(默认5步) → 终验。
        """
        stop = should_stop or (lambda: False)
        speed = float(speed if speed is not None else self.speed)
        accel = float(accel if accel is not None else self.accel)
        v = float(np.clip(speed * 0.7, 0.05, 0.22))
        a = float(np.clip(accel * 0.7, 0.12, 0.7))
        wu = self._place_world_up()
        cls = str(size_class or "").strip().lower()
        if cls == "large":
            cls = "big"
        keep_wrist_raw = self.eye.get("place_view_keep_wrist_sizes", []) or []
        keep_wrist_set = {
            str(x).strip().lower() for x in list(keep_wrist_raw) if str(x).strip()
        }
        keep_wrist = bool(cls) and (
            cls in keep_wrist_set or (cls == "big" and "large" in keep_wrist_set)
        )
        skip_orient = bool(keep_wrist) or bool(
            self.eye.get("place_view_skip_final_orient", False)
        )

        def _finish_observe():
            if place_invalidate_fn is not None:
                try:
                    place_invalidate_fn()
                except Exception as exc:
                    log(f"观察：清旧筐框异常: {exc}")
            settle_s = float(
                np.clip(self.eye.get("place_view_settle_s", 0.35), 0.05, 2.0)
            )
            time.sleep(settle_s)
            log("观察完成：请看画面筐/盘框，再点「对准2」")
            return True

        def _parse_view_pose():
            raw = self.eye.get("place_view_pose")
            if raw is None:
                return None
            if isinstance(raw, dict):
                xyz = raw.get("xyz") or raw.get("position")
                rpy = raw.get("rpy") or raw.get("euler") or raw.get("euler_rpy")
            elif isinstance(raw, (list, tuple)) and len(raw) >= 6:
                xyz, rpy = raw[:3], raw[3:6]
            else:
                return None
            if xyz is None or rpy is None or len(list(xyz)) < 3 or len(list(rpy)) < 3:
                return None
            pos = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=np.float64)
            eul = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
            if not (np.all(np.isfinite(pos)) and all(math.isfinite(x) for x in eul)):
                return None
            return pos, eul

        view_pose = _parse_view_pose()
        view_j = self.eye.get("place_view_joints")
        view_joints = None
        if view_pose is None and view_j is not None and len(list(view_j)) >= 7:
            view_joints = [float(x) for x in list(view_j)[:7]]

        near_m = float(np.clip(self.eye.get("place_view_near_m", 0.035), 0.01, 0.12))
        near_tol_deg = float(
            np.clip(self.eye.get("place_view_near_deg", 12.0), 3.0, 25.0)
        )
        clear_lift = float(
            np.clip(self.eye.get("place_view_clear_lift_m", 0.08), 0.03, 0.20)
        )
        # 到位少步：默认约 10cm/段，最多 5 段
        reach_step = float(
            np.clip(self.eye.get("place_view_reach_step_m", 0.10), 0.06, 0.18)
        )
        reach_max_seg = int(
            np.clip(self.eye.get("place_view_reach_max_seg", 5), 2, 8)
        )
        # 转腕固定约 5 步
        orient_n = int(
            np.clip(self.eye.get("place_view_orient_steps", 5), 3, 10)
        )
        jump = float(
            np.clip(self.eye.get("place_view_ik_max_jump_rad", 0.70), 0.35, 1.2)
        )
        v_obs = float(np.clip(v * 0.9, 0.05, 0.20))
        a_obs = float(np.clip(a * 0.9, 0.10, 0.55))
        v_ori = float(np.clip(v_obs * 0.6, 0.035, 0.12))
        a_ori = float(np.clip(a_obs * 0.6, 0.08, 0.35))

        def _read_pose():
            pose = self.get_pose()
            if pose is None:
                return None
            pos, eul = pose
            return (
                np.array([pos.x, pos.y, pos.z], dtype=np.float64),
                (float(eul.x), float(eul.y), float(eul.z)),
            )

        def _joint_to(pos, rpy, prefix, *, jump_lim=None, spd=None, acc=None, soft=False):
            nonlocal p0, r_cur
            pos_t = np.asarray(pos, dtype=np.float64).reshape(3)
            rpy_t = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
            jmax = float(jump if jump_lim is None else jump_lim)
            vv = float(v_obs if spd is None else spd)
            aa = float(a_obs if acc is None else acc)
            d = float(np.linalg.norm(pos_t - p0))
            g = rotation_geodesic_deg(r_cur, rpy_t)
            if d < 0.012 and g < 2.0:
                return True
            seed = self.read_joints_fresh(tries=2, pause_s=0.02)
            if seed is None:
                log(f"{prefix}：读关节失败")
                return False
            ik = self.ik_to_pose(pos_t, rpy_t, seed_joints=seed, max_jump_rad=jmax)
            if ik is None:
                mid_p, mid_r = lerp_pose_rpy(p0, r_cur, pos_t, rpy_t, 0.5)
                ik = self.ik_to_pose(
                    mid_p, mid_r, seed_joints=seed, max_jump_rad=jmax,
                )
                if ik is None:
                    if not soft:
                        log(f"{prefix}：IK 无近解")
                    return False
                self.set_ee_trace_phase(prefix)
                self.execute_joint_motion(ik, speed=vv, accel=aa, block=True)
                got = _read_pose()
                if got is None:
                    return False
                p0, r_cur = got
                if stop():
                    return False
                seed = self.read_joints_fresh(tries=2, pause_s=0.02)
                if seed is None:
                    return False
                ik = self.ik_to_pose(
                    pos_t, rpy_t, seed_joints=seed, max_jump_rad=jmax,
                )
                if ik is None:
                    if not soft:
                        log(f"{prefix}：半步后仍无解")
                    return False
            max_d = max(
                abs(float(ik[k]) - float(seed[k])) for k in range(min(7, len(ik)))
            )
            if max_d > jmax + 1e-3:
                if not soft:
                    log(
                        f"{prefix}：跳变 {math.degrees(max_d):.0f}°"
                        f" > {math.degrees(jmax):.0f}°"
                    )
                return False
            self.set_ee_trace_phase(prefix)
            self.execute_joint_motion(ik, speed=vv, accel=aa, block=True)
            got = _read_pose()
            if got is None:
                return False
            p0, r_cur = got
            err = float(np.linalg.norm(pos_t - p0))
            if err <= 0.05:
                return True
            if soft:
                return err <= 0.08
            if err > 0.08:
                log(f"{prefix}：到位偏大 {err*1000:.0f}mm")
                return False
            return True

        if view_pose is not None:
            tgt_pos, tgt_rpy = view_pose
            cur = _read_pose()
            if cur is None:
                log("观察失败：无当前位姿")
                return False
            p0, r_cur = cur
            tgt_xyz = np.asarray(tgt_pos, dtype=np.float64).reshape(3).copy()
            d0 = float(np.linalg.norm(tgt_xyz - p0))
            g0 = rotation_geodesic_deg(r_cur, tgt_rpy)
            if d0 <= near_m and (skip_orient or g0 <= near_tol_deg):
                log(f"观察：已在位姿（Δ={d0*1000:.0f}mm / {g0:.1f}°）")
                return _finish_observe()

            r_pick = r_cur
            combine = bool(self.eye.get("place_combine_translate_orient", True))
            log(
                f"观察：抬离 → "
                f"{'到位+转腕同段' if (combine and not skip_orient) else '到位 → 转腕'} "
                f"→ 终验 "
                f"目标xyz=[{tgt_xyz[0]:.3f},{tgt_xyz[1]:.3f},{tgt_xyz[2]:.3f}]"
            )

            # 1) 抬离（1～2 段，仍用当前腕）
            z_now = float(np.dot(p0, wu))
            z_clear = min(z_now + clear_lift, float(np.dot(tgt_xyz, wu)))
            if z_clear < z_now + 0.02:
                z_clear = z_now + min(clear_lift, 0.06)
            if z_clear > z_now + 0.015:
                lat = p0 - z_now * wu
                n_up = 2 if (z_clear - z_now) > 0.05 else 1
                log(f"观察抬离 +{(z_clear - z_now)*1000:.0f}mm ×{n_up}")
                for ui in range(1, n_up + 1):
                    if stop():
                        return False
                    zz = z_now + (z_clear - z_now) * (ui / float(n_up))
                    if not _joint_to(lat + zz * wu, r_pick, "观察抬离", soft=True):
                        log("观察抬离：无解，就地继续到位")
                        break

            travel = float(np.linalg.norm(tgt_xyz - p0))
            geo = 0.0 if skip_orient else rotation_geodesic_deg(r_cur, tgt_rpy)
            r_goal = r_cur if skip_orient else tgt_rpy

            if combine and not skip_orient and (travel > 0.02 or geo >= 2.0):
                # 位置+姿态同段插值（比「先到位再原地转腕」少一次停顿、少顶软限位）
                n_pos = int(math.ceil(travel / max(reach_step, 0.06))) if travel > 0.02 else 1
                n_ori = max(orient_n, int(math.ceil(geo / 12.0))) if geo >= 2.0 else 1
                nseg = int(np.clip(max(n_pos, n_ori), 1, max(reach_max_seg, orient_n)))
                log(
                    f"观察合段 |Δ|={travel*1000:.0f}mm 腕差{geo:.0f}° 分 {nseg} 段"
                )
                p_s, r_s = p0.copy(), r_cur
                j_mix = float(np.clip(jump * 1.15, 0.4, 1.15))
                for si in range(1, nseg + 1):
                    if stop():
                        return False
                    t = si / float(nseg)
                    p_i, r_i = lerp_pose_rpy(p_s, r_s, tgt_xyz, r_goal, t)
                    log(
                        f"观察合段 {si}/{nseg} "
                        f"→[{p_i[0]:.3f},{p_i[1]:.3f},{p_i[2]:.3f}] "
                        f"rpy°=[{math.degrees(r_i[0]):.1f},"
                        f"{math.degrees(r_i[1]):.1f},"
                        f"{math.degrees(r_i[2]):.1f}]"
                    )
                    if not _joint_to(
                        p_i, r_i, "观察合段",
                        jump_lim=j_mix,
                        spd=v_obs if geo < 25 else v_ori,
                        acc=a_obs if geo < 25 else a_ori,
                        soft=(si < nseg),
                    ):
                        if si < nseg:
                            continue
                        mid_p, mid_r = lerp_pose_rpy(p0, r_cur, tgt_xyz, r_goal, 0.5)
                        if not _joint_to(mid_p, mid_r, "观察合段半", soft=True):
                            log("观察失败：合段未完成")
                            return False
                        if not _joint_to(tgt_xyz, r_goal, "观察合段", soft=True):
                            if float(np.linalg.norm(tgt_xyz - p0)) > 0.06:
                                log("观察失败：合段偏差过大")
                                return False
            else:
                # 旧路径：先到位再转腕
                if travel > 0.02:
                    nseg = int(math.ceil(travel / max(reach_step, 0.06)))
                    nseg = int(np.clip(nseg, 1, reach_max_seg))
                    log(f"观察到位 |Δ|={travel*1000:.0f}mm 分 {nseg} 段")
                    p_s = p0.copy()
                    for si in range(1, nseg + 1):
                        if stop():
                            return False
                        t = si / float(nseg)
                        p_i = p_s + (tgt_xyz - p_s) * t
                        log(
                            f"观察到位 {si}/{nseg} "
                            f"→[{p_i[0]:.3f},{p_i[1]:.3f},{p_i[2]:.3f}]"
                        )
                        if not _joint_to(p_i, r_pick, "观察到位", soft=(si < nseg)):
                            if si < nseg:
                                continue
                            mid = 0.5 * (p0 + tgt_xyz)
                            if not _joint_to(mid, r_pick, "观察到位半", soft=True):
                                log("观察失败：到位未完成")
                                return False
                            if not _joint_to(tgt_xyz, r_pick, "观察到位", soft=True):
                                if float(np.linalg.norm(tgt_xyz - p0)) > 0.06:
                                    log("观察失败：到位偏差过大")
                                    return False

                if not skip_orient:
                    geo = rotation_geodesic_deg(r_cur, tgt_rpy)
                    if geo >= 2.0:
                        nseg = orient_n
                        log(f"观察转腕 {geo:.1f}° 分 {nseg} 步")
                        r_s = r_cur
                        p_hold = tgt_xyz.copy()
                        j_ori = float(np.clip(jump * 1.1, 0.4, 1.0))
                        for si in range(1, nseg + 1):
                            if stop():
                                return False
                            t = si / float(nseg)
                            _, r_i = lerp_pose_rpy(p_hold, r_s, p_hold, tgt_rpy, t)
                            log(
                                f"观察转腕 {si}/{nseg} "
                                f"rpy°=[{math.degrees(r_i[0]):.1f},"
                                f"{math.degrees(r_i[1]):.1f},"
                                f"{math.degrees(r_i[2]):.1f}]"
                            )
                            ok = _joint_to(
                                p_hold, r_i, "观察转腕",
                                jump_lim=j_ori, spd=v_ori, acc=a_ori,
                                soft=(si < nseg),
                            )
                            if not ok and si == nseg:
                                g_now = rotation_geodesic_deg(r_cur, tgt_rpy)
                                if g_now > max(near_tol_deg, 18.0):
                                    log(f"观察失败：转腕残差 {g_now:.1f}°")
                                    return False

            # 4) 终验
            got = _read_pose()
            if got is None:
                return False
            p0, r_cur = got
            d_e = float(np.linalg.norm(tgt_xyz - p0))
            g_e = rotation_geodesic_deg(r_cur, tgt_rpy)
            log(
                f"观察终验 |Δ|={d_e*1000:.0f}mm 腕差={g_e:.1f}° "
                f"实际=[{p0[0]:.3f},{p0[1]:.3f},{p0[2]:.3f}]"
            )
            if d_e > 0.06:
                log(f"观察失败：终验 XYZ {d_e*1000:.0f}mm")
                return False
            if not skip_orient and g_e > max(near_tol_deg, 18.0):
                log(f"观察失败：终验腕差 {g_e:.1f}°")
                return False
            log(
                f"观察：已到位（"
                f"{'夹取腕' if skip_orient else '示教观察腕'}）"
            )
            return _finish_observe()

        if view_joints is not None:
            if stop():
                return False
            deg = ", ".join(f"{math.degrees(j):.1f}°" for j in view_joints)
            log(f"观察：关节到位 [{deg}]")
            ok_j = self.execute_joint_motion(
                view_joints, speed=v, accel=a, block=True,
            )
            if stop() or not ok_j:
                log("观察失败：关节运动未完成")
                return False
            return _finish_observe()

        view_lift = float(
            np.clip(self.eye.get("place_view_lift_m", 0.08), 0.0, 0.45)
        )
        if view_lift > 1e-4:
            cur = _read_pose()
            if cur is None:
                log("观察失败：无当前位姿")
                return False
            p0, r_cur = cur
            p_up = p0 + wu * view_lift
            log(f"观察：抬高 +{view_lift*1000:.0f}mm")
            ok = self.move_to_position(
                LbotPosition(float(p_up[0]), float(p_up[1]), float(p_up[2])),
                LbotEuler(float(r_cur[0]), float(r_cur[1]), float(r_cur[2])),
                linear=True, block=True, force=True, speed=v, accel=a,
            )
            if stop() or not ok:
                return False
        return _finish_observe()

    def _place_angle_diff(self, a, b):
        return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)

    def _place_rpy_level_yaw(self, eul_ref, yaw_rad, wu=None, R_pinch_in_ee=None):
        """指平面水平，航向 = yaw_rad（世界系 pinch/水平 X 投影）。"""
        wu = self._place_world_up() if wu is None else np.asarray(wu, dtype=np.float64).reshape(3)
        x_h = np.array(
            [math.cos(float(yaw_rad)), math.sin(float(yaw_rad)), 0.0],
            dtype=np.float64,
        )
        x_h = x_h - float(np.dot(x_h, wu)) * wu
        if float(np.linalg.norm(x_h)) < 1e-6:
            return self._place_level_rpy(eul_ref, R_pinch_in_ee=R_pinch_in_ee)
        return self._place_level_rpy(
            eul_ref, x_hint=x_h, R_pinch_in_ee=R_pinch_in_ee,
        )

    def _place_seed_yaw_nudge(self, seed, dyaw_rad):
        """把 J7 往目标航向拧一点，作 IK 种子（搜索大转腕时必需）。"""
        if seed is None:
            return None
        s = [float(x) for x in seed]
        if len(s) < 7:
            return s
        lo, hi = self.soft_joint_limits_rad()
        s[6] = float(np.clip(s[6] + float(dyaw_rad), lo[6], hi[6]))
        return s

    def _place_ik_pair(self, above, drop, rpy, seed, max_jump_rad):
        """上方+下降都要有解；下降以上方解为种子。"""
        rpy_t = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
        ik_a = self.ik_to_pose(
            above, rpy_t, seed_joints=seed, max_jump_rad=max_jump_rad,
        )
        if ik_a is None:
            return None
        ik_d = self.ik_to_pose(
            drop, rpy_t, seed_joints=ik_a, max_jump_rad=max_jump_rad,
        )
        if ik_d is None:
            ik_d = self.ik_to_pose(
                drop, rpy_t, seed_joints=seed, max_jump_rad=max_jump_rad,
            )
        if ik_d is None:
            return None
        return ik_a, ik_d

    def _ee_pos_for_tip(self, tip_world, R_we, tip_in_ee, mode="full"):
        """
        tip 世界坐标 → tool0 位置。
        mode=full：完整 tip→法兰；z_only：法兰 XY=tip XY，只补高度（易解）。
        """
        tip_w = np.asarray(tip_world, dtype=np.float64).reshape(3)
        tip_ee = np.asarray(tip_in_ee, dtype=np.float64).reshape(3)
        off = np.asarray(R_we, dtype=np.float64).reshape(3, 3) @ tip_ee
        if str(mode).lower() in ("z", "z_only", "height"):
            wu = self._place_world_up()
            return tip_w - float(np.dot(off, wu)) * wu
        return tip_w - off

    def _place_reach_pose(
        self,
        target_pos,
        target_rpy,
        *,
        log=print,
        should_stop=None,
        speed=None,
        accel=None,
        log_prefix="放下",
        prefer_joint=True,
        max_jump_rad=None,
        hold_after=True,
        allow_jump_widen=True,
    ):
        """
        到位：prefer_joint 时 IK→关节 PTP，失败再笛卡尔分步。
        hold_after=False：不打断在途（观察升高禁用 hold，否则会假到位）。
        allow_jump_widen=False：禁止把 max_jump 放宽到 2.8（避免 136° 甩腕翻成水平）。
        """
        stop = should_stop or (lambda: False)
        pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        rpy = (
            float(target_rpy[0]),
            float(target_rpy[1]),
            float(target_rpy[2]),
        )
        if max_jump_rad is None:
            max_jump_rad = float(
                np.clip(
                    self.eye.get("place_ik_max_jump_rad", self.ik_max_jump_rad()),
                    0.5,
                    3.5,
                )
            )
        if prefer_joint and bool(self.eye.get("place_prefer_joint", True)):
            seed = self.read_joints_fresh(tries=2, pause_s=0.02)
            ik = self.ik_to_pose(pos, rpy, seed_joints=seed, max_jump_rad=max_jump_rad)
            if ik is None and seed is not None and allow_jump_widen:
                # 放宽跳变再试（放下大行程）；观察禁用，否则关节弧会把相机甩成水平
                ik = self.ik_to_pose(
                    pos, rpy, seed_joints=seed,
                    max_jump_rad=max(max_jump_rad, 2.8),
                )
            if ik is not None:
                log(f"{log_prefix}：IK 关节到位")
                ok = self.execute_joint_motion(
                    ik, speed=speed, accel=accel, block=True,
                )
                if hold_after:
                    self._hold_current_motion(f"[{log_prefix}]")
                if stop():
                    return False
                return bool(ok)
            log(f"{log_prefix}：IK 无解，改笛卡尔")
        ok = self.move_to_pose_stepped(
            pos, rpy,
            speed=speed, accel=accel,
            log=log, should_stop=stop,
            max_step_m=float(self.eye.get("pinch_align_step_m", 0.045)),
            log_prefix=log_prefix,
        )
        if hold_after:
            self._hold_current_motion(f"[{log_prefix}]")
        return bool(ok)

    def execute_place_drop(
        self,
        place_sample_fn,
        log=print,
        should_stop=None,
        home_joints=None,
        home_pose=None,
        return_home=True,
        speed=None,
        accel=None,
        locked_uvd=None,
        locked_xyz=None,
        tip_in_ee=None,
        pinch_R_in_ee=None,
        thumb_in_ee=None,
        size_class=None,
        place_kind=None,
    ):
        """
        放下：夹取末端(mid=外接圆心)到目标正上方 → 下降 → 指平面水平 → 松手。
        TCP 不是拇指；拇指相对 mid 沿闭合轴偏开约一圈半径。

        place_kind:
          - "plate"：盘装配 mid 上方 → 装配高 → 松手 → 抬 → 回面板
          - "basket"/其它：筐三格口沿路径（默认）

        locked_xyz：对准2 锁的世界坐标（优先，避免臂动后 UV 漂）。
        pinch_R_in_ee：闭合 pinch 旋转在 tool0 系；有则真正做三指平面∥水平。
        thumb_in_ee：可选，仅诊断「拇是否压在盘心竖直线上」。
        """
        stop = should_stop or (lambda: False)
        kind = str(place_kind or "basket").strip().lower()
        is_plate = kind in ("plate", "disk", "pan")
        speed = float(speed if speed is not None else self.speed)
        accel = float(accel if accel is not None else self.accel)
        v_approach = float(np.clip(
            speed * float(self.eye.get("approach_speed_scale", 0.55)), 0.08, 0.28
        ))
        a_approach = float(np.clip(accel * 0.7, 0.15, 0.8))
        wu = self._place_world_up()

        bin_pt = None
        u_p = v_p = d_p = None
        if locked_xyz is not None:
            bin_pt = np.asarray(locked_xyz, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(bin_pt)):
                bin_pt = None
            else:
                cls_s = f" class={size_class}" if size_class else ""
                tag = "盘心" if is_plate else "锁格"
                log(
                    f"放下：{tag}世界点 "
                    f"[{bin_pt[0]:.3f},{bin_pt[1]:.3f},{bin_pt[2]:.3f}]{cls_s}"
                )
        if bin_pt is None and locked_uvd is not None and len(locked_uvd) >= 3:
            u_p, v_p, d_p = locked_uvd[0], locked_uvd[1], locked_uvd[2]
            cls_s = f" class={size_class}" if size_class else ""
            tag = "盘心" if is_plate else "目标格"
            log(
                f"放下：{tag} u={float(u_p):.0f} v={float(v_p):.0f} "
                f"depth={float(d_p)/10:.1f}cm{cls_s}"
            )
        elif bin_pt is None and place_sample_fn is not None:
            try:
                sample = place_sample_fn()
            except Exception as exc:
                log(f"放下：采样异常 {exc}")
                sample = None
            if sample is not None and len(sample) >= 3:
                u_p, v_p, d_p = sample[0], sample[1], sample[2]

        if bin_pt is None:
            require = bool(self.eye.get("place_require_detection", True))
            if u_p is None or v_p is None or d_p is None:
                log(
                    (
                        "放下失败：无盘检测/深度（请先观察并对准2）"
                        if is_plate
                        else "放下失败：无筐检测/深度（请先观察并对准2）"
                    )
                    if require
                    else "放下：无目标，跳过"
                )
                return False
            eye_place = dict(self.eye)
            eye_place["depth_action_max_mm"] = float(
                self.eye.get(
                    "place_depth_max_mm",
                    max(800.0, float(self.eye.get("depth_action_max_mm", 300))),
                )
            )
            eye_place["depth_action_min_mm"] = float(
                self.eye.get(
                    "place_depth_min_mm", self.eye.get("depth_min_valid_mm", 80)
                )
            )
            d_p, why_p = sanitize_action_depth_mm(d_p, eye_place, soft_prior_mm=None)
            if d_p is None:
                log(f"放下失败：深度不可用（{why_p}）")
                return False
            bin_pt = self.object_point_in_world(float(u_p), float(v_p), float(d_p))
            if bin_pt is None:
                log("放下失败：3D 点失败")
                return False
            bin_pt = np.asarray(bin_pt, dtype=np.float64).reshape(3)

        pose_c = self.get_pose()
        if pose_c is None:
            log("放下失败：无当前位姿")
            return False
        pos_c, eul_c = pose_c
        rpy_keep = (float(eul_c.x), float(eul_c.y), float(eul_c.z))
        R_pe = None
        if pinch_R_in_ee is not None:
            R_pe = np.asarray(pinch_R_in_ee, dtype=np.float64).reshape(3, 3)
            if not np.all(np.isfinite(R_pe)):
                R_pe = None
        level_on = resolve_place_level_finger_plane(
            self.eye, size_class=size_class,
        )
        if level_on and R_pe is None:
            log(
                "放下：无 pinch_R_in_ee，无法做三指平面∥水平"
                "（观察腕=法兰竖直≠指平面水平）；保持当前腕"
            )
            level_on = False
        if level_on:
            rpy_level = self._place_level_rpy(eul_c, R_pinch_in_ee=R_pe)
            R_lv = euler_rpy_to_matrix(*rpy_level)
            fl_tilt = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            abs(float(np.dot(R_lv[:, 2], self._place_world_up()))),
                            0.0,
                            1.0,
                        )
                    )
                )
            )
            geo_lv = rotation_geodesic_deg(rpy_keep, rpy_level)
            log(
                f"放下：指平面 level（pinch Z∥重力；"
                f"法兰相对竖直≈{fl_tilt:.0f}°，相对观察腕 {geo_lv:.0f}°）"
            )
        else:
            rpy_level = rpy_keep

        use_tip = bool(self.eye.get("place_use_tip_tcp", True))
        tip_ee = None
        if use_tip and tip_in_ee is not None:
            tip_ee = np.asarray(tip_in_ee, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(tip_ee)):
                tip_ee = None
        if use_tip and tip_ee is None:
            approx = float(self.eye.get("place_tip_approx_m", 0.120))
            if approx > 1e-4:
                tip_ee = np.array([0.0, 0.0, -abs(approx)], dtype=np.float64)
                log(f"放下：无 MJ tip，用近似 tip_ee z=-{approx*1000:.0f}mm")

        if is_plate:
            tip_standoff = float(
                np.clip(
                    self.eye.get("plate_tip_standoff_m", 0.15),
                    0.05,
                    0.35,
                )
            )
            tip_clear = float(
                np.clip(
                    self.eye.get("plate_tip_clearance_m", 0.11),
                    0.02,
                    0.25,
                )
            )
            post_lift = float(
                np.clip(
                    self.eye.get(
                        "plate_post_release_lift_m", tip_standoff,
                    ),
                    0.05,
                    0.35,
                )
            )
            apply_bias = True
            log(
                f"放下(盘装配)：mid→盘心上 {tip_standoff*1000:.0f}mm → "
                f"{tip_clear*1000:.0f}mm 松手 → 抬 {post_lift*1000:.0f}mm"
            )
        else:
            tip_standoff = float(
                np.clip(self.eye.get("place_tip_standoff_m", 0.080), 0.02, 0.35)
            )
            tip_clear = float(
                np.clip(self.eye.get("place_tip_clearance_m", 0.005), 0.0, 0.08)
            )
            post_lift = 0.0
            apply_bias = True
        tip_ee_raw = None
        tip_above = bin_pt + wu * tip_standoff
        tip_drop = bin_pt + wu * tip_clear

        # 世界前/右（与 place_slot_forward、夹取 lift 同口径：前=+fwd，右=+cross）
        fwd = np.asarray(
            self.eye.get("place_slot_forward", [1.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        fn = float(np.linalg.norm(fwd))
        fwd = fwd / fn if fn > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fwd = fwd - float(np.dot(fwd, wu)) * wu
        fn = float(np.linalg.norm(fwd))
        fwd = fwd / fn if fn > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right = np.cross(fwd, wu)
        rn = float(np.linalg.norm(right))
        right = right / rn if rn > 1e-9 else np.array([0.0, -1.0, 0.0], dtype=np.float64)

        # 放下目标侧偏（正=往右/往前移格心；纠「偏左前」→ right>0 且 forward<0）
        def _place_bias_m(key_base, key_by, default=0.0):
            base = float(self.eye.get(key_base, default) or 0.0)
            by = self.eye.get(key_by) or {}
            if isinstance(by, dict) and size_class:
                cls = str(size_class).strip().lower()
                keys = ["big", "large"] if cls in ("big", "large") else [cls]
                for k in keys:
                    if k in by and by[k] is not None:
                        try:
                            return float(by[k])
                        except (TypeError, ValueError):
                            break
            return base

        bias_r = bias_f = drop_down = 0.0
        if apply_bias:
            if is_plate:
                # 与筐同一类 tip 模型误差（实指相对 MJ 偏左前）；盘单独可调
                share = bool(self.eye.get("plate_apply_bias", True))
                def_r = float(self.eye.get("place_bias_right_m", 0.10) or 0.0) if share else 0.0
                def_f = float(self.eye.get("place_bias_forward_m", -0.05) or 0.0) if share else 0.0
                bias_r = float(
                    np.clip(
                        _place_bias_m(
                            "plate_bias_right_m",
                            "plate_bias_right_by_size",
                            default=def_r,
                        ),
                        -0.15,
                        0.15,
                    )
                )
                bias_f = float(
                    np.clip(
                        _place_bias_m(
                            "plate_bias_forward_m",
                            "plate_bias_forward_by_size",
                            default=def_f,
                        ),
                        -0.15,
                        0.15,
                    )
                )
            else:
                bias_r = float(np.clip(_place_bias_m("place_bias_right_m", "place_bias_right_by_size"), -0.15, 0.15))
                bias_f = float(np.clip(_place_bias_m("place_bias_forward_m", "place_bias_forward_by_size"), -0.15, 0.15))
            drop_down = float(
                np.clip(float(self.eye.get("place_drop_down_m", 0.0) or 0.0), 0.0, 0.04)
            )
            if abs(bias_r) > 1e-5 or abs(bias_f) > 1e-5 or drop_down > 1e-5:
                shift = right * bias_r + fwd * bias_f - wu * drop_down
                bin_pt = bin_pt + shift
                tip_above = tip_above + shift
                tip_drop = tip_drop + shift
                tag = "盘心" if is_plate else "格心"
                log(
                    f"放下：目标侧偏 右{bias_r*1000:.0f}mm 前{bias_f*1000:.0f}mm "
                    f"再降{drop_down*1000:.0f}mm "
                    f"→{tag} [{bin_pt[0]:.3f},{bin_pt[1]:.3f},{bin_pt[2]:.3f}]"
                )
            elif is_plate:
                log(
                    "放下(盘)：侧偏=0（若实放偏左前，调 plate_bias_right_m>0 / "
                    "plate_bias_forward_m<0）"
                )

        tip_ee_raw = None
        if tip_ee is not None:
            tip_ee_raw = tip_ee.copy()
            size_info = None
            if size_class:
                size_info = {"class": str(size_class).strip().lower()}
            elif isinstance(getattr(self, "grasp_size_info", None), dict):
                size_info = self.grasp_size_info
            short_m = resolve_pinch_hand_short_m(self.eye, size_info)
            short_m += float(
                np.clip(float(self.eye.get("place_tip_short_extra_m", 0.0) or 0.0), 0.0, 0.02)
            )
            L = float(np.linalg.norm(tip_ee))
            # 各向同性缩 tip：使实 mid≈目标。过缩会沿 tip 世界「前+/右-」→偏前偏左。
            # 上限 30%|tip|；勿再降口沿高度（会与手短双重补偿）。
            short_m = min(short_m, max(0.0, 0.30 * L))
            if short_m > 1e-4 and L > short_m + 1e-3:
                tip_ee = tip_ee * ((L - short_m) / L)
                log(
                    f"放下：tip_ee 手短 {short_m*1000:.0f}mm "
                    f"|tip| {L*1000:.0f}→{float(np.linalg.norm(tip_ee))*1000:.0f}mm"
                    f"（口沿仍+{tip_clear*1000:.0f}mm，不挖筐）"
                )

        # 对准后 mid 常已低于「接近高」：若仍强制去 tip_standoff 会先往上抬（高观察调低后更明显）
        if tip_ee is not None:
            R_now = euler_rpy_to_matrix(
                float(eul_c.x), float(eul_c.y), float(eul_c.z),
            )
            mid_now = (
                np.array(
                    [float(pos_c.x), float(pos_c.y), float(pos_c.z)],
                    dtype=np.float64,
                )
                + R_now @ tip_ee
            )
            h_now = float(np.dot(mid_now - bin_pt, wu))
            h_floor = float(tip_clear) + 0.008
            if h_now < float(tip_standoff) - 0.005 and h_now > h_floor:
                cfg_standoff = float(tip_standoff)
                tip_standoff = float(h_now)
                tip_above = bin_pt + wu * tip_standoff
                log(
                    f"放下：当前 mid 已在目标上 {h_now*1000:.0f}mm"
                    f"（低于接近高 {cfg_standoff*1000:.0f}mm），"
                    f"不再抬升，直接降到松手 {tip_clear*1000:.0f}mm"
                )
            elif h_now <= h_floor:
                tip_standoff = max(float(h_now), h_floor)
                tip_above = bin_pt + wu * tip_standoff
                log(
                    f"放下：当前 mid 已近松手高({h_now*1000:.0f}mm)，跳过抬升"
                )

        # ── tip TCP 选姿（保守）──
        # 上次翻车：以「朝向筐」+最短伸距+搜跳180° → 选出转腕143°。
        # 正确：以上下都可解为前提，优先 tip 竖直（侧向小）再腕差小。
        rpy_place = rpy_level if level_on else rpy_keep
        above = drop = None
        tip_mode = None
        alt_plans = []
        lower = tip_standoff - tip_clear
        jump = float(np.clip(self.eye.get("place_ik_max_jump_rad", 2.8), 0.8, 3.5))
        # 搜索略宽于执行，但禁止 π（会接受翻腕奇异解）
        search_jump = float(
            np.clip(
                self.eye.get("place_ik_search_jump_rad", 2.6),
                jump,
                2.9,
            )
        )
        max_orient_deg = float(
            np.clip(self.eye.get("place_max_orient_deg", 70.0), 20.0, 100.0)
        )
        max_tip_lat = float(
            np.clip(float(self.eye.get("place_tip_max_lat_m", 0.085) or 0.085), 0.03, 0.15)
        )
        yaw_search = self.eye.get("place_tip_yaw_search_deg")
        if not isinstance(yaw_search, (list, tuple)) or not yaw_search:
            yaw_search = [0, -10, 10, -20, 20, -30, 30, -45, 45, -60, 60]

        p0 = np.array([pos_c.x, pos_c.y, pos_c.z], dtype=np.float64)
        seed0 = self.read_joints_fresh(tries=2, pause_s=0.02)
        # 航向基准 = 当前调平腕，禁止用「朝向筐」当 0（易偏 90°）
        yaw0 = float(rpy_place[2])

        yaw_dedup = []
        for dy in yaw_search:
            y = yaw0 + math.radians(float(dy))
            if all(self._place_angle_diff(y, u) > math.radians(2.0) for u in yaw_dedup):
                yaw_dedup.append(float(y))

        def _plan_one(rpy, mode):
            geo = rotation_geodesic_deg(rpy_keep, rpy)
            if geo > max_orient_deg + 1e-6:
                return None
            R = euler_rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            tip_lat = 0.0
            if tip_ee is not None:
                tip_w = R @ tip_ee
                tip_lat = float(np.linalg.norm(tip_w - float(np.dot(tip_w, wu)) * wu))
                if tip_lat > max_tip_lat + 1e-6:
                    return None
            a = self._ee_pos_for_tip(tip_above, R, tip_ee, mode=mode)
            d = self._ee_pos_for_tip(tip_drop, R, tip_ee, mode=mode)
            if seed0 is None:
                return None
            dyaw = self._place_angle_diff(rpy[2], rpy_keep[2])
            seeds = [
                seed0,
                self._place_seed_yaw_nudge(seed0, dyaw),
                self._place_seed_yaw_nudge(seed0, -dyaw),
            ]
            pair = None
            for sd in seeds:
                if sd is None:
                    continue
                pair = self._place_ik_pair(a, d, rpy, sd, search_jump)
                if pair is not None:
                    break
            if pair is None:
                return None
            reach = float(np.linalg.norm(a - float(np.dot(a, wu)) * wu))
            return {
                "mode": str(mode),
                "above": a,
                "drop": d,
                "rpy": tuple(float(x) for x in rpy),
                "reach": reach,
                "dyaw": dyaw,
                "geo": geo,
                "tip_lat": tip_lat,
            }

        if tip_ee is not None:
            candidates = []
            for mode in ("full", "z_only"):
                for yaw in yaw_dedup:
                    if level_on:
                        rpy_try = self._place_rpy_level_yaw(
                            eul_c, yaw, wu=wu, R_pinch_in_ee=R_pe,
                        )
                    else:
                        rpy_try = (
                            float(rpy_keep[0]), float(rpy_keep[1]), float(yaw),
                        )
                    hit = _plan_one(rpy_try, mode)
                    if hit is not None:
                        candidates.append(hit)
                if any(c["mode"] == "full" for c in candidates):
                    break
            # 侧向过严无解：放宽再搜一次
            if not candidates:
                for mode in ("full", "z_only"):
                    for yaw in yaw_dedup:
                        if level_on:
                            rpy_try = self._place_rpy_level_yaw(
                                eul_c, yaw, wu=wu, R_pinch_in_ee=R_pe,
                            )
                        else:
                            rpy_try = (
                                float(rpy_keep[0]), float(rpy_keep[1]), float(yaw),
                            )
                        geo = rotation_geodesic_deg(rpy_keep, rpy_try)
                        if geo > max_orient_deg + 1e-6:
                            continue
                        R = euler_rpy_to_matrix(
                            float(rpy_try[0]), float(rpy_try[1]), float(rpy_try[2]),
                        )
                        tip_w = R @ tip_ee
                        tip_lat = float(
                            np.linalg.norm(tip_w - float(np.dot(tip_w, wu)) * wu)
                        )
                        a = self._ee_pos_for_tip(tip_above, R, tip_ee, mode=mode)
                        d = self._ee_pos_for_tip(tip_drop, R, tip_ee, mode=mode)
                        if seed0 is None:
                            break
                        dyaw = self._place_angle_diff(rpy_try[2], rpy_keep[2])
                        pair = None
                        for sd in (
                            seed0,
                            self._place_seed_yaw_nudge(seed0, dyaw),
                            self._place_seed_yaw_nudge(seed0, -dyaw),
                        ):
                            if sd is None:
                                continue
                            pair = self._place_ik_pair(a, d, rpy_try, sd, search_jump)
                            if pair is not None:
                                break
                        if pair is None:
                            continue
                        reach = float(np.linalg.norm(a - float(np.dot(a, wu)) * wu))
                        candidates.append(
                            {
                                "mode": str(mode),
                                "above": a,
                                "drop": d,
                                "rpy": tuple(float(x) for x in rpy_try),
                                "reach": reach,
                                "dyaw": dyaw,
                                "geo": geo,
                                "tip_lat": tip_lat,
                            }
                        )
                    if any(c["mode"] == "full" for c in candidates):
                        break
                if candidates:
                    log(
                        f"放下：tip 侧向≤{max_tip_lat*1000:.0f}mm 无解，"
                        f"已放宽（仍优先侧向小）"
                    )
            if candidates:
                # 优先：full → tip更竖直 → 腕差小 → 伸距短
                candidates.sort(
                    key=lambda c: (
                        0 if c["mode"] == "full" else 1,
                        float(c.get("tip_lat", 0.0)),
                        c["geo"],
                        c["dyaw"],
                        c["reach"],
                    )
                )
                alt_plans = candidates[:5]
                best = candidates[0]
                tip_mode = best["mode"]
                above = best["above"]
                drop = best["drop"]
                rpy_place = best["rpy"]
                lower = float(np.linalg.norm(above - drop))
                # rotation_geodesic_deg 已返回度，勿再 math.degrees
                geo_deg = float(best["geo"])
                label = f"tip/{tip_mode}"
                if geo_deg >= 2.0:
                    label += f"+转腕{geo_deg:.0f}°"
                log(
                    f"放下：夹取末端(mid)→正上方 {tip_standoff*1000:.0f}mm"
                    f"（{label}，候选{len(candidates)}，"
                    f"腕差上限{max_orient_deg:.0f}°，"
                    f"tip侧向{float(best.get('tip_lat', 0))*1000:.0f}mm），"
                    f"再降到{'装配高' if is_plate else '口沿+'}{tip_clear*1000:.0f}mm"
                )
                log(
                    f"放下目标 tool0 [{above[0]:.3f},{above[1]:.3f},{above[2]:.3f}] "
                    f"tip_ee|Δ|={float(np.linalg.norm(tip_ee))*1000:.0f}mm "
                    f"伸距{best['reach']*1000:.0f}mm（非法兰）"
                )
                # 诊断：mid 目标相对格心；tip 世界偏置拆前/右/上（查「偏左前」）
                R_pl = euler_rpy_to_matrix(
                    float(rpy_place[0]), float(rpy_place[1]), float(rpy_place[2]),
                )
                tip_w_off = R_pl @ tip_ee  # tool0→mid
                mid_tgt = above + tip_w_off
                d_mid = mid_tgt - bin_pt
                log(
                    f"放下★格心 [{bin_pt[0]:.3f},{bin_pt[1]:.3f},{bin_pt[2]:.3f}] "
                    f"mid目标(上方) [{mid_tgt[0]:.3f},{mid_tgt[1]:.3f},{mid_tgt[2]:.3f}] "
                    f"mid−格 前{float(np.dot(d_mid, fwd))*1000:.0f} "
                    f"右{float(np.dot(d_mid, right))*1000:.0f} "
                    f"上{float(np.dot(d_mid, wu))*1000:.0f}mm "
                    f"(期望 前≈0 右≈0 上≈{tip_standoff*1000:.0f})"
                )
                if thumb_in_ee is not None:
                    th_ee = np.asarray(thumb_in_ee, dtype=np.float64).reshape(3)
                    if np.all(np.isfinite(th_ee)):
                        # tool0=above → 拇世界 = above + R@thumb_ee
                        thumb_tgt = above + R_pl @ th_ee
                        d_th = thumb_tgt - bin_pt
                        lat_th = float(np.linalg.norm(
                            d_th - float(np.dot(d_th, wu)) * wu
                        ))
                        lat_mid = float(np.linalg.norm(
                            d_mid - float(np.dot(d_mid, wu)) * wu
                        ))
                        log(
                            f"放下★拇tip(上方) [{thumb_tgt[0]:.3f},{thumb_tgt[1]:.3f},"
                            f"{thumb_tgt[2]:.3f}] 拇−格 前"
                            f"{float(np.dot(d_th, fwd))*1000:.0f} "
                            f"右{float(np.dot(d_th, right))*1000:.0f} "
                            f"上{float(np.dot(d_th, wu))*1000:.0f}mm "
                            f"|侧向|={lat_th*1000:.0f}mm "
                            f"(mid侧向{lat_mid*1000:.0f}；"
                            f"若拇侧向≈0而mid也≈0则像拇压盘心——实为视角/偏置)"
                        )
                log(
                    f"放下★tip_ee→世界 tool0→mid "
                    f"前{float(np.dot(tip_w_off, fwd))*1000:.0f} "
                    f"右{float(np.dot(tip_w_off, right))*1000:.0f} "
                    f"上{float(np.dot(tip_w_off, wu))*1000:.0f}mm "
                    f"rpy°=[{math.degrees(rpy_place[0]):.1f},"
                    f"{math.degrees(rpy_place[1]):.1f},"
                    f"{math.degrees(rpy_place[2]):.1f}]"
                )
                if tip_ee_raw is not None:
                    tip_w_raw = R_pl @ tip_ee_raw
                    log(
                        f"放下★tip_ee满长→世界 "
                        f"前{float(np.dot(tip_w_raw, fwd))*1000:.0f} "
                        f"右{float(np.dot(tip_w_raw, right))*1000:.0f} "
                        f"上{float(np.dot(tip_w_raw, wu))*1000:.0f}mm"
                    )
            else:
                tip_ee = None
                log(
                    f"放下：tip 在腕差≤{max_orient_deg:.0f}°内上下均无解，回退法兰"
                )

        if above is None:
            above = np.asarray(tip_above, dtype=np.float64).copy()
            drop = np.asarray(tip_drop, dtype=np.float64).copy()
            lower = float(np.linalg.norm(above - drop))
            found_flange = False
            for yaw in yaw_dedup:
                rpy_try = (
                    self._place_rpy_level_yaw(
                        eul_c, yaw, wu=wu, R_pinch_in_ee=R_pe,
                    )
                    if level_on
                    else (float(rpy_keep[0]), float(rpy_keep[1]), float(yaw))
                )
                if rotation_geodesic_deg(rpy_keep, rpy_try) > max_orient_deg:
                    continue
                if seed0 is None:
                    break
                dyaw = self._place_angle_diff(rpy_try[2], rpy_keep[2])
                for sd in (seed0, self._place_seed_yaw_nudge(seed0, dyaw)):
                    pair = self._place_ik_pair(
                        above, drop, rpy_try, sd, search_jump,
                    )
                    if pair is None:
                        continue
                    rpy_place = rpy_try
                    found_flange = True
                    geo = rotation_geodesic_deg(rpy_keep, rpy_try)
                    if geo >= 2.0:
                        log(
                            f"放下：法兰回退，转腕 {geo:.0f}° 上下可解"
                        )
                    break
                if found_flange:
                    break
            self.set_ee_trace_phase("放下")
            log(
                "放下：法兰≈格正上方（tip IK 无解回退）"
                + ("" if found_flange else "；上下姿态可能仍难解")
            )
        else:
            self.set_ee_trace_phase("放下")

        above = np.asarray(above, dtype=np.float64).reshape(3)
        drop = np.asarray(drop, dtype=np.float64).reshape(3)
        r_start = rpy_keep
        travel = float(np.linalg.norm(above - p0))
        clear_h = float(np.clip(self.eye.get("place_approach_clear_m", 0.06), 0.0, 0.25))
        geo_yaw = rotation_geodesic_deg(r_start, rpy_place)
        combine = bool(self.eye.get("place_combine_translate_orient", True))

        def _place_update_pose():
            nonlocal p0, r_start, geo_yaw
            pose_u = self.get_pose()
            if pose_u is None:
                return
            pos_u, eul_u = pose_u
            p0 = np.array([pos_u.x, pos_u.y, pos_u.z], dtype=np.float64)
            r_start = (float(eul_u.x), float(eul_u.y), float(eul_u.z))
            geo_yaw = rotation_geodesic_deg(r_start, rpy_place)

        def _place_combined_to_above(rpy_tgt, above_tgt, log_tag="放下合段"):
            """位置+姿态同段插值到上方；中间抬高清障。"""
            nonlocal p0, r_start, geo_yaw, above, drop, rpy_place
            above_tgt = np.asarray(above_tgt, dtype=np.float64).reshape(3)
            rpy_tgt = (float(rpy_tgt[0]), float(rpy_tgt[1]), float(rpy_tgt[2]))
            trav = float(np.linalg.norm(above_tgt - p0))
            geo = rotation_geodesic_deg(r_start, rpy_tgt)
            if trav < 0.012 and geo < 2.0:
                return True
            step_m = float(np.clip(self.eye.get("place_combine_step_m", 0.08), 0.04, 0.15))
            step_deg = float(np.clip(self.eye.get("place_orient_step_deg", 12.0), 5.0, 20.0))
            n_pos = int(math.ceil(trav / step_m)) if trav > 0.02 else 1
            n_ori = int(math.ceil(geo / step_deg)) if geo >= 2.0 else 1
            n_seg = int(np.clip(max(n_pos, n_ori, 2), 2, 14))
            z0 = float(np.dot(p0, wu))
            z1 = float(np.dot(above_tgt, wu))
            z_peak = max(z0, z1) + (clear_h if trav > 0.08 else 0.0)
            log(
                f"{log_tag} |Δ|={trav*1000:.0f}mm 腕差{geo:.0f}° "
                f"分 {n_seg} 段（位姿同插）"
            )
            p_s, r_s = p0.copy(), r_start
            for i in range(1, n_seg + 1):
                if stop():
                    return False
                t = float(i) / float(n_seg)
                p_i, r_i = lerp_pose_rpy(p_s, r_s, above_tgt, rpy_tgt, t)
                # 中段抬高，末段落到 above
                if clear_h > 1e-4 and trav > 0.08 and t < 0.999:
                    z_blend = z0 + (z1 - z0) * t
                    z_arc = z_blend + (z_peak - max(z0, z1)) * (4.0 * t * (1.0 - t))
                    z_now = float(np.dot(p_i, wu))
                    p_i = p_i + (z_arc - z_now) * wu
                ok_i = self._place_reach_pose(
                    p_i, r_i,
                    log=log, should_stop=stop,
                    speed=v_approach * (0.55 if geo >= 25 else 0.7),
                    accel=a_approach * (0.55 if geo >= 25 else 0.7),
                    log_prefix=f"{log_tag}{i}/{n_seg}",
                    hold_after=False,
                    max_jump_rad=max(jump, 1.4),
                )
                if not ok_i:
                    if i < n_seg:
                        continue
                    log(f"{log_tag}：末段失败")
                    return False
                _place_update_pose()
            # 终到 above + 目标腕
            ok_f = self._place_reach_pose(
                above_tgt, rpy_tgt,
                log=log, should_stop=stop,
                speed=v_approach * 0.65, accel=a_approach * 0.65,
                log_prefix=f"{log_tag}到位",
                hold_after=False,
                max_jump_rad=jump,
            )
            _place_update_pose()
            if ok_f:
                above = above_tgt
                rpy_place = rpy_tgt
            return bool(ok_f)

        reached_above = False
        if combine and (travel > 0.02 or geo_yaw >= 3.0):
            reached_above = _place_combined_to_above(rpy_place, above)
            if stop():
                return False
            if not reached_above and alt_plans:
                for alt in alt_plans[1:]:
                    if float(alt["geo"]) >= geo_yaw - 1e-6:
                        continue
                    log(
                        f"放下：合段失败，试更小转腕候选 "
                        f"{float(alt['geo']):.0f}°"
                    )
                    if _place_combined_to_above(
                        alt["rpy"],
                        np.asarray(alt["above"], dtype=np.float64),
                        log_tag="放下合段备",
                    ):
                        drop = np.asarray(alt["drop"], dtype=np.float64)
                        reached_above = True
                        break
                    if stop():
                        return False
        else:
            # 旧路径：腕差小→高位横移；腕差大→先抬高再原地转腕
            clear_pos = None
            if travel > 0.10 and clear_h > 1e-4:
                z_now = float(np.dot(p0, wu))
                z_hi = float(np.dot(above, wu)) + clear_h
                z_use = max(z_now, z_hi)
                if geo_yaw < 12.0:
                    lat = above - float(np.dot(above, wu)) * wu
                    high = lat + z_use * wu
                    log(f"放下：高位接近（行程 {travel*1000:.0f}mm，腕差{geo_yaw:.0f}°）")
                    ok_h = self._place_reach_pose(
                        high, r_start if geo_yaw < 5.0 else rpy_place,
                        log=log, should_stop=stop,
                        speed=v_approach * 0.75, accel=a_approach * 0.75,
                        log_prefix="放下高位",
                        hold_after=False,
                    )
                    if stop() or not ok_h:
                        log("放下失败：高位接近未完成")
                        return False
                    _place_update_pose()
                    clear_pos = p0.copy()
                else:
                    clear_pos = p0 - float(np.dot(p0, wu)) * wu + z_use * wu
                    log(
                        f"放下：先抬高再转腕（行程 {travel*1000:.0f}mm，"
                        f"腕差 {geo_yaw:.0f}°）"
                    )
                    ok_c = self._place_reach_pose(
                        clear_pos, r_start,
                        log=log, should_stop=stop,
                        speed=v_approach * 0.75, accel=a_approach * 0.75,
                        log_prefix="放下抬高",
                        hold_after=False,
                    )
                    if stop() or not ok_c:
                        log("放下失败：抬高未完成")
                        return False
                    _place_update_pose()
                    clear_pos = p0.copy()

            if geo_yaw >= 3.0:
                n_ori = int(np.clip(self.eye.get("place_drop_orient_steps", 5), 2, 10))
                step_deg = float(self.eye.get("place_orient_step_deg", 12.0))
                n_ori = max(n_ori, int(math.ceil(geo_yaw / max(step_deg, 5.0))))
                n_ori = int(np.clip(n_ori, 2, 12))
                p_ori = clear_pos if clear_pos is not None else p0
                log(f"放下：转腕 {geo_yaw:.0f}° 分 {n_ori} 步 @抬高位")
                ok_o = True
                for i in range(1, n_ori + 1):
                    if stop():
                        return False
                    t = float(i) / float(n_ori)
                    _, ri = lerp_pose_rpy(p_ori, r_start, p_ori, rpy_place, t)
                    if not self._place_reach_pose(
                        p_ori, ri,
                        log=log, should_stop=stop,
                        speed=v_approach * 0.4, accel=a_approach * 0.4,
                        log_prefix=f"放下转腕{i}/{n_ori}",
                        hold_after=False,
                        max_jump_rad=max(jump, 1.4),
                    ):
                        log(f"放下：转腕 {i}/{n_ori} 失败，中止转腕")
                        ok_o = False
                        break
                _place_update_pose()
                if not ok_o and alt_plans:
                    for alt in alt_plans[1:]:
                        if alt["geo"] >= geo_yaw - 1e-6:
                            continue
                        log(
                            f"放下：改用更小转腕候选 "
                            f"{float(alt['geo']):.0f}°"
                        )
                        rpy_place = alt["rpy"]
                        above = np.asarray(alt["above"], dtype=np.float64)
                        drop = np.asarray(alt["drop"], dtype=np.float64)
                        geo_yaw = rotation_geodesic_deg(r_start, rpy_place)
                        if geo_yaw < 3.0:
                            break
                        n2 = int(np.clip(math.ceil(geo_yaw / 12.0), 2, 10))
                        p_ori = clear_pos if clear_pos is not None else p0
                        ok_retry = True
                        for j in range(1, n2 + 1):
                            t = float(j) / float(n2)
                            _, ri = lerp_pose_rpy(
                                p_ori, r_start, p_ori, rpy_place, t,
                            )
                            if not self._place_reach_pose(
                                p_ori, ri,
                                log=log, should_stop=stop,
                                speed=v_approach * 0.4, accel=a_approach * 0.4,
                                log_prefix=f"放下补转腕{j}/{n2}",
                                hold_after=False,
                                max_jump_rad=max(jump, 1.4),
                            ):
                                ok_retry = False
                                break
                            if stop():
                                return False
                        _place_update_pose()
                        if ok_retry:
                            break

        if not reached_above:
            ok_a = self._place_reach_pose(
                above, rpy_place,
                log=log, should_stop=stop,
                speed=v_approach * 0.7, accel=a_approach * 0.7,
                log_prefix="放下上方",
                hold_after=False,
                max_jump_rad=jump,
            )
            if stop():
                return False
            if not ok_a:
                recovered = False
                for i, alt in enumerate(alt_plans[1:], start=2):
                    log(
                        f"放下：上方失败，试候选{i} 转腕"
                        f"{float(alt['geo']):.0f}°"
                    )
                    if self._place_reach_pose(
                        alt["above"], alt["rpy"],
                        log=log, should_stop=stop,
                        speed=v_approach * 0.7, accel=a_approach * 0.7,
                        log_prefix=f"放下上方(备{i})",
                        hold_after=False,
                        max_jump_rad=jump,
                    ):
                        rpy_place = alt["rpy"]
                        above = np.asarray(alt["above"], dtype=np.float64)
                        drop = np.asarray(alt["drop"], dtype=np.float64)
                        recovered = True
                        break
                    if stop():
                        return False
                if not recovered:
                    log("放下失败：上方位姿不可达")
                    return False
            else:
                reached_above = True
        if stop():
            return False

        if float(np.linalg.norm(above - drop)) > 1e-4:
            ok_d = self._place_reach_pose(
                drop, rpy_place,
                log=log, should_stop=stop,
                speed=v_approach * 0.5, accel=a_approach * 0.5,
                log_prefix="放下下降",
                hold_after=False,
                max_jump_rad=jump,
            )
            if stop():
                return False
            if not ok_d:
                log("放下失败：下降未到位，不松手")
                return False

        if level_on and R_pe is not None:
            pose_n = self.get_pose()
            if pose_n is not None:
                pos_n, eul_n = pose_n
                rpy_n = self._place_level_rpy(eul_n, R_pinch_in_ee=R_pe)
                geo = rotation_geodesic_deg(
                    (float(eul_n.x), float(eul_n.y), float(eul_n.z)),
                    rpy_n,
                )
                if geo >= 1.5:
                    p_n = np.array(
                        [pos_n.x, pos_n.y, pos_n.z], dtype=np.float64,
                    )
                    log(f"放下：到位后调平指平面（{geo:.1f}°）")
                    ok_lv = self._place_reach_pose(
                        p_n, rpy_n,
                        log=log, should_stop=stop,
                        speed=v_approach * 0.4, accel=a_approach * 0.4,
                        log_prefix="放下调平",
                        max_jump_rad=1.5,
                        hold_after=False,
                    )
                    if not ok_lv:
                        log("放下：调平无解，保持当前姿态松手")

        if stop():
            return False
        # 松手前：用法兰+tip_ee 估 mid，对照格心（查「偏左上」）
        if tip_ee is not None:
            pose_rel = self.get_pose()
            if pose_rel is not None:
                pos_rel, eul_rel = pose_rel
                R_rel = euler_rpy_to_matrix(
                    float(eul_rel.x), float(eul_rel.y), float(eul_rel.z),
                )
                mid_est = np.array(
                    [pos_rel.x, pos_rel.y, pos_rel.z], dtype=np.float64,
                ) + (R_rel @ tip_ee)
                d_rel = mid_est - bin_pt
                log(
                    f"放下★松手前 mid估−格 "
                    f"前{float(np.dot(d_rel, fwd))*1000:.0f} "
                    f"右{float(np.dot(d_rel, right))*1000:.0f} "
                    f"上{float(np.dot(d_rel, wu))*1000:.0f}mm "
                    f"（目标≈口沿+{tip_clear*1000:.0f}；"
                    f"此为 MJ tip 估，非实指；实偏左前靠 bias 纠）"
                )
        open_l6 = [
            int(np.clip(int(v), 0, 255))
            for v in list(self.hand_presets.get("pinch_open", [200, 0, 255, 255, 255, 255]))[:6]
        ]
        log(f"放下：松手 → L6={open_l6}（当前={list(self.hand_cmd) if self.hand_cmd else None}）")
        if not self.hand_open_pinch(force=True):
            log("松手失败，0.25s 后重试…")
            time.sleep(0.25)
            if not self.hand_open_pinch(force=True):
                log("松手仍失败")
                return False
        log(f"放下：松手完成 L6={list(self.hand_cmd) if self.hand_cmd else None}")
        time.sleep(0.15)
        # 盘装配：松手后先抬到 15cm，再回面板（避免斜腕横扫蹭盘）
        if is_plate and post_lift > 1e-4:
            tip_lift = bin_pt + wu * post_lift
            R_lift = euler_rpy_to_matrix(
                float(rpy_place[0]), float(rpy_place[1]), float(rpy_place[2]),
            )
            if tip_ee is not None:
                lift_ee = self._ee_pos_for_tip(
                    tip_lift, R_lift, tip_ee, mode=tip_mode or "full",
                )
            else:
                lift_ee = tip_lift.copy()
            log(f"放下(盘)：抬离 mid→{post_lift*1000:.0f}mm")
            ok_lift = self._place_reach_pose(
                lift_ee, rpy_place,
                log=log, should_stop=stop,
                speed=v_approach * 0.55, accel=a_approach * 0.55,
                log_prefix="放下抬离",
                hold_after=False,
                max_jump_rad=jump,
            )
            if stop():
                return False
            if not ok_lift:
                log("放下：抬离未到位，仍尝试回面板")
        if return_home and home_joints is not None:
            self.set_ee_trace_phase("⑩回原位")
            log("⑩ 回面板（空手）")
            # 空手：勿用 carry_safe 连续笛卡尔。放下刚转腕~50–60°（指平面 level），
            # 斜腕横移易直线卡死（日志曾 191mm/18s），界面 busy 像「相机卡死」。
            self.restore_home(
                home_joints,
                home_pose=home_pose,
                log=log,
                carry_safe=False,
                prepend_world_delta=None,
                skip_settle=True,
            )
        log("放下完成")
        return True

    def teach_mount_offset(self, u, v, depth_mm, log=print):
        from lbot_grasp_utils import pixel_depth_to_camera_point

        cam_pt = pixel_depth_to_camera_point(u, v, depth_mm, self.intrinsics)
        if cam_pt is None:
            return False
        target_ee = np.array(
            [0.0, 0.0, self.eye["target_distance_mm"] / 1000.0]
        )
        new_t = target_ee - self.R_ee_cam @ cam_pt
        self.t_ee_cam = new_t
        save_camera_on_ee_translation(self.config_path, new_t)
        log(f"已更新 camera_on_ee.translation={new_t.tolist()}")
        return True


