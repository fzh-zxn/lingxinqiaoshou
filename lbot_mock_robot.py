"""离线假机械臂：不连 TCP，用 MuJoCo 做 FK/IK，供在家看仿真。"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

from lbot.lbot_api import LbotArm, LbotEuler, LbotPosition
from lbot_grasp_utils import euler_rpy_to_matrix, rotation_matrix_to_rpy
from lbot_mj_hand import (
    TOOL0_SITE,
    apply_hand_mount_rpy,
    apply_o6_hand_to_mjcf,
    collect_hand_joint_maps,
    resolve_arm_joint_signs,
    robot_joints_to_mj_qpos,
)

ARM_JOINT_NAMES = [f"arm_right_R{i}_Joint" for i in range(1, 8)]


class MockLbotRobot:
    """假装已连接的右臂：关节本地维护，运动学走 workstation.mjcf。"""

    def __init__(self, config: dict, mjcf_path: Path):
        if mujoco is None:
            raise RuntimeError("需要安装 mujoco 才能使用离线仿真假机械臂")

        self.host = "mock://home-sim"
        self._connected = False
        self._lock = threading.Lock()
        self._last_error = ""

        robot_cfg = config.get("robot") or {}
        home = list(
            robot_cfg.get(
                "default_home_joints",
                [0.699, -0.22, -0.04, -0.86, -1.38, 0.032, 0.880],
            )
        )
        while len(home) < 7:
            home.append(0.0)
        self._joints = [float(j) for j in home[:7]]

        hp = (config.get("hand_presets") or {}).get("right") or {}
        self._hand = list(hp.get("pinch_open") or hp.get("open") or [255] * 6)
        while len(self._hand) < 6:
            self._hand.append(255)

        eye = config.get("eye_in_hand") or {}
        self._signs = resolve_arm_joint_signs(eye)
        self._mirror_hand = bool(eye.get("model_hand_qpos_mirror", False))

        self.mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        mount_rpy = eye.get("model_hand_mount_rpy")
        if mount_rpy is not None:
            apply_hand_mount_rpy(self.mj_model, mount_rpy)
        self.mj_joints, self.mj_hand_ranges = collect_hand_joint_maps(self.mj_model)

        self._site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, TOOL0_SITE
        )
        if self._site_id < 0:
            raise RuntimeError(f"MJCF 中找不到 site: {TOOL0_SITE}")

        self._dof_ids = []
        for name in ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"MJCF 中找不到关节: {name}")
            self._dof_ids.append(int(self.mj_model.jnt_dofadr[jid]))

        soft = robot_cfg.get("soft_joint_limits_deg") or {}
        lo = soft.get("lower")
        hi = soft.get("upper")
        if lo and hi and len(lo) >= 7 and len(hi) >= 7:
            self._q_lo = np.array([math.radians(float(x)) for x in lo[:7]])
            self._q_hi = np.array([math.radians(float(x)) for x in hi[:7]])
        else:
            self._q_lo = np.full(7, -math.pi)
            self._q_hi = np.full(7, math.pi)

        self._sync_mj(self._joints, self._hand)

    def connect(self, timeout: float = 10.0) -> bool:
        del timeout
        self._connected = True
        print("Mock robot connected (offline MuJoCo sim)", flush=True)
        return True

    def disconnect(self):
        self._connected = False
        print("Mock robot disconnected", flush=True)

    def is_connected(self) -> bool:
        return self._connected

    def get_last_error(self) -> str:
        return self._last_error

    def get_joint_positions(self, arm: LbotArm) -> Optional[List[float]]:
        del arm
        with self._lock:
            return list(self._joints)

    def get_cartesian_pose(
        self, arm: LbotArm
    ) -> Optional[Tuple[LbotPosition, LbotEuler]]:
        with self._lock:
            joints = list(self._joints)
            hand = list(self._hand)
        return self.compute_forward_kinematics(arm, joints, hand_cmd=hand)

    def compute_forward_kinematics(
        self,
        arm: LbotArm,
        joints: List[float],
        hand_cmd: Optional[List[int]] = None,
    ) -> Optional[Tuple[LbotPosition, LbotEuler]]:
        del arm
        with self._lock:
            hand = list(hand_cmd) if hand_cmd is not None else list(self._hand)
            self._sync_mj(joints, hand)
            pos, _R, rpy = self._tool0_pose()
        return (
            LbotPosition(float(pos[0]), float(pos[1]), float(pos[2])),
            LbotEuler(float(rpy[0]), float(rpy[1]), float(rpy[2])),
        )

    def compute_inverse_kinematics(
        self,
        arm: LbotArm,
        position: LbotPosition,
        euler: LbotEuler,
        initial_joints: List[float] = None,
    ) -> Optional[List[float]]:
        del arm
        if not self.is_connected():
            self._last_error = "not connected"
            return None
        seed = initial_joints
        if seed is None:
            with self._lock:
                seed = list(self._joints)
        target = np.array(
            [float(position.x), float(position.y), float(position.z)],
            dtype=np.float64,
        )
        rpy = (float(euler.x), float(euler.y), float(euler.z))
        with self._lock:
            hand = list(self._hand)
            sol = self._solve_ik(seed, target, rpy, hand)
        if sol is None:
            self._last_error = "mock IK failed"
        return sol

    def move_to_joint_target(
        self,
        arm: LbotArm,
        target_joints: List[float],
        speed: float = 0.2,
        accel: float = 0.5,
        block: bool = True,
    ) -> bool:
        del arm, accel
        tgt = [float(j) for j in list(target_joints)[:7]]
        while len(tgt) < 7:
            tgt.append(0.0)
        tgt = self._clamp_joints(tgt)
        if not block:
            with self._lock:
                self._joints = tgt
                self._sync_mj(self._joints, self._hand)
            return True
        with self._lock:
            start = list(self._joints)
        max_d = max(abs(a - b) for a, b in zip(tgt, start)) or 1e-6
        dur = max(max_d / max(float(speed), 1e-3), 0.05)
        t0 = time.time()
        while True:
            u = min(1.0, (time.time() - t0) / dur)
            cur = [(1.0 - u) * a + u * b for a, b in zip(start, tgt)]
            with self._lock:
                self._joints = cur
                self._sync_mj(self._joints, self._hand)
            if u >= 1.0:
                break
            time.sleep(0.02)
        return True

    def move_to_pose_target(
        self,
        arm: LbotArm,
        position: LbotPosition,
        euler: LbotEuler,
        speed: float = 0.2,
        accel: float = 0.5,
        block: bool = True,
    ) -> bool:
        with self._lock:
            seed = list(self._joints)
        ik = self.compute_inverse_kinematics(arm, position, euler, seed)
        if ik is None:
            return False
        return self.move_to_joint_target(arm, ik, speed, accel, block)

    def linear_move_to_pose(
        self,
        arm: LbotArm,
        position: LbotPosition,
        euler: LbotEuler,
        speed: float = 0.2,
        accel: float = 0.5,
        block: bool = True,
    ) -> bool:
        return self.move_to_pose_target(arm, position, euler, speed, accel, block)

    def pose_follow(
        self, arm: LbotArm, position: LbotPosition, euler: LbotEuler
    ) -> bool:
        return self.move_to_pose_target(
            arm, position, euler, speed=0.5, accel=1.0, block=False
        )

    def l6_set_position(self, arm: LbotArm, position: List[int]) -> bool:
        del arm
        with self._lock:
            self._hand = [int(x) for x in list(position)[:6]]
            while len(self._hand) < 6:
                self._hand.append(255)
            self._sync_mj(self._joints, self._hand)
        return True

    def l6_set_velocity(self, arm: LbotArm, velocity: List[int]) -> bool:
        del arm, velocity
        return True

    def l6_set_effort(self, arm: LbotArm, effort: List[int]) -> bool:
        del arm, effort
        return True

    def get_current_tool_frame(
        self, arm: LbotArm
    ) -> Optional[Tuple[str, LbotPosition, LbotEuler]]:
        del arm
        return ("tool0", LbotPosition(0, 0, 0), LbotEuler(0, 0, 0))

    def get_state(self):
        return None

    def _clamp_joints(self, joints: List[float]) -> List[float]:
        return [
            float(np.clip(j, self._q_lo[i], self._q_hi[i]))
            for i, j in enumerate(joints[:7])
        ]

    def _sync_mj(self, joints, hand_cmd):
        mj_q = robot_joints_to_mj_qpos(joints, self._signs)
        for i, value in enumerate(mj_q):
            name = ARM_JOINT_NAMES[i]
            if name in self.mj_joints:
                self.mj_data.qpos[self.mj_joints[name]] = value
        apply_o6_hand_to_mjcf(
            self.mj_data,
            self.mj_joints,
            self.mj_hand_ranges,
            hand_cmd,
            mirror_qpos=self._mirror_hand,
        )
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _tool0_pose(self):
        p = self.mj_data.site_xpos[self._site_id].copy()
        R = self.mj_data.site_xmat[self._site_id].reshape(3, 3).copy()
        rpy = rotation_matrix_to_rpy(R)
        return p, R, rpy

    def _solve_ik(self, seed, target_pos, target_rpy, hand_cmd, max_iters=80):
        q = np.asarray(seed[:7], dtype=np.float64).copy()
        target_R = euler_rpy_to_matrix(*target_rpy)
        damp = 1e-3
        w_ori = 0.35
        jacp = np.zeros((3, self.mj_model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.mj_model.nv), dtype=np.float64)

        for _ in range(max_iters):
            self._sync_mj(q.tolist(), hand_cmd)
            p, R, _rpy = self._tool0_pose()
            dp = target_pos - p
            R_err = target_R @ R.T
            w = 0.5 * np.array(
                [
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ],
                dtype=np.float64,
            )
            err = np.concatenate([dp, w_ori * w])
            if float(np.linalg.norm(err)) < 1.5e-4:
                return self._clamp_joints(q.tolist())

            mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, self._site_id)
            J = np.zeros((6, 7), dtype=np.float64)
            for i, dof in enumerate(self._dof_ids):
                s = float(self._signs[i])
                J[0:3, i] = jacp[:, dof] * s
                J[3:6, i] = jacr[:, dof] * s

            jjt = J @ J.T + damp * np.eye(6)
            try:
                dq = J.T @ np.linalg.solve(jjt, err)
            except np.linalg.LinAlgError:
                return None
            step = float(np.linalg.norm(dq))
            if step > 0.25:
                dq *= 0.25 / step
            q = np.clip(q + dq, self._q_lo, self._q_hi)

        self._sync_mj(q.tolist(), hand_cmd)
        p, _R, _rpy = self._tool0_pose()
        if float(np.linalg.norm(target_pos - p)) < 0.012:
            return self._clamp_joints(q.tolist())
        return None
