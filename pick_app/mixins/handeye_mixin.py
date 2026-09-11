"""Hand-eye ChArUco capture / solve inside the main pick UI."""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from calibrate_handeye_charuco import (
    CHARUCO_SQUARES,
    MIN_CORNERS,
    MIN_SAMPLES,
    _rot_log,
    enumerate_handeye_solutions,
    estimate_board_pose,
    make_board,
    r_t_to_yaml_mount,
    save_samples,
    write_camera_on_ee,
)
from lbot_grasp_utils import build_camera_on_ee, pose_to_matrix, rotation_geodesic_deg

# 相对「看板位」的保守偏移（米 / rad）。相邻位拉开，避免「太近拒记」。
# 偏近(z更负)仍克制；姿态差加大一点以改善 AX=XB。
_HANDEYE_SAMPLE_DELTAS = (
    ("S1看板", (0.000, 0.000, 0.000), (0.00, 0.00, 0.00)),
    ("S2稍远", (0.000, 0.000, 0.045), (0.00, 0.00, 0.00)),
    ("S3再远", (0.020, 0.000, 0.080), (0.00, 0.00, 0.00)),
    ("S4右移", (0.000, -0.070, 0.020), (0.00, 0.00, 0.00)),
    ("S5左移", (0.000, 0.070, 0.020), (0.00, 0.00, 0.00)),
    ("S6前俯", (0.050, 0.000, 0.010), (0.00, -0.18, 0.00)),
    ("S7后仰", (-0.045, -0.025, 0.020), (0.00, 0.18, 0.00)),
    ("S8滚左", (0.015, -0.055, 0.025), (0.25, 0.00, 0.00)),
    ("S9滚右", (0.015, 0.055, 0.025), (-0.25, 0.00, 0.00)),
    ("S10偏左", (0.015, -0.065, 0.040), (0.00, 0.00, 0.20)),
    ("S11偏右", (0.015, 0.065, 0.040), (0.00, 0.00, -0.20)),
    ("S12轻斜", (0.040, -0.070, -0.005), (0.18, -0.16, 0.10)),
)

_HANDEYE_SPEED_CAP = 0.06
_HANDEYE_MIN_BOARD_Z = 0.20  # m；更近则警告，仍允许手动记录
_HANDEYE_SETTLE_S = 0.5  # 到位后停稳再读实机位姿
_HANDEYE_ARRIVE_TOL_M = 0.006  # 手眼采样要求实机距目标 ≤6mm
_HANDEYE_ARRIVE_ANG_DEG = 4.0

class HandeyeMixin:
    """UI：手眼标定模式（调姿用主界面运动区，拍照用本区按钮）。"""

    def _handeye_out_dir(self) -> Path:
        return Path(__file__).resolve().parents[2] / "calib_handeye"

    def _handeye_base_pose(self):
        robot = (self.config.get("robot") or {})
        pose = robot.get("screw_board_view_pose") or {}
        xyz = list(pose.get("xyz") or [0.3496, 0.0183, -0.100])
        rpy = list(pose.get("rpy") or [0.0015, 0.0004, 1.5724])
        return (
            [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            [float(rpy[0]), float(rpy[1]), float(rpy[2])],
        )

    def _handeye_sample_poses(self):
        """返回 [(name, xyz, rpy), ...]，相对当前 yaml 看板位。"""
        base_xyz, base_rpy = self._handeye_base_pose()
        out = []
        for name, dxyz, drpy in _HANDEYE_SAMPLE_DELTAS:
            xyz = [base_xyz[i] + float(dxyz[i]) for i in range(3)]
            rpy = [base_rpy[i] + float(drpy[i]) for i in range(3)]
            out.append((name, xyz, rpy))
        return out

    def _handeye_speed_accel(self):
        try:
            speed = float(self.move_speed.get())
        except Exception:
            speed = _HANDEYE_SPEED_CAP
        try:
            accel = float(self.move_accel.get())
        except Exception:
            accel = 0.2
        return min(speed, _HANDEYE_SPEED_CAP), min(accel, 0.25)

    def _handeye_fill_pose(self, xyz, rpy, label):
        if hasattr(self, "pose_xyz_vars") and hasattr(self, "pose_rpy_vars"):
            for var, value in zip(self.pose_xyz_vars, xyz):
                var.set(round(float(value), 4))
            for var, value in zip(self.pose_rpy_vars, rpy):
                var.set(round(float(value), 4))
        self.log(
            f"手眼采样位 {label}: xyz={[round(v, 4) for v in xyz]} "
            f"rpy={[round(v, 4) for v in rpy]}"
        )

    def _handeye_ensure_right_arm(self):
        from lbot.lbot_robot import LbotArm

        if int(getattr(self.controller, "arm", LbotArm.RIGHT_ARM)) == int(
            LbotArm.RIGHT_ARM
        ):
            return True
        self.log("手眼采样：请先切到「右臂(相机)」")
        try:
            if hasattr(self, "control_arm_var"):
                self.control_arm_var.set("right")
                self._on_control_arm_change()
        except Exception as exc:
            self.log(f"切换右臂失败: {exc}")
            return False
        return True

    def _handeye_pose_err(self, xyz, rpy):
        """实机相对目标的平移(m)与姿态角(°)。读失败返回 (None, None)。"""
        try:
            pose = self.controller.get_pose()
        except Exception:
            pose = None
        if pose is None:
            return None, None
        pos, eul = pose
        err = float(
            np.linalg.norm(
                np.array(
                    [float(pos.x) - float(xyz[0]),
                     float(pos.y) - float(xyz[1]),
                     float(pos.z) - float(xyz[2])],
                    dtype=np.float64,
                )
            )
        )
        try:
            geo = float(
                rotation_geodesic_deg(
                    (float(eul.x), float(eul.y), float(eul.z)),
                    (float(rpy[0]), float(rpy[1]), float(rpy[2])),
                )
            )
        except Exception:
            geo = None
        return err, geo

    def _handeye_move_to(self, xyz, rpy, label) -> bool:
        if self.controller.robot is None:
            self.log(f"{label}：未连接机械臂")
            return False
        speed, accel = self._handeye_speed_accel()
        self._handeye_fill_pose(xyz, rpy, label)
        # 每段等到位，避免「中间点还在飞、末点假成功」
        ok = bool(
            self.controller.execute_pose_motion_safe(
                list(xyz),
                list(rpy),
                speed=speed,
                accel=accel,
                log=self.log,
                wait_each=True,
            )
        )
        if not ok:
            return False
        err, geo = self._handeye_pose_err(xyz, rpy)
        if (
            err is not None
            and err <= _HANDEYE_ARRIVE_TOL_M
            and (geo is None or geo <= _HANDEYE_ARRIVE_ANG_DEG)
        ):
            self.log(
                f"{label}: 到位确认 Δ={err*1000:.1f}mm"
                f"{'' if geo is None else f' ∠={geo:.1f}°'}"
            )
            return True
        self.log(
            f"{label}: 未严到位"
            f"{'' if err is None else f' Δ={err*1000:.1f}mm'}"
            f"{'' if geo is None else f' ∠={geo:.1f}°'}，慢速补正…"
        )
        try:
            ok2 = bool(
                self.controller.execute_pose_motion(
                    list(xyz),
                    list(rpy),
                    speed=min(speed, 0.04),
                    accel=min(accel, 0.12),
                    block=True,
                    linear=True,
                    arrive_tol_m=_HANDEYE_ARRIVE_TOL_M,
                )
            )
        except Exception as exc:
            self.log(f"{label}: 补正异常 {exc}")
            ok2 = False
        err2, geo2 = self._handeye_pose_err(xyz, rpy)
        if (
            ok2
            and err2 is not None
            and err2 <= 0.008
            and (geo2 is None or geo2 <= 5.0)
        ):
            self.log(
                f"{label}: 补正后 Δ={err2*1000:.1f}mm"
                f"{'' if geo2 is None else f' ∠={geo2:.1f}°'}"
            )
            return True
        self.log(
            f"{label}: 仍未到位"
            f"{'' if err2 is None else f' Δ={err2*1000:.1f}mm'}，跳过本采样位"
        )
        return False

    def _handeye_settle(self, settle_s=None) -> bool:
        """到位后停稳，避免用运动中的法兰/模糊帧。"""
        settle_s = float(_HANDEYE_SETTLE_S if settle_s is None else settle_s)
        self.log(f"手眼停稳 {settle_s:.1f}s，再读实机位姿…")
        t0 = time.time()
        while time.time() - t0 < settle_s:
            if getattr(self, "_handeye_seq_stop", False):
                return False
            time.sleep(0.05)
        return True

    def _handeye_read_flange_Tg(self):
        """多次读实机笛卡尔位姿 → 4x4（不用指令目标、不用运动中缓存）。"""
        for _ in range(6):
            if getattr(self, "_handeye_seq_stop", False):
                return None
            pose = None
            try:
                pose = self.controller.get_pose()
            except Exception:
                pose = None
            if pose is None:
                try:
                    pose = self.controller.get_live_pose()
                except Exception:
                    pose = None
            if pose is not None:
                try:
                    return pose_to_matrix(pose[0], pose[1])
                except Exception:
                    pass
            time.sleep(0.05)
        return None

    def _handeye_wait_ready(self, timeout_s=4.0, min_corners=60):
        """到位后等 ChArUco READY。返回 (ok, n_corners, board_z or None)。"""
        t0 = time.time()
        last_n = 0
        last_z = None
        while time.time() - t0 < timeout_s:
            if getattr(self, "_handeye_seq_stop", False):
                return False, last_n, last_z
            with getattr(self, "lock", threading.Lock()):
                live = dict(self._handeye_live) if getattr(self, "_handeye_live", None) else None
            if live:
                last_n = int(live.get("n_corners") or 0)
                pe = live.get("pose_est")
                if pe is not None:
                    last_z = float(pe[1][2])
                if live.get("ready") and last_n >= min_corners:
                    if last_z is not None and last_z < _HANDEYE_MIN_BOARD_Z:
                        self.log(
                            f"⚠ board_z={last_z:.3f}m < {_HANDEYE_MIN_BOARD_Z}m，偏近，跳过自动记录"
                        )
                        return False, last_n, last_z
                    return True, last_n, last_z
            time.sleep(0.12)
        return False, last_n, last_z

    def _handeye_record_now(self, settle: bool = True) -> bool:
        """停稳 → 等 READY → 用实机位姿记录；成功 True。"""
        if settle and not self._handeye_settle():
            return False
        ready, n_c, bz = self._handeye_wait_ready()
        if not ready:
            self.log(
                f"手眼记录跳过：未 READY(corners={n_c}"
                f"{'' if bz is None else f', z={bz:.3f}m'})"
            )
            return False
        before = len(getattr(self, "_handeye_samples", []) or [])
        self.on_handeye_record(use_fresh_flange=True)
        after = len(getattr(self, "_handeye_samples", []) or [])
        return after > before

    def on_handeye_goto_sample(self, idx: int, and_record: bool = False):
        """点击采样位：慢速安全到位；可选到位后自动记录。"""
        if not bool(getattr(self, "handeye_mode_var", None) and self.handeye_mode_var.get()):
            self.log("请先勾选「手眼标定」")
            return
        poses = self._handeye_sample_poses()
        if idx < 0 or idx >= len(poses):
            return
        if self.busy:
            self.log("忙，忽略手眼去采样位（可点「停止/解锁」）")
            return
        if not self._handeye_ensure_right_arm():
            return
        name, xyz, rpy = poses[idx]
        self._handeye_sample_idx = idx
        self._handeye_seq_stop = False

        def work():
            self.busy = True
            if hasattr(self, "_begin_motion"):
                try:
                    self._begin_motion()
                except Exception:
                    pass
            try:
                self.log(f"手眼→{name}（speed≤{_HANDEYE_SPEED_CAP}）…")
                ok = self._handeye_move_to(xyz, rpy, name)
                if not ok:
                    self.log(f"手眼→{name}: 运动失败/超时")
                    return
                self.log(f"手眼→{name}: 到位")
                if and_record:
                    if self._handeye_record_now(settle=True):
                        self.log(f"手眼→{name}: 已自动记录（停稳+实机位姿）")
                    else:
                        self.log(f"手眼→{name}: 自动记录失败，可手动「记录姿态」")
            except Exception as exc:
                self.log(f"手眼→{name} 异常: {exc}")
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def on_handeye_next_sample(self, and_record: bool = True):
        poses = self._handeye_sample_poses()
        idx = int(getattr(self, "_handeye_sample_idx", -1)) + 1
        if idx >= len(poses):
            self.log("手眼采样位已走完，可「求解」或点 S1 重来")
            return
        self.on_handeye_goto_sample(idx, and_record=and_record)

    def on_handeye_stop_sequence(self):
        self._handeye_seq_stop = True
        self.log("手眼自动序列：请求停止（当前段结束后停）")

    def on_handeye_auto_sequence(self):
        """确认后按 S1→S12 慢速到位并自动记录；可随时「停序列」。"""
        if not bool(getattr(self, "handeye_mode_var", None) and self.handeye_mode_var.get()):
            self.log("请先勾选「手眼标定」")
            return
        if self.busy:
            self.log("忙，忽略自动序列")
            return
        from tkinter import messagebox

        if not messagebox.askyesno(
            "手眼自动采样",
            "将按 S1→S12 慢速安全路径移动并尝试自动记录。\n\n"
            "请确认：\n"
            "· 标定板固定、桌面无障碍\n"
            "· 已切右臂(相机)\n"
            "· 手悬停在「停止/解锁」/「停序列」\n"
            "· 偏近或看不清板的位会跳过\n\n"
            "继续？",
        ):
            return
        if not self._handeye_ensure_right_arm():
            return
        self._handeye_seq_stop = False
        poses = self._handeye_sample_poses()

        def work():
            self.busy = True
            if hasattr(self, "_begin_motion"):
                try:
                    self._begin_motion()
                except Exception:
                    pass
            try:
                self.log(f"手眼自动序列开始（{len(poses)} 位，speed≤{_HANDEYE_SPEED_CAP}）")
                for i, (name, xyz, rpy) in enumerate(poses):
                    if getattr(self, "_handeye_seq_stop", False):
                        self.log("手眼自动序列已停止")
                        break
                    self._handeye_sample_idx = i
                    self.log(f"手眼自动 [{i+1}/{len(poses)}] → {name}")
                    ok = self._handeye_move_to(xyz, rpy, name)
                    if not ok:
                        self.log(f"  {name}: 运动失败，跳过")
                        continue
                    if getattr(self, "_handeye_seq_stop", False):
                        break
                    if self._handeye_record_now(settle=True):
                        self.log(f"  {name}: 已记录（停稳+实机位姿）")
                    else:
                        self.log(f"  {name}: 跳过（未就绪/太近/已停）")
                n = len(getattr(self, "_handeye_samples", []) or [])
                self.log(f"手眼自动序列结束，当前样本 {n} 组；够 {MIN_SAMPLES} 可「求解」")
                self._handeye_set_status(f"手眼ON | 样本 {n} | 自动序列结束")
            except Exception as exc:
                self.log(f"手眼自动序列异常: {exc}")
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def _handeye_ensure_detector(self):
        if getattr(self, "_handeye_detector", None) is None:
            det, board = make_board()
            self._handeye_detector = det
            self._handeye_board = board

    def _handeye_K_dist(self):
        intr = self.controller.intrinsics_base
        fx, fy = float(intr["fx"]), float(intr["fy"])
        cx, cy = float(intr["cx"]), float(intr["cy"])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((5, 1), dtype=np.float64)
        return K, dist

    def _handeye_set_status(self, text: str):
        var = getattr(self, "handeye_status_var", None)
        if var is not None:
            try:
                var.set(text)
            except Exception:
                pass

    def on_handeye_toggle(self):
        on = bool(self.handeye_mode_var.get())
        bar = getattr(self, "handeye_sample_bar", None)
        if bar is not None:
            try:
                if on:
                    bar.pack(fill="x", pady=(2, 0))
                else:
                    bar.pack_forget()
            except Exception:
                pass
        if on:
            self._handeye_ensure_detector()
            if not hasattr(self, "_handeye_samples") or self._handeye_samples is None:
                self._handeye_samples = []
            self._handeye_live = None
            self._handeye_sample_idx = -1
            self._handeye_seq_stop = False
            n = len(self._handeye_samples)
            self.log(
                "手眼标定模式 ON：点 S1–S12 去采样位（慢速）；"
                "「去并记」到位后自动记录；或「自动采12组」。"
                f" 已有 {n} 组。"
            )
            self._handeye_set_status(
                f"手眼ON | 样本 {n} | 点采样位或自动序列"
            )
        else:
            self._handeye_seq_stop = True
            self.log("手眼标定模式 OFF")
            self._handeye_set_status("手眼OFF")

    def _handeye_process_frame(self, frame_bgr):
        """视觉线程：检测 ChArUco，写 _handeye_live，返回叠图层。"""
        if not bool(getattr(self, "handeye_mode_var", None) and self.handeye_mode_var.get()):
            return frame_bgr
        self._handeye_ensure_detector()
        K, dist = self._handeye_K_dist()
        min_c = int(getattr(self, "_handeye_min_corners", MIN_CORNERS))
        pose_est, n_c, n_m, mcorn, mids, ccorn, cids = estimate_board_pose(
            self._handeye_detector,
            self._handeye_board,
            frame_bgr,
            K,
            dist,
            min_corners=min_c,
        )

        Tg = None
        try:
            pose = (
                self.controller.get_live_pose()
                if getattr(self, "busy", False)
                else self.controller.get_pose()
            )
            if pose is None:
                pose = self.controller.get_pose()
            if pose is not None:
                Tg = pose_to_matrix(pose[0], pose[1])
        except Exception:
            Tg = None

        ready = pose_est is not None and Tg is not None
        with getattr(self, "lock", threading.Lock()):
            self._handeye_live = {
                "ready": ready,
                "n_corners": n_c,
                "n_markers": n_m,
                "pose_est": pose_est,
                "Tg": None if Tg is None else Tg.copy(),
                "frame": frame_bgr.copy() if ready else None,
                "t": datetime.now().isoformat(timespec="seconds"),
            }

        vis = frame_bgr
        if mids is not None and mcorn is not None:
            cv2.aruco.drawDetectedMarkers(vis, mcorn, mids)
        if cids is not None and ccorn is not None:
            for p in np.asarray(ccorn).reshape(-1, 2):
                cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)

        n_s = len(getattr(self, "_handeye_samples", []) or [])
        color = (0, 220, 0) if ready else (0, 165, 255)
        tag = "READY" if ready else "wait"
        line = f"HandEye {tag}  samples={n_s}  corners={n_c}/{104}  markers={n_m}"
        cv2.putText(vis, line, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(vis, line, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)
        self._handeye_set_status(
            f"手眼ON | 样本 {n_s} | corners={n_c} | {tag}"
        )
        return vis

    def on_handeye_record(self, use_fresh_flange: bool = True):
        if not bool(self.handeye_mode_var.get()):
            self.log("请先勾选「手眼标定」")
            return
        with getattr(self, "lock", threading.Lock()):
            live = dict(self._handeye_live) if getattr(self, "_handeye_live", None) else None
        if not live or not live.get("ready") or live.get("pose_est") is None:
            self.log(
                f"未记录：需检出角点≥{getattr(self, '_handeye_min_corners', MIN_CORNERS)} "
                "且法兰位姿可读（看画面 READY）"
            )
            return
        Rc, tc = live["pose_est"]
        n_c = int(live["n_corners"])

        # 默认用记录瞬间的实机法兰，不用运动中缓存 / 指令目标
        Tg = None
        if use_fresh_flange:
            Tg = self._handeye_read_flange_Tg()
        if Tg is None:
            Tg = live.get("Tg")
        if Tg is None:
            self.log("未记录：无法读取实机法兰位姿")
            return
        Rg, tg = Tg[:3, :3].copy(), Tg[:3, 3].copy()

        samples = getattr(self, "_handeye_samples", None)
        if samples is None:
            samples = []
            self._handeye_samples = samples

        if samples:
            d = float(np.linalg.norm(tg - samples[-1]["tg"]))
            dR = float(
                np.degrees(np.linalg.norm(_rot_log(Rg.T @ samples[-1]["Rg"])))
            )
            if d < 0.02 and dR < 5.0:
                self.log(
                    f"与上一样本太近 (Δt={d*1000:.0f}mm ΔR={dR:.1f}°)，"
                    "请用「机械臂运动」换姿态再记"
                )
                return

        out_dir = self._handeye_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = len(samples) + 1
        img_path = out_dir / f"pose_{idx:02d}.png"
        frame = live.get("frame")
        if frame is not None:
            cv2.imwrite(str(img_path), frame)

        samples.append(
            {
                "Rg": Rg,
                "tg": tg,
                "Rc": Rc.copy(),
                "tc": tc.copy(),
                "n_corners": n_c,
                "image": img_path.name,
            }
        )
        self.log(
            f"手眼 #{len(samples)} 已记录 corners={n_c}  "
            f"实机flange={np.round(tg, 3).tolist()}  board_z={tc[2]:.3f}m"
        )
        self._handeye_set_status(
            f"手眼ON | 样本 {len(samples)} | 刚记录 #{len(samples)}"
        )

    def on_handeye_clear(self):
        self._handeye_samples = []
        self.log("手眼样本已清空")
        self._handeye_set_status("手眼ON | 样本 0 | 已清空")

    def on_handeye_solve(self):
        samples = getattr(self, "_handeye_samples", None) or []
        if len(samples) < MIN_SAMPLES:
            self.log(f"样本不足 ({len(samples)}<{MIN_SAMPLES})，请继续记录")
            return
        if self.busy:
            self.log("忙，忽略手眼求解")
            return

        def work():
            try:
                self.busy = True
                out_dir = self._handeye_out_dir()
                save_samples(
                    out_dir,
                    samples,
                    {
                        "time": datetime.now().isoformat(),
                        "n": len(samples),
                        "source": "pick_app",
                        "board": "CC300-20-15",
                        "squares": list(CHARUCO_SQUARES),
                    },
                )
                Rg = [s["Rg"] for s in samples]
                tg = [s["tg"] for s in samples]
                Rc = [s["Rc"] for s in samples]
                tc = [s["tc"] for s in samples]

                candidates, best = enumerate_handeye_solutions(
                    self.config, Rg, tg, Rc, tc,
                )
                for c in candidates:
                    if "error" in c:
                        self.log(f"手眼[{c['name']}] 失败: {c['error']}")
                        continue
                    self.log(
                        f"手眼[{c['name']}] AX={c['ang']:.2f}°/{c['mm']:.1f}mm  "
                        f"板RMS={c['board_rms_mm']:.1f}mm  "
                        f"t={np.round(c['t'], 4).tolist()}"
                    )

                if best is None:
                    self.log("手眼求解全部失败")
                    return

                R, t = best["R"], best["t"]
                ang, mm = best["ang"], best["mm"]
                board_rms = float(best["board_rms_mm"])
                R_old, t_old = build_camera_on_ee(self.config, fold_servo_xy=True)
                self.log(
                    f"手眼选用 {best['name']} | 板RMS={board_rms:.1f}mm | "
                    f"相对当前外参 Δt(mm)="
                    f"{np.round((t - t_old) * 1000, 1).tolist()}"
                )
                if str(best["name"]).startswith(("fixR_yaml", "yamlR")):
                    self.log(
                        "（固定 yaml 旋转只解平移；若板RMS仍大，说明姿态假设不成立）"
                    )
                if board_rms > 10.0 or mm > 10.0 or ang > 3.0:
                    self.log(
                        f"⚠ 质量偏弱(板RMS{board_rms:.0f}mm / "
                        f"AX转{ang:.1f}°/移{mm:.0f}mm)，建议多采斜视；暂勿写回"
                    )

                mount_old = self.config["eye_in_hand"]["camera_on_ee"]
                mount_new = r_t_to_yaml_mount(
                    R,
                    t,
                    keep_flip_y=bool(mount_old.get("flip_y", True)),
                    keep_flip_x=bool(mount_old.get("flip_x", False)),
                    ref_rpy=mount_old.get("euler_rpy"),
                )
                self._handeye_last_mount = mount_new
                self._handeye_last_stats = {
                    "method": best["name"],
                    "ang": ang,
                    "mm": mm,
                    "board_rms_mm": board_rms,
                }

                snippet = {
                    "camera_on_ee": mount_new,
                    "camera_on_ee_prefer_yaml": True,
                    "_meta": {
                        "method": best["name"],
                        "pair_rot_err_deg": ang,
                        "pair_trans_err_mm": mm,
                        "board_rms_mm": board_rms,
                        "n_samples": len(samples),
                        "note": "勿改 plate_bias_*；确认后再点「写回外参」",
                    },
                }
                out_yaml = out_dir / "camera_on_ee_snippet.yaml"
                with out_yaml.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        snippet, f, allow_unicode=True, sort_keys=False,
                    )
                self.log(f"手眼建议已写 {out_yaml.name}（未改 grasp_config）")
                self._handeye_set_status(
                    f"求解完 {best['name']} | 板{board_rms:.0f}mm "
                    f"AX{mm:.0f}mm | 可「写回外参」"
                )
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def on_handeye_write_yaml(self):
        mount = getattr(self, "_handeye_last_mount", None)
        if mount is None:
            # try load snippet
            path = self._handeye_out_dir() / "camera_on_ee_snippet.yaml"
            if path.is_file():
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                mount = data.get("camera_on_ee")
        if not mount:
            self.log("无手眼结果可写回：请先「求解」")
            return
        stats = getattr(self, "_handeye_last_stats", {}) or {}
        mm = float(stats.get("mm", 99))
        if mm > 15.0:
            from tkinter import messagebox

            if not messagebox.askyesno(
                "手眼写回",
                f"残差约 {mm:.0f}mm，偏大，仍要写入 camera_on_ee？\n"
                "（不会改 plate_bias）",
            ):
                return
        cfg_path = Path(getattr(self.controller, "config_path", None) or "")
        if not cfg_path.is_file():
            cfg_path = Path(__file__).resolve().parents[2] / "grasp_config.yaml"
        old, new = write_camera_on_ee(cfg_path, mount)
        # sync memory
        self.config["eye_in_hand"]["camera_on_ee"] = dict(new)
        self.config["eye_in_hand"]["camera_on_ee_prefer_yaml"] = True
        self.controller.config = self.config
        self.controller.eye = self.config["eye_in_hand"]
        self.controller.R_ee_cam, self.controller.t_ee_cam = build_camera_on_ee(
            self.config, fold_servo_xy=True,
        )
        if hasattr(self, "extrinsic_var"):
            self.extrinsic_var.set(self._extrinsic_summary())
        self.log(
            f"已写回 camera_on_ee translation={new.get('translation')} "
            f"（旧 {old.get('translation')}）；plate_bias 未改。建议重启或再验参。"
        )
