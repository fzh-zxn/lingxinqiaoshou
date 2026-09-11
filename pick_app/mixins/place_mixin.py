"""Place sampling, slots, pinch tip, depth gate"""
from __future__ import annotations

from pick_app.deps import *
from pick_app.constants import *


class PlaceMixin:
    @staticmethod
    def _place_class_is_basket(cls):
        s = str(cls or "").strip().lower()
        if not s:
            return False
        return s in (
            "basket", "bin", "kuang", "框", "筐", "筐子", "篮子",
        ) or ("basket" in s) or ("筐" in s)

    @staticmethod
    def _place_class_is_plate(cls):
        s = str(cls or "").strip().lower()
        if not s:
            return False
        return s in (
            "plate", "pan", "disk", "盘", "盘子", "碟",
        ) or ("plate" in s) or ("盘" in s)

    def _place_intent_is_plate(self):
        """UI 意图：勾盘 / 盘+筐且 prefer → 对准2 优先盘（不要求当前帧已检出）。"""
        if self.yolo_place is None or not bool(self.detect_plate_var.get()):
            return False
        if not bool(self.detect_basket_var.get()):
            return True
        eye = self.config.get("eye_in_hand") or {}
        return bool(eye.get("place_prefer_plate", True))

    def _place_target_is_plate(self):
        """
        当前放置目标是否为盘装配（圆心对准）。

        优先级：
          1) 对准2 已锁的 _place_locked_kind
          2) 有冻结盘心且勾盘（多颗复用，可跳过再次对准2）
          3) 仅勾盘 → 盘
          4) 勾盘+筐且 prefer：须画面已是盘心且 UV/深度有效
          5) 否则筐
        """
        locked = getattr(self, "_place_locked_kind", None)
        if locked == "plate":
            return True
        if locked == "basket":
            return False
        frozen = getattr(self, "_plate_frozen_xyz", None)
        if (
            frozen is not None
            and self.yolo_place is not None
            and bool(self.detect_plate_var.get())
        ):
            if not bool(self.detect_basket_var.get()):
                return True
            eye = self.config.get("eye_in_hand") or {}
            if bool(eye.get("place_prefer_plate", True)):
                return True
        if not self._place_intent_is_plate():
            return False
        if not bool(self.detect_basket_var.get()):
            return True
        with self.lock:
            cls = self.latest.get("place_class")
            pu = self.latest.get("place_u")
            pv = self.latest.get("place_v")
            pd = self.latest.get("place_depth_mm")
        if self._place_class_is_basket(cls):
            return False
        if pu is None or pv is None:
            return False
        if pd is None or not np.isfinite(float(pd)):
            return False
        return self._place_class_is_plate(cls) or not cls

    def _arm_joint_signs(self, side=None):
        """side=None 时跟当前控制臂；也可显式 'left'|'right'。"""
        if side is None:
            side = "left" if self._mj_control_is_left() else "right"
        side = "left" if str(side).lower() == "left" else "right"
        cache_attr = "_mj_arm_joint_signs_left" if side == "left" else "_mj_arm_joint_signs"
        signs = getattr(self, cache_attr, None)
        if signs is None:
            signs = resolve_arm_joint_signs(
                self.config.get("eye_in_hand", {}), side=side,
            )
            setattr(self, cache_attr, signs)
        return signs

    def _mj_control_is_left(self):
        return int(getattr(self.controller, "arm", LbotArm.RIGHT_ARM)) == int(
            LbotArm.LEFT_ARM
        )

    def _write_mj_arm_joints(self, joints, side):
        """side: 'left' | 'right' → 写入对应臂 7 轴。"""
        if joints is None or len(list(joints)) < 1:
            return
        mj_q = robot_joints_to_mj_qpos(joints, self._arm_joint_signs(side=side))
        letter = "L" if side == "left" else "R"
        prefix = "left" if side == "left" else "right"
        for index, value in enumerate(mj_q, start=1):
            name = f"arm_{prefix}_{letter}{index}_Joint"
            if name in self.mj_joints:
                self.mj_data.qpos[self.mj_joints[name]] = value

    def _pinch_kin_kwargs(self):
        eye = self.config.get("eye_in_hand", {})
        return {
            "tip_frac": float(eye.get("pinch_pad_tip_frac", 0.08)),
            "tip_mode": str(eye.get("pinch_tip_mode", "tip")),
            "grasp_mode": pinch_grasp_mode(eye=eye),
        }

    def _apply_mj_qpos(self, joints, hand_cmd):
        """写入 qpos 并 mj_forward（调用方需已持 _mj_lock）。按控制臂写左/右，另一侧保留缓存。"""
        is_left = self._mj_control_is_left()
        j7 = list(joints)[:7] if joints is not None else None
        if j7 is not None:
            while len(j7) < 7:
                j7.append(0.0)
            if is_left:
                self._mj_last_joints_left = list(j7)
            else:
                self._mj_last_joints_right = list(j7)

        if is_left:
            self._write_mj_arm_joints(j7, "left")
            other = getattr(self, "_mj_last_joints_right", None)
            if other is not None:
                self._write_mj_arm_joints(other, "right")
        else:
            self._write_mj_arm_joints(j7, "right")
            other = getattr(self, "_mj_last_joints_left", None)
            if other is not None:
                self._write_mj_arm_joints(other, "left")

        mirror = bool(
            self.config["eye_in_hand"].get("model_hand_qpos_mirror", False)
        )
        # 当前控制臂的手；另一侧手保持上次写入的 qpos
        _apply_o6_hand_to_mjcf(
            self.mj_data,
            self.mj_joints,
            self.mj_hand_ranges,
            hand_cmd,
            mirror_qpos=mirror,
            side="left" if is_left else "right",
        )
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _sync_mj_state(self, joints, hand_cmd=None, block=True):
        """把关节+手指令写入 MuJoCo（供捏取 frame 计算）。"""
        if self.mj_model is None or self.mj_data is None:
            return False
        if hand_cmd is None:
            if self._model_hand_cmd is not None:
                hand_cmd = list(self._model_hand_cmd)
            elif self.busy:
                hand_cmd = list(
                    self.controller.hand_presets.get(
                        "pinch_open", DEFAULT_PINCH_OPEN
                    )
                )
            else:
                hand_cmd = resolve_hand_l6_cmd(
                    None,
                    getattr(self.controller, "hand_cmd", None),
                    self.controller.hand_presets.get(
                        "pinch_open", DEFAULT_PINCH_OPEN
                    ),
                )
        if block:
            with self._mj_lock:
                self._apply_mj_qpos(joints, hand_cmd)
            return True
        if not self._mj_lock.acquire(blocking=False):
            return False
        try:
            self._apply_mj_qpos(joints, hand_cmd)
            return True
        finally:
            self._mj_lock.release()

    def compute_pinch_frame_world(self):
        """当前姿态 + 手指令下的捏取 frame（世界/工作系）。"""
        if mujoco is None or self.mj_model is None:
            return None
        joints = (
            self._model_joints
            if self._model_joints is not None
            else (
                self.controller.get_live_joints()
                if self.busy
                else self.controller.get_joints()
            )
        )
        if not self._sync_mj_state(joints):
            return None
        return _compute_o6_pinch_kinematics(
            self.mj_model, self.mj_data, **self._pinch_kin_kwargs(),
        )

    def compute_pinch_at_joints(self, joints, hand_cmd=None):
        if mujoco is None or self.mj_model is None:
            return None
        if not self._sync_mj_state(joints, hand_cmd=hand_cmd):
            return None
        return _compute_o6_pinch_kinematics(
            self.mj_model, self.mj_data, **self._pinch_kin_kwargs(),
        )

    def compute_pinch_frame_for_alignment(self):
        """⑥ 对齐用 grasp 闭合手 frame（与 ⑧ 三指闭合几何一致）。"""
        grasp_close = list(
            self.controller.hand_presets.get("grasp", DEFAULT_GRASP_PRESET)
        )
        joints = (
            self._model_joints
            if self._model_joints is not None
            else (
                self.controller.get_live_joints()
                if self.busy
                else self.controller.get_joints()
            )
        )
        return self.compute_pinch_at_joints(joints, hand_cmd=grasp_close)

    def _sample(self):
        with self.lock:
            if self.latest["u"] is None:
                return None
            d = self.latest["depth_mm"]
            eye = self.config["eye_in_hand"]
            min_mm = float(eye.get("depth_min_valid_mm", 80))
            max_mm = float(eye.get("depth_action_max_mm", 300))
            if d is not None and np.isfinite(d):
                d = float(d)
                if d < min_mm or d > max_mm:
                    d = self._last_reliable_depth_mm
            return (
                self.latest["u"],
                self.latest["v"],
                d,
            )

    def _invalidate_place(self):
        """抬高扩视野前清掉近距旧筐框，强制用新视角重检。"""
        with self.lock:
            self.latest["place_box"] = None
            self.latest["place_u"] = None
            self.latest["place_v"] = None
            self.latest["place_depth_mm"] = None
            self.latest["place_class"] = None

    def _sample_place(self):
        """放置目标 (u, v, depth_mm)。盘已对准2锁定/冻结时只用锁点，勿跟漂。"""
        if getattr(self, "_place_locked_kind", None) == "plate":
            if self._locked_place_uvd is not None:
                u, v, d = self._locked_place_uvd[:3]
                return (float(u), float(v), float(d))
        frozen_uvd = getattr(self, "_plate_frozen_uvd", None)
        if (
            frozen_uvd is not None
            and getattr(self, "_plate_frozen_xyz", None) is not None
            and not getattr(self, "_place_aligning", False)
            and bool(self.detect_plate_var.get())
            and self._place_intent_is_plate()
        ):
            u, v, d = frozen_uvd[:3]
            return (float(u), float(v), float(d))
        with self.lock:
            pu = self.latest.get("place_u")
            pv = self.latest.get("place_v")
            pd = self.latest.get("place_depth_mm")
        if pu is None or pv is None:
            return None
        eye = self.config["eye_in_hand"]
        min_mm = float(eye.get("place_depth_min_mm", eye.get("depth_min_valid_mm", 80)))
        max_mm = float(
            eye.get(
                "place_depth_max_mm",
                max(800.0, float(eye.get("depth_action_max_mm", 300))),
            )
        )
        if pd is not None and np.isfinite(pd):
            pd = float(pd)
            if pd < min_mm or pd > max_mm:
                return None
        else:
            return None
        return (float(pu), float(pv), float(pd))

    def _compute_basket_slots(self, basket_hit):
        """筐 OBB → 沿长边三格（世界前=大 / 中 / 近=小）。"""
        if basket_hit is None:
            return []
        eye = self.config.get("eye_in_hand") or {}
        fwd = eye.get("place_slot_forward", [1.0, 0.0, 0.0])

        def _world(u, v, depth_mm):
            try:
                return self.controller.object_point_in_world(u, v, depth_mm)
            except Exception:
                return None

        return split_basket_slots_from_hit(
            basket_hit, world_of_uv=_world, forward=np.asarray(fwd, dtype=np.float64),
        )

    def _pinch_mid_in_ee(self, joints=None, hand_cmd=None):
        """夹取末端(pinch mid)相对 tool0，EE 系；放下用 tip 而非法兰。"""
        pack = self._pinch_tip_and_R_in_ee(joints=joints, hand_cmd=hand_cmd)
        return None if pack is None else pack[0]

    def _pinch_tip_and_R_in_ee(self, joints=None, hand_cmd=None):
        """
        闭合 pinch mid 与 R_pinch_in_ee（R_ee.T @ R_pinch）。
        放下 tip TCP = 外接圆心 mid（不是拇指）；可选第三项 thumb_ee 供诊断。
        返回 (tip_ee, R_pe) 或 (tip_ee, R_pe, thumb_ee)。
        """
        if self.mj_model is None:
            return None
        joints = joints if joints is not None else self.controller.get_joints()
        if joints is None:
            return None
        if hand_cmd is None:
            size_info = (
                getattr(self, "_locked_size_info", None)
                or getattr(self.controller, "grasp_size_info", None)
            )
            if not size_info and getattr(self, "_place_size_class", None):
                size_info = {"class": self._place_size_class}
            try:
                hand_cmd = grasp_close_preset(
                    self.controller,
                    size_info=size_info,
                    eye=self.config.get("eye_in_hand") or {},
                )
            except Exception:
                hand_cmd = list(
                    self.controller.hand_presets.get("grasp", DEFAULT_GRASP_PRESET)
                )
        pinch = self.compute_pinch_at_joints(joints, hand_cmd=hand_cmd)
        tool0 = self._mj_tool0_pose(joints, hand_cmd=hand_cmd)
        if pinch is None or tool0 is None or pinch.get("pinch_mid_m") is None:
            return None
        R_pw = pinch.get("R_world")
        if R_pw is None:
            return None
        mid_mj = np.asarray(pinch["pinch_mid_m"], dtype=np.float64).reshape(3)
        p_mj, R_mj = tool0
        tip_ee = R_mj.T @ (mid_mj - p_mj)
        R_pe = R_mj.T @ np.asarray(R_pw, dtype=np.float64).reshape(3, 3)
        thumb_ee = None
        th = pinch.get("thumb_tip_m")
        if th is not None:
            thumb_mj = np.asarray(th, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(thumb_mj)):
                thumb_ee = R_mj.T @ (thumb_mj - p_mj)
        return tip_ee, R_pe, thumb_ee

    def _resolve_place_slot_uvd(self, log=None):
        """
        按锁定尺寸选筐格 (u,v,rim_mm) 并写 _locked_place_xyz。
        大→最前格，中→中间，小→最近格；失败回退对准2锁点。
        """
        log = log or self.log
        cls = getattr(self, "_place_size_class", None)
        if not cls:
            size_info = getattr(self, "_locked_size_info", None)
            if isinstance(size_info, dict):
                cls = size_info.get("class")
        hit = getattr(self, "_basket_cache_hit", None)
        slots = getattr(self, "_basket_slots", None) or []
        if hit is not None and not slots:
            slots = self._compute_basket_slots(hit)
            self._basket_slots = slots
        slot = slot_for_size_class(slots, cls) if slots else None
        if slot is not None:
            dmm = slot.depth_mm
            if dmm is None and hit is not None:
                dmm = hit.rim_mm
            if dmm is None and self._locked_place_uvd is not None:
                dmm = self._locked_place_uvd[2]
            if dmm is not None and np.isfinite(dmm):
                zh = {"big": "大", "medium": "中", "small": "小"}.get(
                    slot.label, slot.label
                )
                log(
                    f"放下格：{zh}({slot.label}) "
                    f"u={slot.center_uv[0]:.0f} v={slot.center_uv[1]:.0f} "
                    f"← class={cls or '?'}"
                )
                uvd = (
                    float(slot.center_uv[0]),
                    float(slot.center_uv[1]),
                    float(dmm),
                )
                if slot.world_xyz is not None and np.all(np.isfinite(slot.world_xyz)):
                    self._locked_place_xyz = np.asarray(
                        slot.world_xyz, dtype=np.float64
                    ).reshape(3)
                else:
                    pt = self.controller.object_point_in_world(*uvd)
                    self._locked_place_xyz = (
                        np.asarray(pt, dtype=np.float64).reshape(3)
                        if pt is not None else None
                    )
                return uvd
        if self._locked_place_uvd is not None:
            log("放下格：无三格，用对准2锁点")
            uvd = tuple(float(x) for x in self._locked_place_uvd[:3])
            if self._locked_place_xyz is None:
                pt = self.controller.object_point_in_world(*uvd)
                if pt is not None:
                    self._locked_place_xyz = np.asarray(pt, dtype=np.float64).reshape(3)
            return uvd
        sample = self._sample_place()
        if sample is not None:
            log("放下格：无锁点，用当前筐心检测")
            return sample
        return None

    def _lock_place_slot_after_align(self, u, v, depth_mm, log=None):
        """对准2 后按尺寸锁格 UV + 世界点（放下不再跟漂 UV）。"""
        log = log or self.log
        self._locked_place_uvd = (float(u), float(v), float(depth_mm))
        self._locked_place_xyz = None

        # 盘装配：锁光轴对准的圆心，禁止筐三格重映射
        if self._place_target_is_plate():
            pt = self.controller.object_point_in_world(
                float(u), float(v), float(depth_mm),
            )
            if pt is not None:
                self._locked_place_xyz = np.asarray(pt, dtype=np.float64).reshape(3)
            self._place_locked_kind = "plate"
            # 长期冻结：第二颗及以后可跳过对准2
            if self._locked_place_xyz is not None:
                self._plate_frozen_xyz = self._locked_place_xyz.copy()
                self._plate_frozen_uvd = (
                    float(u), float(v), float(depth_mm),
                )
            # 冻结 latest，后续视觉/采样不再被持物螺母改写
            with self.lock:
                self.latest["place_u"] = float(u)
                self.latest["place_v"] = float(v)
                self.latest["place_depth_mm"] = float(depth_mm)
                self.latest["place_class"] = "plate"
            p = self._locked_place_xyz
            if p is not None:
                log(
                    f"对准2锁定盘心 "
                    f"u={float(u):.0f} v={float(v):.0f} "
                    f"depth={float(depth_mm)/10:.1f}cm "
                    f"世界[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]"
                    f"（已冻结；下颗可观察后直接放下）"
                )
            else:
                log(
                    f"对准2锁定盘心 UV "
                    f"u={float(u):.0f} v={float(v):.0f} "
                    f"depth={float(depth_mm)/10:.1f}cm（3D 失败）"
                )
            return True

        self._place_locked_kind = "basket"
        cls = getattr(self, "_place_size_class", None)
        hit = getattr(self, "_basket_cache_hit", None)
        slots = []
        if hit is not None:
            slots = self._compute_basket_slots(hit)
            self._basket_slots = slots
        slot = slot_for_size_class(slots, cls) if slots else None
        if slot is not None:
            dmm = slot.depth_mm if slot.depth_mm is not None else float(depth_mm)
            if hit is not None and hit.rim_mm is not None and slot.depth_mm is None:
                dmm = float(hit.rim_mm)
            u_s, v_s = float(slot.center_uv[0]), float(slot.center_uv[1])
            self._locked_place_uvd = (u_s, v_s, float(dmm))
            # 格心：用对准点定水平面 + 格 UV 射线求交（勿远格套近处 rim 深度）
            pt_align = self.controller.object_point_in_world(
                float(u), float(v), float(depth_mm),
            )
            pt = None
            plane_note = ""
            if pt_align is not None:
                eye = self.config.get("eye_in_hand") or {}
                wu = np.asarray(
                    eye.get("pinch_world_up", [0.0, 0.0, 1.0]), dtype=np.float64
                ).reshape(3)
                wn = float(np.linalg.norm(wu))
                wu = wu / wn if wn > 1e-9 else np.array([0.0, 0.0, 1.0])
                pt = self.controller.object_point_on_plane(u_s, v_s, pt_align, wu)
                if pt is not None:
                    plane_note = "平面投"
            if pt is None:
                pt = self.controller.object_point_in_world(u_s, v_s, float(dmm))
                plane_note = "深度投"
            if pt is not None:
                self._locked_place_xyz = np.asarray(pt, dtype=np.float64).reshape(3)
            zh = {"big": "大", "medium": "中", "small": "小"}.get(slot.label, slot.label)
            log(
                f"对准2锁定格 {zh}({slot.label}) "
                f"u={u_s:.0f} v={v_s:.0f} depth={float(dmm)/10:.1f}cm "
                f"← class={cls or '?'} {plane_note}"
            )
            if self._locked_place_xyz is not None:
                p = self._locked_place_xyz
                if pt_align is not None:
                    pa = np.asarray(pt_align, dtype=np.float64).reshape(3)
                    dlt = p - pa
                    eye = self.config.get("eye_in_hand") or {}
                    wu = np.asarray(
                        eye.get("pinch_world_up", [0.0, 0.0, 1.0]), dtype=np.float64
                    ).reshape(3)
                    wn = float(np.linalg.norm(wu))
                    wu = wu / wn if wn > 1e-9 else np.array([0.0, 0.0, 1.0])
                    fwd = np.asarray(
                        eye.get("place_slot_forward", [1.0, 0.0, 0.0]), dtype=np.float64
                    ).reshape(3)
                    fwd = fwd - float(np.dot(fwd, wu)) * wu
                    fn = float(np.linalg.norm(fwd))
                    fwd = fwd / fn if fn > 1e-9 else np.array([1.0, 0.0, 0.0])
                    right = np.cross(fwd, wu)
                    rn = float(np.linalg.norm(right))
                    right = right / rn if rn > 1e-9 else np.array([0.0, -1.0, 0.0])
                    log(
                        f"对准2★格心世界 [{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}] "
                        f"相对对准点 前{float(np.dot(dlt, fwd))*1000:.0f} "
                        f"右{float(np.dot(dlt, right))*1000:.0f} "
                        f"上{float(np.dot(dlt, wu))*1000:.0f}mm "
                        f"(对准uv=({float(u):.0f},{float(v):.0f}) {plane_note})"
                    )
        else:
            pt = self.controller.object_point_in_world(
                float(u), float(v), float(depth_mm),
            )
            if pt is not None:
                self._locked_place_xyz = np.asarray(pt, dtype=np.float64).reshape(3)
            if not cls:
                log(
                    f"对准2锁定对准点（无尺寸档→未按大中小分格，易放错格） "
                    f"u={float(u):.0f} v={float(v):.0f} "
                    f"depth={float(depth_mm)/10:.1f}cm"
                )
            elif not slots:
                log(
                    f"对准2锁定对准点（无三格） class={cls} "
                    f"u={float(u):.0f} v={float(v):.0f}"
                )
            else:
                log(
                    f"对准2：class={cls} 未匹配到格，锁对准点 "
                    f"u={float(u):.0f} v={float(v):.0f}"
                )
        return True

    def _measure_raw_depth_mm(self):
        """
        本帧框内原始深度（mm），绕过 DepthTracker 记忆。
        返回 (u, v, depth_mm)；无检测或质量不足时 depth 为 None。
        """
        with self.lock:
            depth_aligned = self.latest.get("depth_aligned")
            box = self.latest.get("box")
            u0 = self.latest.get("u")
            v0 = self.latest.get("v")
        if depth_aligned is None or box is None:
            return u0, v0, None
        eye = self.config["eye_in_hand"]
        R_wc = self.controller.camera_rotation_world()
        u, v, raw = measure_detection_uvd(
            depth_aligned,
            list(box) + [0.0, 0],
            eye,
            prior_mm=self._last_reliable_depth_mm,
            freeze_below_mm=0.0,
            R_world_cam=R_wc,
        )
        if raw is None or not np.isfinite(raw) or float(raw) <= 1.0:
            return (u if u is not None else u0), (v if v is not None else v0), None
        return float(u), float(v), float(raw)

    def _ensure_align_working_depth(self, log=None, stop=None):
        """
        对准前把工作距推到 ~align_working_depth_mm（默认 25cm）。
        过近时小螺母易定档/深度错；过远则沿光轴靠近一点。
        成功后清空旧可靠深度，避免「远一层」误拒新工作距。
        """
        log = log or self.log
        stop = stop or (lambda: self._align_stop or not self.running)
        eye = self.config["eye_in_hand"]
        if not bool(eye.get("align_working_depth_enable", True)):
            return True
        target = float(eye.get("align_working_depth_mm", 250.0))
        tol = float(eye.get("align_working_depth_tol_mm", 30.0))
        max_step = float(eye.get("align_working_depth_max_step_m", 0.08))
        max_tries = int(eye.get("align_working_depth_max_tries", 8))
        settle_s = float(eye.get("depth_retreat_settle_s", 0.18))

        # 故意换距：旧 ~24cm 可靠值不能挡 30cm
        self._last_reliable_depth_mm = None
        self._reset_depth_memory()

        for attempt in range(max_tries):
            if stop():
                log("对准工作距调整被中止")
                return False
            time.sleep(settle_s)
            _u, _v, raw = self._measure_raw_depth_mm()
            if raw is None or not np.isfinite(raw):
                log(f"对准工作距：无深度，回退 {max_step*1000:.0f}mm…")
                if not self.controller.retreat_along_optical_axis(max_step):
                    log("对准工作距：回退失败")
                    return False
                self._reset_depth_memory()
                continue
            d = float(raw)
            err = d - target
            if abs(err) <= tol:
                self._last_reliable_depth_mm = d
                self._reset_depth_memory()
                log(
                    f"对准工作距就绪 {d/10:.1f}cm"
                    f"（目标 {target/10:.0f}±{tol/10:.0f}cm）"
                )
                return True
            step = min(abs(err) / 1000.0, max_step)
            if err < 0:
                # 过近 → 回退
                log(
                    f"对准工作距 {d/10:.1f}cm < {target/10:.0f}cm，"
                    f"回退 {step*1000:.0f}mm（{attempt+1}/{max_tries}）…"
                )
                ok = self.controller.retreat_along_optical_axis(step)
            else:
                log(
                    f"对准工作距 {d/10:.1f}cm > {target/10:.0f}cm，"
                    f"靠近 {step*1000:.0f}mm（{attempt+1}/{max_tries}）…"
                )
                ok = self.controller.approach_along_optical_axis(step)
            if not ok:
                log("对准工作距：光轴移动失败")
                return False
            self._reset_depth_memory()
            time.sleep(max(settle_s, 0.22))

        # 最后一测：未进容差也尽量带当前深度继续（XY 对准仍可做）
        _u, _v, raw = self._measure_raw_depth_mm()
        if raw is not None and np.isfinite(raw):
            self._last_reliable_depth_mm = float(raw)
            log(
                f"对准工作距未完全到位 {float(raw)/10:.1f}cm"
                f"（目标 {target/10:.0f}cm），继续对准"
            )
            return True
        log("对准工作距调整失败：仍无有效深度")
        return False

    def _ensure_depth_before_action(self, log=None, stop=None):
        """
        对准/夹取前确认本帧真实物体深度。

        - 12–16cm 是正常工作距，必须接受（勿再用 ≥17cm 门槛拒掉）。
        - 仅当空洞/异常贴脸/过远背景时才回退或中止。
        - 多帧共识后再就绪，降低单帧飞点。
        - 成功后 reset tracker，不 stamp，避免记忆拖歪后续 UV。
        """
        log = log or self.log
        stop = stop or (lambda: self._align_stop or not self.running)
        eye = self.config["eye_in_hand"]
        # 螺母对准：先把距推到 ~25cm，再做共识（小目标过近易错）
        if bool(eye.get("align_working_depth_enable", True)):
            if not self._ensure_align_working_depth(log=log, stop=stop):
                return False
        step_m = float(eye.get("depth_retreat_step_m", 0.01))
        max_tries = int(eye.get("depth_retreat_max_tries", 12))
        settle_s = float(eye.get("depth_retreat_settle_s", 0.18))
        need_n = int(max(1, eye.get("depth_consensus_frames", 3)))
        cons_tol = float(eye.get("depth_consensus_tol_mm", 14.0))
        near_pct = float(eye.get("depth_consensus_near_percentile", 35.0))
        soft_prior = self._last_reliable_depth_mm
        far_rejects = 0
        accepted = []

        self._reset_depth_memory()

        for attempt in range(max_tries + 1):
            if stop():
                log("深度确认被中止")
                return False
            time.sleep(settle_s)
            _u, _v, raw = self._measure_raw_depth_mm()
            ok, why = depth_acceptable_for_action(raw, eye, soft_prior_mm=soft_prior)
            if ok:
                accepted.append(float(raw))
                if len(accepted) > need_n:
                    accepted = accepted[-need_n:]
                fused = consensus_depth_mm(
                    accepted, tol_mm=cons_tol, near_percentile=near_pct
                )
                if fused is None:
                    if len(accepted) >= need_n:
                        log(
                            f"深度帧间抖动 "
                            f"{min(accepted)/10:.1f}–{max(accepted)/10:.1f}cm，重测…"
                        )
                        accepted = accepted[-1:]
                    continue
                if len(accepted) < need_n:
                    continue
                self._last_reliable_depth_mm = float(fused)
                self._reset_depth_memory()
                time.sleep(settle_s * 0.5)
                if attempt == 0:
                    log(f"深度就绪 {fused/10:.1f}cm（共识{need_n}帧）")
                else:
                    log(
                        f"深度就绪 {fused/10:.1f}cm"
                        f"（共识{need_n}帧，已回退 {attempt}×{step_m*1000:.0f}mm）"
                    )
                return True

            accepted = []
            # 过远背景：回退帮不上，连续两次则中止（避免当成 40cm 物体猛冲）
            # 「远一层」= 孔/桌面误采，应重测/轻回退，勿当背景墙直接中止
            is_layer = raw is not None and ("远一层" in why or "孔/桌面" in why)
            is_far = raw is not None and (
                ("背景" in why or "飙远" in why) and not is_layer
            )
            if is_layer:
                log(f"{why}，重测…")
                self._reset_depth_memory()
                continue
            if is_far:
                far_rejects += 1
                log(f"{why}")
                if far_rejects >= 2 or attempt >= 1:
                    log("深度疑似背景墙，中止对准/夹取（请检查目标是否在框内）")
                    return False
                self._reset_depth_memory()
                continue

            if attempt >= max_tries:
                log(f"{why}")
                break
            log(
                f"{why}，沿光轴回退 {step_m*1000:.0f}mm "
                f"（{attempt + 1}/{max_tries}）…"
            )
            # 阻塞回退：等本段走完再测，禁止空转叠指令
            if not self.controller.retreat_along_optical_axis(step_m):
                log("回退运动失败，中止")
                return False
            self._reset_depth_memory()
            # 到位后再 settle，避免沿用运动中的深度帧
            time.sleep(max(settle_s, 0.25))
            accepted = []
            far_rejects = 0

        log("多次回退后仍无可用物体深度，中止（请检查深度图/目标）")
        return False

