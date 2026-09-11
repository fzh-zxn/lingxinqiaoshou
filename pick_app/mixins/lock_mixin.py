"""Object lock, depth memory, TCP diagnostics"""
from __future__ import annotations

from pick_app.deps import *
from pick_app.constants import *


class LockMixin:
    def log(self, message):
        self.log_var.set(str(message))
        print(message, flush=True)

    @staticmethod
    def _fmt_xyz(point, label):
        if point is None:
            return f"  {label}: —"
        p = np.asarray(point, dtype=np.float64).reshape(3)
        return f"  {label}: [{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}] m"

    def _log_grasp_marker_coords(self, tag, u=None, v=None, depth_mm=None, plan=None):
        """打印模型里各彩色球坐标，便于对照「对不准」原因（pinch_debug）。"""
        eye = self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {}
        if not eye_debug(eye):
            return
        cx = float(self.controller.intrinsics.get("cx", 320))
        cy = float(self.controller.intrinsics.get("cy", 240))
        self.log(f"── 抓取标记 [{tag}] ──")
        if u is not None and v is not None:
            err = math.hypot(float(u) - cx, float(v) - cy)
            self.log(
                f"  像素 抓取点=({float(u):.0f},{float(v):.0f}) "
                f"光轴=({cx:.0f},{cy:.0f}) 偏心={err:.0f}px"
            )
        if depth_mm is not None and np.isfinite(depth_mm):
            self.log(f"  深度 {float(depth_mm)/10:.1f}cm")

        pose = self.controller.get_pose()
        if pose is not None:
            pos, eul = pose
            self.log(
                f"  实机 tool0 xyz=[{pos.x:.3f},{pos.y:.3f},{pos.z:.3f}] "
                f"rpy°=[{math.degrees(eul.x):.1f},{math.degrees(eul.y):.1f},"
                f"{math.degrees(eul.z):.1f}]"
            )

        joints = self.controller.get_live_joints() if self.busy else self.controller.get_joints()
        t0 = self._mj_tool0_pos(joints) if self.mj_model is not None else None
        self.log(self._fmt_xyz(t0, "MJ tool0(当前)"))

        obj_mj = getattr(self, "_locked_object_mj", None)
        obj_robot = getattr(self, "_locked_object_robot", None)
        self.log(self._fmt_xyz(obj_mj, "青球⑤物体(MJ)"))
        self.log(self._fmt_xyz(obj_robot, "青球⑤物体(robot)"))

        if u is not None and v is not None and depth_mm is not None and pose is not None:
            live_mj = live_robot = None
            if self.mj_model is not None:
                self._sync_mj_state(joints)
                live_mj = _object_point_in_mj(
                    self.mj_model, self.mj_data, u, v, depth_mm,
                    self.controller.intrinsics, self.mj_cam_optical_in_body,
                    **self.controller._vision_ray_params(),
                )
            live_robot = self.controller.object_point_in_world(u, v, depth_mm)
            self.log(self._fmt_xyz(live_mj, "即时射线(MJ)"))
            self.log(self._fmt_xyz(live_robot, "即时射线(robot)"))
            if obj_robot is not None and live_robot is not None:
                drift = float(np.linalg.norm(obj_robot - live_robot)) * 1000.0
                self.log(f"  锁定 vs 即时(robot) 偏差 {drift:.1f}mm")
            if live_mj is not None and live_robot is not None:
                # 两套射线差（应≈锁定时打印的差）；拆 XYZ 便于对侧偏
                T_r, T_m = tool0_matrices_from_poses(pose, self._mj_tool0_pose(joints))
                live_mj_as_r = point_mj_to_robot(live_mj, T_r, T_m)
                if live_mj_as_r is not None:
                    d = live_robot - live_mj_as_r
                    self.log(
                        f"  射线差 robot−(MJ映射到robot) Δxyz_mm="
                        f"[{d[0]*1000:.1f},{d[1]*1000:.1f},{d[2]*1000:.1f}] "
                        f"|Δ|={float(np.linalg.norm(d))*1000:.1f}mm"
                    )
            if pose is not None and obj_robot is not None:
                reproj = world_point_to_image_pixel(
                    obj_robot, pose, self.controller.intrinsics,
                    self.controller.R_ee_cam, self.controller.t_ee_cam,
                    xy_rotate_deg=self.controller.xy_rotate_deg,
                    flip_x=self.controller.servo_flip_x,
                    flip_y=self.controller.servo_flip_y,
                    depth_sign=float(self.config["eye_in_hand"].get("camera_depth_sign", -1)),
                )
                if reproj is not None:
                    ru, rv = reproj
                    self.log(
                        f"  重投影 锁定物→像素 ({ru:.0f},{rv:.0f}) "
                        f"vs 抓取点 ({float(u):.0f},{float(v):.0f}) "
                        f"Δ=({ru-float(u):+.0f},{rv-float(v):+.0f})px"
                    )

        if plan is not None:
            self.log(self._fmt_xyz(plan.get("obj_world"), "青球⑤规划"))
            self.log(self._fmt_xyz(plan.get("obj_target"), "黄球预捏合"))
            self.log(self._fmt_xyz(plan.get("target_pos"), "红球⑥tool0目标(MJ)"))
            self.log(self._fmt_xyz(plan.get("ee_sim"), "灰球预抓EE(MJ)"))
            self.log(self._fmt_xyz(plan.get("open_center_m"), "三指mid张手(MJ)"))
            self.log(self._fmt_xyz(plan.get("grasp_center_m"), "三指mid闭合(MJ)"))
            self.log(self._fmt_xyz(plan.get("thumb_tip_m"), "拇指tip闭合"))
            self.log(self._fmt_xyz(plan.get("index_tip_m"), "食指tip闭合"))
            self.log(self._fmt_xyz(plan.get("middle_tip_m"), "中指tip闭合"))
            yellow = plan.get("obj_target")
            close_m = plan.get("grasp_center_m")
            if yellow is not None and close_m is not None:
                gap = float(
                    np.linalg.norm(
                        np.asarray(close_m, dtype=np.float64)
                        - np.asarray(yellow, dtype=np.float64)
                    )
                ) * 1000.0
                self.log(f"  闭合三指mid→黄球(当前姿态) {gap:.1f}mm（⑥ 要消掉的间隙）")
            ik = plan.get("robot_ik")
            if ik is not None:
                tp = ik.get("target_pos")
                tr = ik.get("target_rpy")
                if tp is not None:
                    self.log(
                        f"  红球⑥tool0(robot) xyz=[{tp[0]:.3f},{tp[1]:.3f},{tp[2]:.3f}]"
                        + (
                            f" rpy°=[{math.degrees(tr[0]):.1f},"
                            f"{math.degrees(tr[1]):.1f},{math.degrees(tr[2]):.1f}]"
                            if tr is not None else ""
                        )
                    )
            planned_red = plan.get("target_pos")
            if t0 is not None and planned_red is not None:
                err_mm = float(
                    np.linalg.norm(t0 - np.asarray(planned_red, dtype=np.float64))
                ) * 1000.0
                self.log(f"  当前 tool0 → 红球目标 距离 {err_mm:.1f}mm")
            gray = plan.get("ee_sim")
            if planned_red is not None and gray is not None:
                rg = float(
                    np.linalg.norm(
                        np.asarray(planned_red, dtype=np.float64)
                        - np.asarray(gray, dtype=np.float64)
                    )
                ) * 1000.0
                self.log(f"  红球相对灰球(预抓) {rg:.1f}mm（⑥ 姿态对齐位移，≠误差）")

        pinch = getattr(self, "_pinch_info", None)
        if pinch is not None:
            self.log(self._fmt_xyz(pinch.get("pinch_mid_m"), "三指mid(当前MJ·随手姿)"))

    def _flush_model_canvas(self):
        """拖拽/缩放后立即重绘模型（不等 ui tick）。"""
        if self.mj_renderer is None or self.mj_cam is None:
            return
        try:
            photo = self._render_model(self._joints_for_model_view())
        except Exception:
            return
        if photo is None:
            return
        self.model_photo = photo
        if not self.model_canvas.winfo_exists():
            return
        self.model_canvas.delete("all")
        cw, ch = self._model_view_dims()
        self.model_canvas.create_image(cw // 2, ch // 2, image=self.model_photo)

    def _joints_for_model_view(self):
        if self._model_joints is not None:
            return list(self._model_joints)
        if self.busy:
            return list(self.controller.get_live_joints())
        return list(self.controller.get_joints())

    def _clear_locked_object(self, clear_place_size=False):
        """清除紫/青锁定球与像素锁（抓取结束或重新对准前调用）。"""
        self._locked_object_mj = None
        self._locked_object_robot = None
        self._locked_surface_robot = None
        self._locked_detection = None
        self._vision_obj_world = None
        self._locked_size_info = None
        self._last_size_info = None
        if clear_place_size:
            self._place_size_class = None
        if getattr(self, "controller", None) is not None:
            self.controller.grasp_size_info = None
        self._det_hold_box = None
        self._det_hold_n = 0

    def _remember_place_size(self, size_info):
        """夹取定档后记住放下用尺寸（观察/对准2 不清）。"""
        cls = None
        if isinstance(size_info, dict):
            cls = size_info.get("class")
        if cls:
            self._place_size_class = str(cls).strip().lower()
            if self._place_size_class == "large":
                self._place_size_class = "big"

    def _reset_depth_memory(self):
        """清深度记忆与 latest，强制下一帧重新测（禁止沿用残留）。"""
        self._depth_tracker.reset()
        self._ring_depth_buf.reset()
        if getattr(self, "_size_bank", None) is not None:
            self._size_bank.reset()
        with self.lock:
            self.latest["depth_mm"] = None

    def _after_pick_or_align_cleanup(self, clear_lock=True):
        """动作结束统一收尾，避免状态残留成石山。"""
        if clear_lock:
            self._clear_locked_object()
            self._pinch_plan = None
        self._reset_depth_memory()
        self.busy = False
        self._align_stop = False
        self._place_aligning = False

    def lock_object_for_grasp(
        self, u, v, depth_mm, log=None, source="pregrasp", apply_bias=True,
    ):
        """
        预抓后锁定物体 3D（模型青球与 ⑤ 规划不再随检测变化）。

        以实机工作系点为准（yaml camera_on_ee / 验参同一套），再映射进 MuJoCo，
        避免 MJ 相机射线与手调外参两套点导致 ⑥ 规划偏几厘米。
        """
        log = log or self.log
        if u is None or v is None or depth_mm is None:
            return False
        depth_mm = float(depth_mm)
        eye = self.config["eye_in_hand"]
        ok_d, why_d = depth_acceptable_for_action(
            depth_mm, eye, soft_prior_mm=self._last_reliable_depth_mm,
        )
        if not ok_d:
            log(f"物体锁定拒绝({source})：{why_d}")
            return False
        self._last_reliable_depth_mm = depth_mm
        joints = self.controller.get_joints()
        if self.mj_model is not None and self.mj_data is not None:
            self._sync_mj_state(joints)

        obj_robot = self.controller.object_point_in_world(u, v, depth_mm)
        obj_mj_ray = None
        if self.mj_model is not None and self.mj_data is not None:
            obj_mj_ray = _object_point_in_mj(
                self.mj_model, self.mj_data, u, v, depth_mm,
                self.controller.intrinsics, self.mj_cam_optical_in_body,
                **self.controller._vision_ray_params(),
            )

        if obj_robot is None and obj_mj_ray is None:
            log("物体锁定失败：无法计算 3D 点")
            return False

        # 先定档（下偏量依赖外径 AF）；YOLO 尺寸类仅作可选先验
        size_info = None
        det_cfg = self.config.get("detection") or {}
        with self.lock:
            box = self.latest.get("box")
            size_info = self.latest.get("size_info")
            yolo_lock = self.latest.get("yolo_class")
            yolo_conf_lock = self.latest.get("confidence")
        if box is not None:
            measured = measure_and_classify_nut(
                box,
                depth_mm,
                self.controller.intrinsics,
                det_cfg,
                yolo_cls=(
                    yolo_lock
                    if (yolo_lock in nut_size_classes())
                    else None
                ),
                yolo_conf=yolo_conf_lock,
            )
            if measured is not None:
                size_info = measured

        # 侧偏 + 世界系下偏（上表面识别点 → 略低于表面，便于捏住）
        obj_robot_raw = (
            np.asarray(obj_robot, dtype=np.float64).reshape(3).copy()
            if obj_robot is not None else None
        )
        if obj_robot is not None and apply_bias:
            pose_r = self.controller.get_pose()
            if pose_r is not None:
                pos_r, eul_r = pose_r
                ee_r = np.array([pos_r.x, pos_r.y, pos_r.z], dtype=np.float64)
                R_we = euler_rpy_to_matrix(eul_r.x, eul_r.y, eul_r.z)
                up_r = self.controller._pinch_object_up(R_we)
                obj_robot = apply_pinch_grasp_bias(
                    obj_robot, ee_r, up_r, eye, log=log, size_info=size_info,
                    detection=det_cfg,
                )

        obj_mj = None
        lock_tag = ""
        if obj_robot is not None:
            self._locked_object_robot = np.asarray(obj_robot, dtype=np.float64).copy()
            # 上表面点（未下偏）：彩色叠加以此反投影，应落在通孔中心
            self._locked_surface_robot = (
                np.asarray(obj_robot_raw, dtype=np.float64).copy()
                if obj_robot_raw is not None
                else self._locked_object_robot.copy()
            )
            pose_r = self.controller.get_pose()
            pose_mj = self._mj_tool0_pose(joints)
            T_r, T_m = tool0_matrices_from_poses(pose_r, pose_mj)
            mapped = point_robot_to_mj(obj_robot, T_r, T_m)
            mapped_raw = point_robot_to_mj(obj_robot_raw, T_r, T_m)
            if mapped is not None:
                obj_mj = mapped
                lock_tag = "实机→MJ"
                if obj_mj_ray is not None and mapped_raw is not None:
                    # 必须与未下偏的射线比；否则 ΔZ≈world_down 会被误当成外参误差
                    d_vec = mapped_raw - obj_mj_ray
                    d_mm = float(np.linalg.norm(d_vec)) * 1000.0
                    bias_mm = float(np.linalg.norm(
                        np.asarray(obj_robot) - obj_robot_raw
                    )) * 1000.0
                    log(
                        f"锁定物体以实机点为准映射到 MJ"
                        f"（未下偏 vs MJ射线 |Δ|={d_mm:.1f}mm "
                        f"Δxyz_mm=[{d_vec[0]*1000:.1f},{d_vec[1]*1000:.1f},"
                        f"{d_vec[2]*1000:.1f}]；下偏后另有 {bias_mm:.1f}mm）"
                    )
                    # 外参/相机模型侧向差：只把 XY（垂直 world_up）拉向 MJ 射线，保留实机下偏 Z
                    # 日志常见 Δy≈5–7mm → 规划青球偏左 → 实机抓偏左；★C 仍「对青」好看
                    wu = np.asarray(
                        eye.get("pinch_world_up", [0.0, 0.0, 1.0]), dtype=np.float64
                    ).reshape(3)
                    wn = float(np.linalg.norm(wu))
                    wu = wu / wn if wn > 1e-9 else np.array([0.0, 0.0, 1.0])
                    d_lat = d_vec - float(np.dot(d_vec, wu)) * wu
                    lat_mm = float(np.linalg.norm(d_lat)) * 1000.0
                    fuse_mm = float(eye.get("lock_mj_ray_fuse_lateral_mm", 4.0))
                    if lat_mm >= fuse_mm and bool(eye.get("lock_mj_ray_fuse_lateral", True)):
                        obj_mj = np.asarray(mapped, dtype=np.float64).reshape(3) - d_lat
                        # 同步实机锁点侧向，避免 robot/MJ 再漂开
                        p_fix = point_mj_to_robot(obj_mj, T_r, T_m)
                        if p_fix is not None:
                            self._locked_object_robot = np.asarray(
                                p_fix, dtype=np.float64
                            ).reshape(3)
                        lock_tag = "实机→MJ+侧向融射线"
                        log(
                            f"  ⇒ 锁点侧向融 MJ 射线 {lat_mm:.1f}mm "
                            f"（Δlat_mm=[{d_lat[0]*1000:.1f},{d_lat[1]*1000:.1f},"
                            f"{d_lat[2]*1000:.1f}]；查 camera_on_ee 若反复 >5mm）"
                        )
            else:
                log("锁定：实机→MJ 映射失败，回退 MJ 相机射线")
        else:
            self._locked_object_robot = None
            self._locked_surface_robot = None

        if obj_mj is None and obj_mj_ray is not None:
            obj_mj = np.asarray(obj_mj_ray, dtype=np.float64).copy()
            lock_tag = "MJ相机射线"
            if apply_bias and self._locked_object_robot is None:
                pose_mj = self._mj_tool0_pose(joints)
                if pose_mj is not None:
                    ee_m, R_m = pose_mj
                    R_cam = R_m @ np.asarray(
                        self.controller.R_ee_cam, dtype=np.float64
                    ).reshape(3, 3)
                    up_m = object_up_direction_world(
                        R_cam,
                        grasp_point_mode=eye.get("grasp_point_mode", "bbox_center"),
                        grasp_point_side=eye.get("grasp_point_side", "right"),
                        xy_rotate_deg=float(eye.get("servo_xy_rotate_deg", 90.0)),
                        flip_x=bool(eye.get("servo_flip_x", True)),
                        flip_y=bool(eye.get("servo_flip_y", False)),
                        world_up_fallback=eye.get("pinch_world_up", [0.0, 0.0, 1.0]),
                    )
                    obj_mj = apply_pinch_grasp_bias(
                        obj_mj, ee_m, up_m, eye, log=log, size_info=size_info,
                        detection=det_cfg,
                    )

        if obj_mj is not None:
            self._locked_object_mj = np.asarray(obj_mj, dtype=np.float64).copy()
        else:
            self._locked_object_mj = None

        self._locked_detection = (float(u), float(v), depth_mm)
        self._locked_size_info = size_info
        self._last_size_info = size_info
        self._remember_place_size(size_info)
        if getattr(self, "controller", None) is not None:
            self.controller.grasp_size_info = size_info
        if size_info is not None:
            try:
                eye = self.config.get("eye_in_hand", {})
                close_sz = grasp_close_preset(
                    self.controller, size_info=size_info, eye=eye,
                )
                self.controller.hand_presets["pinch_close"] = list(close_sz)
                self.controller.hand_presets["grasp"] = list(close_sz)
                stab_sz = grasp_stabilize_preset(
                    self.controller, size_info=size_info, eye=eye,
                )
                if stab_sz is not None:
                    self.controller.hand_presets["pinch_stabilize"] = list(stab_sz)
                if hasattr(self, "hand_grasp_vars"):
                    for var, v in zip(self.hand_grasp_vars, close_sz):
                        var.set(int(v))
                from lbot_grasp_utils import resolve_pinch_standoff_m
                so_m = resolve_pinch_standoff_m(eye, size_info)
                log(
                    f"闭合手按尺寸 {size_info.get('class')} "
                    f"mode={pinch_grasp_mode(eye=eye)} → 主抓L6={close_sz}"
                    + (f" 固定L6={stab_sz}" if stab_sz else "")
                    + f" standoff={so_m*1000:.0f}mm"
                )
            except Exception as _e:
                log(f"尺寸闭合档写入跳过: {_e}")
        ref = (
            self._locked_object_robot
            if self._locked_object_robot is not None
            else self._locked_object_mj
        )
        mode = eye.get("grasp_point_mode", "bbox_center")
        side = eye.get("grasp_point_side", "right")
        mode_note = str(mode)
        if bool(eye.get("grasp_use_ring_center", True)):
            mode_note = "ring_center（孔核环心）"
        elif str(mode).lower() in ("hand_top", "hand_upper", "side_top", "world_top", "world_up"):
            mode_note = f"{mode}/{side}（世界上边投影选边）"
        elif str(mode).lower() in ("bbox_center", "center", ""):
            mode_note = "bbox_center（框心）"
        log(
            f"物体已锁定({source}/{lock_tag or '—'}) @ "
            f"[{ref[0]:.3f}, {ref[1]:.3f}, {ref[2]:.3f}] m "
            f"depth={depth_mm/10:.1f}cm 抓取点={mode_note}"
        )
        if size_info is not None:
            yolo_note = ""
            with self.lock:
                yolo_cls = self.latest.get("yolo_class")
            if yolo_cls:
                yolo_note = f" YOLO={yolo_cls}"
            amb = "（模糊，请再靠近/校准 size_scale）" if size_info.get("ambiguous") else ""
            log(
                f"尺寸定档: {size_info.get('class')} "
                f"六边形对边={size_info.get('hex_mm', size_info.get('long_mm', 0)):.1f}mm "
                f"距标称{size_info.get('dist_mm', 0):.1f}mm "
                f"raw={size_info.get('hex_raw_mm', size_info.get('long_raw_mm', 0)):.1f} "
                f"mode={((size_info.get('calib') or {}).get('mode', '?'))}"
                f"{yolo_note}{amb}"
            )
            if (
                yolo_cls
                and yolo_cls in nut_size_classes()
                and size_info.get("class")
                and yolo_cls != size_info.get("class")
            ):
                log(
                    f"⚠ YOLO={yolo_cls} 与深度档={size_info.get('class')} 不一致 → 以深度为准"
                )
        self._log_grasp_marker_coords(f"锁定·{source}", u, v, depth_mm)
        if self.mj_model is not None and not self.busy and apply_bias:
            pinch_open = list(
                self.controller.hand_presets.get("pinch_open", DEFAULT_PINCH_OPEN)
            )
            snap = self._compute_pinch_plan_mj(
                u, v, depth_mm, joints, pinch_open,
                log=self.log,
                already_at_pregrasp=True,
            )
            if snap is not None:
                self._pinch_plan = snap
                self._pinch_plan_time = time.time()
        return True

    def _log_hand_mj_preset_compare(self):
        """启动时：yaml/面板 pinch_open/close 在 MJ 下的三指几何（对照实机 ⑧）。"""
        if self.mj_model is None or self.mj_data is None:
            return
        hp = self.controller.hand_presets
        pinch_open = list(hp.get("pinch_open", DEFAULT_PINCH_OPEN))
        pinch_close = list(
            hp.get("grasp", hp.get("pinch_close", DEFAULT_GRASP_PRESET))
        )
        joints = list(self.controller.default_home_joints)
        eye = self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {}
        mirror = bool(eye.get("model_hand_qpos_mirror", False))
        self.log("── O6 手预设 vs MJ（home 关节 FK）──")
        with self._mj_lock:
            log_hand_preset_mj_compare(
                self.mj_model,
                self.mj_data,
                self.mj_joints,
                self.mj_hand_ranges,
                pinch_open,
                pinch_close,
                joints,
                self._arm_joint_signs(),
                self._pinch_kin_kwargs(),
                mirror_qpos=mirror,
                log=self.log,
            )

    def _log_pinch_tcp_diag(self, phase, plan=None):
        """
        夹取关键步诊断（盯 ★ 行）：
        A  API tool0 vs 规划目标（笛卡尔到位）
        B  实机关节→MJ FK tool0 vs 规划 MJ tool0
        C  闭合 mid→黄/青（默认只打 ★C+判读；全文开 pinch_debug）
        """
        eye = self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {}
        debug = eye_debug(eye)
        _full = self.log

        def _diag_line_always(s):
            s = str(s)
            return any(
                k in s
                for k in (
                    "── TCP",
                    "★A ",
                    "★B ",
                    "★C ",
                    "★C口径",
                    "★D ",
                    "★Z ",
                    "⇒",
                    "张→闭",
                    "张手mid",
                    "规划黄",
                    "闭合mid(T_mr",
                    "闭合mid(robot",
                    "规划tool0",
                    "规划接触",
                )
            )

        def log(msg):
            if debug or self.busy or _diag_line_always(msg):
                _full(msg)

        obj_r = getattr(self, "_locked_object_robot", None)
        pose = self.controller.get_pose()
        joints = self.controller.get_live_joints()
        if joints is None:
            joints = self.controller.get_joints()
        if pose is None or joints is None:
            _full(f"── TCP诊断[{phase}]：无位姿/关节，跳过 ──")
            return
        pos, eul = pose
        ee = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        jdeg = [round(math.degrees(float(j)), 1) for j in list(joints)[:7]]
        log(f"── TCP诊断[{phase}] ──")
        log(
            f"  API tool0 xyz=[{ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f}] "
            f"rpy°=[{math.degrees(eul.x):.1f},{math.degrees(eul.y):.1f},"
            f"{math.degrees(eul.z):.1f}] joints°={jdeg}"
        )
        if obj_r is not None:
            obj_r = np.asarray(obj_r, dtype=np.float64).reshape(3)
            log(
                f"  锁定物(robot) [{obj_r[0]:.4f},{obj_r[1]:.4f},{obj_r[2]:.4f}] "
                f"tool0→物 |Δ|={float(np.linalg.norm(obj_r - ee))*1000:.1f}mm"
            )

        def _along_lat_from_axis(p, obj, axis_from_obj):
            """p−obj 沿「物体→手侧」轴分解；axis 用黄−青或同向向量。"""
            d = np.asarray(p, dtype=np.float64).reshape(3) - np.asarray(
                obj, dtype=np.float64
            ).reshape(3)
            u = np.asarray(axis_from_obj, dtype=np.float64).reshape(3)
            un = float(np.linalg.norm(u))
            if un < 1e-9:
                return float(np.linalg.norm(d)) * 1000.0, float("nan")
            u = u / un
            along = float(np.dot(d, u)) * 1000.0  # >0 = 在黄球一侧（远离青球）
            lat = float(np.linalg.norm(d - u * (along / 1000.0))) * 1000.0
            return along, lat

        api_err_mm = None
        api_geo = None
        if plan is not None:
            rik = plan.get("robot_ik") or {}
            tp = rik.get("target_pos")
            tr = rik.get("target_rpy")
            if tp is None:
                tp = plan.get("target_pos_robot")
            if tp is not None:
                tp = np.asarray(tp, dtype=np.float64).reshape(3)
                err = ee - tp
                api_err_mm = float(np.linalg.norm(err)) * 1000.0
                if tr is not None:
                    try:
                        api_geo = rotation_geodesic_deg(
                            (eul.x, eul.y, eul.z),
                            (float(tr[0]), float(tr[1]), float(tr[2])),
                        )
                    except Exception:
                        api_geo = None
                msg = (
                    f"  ★A API到位 实际−规划 |Δ|={api_err_mm:.1f}mm "
                    f"Δxyz_mm=[{err[0]*1000:.1f},{err[1]*1000:.1f},{err[2]*1000:.1f}]"
                )
                if api_geo is not None:
                    msg += f" 姿态={api_geo:.1f}°"
                _eye = self.config.get("eye_in_hand", {}) if isinstance(
                    self.config, dict
                ) else {}
                _pm = str(_eye.get("pinch_path_mode", "cart")).strip().lower()
                if _pm in (
                    "mj_ik_smooth", "cart_mj_polish", "mj_ik_joint", "mj_joint",
                    "preview_joint", "preview_ik", "mj_ik_stepped",
                    "mj_ik_multi", "preview_stepped",
                ):
                    msg += "  （MJ-IK 模式下盯★B/C；★A 相对旧笛卡尔目标可偏）"
                else:
                    msg += "  （⑥期望 ≲10mm / ≲3.5°）"
                log(msg)

        if self.mj_model is None:
            return

        pinch_close = None
        if plan is not None:
            hc = plan.get("grasp_close_hand") or plan.get("hand_close")
            if hc is not None:
                pinch_close = list(hc)
        if pinch_close is None:
            size_info = None
            if plan is not None:
                size_info = plan.get("size_info")
            if size_info is None:
                size_info = (
                    getattr(self, "_locked_size_info", None)
                    or getattr(self, "_last_size_info", None)
                )
            pinch_close = grasp_close_preset(
                self.controller, size_info=size_info, eye=eye,
            )
        pinch_open = list(
            self.controller.hand_presets.get("pinch_open", DEFAULT_PINCH_OPEN)
        )
        kin_kw = self._pinch_kin_kwargs()

        # 用闭合手同步：与规划几何一致
        if not self._sync_mj_state(joints, hand_cmd=pinch_close):
            log("  MJ 同步失败，后续跳过")
            return
        tool0_mj = self._mj_tool0_pose(joints, hand_cmd=pinch_close)
        if tool0_mj is None:
            log("  无 MJ tool0")
            return
        p_m, R_m = tool0_mj
        p_m = np.asarray(p_m, dtype=np.float64).reshape(3)
        T_r, T_m = tool0_matrices_from_poses(pose, tool0_mj)

        yellow_m = obj_m = None
        mj_tool0_err = None
        if plan is not None:
            yellow_m = plan.get("obj_target")
            obj_m = plan.get("obj_world")
            T_ee = plan.get("T_ee_des")
            # ⑦后精修目标是接触位，勿再用黄球 T_ee_des 骂 ★B
            if phase and ("⑦" in str(phase) or "⑧" in str(phase) or "闭合" in str(phase)):
                T_contact = plan.get("T_ee_contact")
                if T_contact is not None:
                    T_ee = T_contact
            if T_ee is not None:
                T_ee = np.asarray(T_ee, dtype=np.float64).reshape(4, 4)
                p_des = T_ee[:3, 3]
                d_t = p_m - p_des
                mj_tool0_err = float(np.linalg.norm(d_t)) * 1000.0
                # 姿态：测地近似用旋转矩阵
                R_des = T_ee[:3, :3]
                R_err = R_m.T @ R_des
                cos_th = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
                mj_att = math.degrees(math.acos(cos_th))
                tgt_tag = "T_ee_contact" if (
                    phase and "⑦" in str(phase) and plan.get("T_ee_contact") is not None
                ) else "T_ee_des"
                log(
                    f"  ★B 关节→MJ FK tool0−规划MJ |Δ|={mj_tool0_err:.1f}mm "
                    f"Δxyz_mm=[{d_t[0]*1000:.1f},{d_t[1]*1000:.1f},{d_t[2]*1000:.1f}] "
                    f"姿态≈{mj_att:.1f}° ({tgt_tag})"
                )
                log(
                    f"  规划tool0 [{p_des[0]:.4f},{p_des[1]:.4f},{p_des[2]:.4f}] "
                    f"实际MJ [{p_m[0]:.4f},{p_m[1]:.4f},{p_m[2]:.4f}]"
                )
            elif plan.get("T_ee_des") is not None:
                p6 = np.asarray(plan["T_ee_des"], dtype=np.float64).reshape(4, 4)[:3, 3]
                log(
                    f"  规划tool0(⑥黄) [{p6[0]:.4f},{p6[1]:.4f},{p6[2]:.4f}]"
                )
            T_ct = plan.get("T_ee_contact")
            if T_ct is not None and phase and "⑦" in str(phase):
                pc = np.asarray(T_ct, dtype=np.float64).reshape(4, 4)[:3, 3]
                log(
                    f"  规划接触tool0(⑦青) [{pc[0]:.4f},{pc[1]:.4f},{pc[2]:.4f}]"
                )
            if debug:
                log(
                    f"  MJ tool0(关节FK) [{p_m[0]:.4f},{p_m[1]:.4f},{p_m[2]:.4f}]"
                )

        info_c = _compute_o6_pinch_kinematics(self.mj_model, self.mj_data, **kin_kw)
        mid_close_m = None
        if info_c is not None and info_c.get("pinch_mid_m") is not None:
            mid_close_m = np.asarray(info_c["pinch_mid_m"], dtype=np.float64).reshape(3)
            gmode = str(info_c.get("grasp_mode") or kin_kw.get("grasp_mode") or "")
            if gmode == "power5":
                circ_r = float(info_c.get("circumradius_mm") or 0.0)
                mode = info_c.get("tcp_origin_mode") or "circumcenter"
                log(
                    f"  外接圆(指令FK) R={circ_r:.1f}mm mode={mode} mid=["
                    f"{mid_close_m[0]:.3f},{mid_close_m[1]:.3f},{mid_close_m[2]:.3f}] "
                    f"sep={float(info_c.get('pinch_sep_mm', 0)):.1f}mm"
                )
                for key, lab in (
                    ("ring_tip_m", "无"),
                    ("index_tip_m", "食"),
                    ("pinky_tip_m", "小"),
                ):
                    tip = info_c.get(key)
                    if tip is None:
                        continue
                    t = np.asarray(tip, dtype=np.float64).reshape(3)
                    log(
                        f"    {lab}tip [{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]"
                    )

        mid_to_y = mid_along = mid_lat = None
        if plan is not None and mid_close_m is not None:
            if yellow_m is not None and obj_m is not None:
                yellow_m = np.asarray(yellow_m, dtype=np.float64).reshape(3)
                obj_m = np.asarray(obj_m, dtype=np.float64).reshape(3)
                mid_to_y = float(np.linalg.norm(mid_close_m - yellow_m)) * 1000.0
                mid_along, mid_lat = _along_lat_from_axis(
                    mid_close_m, obj_m, yellow_m - obj_m
                )
                standoff = float(np.linalg.norm(yellow_m - obj_m)) * 1000.0
                log(
                    f"  青球(抓取点) [{obj_m[0]:.4f},{obj_m[1]:.4f},{obj_m[2]:.4f}] "
                    f"黄球(standoff) [{yellow_m[0]:.4f},{yellow_m[1]:.4f},{yellow_m[2]:.4f}]"
                )
                log(
                    f"  ★C 闭合mid(MJ)→黄球 {mid_to_y:.1f}mm；"
                    f"→青球 along={mid_along:.1f}mm 侧向={mid_lat:.1f}mm"
                )
                ph = str(phase or "")
                if any(k in ph for k in ("⑦", "⑧", "闭合", "接触")):
                    log(
                        f"  ★C口径 ⑦/⑧: 闭合mid→青 along≈0(≤3佳)、侧向≈0；"
                        f"→黄≈standoff {standoff:.0f}mm"
                    )
                else:
                    log(
                        f"  ★C口径 ⑥: →黄≈0、侧向≈0、along≈{standoff:.0f}"
                    )
                st = yellow_m - obj_m
                log(
                    f"  规划黄−青 |Δ|={float(np.linalg.norm(st))*1000:.1f}mm "
                    f"Δxyz_mm=[{st[0]*1000:.1f},{st[1]*1000:.1f},{st[2]*1000:.1f}]"
                )

        # 张手 mid（实机此步仍张手；闭合 mid 为「若此刻合指」几何）
        mid_open_m = None
        mid_open_along = mid_open_lat = None
        info_o = None
        if self._sync_mj_state(joints, hand_cmd=pinch_open):
            info_o = _compute_o6_pinch_kinematics(
                self.mj_model, self.mj_data, **kin_kw
            )
            if info_o is not None and info_o.get("pinch_mid_m") is not None:
                mid_open_m = np.asarray(
                    info_o["pinch_mid_m"], dtype=np.float64
                ).reshape(3)
                if yellow_m is not None and obj_m is not None:
                    mid_open_along, mid_open_lat = _along_lat_from_axis(
                        mid_open_m, obj_m, yellow_m - obj_m
                    )
                    log(
                        f"  张手mid(MJ)→青 along={mid_open_along:.1f}mm "
                        f"侧向={mid_open_lat:.1f}mm "
                        f"（仅参考；判抓取看闭合mid）"
                    )
        if mid_open_m is not None and mid_close_m is not None:
            s = mid_close_m - mid_open_m
            log(
                f"  张→闭 mid(MJ·运动学) |Δ|={float(np.linalg.norm(s))*1000:.1f}mm "
                f"Δxyz_mm=[{s[0]*1000:.1f},{s[1]*1000:.1f},{s[2]*1000:.1f}] "
                f"（非到位误差；tool0在法兰，张/闭不改变臂FK）"
            )

        # robot 系 mid：用 tool0 + R @ mid_ee（与锁存同系，勿用易漂的 T_mr 映青）
        if mid_close_m is not None and obj_r is not None:
            # mid 在 tool0 系（MJ FK）；API 姿态与 MJ 姿态已对齐时可用于估实机 mid
            mid_ee = None
            if tool0_mj is not None:
                R_mj = np.asarray(tool0_mj[1], dtype=np.float64).reshape(3, 3)
                p_mj = np.asarray(tool0_mj[0], dtype=np.float64).reshape(3)
                mid_ee = R_mj.T @ (
                    np.asarray(mid_close_m, dtype=np.float64).reshape(3) - p_mj
                )
            R_api = euler_rpy_to_matrix(eul.x, eul.y, eul.z)
            mid_est = None
            if mid_ee is not None:
                mid_est = ee + R_api @ mid_ee
            eye_z = self.config.get("eye_in_hand", {}) if isinstance(
                self.config, dict
            ) else {}
            down_z, down_src = pinch_grasp_down_m(
                eye_z,
                getattr(self, "_locked_size_info", None),
                detection=self.config.get("detection") if isinstance(self.config, dict) else None,
                with_source=True,
            )
            raw_z = float(obj_r[2] + down_z)  # 锁存已含 −down；回加得未下偏上表面
            hand_mm = (
                float(np.linalg.norm(mid_ee)) * 1000.0 if mid_ee is not None else float("nan")
            )
            to_lock_mm = float(np.linalg.norm(obj_r - ee)) * 1000.0
            if mid_est is not None:
                dz_lock = float(mid_est[2] - obj_r[2]) * 1000.0
                dz_raw = float(mid_est[2] - raw_z) * 1000.0
                log(
                    f"  ★Z 高度(robot): mid估 z={mid_est[2]:.4f} 锁存(含下偏) z={obj_r[2]:.4f} "
                    f"上表面估 z={raw_z:.4f}（下偏 {down_z*1000:.1f}mm/{down_src}）"
                )
                log(
                    f"  ★Z mid−锁存 Δz={dz_lock:+.1f}mm  mid−上表面 Δz={dz_raw:+.1f}mm "
                    f"（+为偏高；预期 mid≈锁存→Δz≈0） "
                    f"|tool0→mid|_ee={hand_mm:.0f}mm |tool0→锁存|={to_lock_mm:.0f}mm"
                )
                if abs(hand_mm - to_lock_mm) > 15.0:
                    log(
                        f"  ★Z ⇒ 手长不一致：MJ指尖距法兰 {hand_mm:.0f}mm，"
                        f"法兰→锁存 {to_lock_mm:.0f}mm（差 {hand_mm-to_lock_mm:.0f}mm）；"
                        f"若实指比模型短，mid 会比 MJ★C 偏高"
                    )
            if (
                yellow_m is not None
                and obj_m is not None
                and T_r is not None
            ):
                mid_r = point_mj_to_robot(mid_close_m, T_r, T_m)
                y_r = point_mj_to_robot(yellow_m, T_r, T_m)
                obj_r_mapped = point_mj_to_robot(obj_m, T_r, T_m)
                if mid_r is not None and y_r is not None and obj_r_mapped is not None:
                    axis = y_r - obj_r_mapped
                    along_r, lat_r = _along_lat_from_axis(mid_r, obj_r_mapped, axis)
                    to_y_r = float(np.linalg.norm(mid_r - y_r)) * 1000.0
                    lock_err = float(np.linalg.norm(obj_r_mapped - obj_r)) * 1000.0
                    log(
                        f"  闭合mid(T_mr映)→黄 {to_y_r:.1f}mm；"
                        f"→映青 along={along_r:.1f}mm 侧向={lat_r:.1f}mm "
                        f"锁存↔映青 {lock_err:.1f}mm（仅参考，易随位姿漂）"
                    )

        # ★D 实机 L6 vs MJ 预设；level 捏取蹭顶面风险
        ph_d = str(phase or "")
        after_close = "⑧" in ph_d or "闭合" in ph_d
        after_contact_d = after_close or any(k in ph_d for k in ("⑦", "接触"))
        level_risk = None
        if after_contact_d and plan is not None and self.mj_model is not None:
            preset_close = list(
                plan.get("grasp_close_hand")
                or self.controller.hand_presets.get(
                    "grasp",
                    self.controller.hand_presets.get(
                        "pinch_close", DEFAULT_GRASP_PRESET
                    ),
                )
            )
            actual_hand = getattr(self.controller, "hand_cmd", None)
            if actual_hand is not None and len(actual_hand) >= 6:
                actual_hand = [int(v) for v in actual_hand[:6]]
            hand_for_geom = (
                actual_hand if after_close and actual_hand is not None else preset_close
            )
            hand_tag = "实机L6" if after_close and actual_hand is not None else "预设close"
            if after_close and actual_hand is not None:
                delta_l6 = [
                    int(actual_hand[i]) - int(preset_close[i]) for i in range(6)
                ]
                if any(abs(d) > 2 for d in delta_l6):
                    log(
                        f"  ★D 实机L6={actual_hand} ≠ 预设close={preset_close} "
                        f"Δ={delta_l6}（面板/yaml 与 ⑧ 下发不一致）"
                    )
                else:
                    log(
                        f"  ★D 实机L6={actual_hand} ≈ 预设close={preset_close}"
                    )
            else:
                log(f"  ★D 对照 L6({hand_tag})={hand_for_geom}")

            if hand_for_geom != list(pinch_close) and info_c is not None:
                if self._sync_mj_state(joints, hand_cmd=hand_for_geom):
                    info_act = _compute_o6_pinch_kinematics(
                        self.mj_model, self.mj_data, **kin_kw
                    )
                    if info_act is not None and info_act.get("pinch_mid_m") is not None:
                        mid_act = np.asarray(
                            info_act["pinch_mid_m"], dtype=np.float64
                        ).reshape(3)
                        if mid_close_m is not None and obj_m is not None and yellow_m is not None:
                            a_along, a_lat = _along_lat_from_axis(
                                mid_act, obj_m, yellow_m - obj_m
                            )
                            d_preset = float(
                                np.linalg.norm(mid_act - mid_close_m)
                            ) * 1000.0
                            log(
                                f"  ★D {hand_tag}闭合mid→青 along={a_along:.1f}mm "
                                f"侧向={a_lat:.1f}mm（相对预设close mid Δ={d_preset:.1f}mm）"
                            )
                            if d_preset > 3.0 and after_close:
                                log(
                                    "  ★D ⇒ 实机 L6 与预设 close 几何差 >3mm，"
                                    "MJ ★C 好仍可能蹭顶/夹空"
                                )

            eye_d = self.config.get("eye_in_hand", {}) if isinstance(
                self.config, dict
            ) else {}
            size_info = getattr(self, "_locked_size_info", None) or getattr(
                self, "_last_size_info", None
            )
            kin_for_level = info_c
            if after_close and hand_for_geom != list(pinch_close):
                if self._sync_mj_state(joints, hand_cmd=hand_for_geom):
                    kin_for_level = _compute_o6_pinch_kinematics(
                        self.mj_model, self.mj_data, **kin_kw
                    )
            if kin_for_level is not None and obj_m is not None:
                det_d = self.config.get("detection") if isinstance(
                    self.config, dict
                ) else {}
                down_m, down_src = pinch_grasp_down_m(
                    eye_d, size_info, detection=det_d, with_source=True,
                )
                # tip vs pad：同关节再算一次 pad mid，看接触模型高度差
                info_pad = None
                tip_mode_now = str(kin_kw.get("tip_mode", "tip")).strip().lower()
                if tip_mode_now != "pad_plane" and self._sync_mj_state(
                    joints, hand_cmd=hand_for_geom,
                ):
                    kw_pad = dict(kin_kw)
                    kw_pad["tip_mode"] = "pad_plane"
                    info_pad = _compute_o6_pinch_kinematics(
                        self.mj_model, self.mj_data, **kw_pad
                    )
                lvl = diagnose_level_pinch_top_rub(
                    obj_m,
                    kin_for_level,
                    eye_d,
                    size_info,
                    grasp_down_m=down_m,
                    detection=det_d,
                    pinch_info_pad=info_pad,
                    pinch_info_open=info_o if mid_open_m is not None else None,
                )
                if lvl.get("level_mode"):
                    tips_s = ""
                    tip_bt = lvl.get("tip_below_top_mm") or []
                    if tip_bt and lvl.get("tip_labels"):
                        tips_s = " ".join(
                            f"{lab}{z:.0f}"
                            for lab, z in zip(lvl["tip_labels"], tip_bt)
                        )
                    tip_pad_s = ""
                    if lvl.get("tip_vs_pad_z_mm") is not None:
                        tip_pad_s = (
                            f" tip−padZ={lvl['tip_vs_pad_z_mm']:.1f}mm"
                        )
                    open_s = ""
                    if lvl.get("open_mid_below_top_mm") is not None:
                        open_s = (
                            f" 张手mid顶下={lvl['open_mid_below_top_mm']:.0f}mm"
                        )
                    log(
                        f"  ★D level: mid顶下={lvl.get('mid_below_top_mm', float('nan')):.1f}mm "
                        f"（期望侧中≈{lvl.get('expect_below_top_mm', float('nan')):.0f}mm "
                        f"H={lvl.get('nut_height_mm', float('nan')):.0f} "
                        f"下偏={lvl.get('grasp_down_mm', float('nan')):.1f}mm/{down_src} "
                        f"AF={lvl.get('outer_af_mm', float('nan')):.0f}）"
                        f" mid−侧中={lvl.get('mid_above_side_mm', float('nan')):.1f}mm"
                        f"{tip_pad_s}{open_s}"
                        f" Z∥重力={lvl.get('z_tilt_from_up_deg', float('nan')):.1f}° "
                        f"sep={lvl.get('pinch_sep_mm', float('nan')):.1f}mm"
                        + (f" 指尖顶下{tips_s}mm" if tips_s else "")
                    )
                    if lvl.get("risk") == "high":
                        level_risk = "high"
                        log(f"  ★D ⇒ 蹭顶面风险高：{lvl.get('hint', '')}")
                    elif lvl.get("risk") == "med":
                        level_risk = "med"
                        log(f"  ★D ⇒ 蹭顶面风险中：{lvl.get('hint', '')}")
                    elif lvl.get("risk") is None and resolve_pinch_level_finger_plane(
                        eye_d, size_info=size_info
                    ):
                        log(
                            "  ★D ⇒ level 闭合高度≈侧中心（几何 OK；"
                            "若仍蹭顶查实机 O6 是否与 MJ close 一致）"
                        )

        # 一句话判读
        tips = []
        _eye2 = self.config.get("eye_in_hand", {}) if isinstance(
            self.config, dict
        ) else {}
        _pm2 = str(_eye2.get("pinch_path_mode", "cart")).strip().lower()
        _mj_mode = _pm2 in (
            "mj_ik_smooth", "cart_mj_polish", "mj_ik_joint", "mj_joint",
            "preview_joint", "preview_ik", "mj_ik_stepped",
            "mj_ik_multi", "preview_stepped",
        )
        if api_err_mm is not None:
            if _mj_mode:
                if mj_tool0_err is not None and mj_tool0_err <= 15.0:
                    tips.append(f"A相对笛卡尔{api_err_mm:.0f}mm(可忽略)")
                elif api_err_mm <= 10.0 and (api_geo is None or api_geo <= 3.5):
                    tips.append("A笛卡尔OK")
                else:
                    tips.append(f"A笛卡尔偏({api_err_mm:.0f}mm/{api_geo or 0:.1f}°)")
            elif api_err_mm <= 10.0 and (api_geo is None or api_geo <= 3.5):
                tips.append("A到位OK")
            else:
                tips.append(f"A未到位({api_err_mm:.0f}mm/{api_geo or 0:.1f}°)")
        if mj_tool0_err is not None:
            if mj_tool0_err <= 12.0:
                tips.append("B关节FK尚可")
            else:
                tips.append(f"B偏大({mj_tool0_err:.0f}mm)")
            if api_err_mm is not None and api_err_mm <= 10.0 and mj_tool0_err > 15.0:
                tips.append("B≫A→勿信笛卡尔到位")
        if mid_to_y is not None:
            ph = str(phase or "")
            after_contact = any(k in ph for k in ("⑦", "⑧", "闭合", "接触"))
            if after_contact:
                st = None
                if yellow_m is not None and obj_m is not None:
                    st = float(np.linalg.norm(
                        np.asarray(yellow_m) - np.asarray(obj_m)
                    )) * 1000.0
                along_ok = mid_along is not None and abs(mid_along) <= 3.0
                lat_ok = mid_lat is not None and mid_lat <= 5.0
                y_ok = st is not None and mid_to_y is not None and abs(mid_to_y - st) <= 5.0
                y_note = ""
                if st is not None:
                    y_note = f"，→黄{mid_to_y:.0f}/standoff{st:.0f}"
                if along_ok and lat_ok:
                    tips.append(
                        f"C闭合几何OK along={mid_along:.0f} 侧向={mid_lat:.0f}{y_note}"
                    )
                elif mid_along is not None and mid_along > 3.0:
                    tips.append(
                        f"C未到底(闭合mid高于青球 along={mid_along:.0f}mm"
                        f"→可能蹭顶面){y_note}"
                    )
                else:
                    tips.append(
                        f"C接触偏 along={mid_along if mid_along is not None else float('nan'):.0f} "
                        f"侧向={mid_lat if mid_lat is not None else float('nan'):.0f}"
                        f"{y_note}"
                    )
                if not y_ok and st is not None:
                    tips.append(f"C→黄距异常(期望≈{st:.0f}mm)")
                if level_risk == "high":
                    tips.append("D level蹭顶面风险高")
                elif level_risk == "med":
                    tips.append("D level高度略偏上")
            elif mid_to_y <= 8.0 and mid_lat is not None and mid_lat <= 8.0:
                tips.append("C捏取几何OK")
            else:
                tips.append(
                    f"C歪 mid→黄{mid_to_y:.0f}mm 侧向{mid_lat if mid_lat is not None else float('nan'):.0f}mm"
                )
        if tips:
            log(f"  ⇒ 判读: {'；'.join(tips)}")

    def _ensure_locked_for_extrinsic(self, log=None):
        """
        外参验参/校准专用锁定：
        - 保留对准时的 UV/depth（若有），但世界点一律按当前外参重算，且不加抓取侧偏
        - 避免对准锁里的 pinch_grasp_bias（约3cm）把验参指尖拽歪，看起来像瞎动
        """
        log = log or self.log
        u = v = depth_mm = None
        if self._locked_detection is not None:
            u, v, depth_mm = self._locked_detection
            log(
                f"外参沿用对准 UV=({u:.0f},{v:.0f}) depth={float(depth_mm)/10:.1f}cm，"
                f"按当前外参重算世界点（无抓取侧偏）"
            )
        else:
            if not self._ensure_depth_before_action(
                log=log,
                stop=lambda: self._align_stop or not self.running,
            ):
                log("外参锁定失败：深度未就绪")
                return None
            sample = self._sample()
            if sample is None:
                log("外参锁定失败：无检测点（请先对准或勾选追踪颜色）")
                return None
            u, v, depth_mm = sample
        if not self.lock_object_for_grasp(
            u, v, depth_mm, log=log, source="extrinsic", apply_bias=False,
        ):
            return None
        if self._locked_object_robot is None:
            log("外参锁定失败：无实机系物体点")
            return None
        return np.asarray(self._locked_object_robot, dtype=np.float64)

    def _refresh_locked_object_after_extrinsic_calib(self, log=None):
        """t 更新后：同一 UV/depth 重算锁定世界点（保持锁定不丢；无抓取侧偏）。"""
        log = log or self.log
        if self._locked_detection is None:
            return False
        u, v, depth_mm = self._locked_detection
        return self.lock_object_for_grasp(
            u, v, depth_mm, log=log,
            source="extrinsic_recalc", apply_bias=False,
        )

    def lock_object_at_pregrasp(self, u, v, depth_mm, log=None):
        """④ 后校验：优先保留对准锁定；仅当检测仍可靠时才刷新 3D。"""
        log = log or self.log
        eye = self.config["eye_in_hand"]
        prior_u = prior_v = prior_d = None
        if self._locked_detection is not None:
            prior_u, prior_v, prior_d = self._locked_detection
        if depth_mm is not None:
            prior_d = float(depth_mm) if prior_d is None else float(prior_d)

        def _has_align_lock():
            return (
                self._locked_object_mj is not None
                or self._locked_object_robot is not None
            )

        def _keep_align_lock(reason):
            if _has_align_lock():
                log(f"{reason}，保留对准阶段锁定物体（青球不变）")
                return True
            return False

        # 手偏移后画面本就会偏，默认应已有对准锁
        if _has_align_lock() and bool(eye.get("pregrasp_prefer_align_lock", True)):
            # 仍尝试读 ROI 深度做日志；异常则直接保留对准锁
            with self.lock:
                depth_img = self.latest.get("depth_aligned")
                box = self.latest.get("box")
            if depth_img is not None and box is not None and prior_d is not None:
                det_stub = [box[0], box[1], box[2], box[3], 1.0, 0]
                _u_m, _v_m, d_m = measure_detection_uvd(
                    depth_img, det_stub, eye,
                    prior_mm=None, freeze_below_mm=0.0,
                    R_world_cam=self.controller.camera_rotation_world(),
                )
                if d_m is not None:
                    d_m = float(d_m)
                    max_jump = float(eye.get("pregrasp_depth_max_jump_mm", 60))
                    if abs(d_m - float(prior_d)) > max_jump:
                        log(
                            f"预抓 ROI 深度 {d_m/10:.1f}cm 相对对准 "
                            f"{float(prior_d)/10:.1f}cm 跳变过大，忽略"
                        )
                    else:
                        log(
                            f"预抓 ROI 深度 {d_m/10:.1f}cm "
                            f"（已有对准锁，不改青球位置）"
                        )
            else:
                log("预抓后沿用对准锁定物体")
            return True

        with self.lock:
            depth_img = self.latest.get("depth_aligned")
            box = self.latest.get("box")
        if depth_img is not None and box is not None:
            det_stub = [box[0], box[1], box[2], box[3], 1.0, 0]
            u_m, v_m, d_m = measure_detection_uvd(
                depth_img,
                det_stub,
                eye,
                prior_mm=None,
                freeze_below_mm=0.0,
                R_world_cam=self.controller.camera_rotation_world(),
            )
            if u_m is not None and d_m is not None:
                d_m = float(d_m)
                max_jump = float(eye.get("pregrasp_depth_max_jump_mm", 60))
                if prior_d is not None and abs(d_m - float(prior_d)) > max_jump:
                    log(
                        f"预抓 ROI 深度异常 {d_m/10:.1f}cm "
                        f"(相对先前 {float(prior_d)/10:.1f}cm 跳变 "
                        f"{abs(d_m - float(prior_d)):.0f}mm > {max_jump:.0f}mm)"
                    )
                    if _keep_align_lock("深度飞点"):
                        return True
                    depth_mm = float(prior_d)
                    if prior_u is not None:
                        u, v = float(prior_u), float(prior_v)
                else:
                    u, v, depth_mm = float(u_m), float(v_m), d_m
                    log(
                        f"预抓 ROI 深度 {depth_mm/10:.1f}cm "
                        f"({eye.get('depth_roi_mode', 'trimmed')})"
                    )

        if u is None or v is None or depth_mm is None:
            if _keep_align_lock("预抓后无可靠检测"):
                return True
            log("预抓后物体锁定失败：无 u/v/depth")
            return False

        cx = float(self.controller.intrinsics.get("cx", 320))
        cy = float(self.controller.intrinsics.get("cy", 240))
        err_px = math.hypot(float(u) - cx, float(v) - cy)
        max_ecc = float(eye.get("pregrasp_max_ecc_px", 220))
        if err_px > max_ecc:
            log(
                f"预抓后检测偏心 {err_px:.0f}px > {max_ecc:.0f}px "
                f"(抓取点=({float(u):.0f},{float(v):.0f}))"
            )
            if _keep_align_lock("检测跑偏"):
                return True
            log("拒绝锁定以免乱飞（此前也未对准锁定）")
            return False

        return self.lock_object_for_grasp(u, v, depth_mm, log=log, source="pregrasp")

    def lock_object_after_align(self, u, v, depth_mm, log=None):
        """② 对准+融合后锁定物体 3D（后续预抓手偏移不再改青球）。"""
        return self.lock_object_for_grasp(
            u, v, depth_mm, log=log or self.log, source="align",
        )

    def _detection_for_planning(self, sample=None):
        """优先用对准后锁定的 u/v/depth。"""
        if self._locked_detection is not None:
            return self._locked_detection
        if sample is not None:
            u, v, d = sample
            return float(u), float(v), d
        s = self._sample()
        if s is None:
            return None
        return s

