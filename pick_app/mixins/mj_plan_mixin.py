"""MuJoCo load, IK preview, pick plan, path builders"""
from __future__ import annotations

from pick_app.deps import *
from pick_app.constants import *
from pick_app.helpers import *


class MjPlanMixin:
    def _load_mujoco(self):
        if mujoco is None or Image is None:
            self.model_canvas.create_text(
                MODEL_W // 2, MODEL_H // 2,
                text="未安装 mujoco / pillow", fill="#ccc", font=uifont.ui_font(FONT_SM),
            )
            return
        path = find_mjcf()
        if path is None:
            self.model_canvas.create_text(
                MODEL_W // 2, MODEL_H // 2,
                text=f"找不到模型\n{MJCF_RELATIVE}", fill="#ccc", font=uifont.ui_font(FONT_SM),
            )
            return
        try:
            self.mj_model = mujoco.MjModel.from_xml_path(str(path))
            self.mj_data = mujoco.MjData(self.mj_model)
            clear_pad_plane_cache()
            mount_rpy = self.config["eye_in_hand"].get("model_hand_mount_rpy")
            if mount_rpy is not None and apply_hand_mount_rpy(self.mj_model, mount_rpy):
                self.log(
                    f"模型手安装角(覆盖 MJCF): "
                    f"rpy°={[round(math.degrees(x), 1) for x in mount_rpy]}"
                )
            # 加粗模型自带坐标系可视化（若启用 site frame）
            self.mj_model.vis.scale.framelength = 0.12
            self.mj_model.vis.scale.framewidth = 0.008
            rw, rh = _mj_prepare_offscreen(self.mj_model, MODEL_W, MODEL_H)
            self.mj_renderer = mujoco.Renderer(
                self.mj_model, height=rh, width=rw
            )
            self._mj_render_size = (rw, rh)
            self.mj_cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.mj_model, self.mj_cam)
            self.mj_cam.distance = self._cam_default["distance"]
            self.mj_cam.azimuth = self._cam_default["azimuth"]
            self.mj_cam.elevation = self._cam_default["elevation"]
            self._cam_default["lookat"] = self.mj_cam.lookat.copy()
            self.mj_vopt = mujoco.MjvOption()
            self.mj_joints, self.mj_hand_ranges = collect_hand_joint_maps(self.mj_model)
            self.mj_hand_l6_map = list(O6_L6_MASTER_JOINTS) + [
                (None, dep) for dep, _m, _r in O6_L6_MIMIC_JOINTS
            ]
            try:
                _, _, optical_in_body = camera_on_ee_from_mjcf(path)
                self.mj_cam_optical_in_body = np.asarray(
                    optical_in_body, dtype=np.float64
                )
            except Exception:
                self.mj_cam_optical_in_body = np.array(
                    [0.0, 0.0, -0.019], dtype=np.float64
                )
            # yaml 手调外参 → 覆盖 MJ 腕相机 body，使模型光轴与实机射线一致
            self._sync_mj_camera_mount_from_yaml(write_mjcf=True, path=path)
            self.mj_frame_markers = filter_frame_markers(self.mj_model)
            self._mj_arm_joint_signs = resolve_arm_joint_signs(
                self.config.get("eye_in_hand", {}), side="right",
            )
            self._mj_arm_joint_signs_left = resolve_arm_joint_signs(
                self.config.get("eye_in_hand", {}), side="left",
            )
            flip_js = [
                f"J{i}" for i, s in enumerate(self._mj_arm_joint_signs, start=1) if s < 0
            ]
            flip_js_l = [
                f"J{i}"
                for i, s in enumerate(self._mj_arm_joint_signs_left, start=1)
                if s < 0
            ]
            self.log(
                f"模型已加载: {path.name}；已标 "
                + ", ".join(m[0] for m in self.mj_frame_markers)
                + (
                    f"；右臂符号取反: {','.join(flip_js)}"
                    if flip_js
                    else "；右臂符号与实机同向"
                )
                + (
                    f"；左臂取反: {','.join(flip_js_l)}"
                    if flip_js_l
                    else "；左臂同向"
                )
            )
            self._log_hand_mj_preset_compare()
        except Exception as error:
            self.log(f"模型加载失败: {error}")
            self.model_canvas.create_text(
                MODEL_W // 2, MODEL_H // 2,
                text=f"模型加载失败\n{error}", fill="#f88", font=uifont.ui_font(FONT_SM),
            )

    def _sync_mj_camera_mount_from_yaml(self, write_mjcf=False, path=None, force=False):
        """
        把 controller.t_ee_cam（yaml）写进 MuJoCo arm_right_camera，
        消除「实机射线 vs MJ 相机射线」~2cm 系统差。
        prefer_yaml=false 且非 force 时跳过（以外参←模型为准）。
        """
        if self.mj_model is None:
            return False
        prefer = bool(
            self.controller.eye.get("camera_on_ee_prefer_yaml", False)
        )
        if not prefer and not force:
            return False
        t = np.asarray(self.controller.t_ee_cam, dtype=np.float64).reshape(3)
        # MJ body 只吃 SO(3)；光学 flip 留在 Python R_ee_cam（det 可能为 −1）
        eul = self.controller.eye.get("camera_on_ee", {}).get(
            "euler_rpy", [0.0, 0.0, 0.0]
        )
        opt = np.asarray(self.mj_cam_optical_in_body, dtype=np.float64).reshape(3)
        result = apply_camera_on_ee_to_mj_model(
            self.mj_model, t, opt, euler_rpy=eul,
        )
        if result is None:
            self.log("MJ 相机外参同步失败：无 arm_right_camera")
            return False
        old_b, new_b = result
        gap = float(np.linalg.norm(new_b - old_b)) * 1000.0
        if gap > 0.5:
            self.log(
                f"MJ 腕相机已对齐 yaml 光心 t={np.round(t, 4).tolist()} "
                f"（body [{old_b[0]:.4f},{old_b[1]:.4f},{old_b[2]:.4f}] → "
                f"[{new_b[0]:.4f},{new_b[1]:.4f},{new_b[2]:.4f}]，"
                f"|Δbody|={gap:.1f}mm）"
            )
        if write_mjcf:
            mjcf = path if path is not None else find_mjcf()
            if mjcf is not None:
                try:
                    write_mjcf_camera_body_pos(mjcf, new_b)
                    self.log(f"已写回 MJCF 相机 body pos → {mjcf.name}")
                except Exception as exc:
                    self.log(f"写回 MJCF 相机位姿失败: {exc}")
        return True

    def _model_view_dims(self):
        try:
            cw = int(self.model_canvas.winfo_width())
            ch = int(self.model_canvas.winfo_height())
        except tk.TclError:
            cw, ch = 0, 0
        cw = max(cw, MODEL_MIN_W)
        ch = max(ch, MODEL_MIN_H)
        return cw, ch

    def _mj_preview_ik_delta(self, delta_tool0, joints, hand_cmd, max_jump_rad=None):
        """模型预览：MuJoCo tool0 系增量 → robot IK（与实机同跳变上限）。"""
        if max_jump_rad is None:
            max_jump_rad = self.controller.ik_max_jump_rad()
        soft = self.controller.soft_joint_limits_rad()
        if self.controller.robot is None or self.mj_model is None:
            return self.controller.ik_after_delta_ee(
                delta_tool0, seed_joints=joints, max_jump_rad=max_jump_rad,
            )
        return mj_tool0_delta_to_robot_ik(
            self.controller.robot,
            self.mj_model,
            self.mj_data,
            joints,
            hand_cmd,
            self._mj_sync_fn,
            delta_tool0,
            max_jump_rad=max_jump_rad,
            soft_limits_rad=soft,
            config=self.config,
        )

    def _mj_preview_ik_pose(self, target_pos_mj, target_rot_mj, joints, hand_cmd, max_jump_rad=None):
        """模型预览：MuJoCo 世界系 tool0 目标位姿 → robot IK。"""
        if max_jump_rad is None:
            max_jump_rad = self.controller.ik_max_jump_rad()
        soft = self.controller.soft_joint_limits_rad()
        if self.controller.robot is None or self.mj_model is None:
            return None
        return mj_tool0_pose_to_robot_ik(
            self.controller.robot,
            self.mj_model,
            self.mj_data,
            joints,
            hand_cmd,
            self._mj_sync_fn,
            target_pos_mj,
            target_rot_mj,
            max_jump_rad=max_jump_rad,
            soft_limits_rad=soft,
            config=self.config,
        )

    def _mj_tool0_pos(self, joints, hand_cmd=None):
        pose = self._mj_tool0_pose(joints, hand_cmd=hand_cmd)
        return pose[0].copy() if pose is not None else None

    def _mj_tool0_pose(self, joints, hand_cmd=None):
        """返回 (pos, R_world) ；失败返回 None。"""
        if self.mj_model is None or not self._sync_mj_state(joints, hand_cmd=hand_cmd):
            return None
        pose = _mj_object_pose(
            self.mj_model, self.mj_data, "site", "arm_right_tool0",
        )
        if pose is None:
            return None
        return pose[0].copy(), pose[1].copy()

    def _append_tool0_world_path(
        self, kfs, joints, hand, world_delta, label, plan, n_steps, log=print,
    ):
        """
        把世界系平移拆成 n_steps 段，每段走 MuJoCo tool0 IK。
        避免只插值关节导致“盖着盖着就歪了”。
        """
        world_delta = np.asarray(world_delta, dtype=np.float64).reshape(3)
        total = float(np.linalg.norm(world_delta))
        if total < 1e-6 or n_steps < 1:
            return joints
        step = world_delta / float(n_steps)
        for i in range(1, int(n_steps) + 1):
            pose = self._mj_tool0_pose(joints, hand_cmd=hand)
            if pose is None:
                log(f"{label}: 中间点 {i}/{n_steps} 无 tool0，中止")
                break
            _p, R = pose
            delta_tool0 = R.T @ step
            j_next = self._mj_preview_ik_delta(delta_tool0, joints, hand)
            if j_next is None:
                log(f"{label}: 中间点 {i}/{n_steps} IK 失败，停在此")
                break
            joints = list(j_next)
            kfs.append({
                "label": f"{label} ({i}/{n_steps})",
                "joints": list(joints),
                "hand": list(hand),
                "plan": plan,
                "micro": True,
            })
        return joints

    def _append_tool0_pose_path(
        self, kfs, joints, hand, T_des, label, plan, n_steps, log=print,
    ):
        """
        ⑥ 对齐：在 MJ 世界系对 tool0 位姿做插值，逐步 IK。
        比关节空间 lerp 更贴近规划几何，轨迹不会“绕远路”。
        """
        T_des = np.asarray(T_des, dtype=np.float64).reshape(4, 4)
        p_des = T_des[:3, 3]
        R_des = T_des[:3, :3]
        n_steps = max(int(n_steps), 1)
        pose_start = self._mj_tool0_pose(joints, hand_cmd=hand)
        if pose_start is None:
            log(f"{label}: 无起始 tool0，跳过")
            return joints
        p_start, R_start = pose_start
        jump_cap = float(self.controller.ik_max_jump_rad())
        jump_cap = float(np.clip(jump_cap, 0.35, 2.0))
        for i in range(1, n_steps + 1):
            t = _smoothstep(float(i) / float(n_steps))
            p_t = (1.0 - t) * p_start + t * p_des
            R_t = _slerp_rot3(R_start, R_des, t)
            j_next = self._mj_preview_ik_pose(
                p_t, R_t, joints, hand, max_jump_rad=jump_cap,
            )
            if j_next is None:
                log(f"{label}: 中间点 {i}/{n_steps} IK 失败，停在此")
                break
            joints = list(j_next)
            kfs.append({
                "label": f"{label} ({i}/{n_steps})",
                "joints": list(joints),
                "hand": list(hand),
                "plan": plan,
                "micro": True,
            })
        return joints

    def _plan_mj_ik_joint_waypoints(
        self,
        T_des,
        joints,
        hand_cmd,
        n_steps,
        jump_cap,
        log=print,
        log_prefix="MJ-IK",
        taper_power=0.72,
        fixed_rotation=False,
        allow_partial=True,
    ):
        """
        MJ tool0 位姿插值 → 逐点 robot IK。
        fixed_rotation=True：全程锁死目标姿态（⑦ 竖直下落勿甩腕/换肘）。
        allow_partial：过半后某点 IK 失败时，保留已有路点（旧行为；否则整段作废）。
        """
        T_des = np.asarray(T_des, dtype=np.float64).reshape(4, 4)
        p_des = T_des[:3, 3]
        R_des = T_des[:3, :3]
        pose0 = self._mj_tool0_pose(joints, hand_cmd=hand_cmd)
        if pose0 is None:
            log(f"{log_prefix}：无起始 MJ tool0")
            return None
        p0, R0 = pose0
        travel = float(np.linalg.norm(p_des - p0))
        n_steps = int(max(int(n_steps), 1))
        ts = _tapered_path_ts(n_steps, power=taper_power)
        j = list(joints)
        wps = []
        seg_mm = []
        jump_deg_max = 0.0
        p_prev = p0
        jump_hard = float(np.clip(max(float(jump_cap) * 1.35, float(jump_cap) + 0.25), 0.5, 2.5))
        partial = False
        for i, t in enumerate(ts, start=1):
            p_t = (1.0 - t) * p0 + t * p_des
            R_t = R_des if fixed_rotation else _slerp_rot3(R0, R_des, t)
            j_next = self._mj_preview_ik_pose(
                p_t, R_t, j, hand_cmd, max_jump_rad=jump_cap,
            )
            if j_next is None and jump_hard > float(jump_cap) + 1e-6:
                # 单点放宽再试（转腕中段常卡在 jump）
                j_next = self._mj_preview_ik_pose(
                    p_t, R_t, j, hand_cmd, max_jump_rad=jump_hard,
                )
                if j_next is not None:
                    log(
                        f"{log_prefix}：路点 {i}/{n_steps} 放宽跳 "
                        f"{math.degrees(jump_cap):.0f}→{math.degrees(jump_hard):.0f}° 成功"
                    )
            if j_next is None:
                log(f"{log_prefix}：路点 {i}/{n_steps} IK 失败")
                if (
                    allow_partial
                    and i > max(1, n_steps // 2)
                    and wps
                ):
                    partial = True
                    log(
                        f"{log_prefix}：保留已规划 {len(wps)} 点"
                        f"（过半失败，交残缺路点；执行端会终验）"
                    )
                    break
                return None
            ddeg = math.degrees(
                max(abs(float(a) - float(b)) for a, b in zip(j_next, j))
            )
            jump_deg_max = max(jump_deg_max, ddeg)
            wps.append(list(j_next))
            j = list(j_next)
            seg_mm.append(float(np.linalg.norm(p_t - p_prev)) * 1000.0)
            p_prev = p_t
        if not partial:
            R_f = R_des if fixed_rotation else R_des
            j_f = self._mj_preview_ik_pose(
                p_des, R_f, j, hand_cmd, max_jump_rad=jump_cap,
            )
            if j_f is None and jump_hard > float(jump_cap) + 1e-6:
                j_f = self._mj_preview_ik_pose(
                    p_des, R_f, j, hand_cmd, max_jump_rad=jump_hard,
                )
            if j_f is not None:
                if not wps or max(
                    abs(float(a) - float(b)) for a, b in zip(j_f, wps[-1])
                ) > math.radians(0.6):
                    wps.append(list(j_f))
                else:
                    wps[-1] = list(j_f)
                if wps and len(wps) >= 2:
                    jump_deg_max = max(
                        jump_deg_max,
                        math.degrees(
                            max(
                                abs(float(a) - float(b))
                                for a, b in zip(wps[-1], wps[-2])
                            )
                        ),
                    )
            elif not wps:
                log(f"{log_prefix}：终点 IK 失败且无中间路点")
                return None
        # 残缺/终点未贴准：记录末点误差（不在这里整段作废——由重试/终验处理）
        pose_end = self._mj_tool0_pose(wps[-1], hand_cmd=hand_cmd) if wps else None
        if pose_end is not None:
            end_err_mm = float(np.linalg.norm(pose_end[0] - p_des)) * 1000.0
            if end_err_mm > 25.0 or partial:
                log(
                    f"{log_prefix}：末点距目标 {end_err_mm:.0f}mm"
                    + ("（残缺路点）" if partial else "")
                )
        seg_txt = "/".join(f"{s:.0f}" for s in seg_mm) if seg_mm else "-"
        fix_s = " 定姿" if fixed_rotation else ""
        log(
            f"{log_prefix}：规划 {len(wps)} 关节路点{fix_s}"
            f"（行程 {travel * 1000:.0f}mm 段={seg_txt} "
            f"单段最大关节跳 {jump_deg_max:.0f}°）"
        )
        return wps

    def _exec_mj_ik_smooth_to_T(
        self,
        T_des,
        speed=None,
        accel=None,
        log=print,
        should_stop=None,
        log_prefix="MJ-IK",
        step_m=None,
        max_steps=None,
    ):
        """
        根本路径：全程 MJ 位姿插值→机器人 IK→平滑关节执行。
        不走笛卡尔（笛卡尔与 MJ 为不同冗余支，无法靠「到位后精修」跳过去）。
        """
        stop = should_stop or (lambda: False)
        ctrl = self.controller
        eye = self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {}
        T_des = np.asarray(T_des, dtype=np.float64).reshape(4, 4)
        p_des = T_des[:3, 3]
        pinch_open = list(
            ctrl.hand_presets.get("pinch_open", DEFAULT_PINCH_OPEN)
        )
        ctrl.refresh_live_state()
        joints = ctrl.get_live_joints()
        if joints is None:
            joints = ctrl.get_joints()
        if joints is None:
            log(f"{log_prefix}：无当前关节")
            return False
        pose0 = self._mj_tool0_pose(joints, hand_cmd=pinch_open)
        if pose0 is None:
            log(f"{log_prefix}：无 MJ tool0")
            return False
        travel = float(np.linalg.norm(pose0[0] - p_des))
        step = float(
            step_m
            if step_m is not None
            else eye.get("pinch_align_step_m", 0.045)
        )
        cap = int(
            max_steps
            if max_steps is not None
            else eye.get("pinch_mj_ik_max_steps", 4)
        )
        cap = int(np.clip(cap, 1, 8))
        # 少路点：中间不刹停靠 blend；过密反而又顿又漂
        if travel < 0.012:
            n_steps = 1
        elif travel < 0.04:
            n_steps = 2
        else:
            n_steps = int(
                np.clip(math.ceil(travel / max(step, 0.02)), 2, cap)
            )
        jump_cap = float(ctrl.ik_max_jump_rad())
        jump_cap = float(np.clip(jump_cap, 0.35, 2.0))
        taper = float(eye.get("pinch_mj_ik_taper", 0.72))
        wps = self._plan_mj_ik_joint_waypoints(
            T_des, joints, pinch_open, n_steps, jump_cap,
            log=log, log_prefix=log_prefix, taper_power=taper,
        )
        if not wps:
            return False
        v = float(speed if speed is not None else ctrl.speed)
        a = float(accel if accel is not None else ctrl.accel)
        blend = float(eye.get("pinch_joint_blend_deg", 25.0))
        ok = ctrl.execute_joint_path_smooth(
            wps,
            speed=v, accel=a, log=log, should_stop=stop,
            log_prefix=log_prefix,
            blend_deg=blend,
        )
        if stop():
            return False
        time.sleep(0.06)
        ctrl.refresh_live_state()
        j_live = ctrl.get_live_joints()
        if j_live is None:
            j_live = ctrl.get_joints()
        polish_mm = float(eye.get("pinch_mj_ik_polish_mm", 5.0))
        if j_live is not None:
            pose_f = self._mj_tool0_pose(j_live, hand_cmd=pinch_open)
            if pose_f is not None:
                p_act = pose_f[0]
                err = float(np.linalg.norm(p_act - p_des)) * 1000.0
                log(
                    f"{log_prefix}：终验 MJ tool0 |Δ|={err:.1f}mm（补步阈{polish_mm:.0f}mm）"
                )
                log(
                    f"{log_prefix}：目标 [{p_des[0]:.4f},{p_des[1]:.4f},{p_des[2]:.4f}] "
                    f"实际 [{p_act[0]:.4f},{p_act[1]:.4f},{p_act[2]:.4f}] "
                    f"Δmm=[{(p_act[0]-p_des[0])*1000:.1f},"
                    f"{(p_act[1]-p_des[1])*1000:.1f},"
                    f"{(p_act[2]-p_des[2])*1000:.1f}]"
                )
                err_final = err
                if err > polish_mm and not stop():
                    for attempt in range(1, 4):
                        j_f = self._mj_preview_ik_pose(
                            p_des, T_des[:3, :3], j_live, pinch_open,
                            max_jump_rad=jump_cap,
                        )
                        if j_f is None:
                            log(
                                f"{log_prefix}：补步 IK 失败 ({attempt}/3，"
                                f"跳变上限 {math.degrees(jump_cap):.0f}°)"
                            )
                            break
                        log(f"{log_prefix}：终验偏大 → 补步 {attempt}/3")
                        ctrl.execute_joint_motion(
                            j_f, speed=v * 0.7, accel=a * 0.7, block=True,
                        )
                        ctrl.refresh_live_state()
                        j_live = ctrl.get_live_joints() or j_f
                        pose2 = self._mj_tool0_pose(j_live, hand_cmd=pinch_open)
                        if pose2 is None:
                            continue
                        err_final = float(np.linalg.norm(pose2[0] - p_des)) * 1000.0
                        log(f"{log_prefix}：补步后 |Δ|={err_final:.1f}mm")
                        if err_final <= polish_mm:
                            break
                    if err_final > polish_mm:
                        log(
                            f"{log_prefix}：⚠ 补步后仍差 {err_final:.1f}mm"
                            f"（闭合 mid 可能未到底→蹭顶面/夹空）"
                        )
        return bool(ok)

    def _exec_mj_ik_polish(
        self,
        T_des,
        speed=None,
        accel=None,
        log=print,
        should_stop=None,
        log_prefix="MJ精修",
        err_log_mm=8.0,
        force=False,
    ):
        """兼容旧回调：改为整段平滑 MJ-IK（不再做「小跳变伪精修」）。"""
        return self._exec_mj_ik_smooth_to_T(
            T_des,
            speed=speed, accel=accel, log=log, should_stop=should_stop,
            log_prefix=log_prefix,
        )

    def _exec_replay_waypoints(
        self,
        waypoints,
        speed=None,
        accel=None,
        log=print,
        should_stop=None,
        log_prefix="Replay",
        seed_joints=None,
        T_des=None,
    ):
        """
        Plan-once 回放：只执行已规划关节路点，不现场重算 IK。
        seed 漂移：tol 内直接播；超 tol 单段 bridge 到首路点；超 abort 则失败。
        """
        stop = should_stop or (lambda: False)
        ctrl = self.controller
        eye = self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {}
        wps = [list(w) for w in (waypoints or []) if w is not None]
        if not wps:
            log(f"{log_prefix}：无路点")
            return False

        ctrl.refresh_live_state()
        live = ctrl.get_live_joints()
        if live is None:
            live = ctrl.get_joints()
        tol = float(eye.get("plan_replay_seed_tol_deg", 8.0))
        abort = float(eye.get("plan_replay_seed_abort_deg", 25.0))
        seed = seed_joints if seed_joints is not None else wps[0]
        ok_seed, ddeg, action = seed_drift_status(live, seed, tol, abort)
        if action == "abort":
            log(
                f"{log_prefix}：种子漂移 {ddeg:.1f}° > abort {abort:.0f}°，"
                f"中止回放（请重锁/重规划）"
            )
            return False
        v = float(speed if speed is not None else ctrl.speed)
        a = float(accel if accel is not None else ctrl.accel)
        if action == "bridge":
            log(
                f"{log_prefix}：种子漂移 {ddeg:.1f}° → 单段 bridge 到路点[0]"
            )
            if not ctrl.execute_joint_motion(
                wps[0], speed=v * 0.75, accel=a * 0.75, block=True,
            ):
                log(f"{log_prefix}：bridge 失败")
                return False
            if stop():
                return False
        else:
            log(f"{log_prefix}：种子 OK（Δ≤{ddeg:.1f}°），回放 {len(wps)} 路点")

        blend = float(eye.get("pinch_joint_blend_deg", 25.0))
        ok = ctrl.execute_joint_path_smooth(
            wps,
            speed=v, accel=a, log=log, should_stop=stop,
            log_prefix=log_prefix,
            blend_deg=blend,
        )
        if stop():
            return False
        time.sleep(0.05)
        ctrl.refresh_live_state()
        # 终验：Plan-once 默认不 polish；位置或姿态超阈则自动补 1 步
        # 姿态差 2° × 手长~130mm ≈ mid 侧向 4~5mm（日志 C歪 常见根因）
        polish_on = bool(eye.get("pinch_mj_ik_polish", False))
        polish_mm = float(eye.get("pinch_mj_ik_polish_mm", 15.0))
        auto_polish_mm = float(eye.get("pinch_replay_auto_polish_mm", polish_mm))
        auto_polish_deg = float(eye.get("pinch_replay_auto_polish_deg", 1.5))
        if T_des is not None:
            j_live = ctrl.get_live_joints() or ctrl.get_joints()
            pinch_open = list(
                ctrl.hand_presets.get("pinch_open", DEFAULT_PINCH_OPEN)
            )
            if j_live is not None:
                pose_f = self._mj_tool0_pose(j_live, hand_cmd=pinch_open)
                if pose_f is not None:
                    T = np.asarray(T_des, dtype=np.float64).reshape(4, 4)
                    p_des = T[:3, 3]
                    R_des = T[:3, :3]
                    p_now, R_now = pose_f
                    err = float(np.linalg.norm(p_now - p_des)) * 1000.0
                    R_rel = R_now.T @ R_des
                    tr = float(np.clip((float(np.trace(R_rel)) - 1.0) * 0.5, -1.0, 1.0))
                    err_deg = float(math.degrees(math.acos(tr)))
                    do_polish = False
                    polish_why = ""
                    if polish_on and (err > polish_mm or err_deg > auto_polish_deg):
                        do_polish = True
                        polish_why = f"|Δ|={err:.1f}mm 姿态={err_deg:.1f}°"
                    elif (not polish_on) and auto_polish_mm > 0.0 and (
                        err > auto_polish_mm or err_deg > auto_polish_deg
                    ):
                        do_polish = True
                        polish_why = f"|Δ|={err:.1f}mm 姿态={err_deg:.1f}°"
                    if do_polish and not polish_on:
                        log(
                            f"{log_prefix}：终验 MJ tool0 {polish_why} "
                            f"> 补步阈{auto_polish_mm:.0f}mm/"
                            f"{auto_polish_deg:.1f}° → 补 1 次"
                        )
                    else:
                        log(
                            f"{log_prefix}：终验 MJ tool0 |Δ|={err:.1f}mm "
                            f"姿态={err_deg:.1f}°"
                            + (
                                f"（回放，不重规划）"
                                if not do_polish
                                else f"（polish 阈{polish_mm:.0f}mm）"
                            )
                        )
                    if do_polish and not stop():
                        if polish_on:
                            log(f"{log_prefix}：polish 开启且偏大 → 单次补步")
                        jump_cap = float(ctrl.ik_max_jump_rad())
                        jump_cap = float(np.clip(jump_cap, 0.35, 2.0))
                        j_f = self._mj_preview_ik_pose(
                            p_des, R_des, j_live, pinch_open,
                            max_jump_rad=jump_cap,
                        )
                        if j_f is not None:
                            ctrl.execute_joint_motion(
                                j_f, speed=v * 0.7, accel=a * 0.7, block=True,
                            )
                            time.sleep(0.05)
                            ctrl.refresh_live_state()
                            j_live = ctrl.get_live_joints() or ctrl.get_joints()
                            if j_live is not None:
                                pose_f = self._mj_tool0_pose(
                                    j_live, hand_cmd=pinch_open
                                )
                                if pose_f is not None:
                                    err = float(
                                        np.linalg.norm(pose_f[0] - p_des)
                                    ) * 1000.0
                        else:
                            log(f"{log_prefix}：补步 IK 失败，保持当前关节")
                    # ⑥/⑦ 终验仍远：标失败，避免「假成功」后半空合手
                    fail_mm = float(
                        eye.get("pinch_replay_fail_err_mm", 40.0)
                    )
                    if err > fail_mm:
                        log(
                            f"{log_prefix}：终验仍偏 {err:.0f}mm > "
                            f"{fail_mm:.0f}mm，本段失败"
                        )
                        return False
        return bool(ok)

    def _exec_align_mj_ik_joints(
        self,
        plan,
        speed=None,
        accel=None,
        log=print,
        should_stop=None,
        target_pos=None,
        target_rpy=None,
    ):
        """
        实机 ⑥：优先 Plan-once 回放 waypoints_6；否则回退现场 MJ-IK。
        """
        stop = should_stop or (lambda: False)
        ctrl = self.controller
        eye = self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {}
        path_mode = str(eye.get("pinch_path_mode", "mj_ik_smooth")).strip().lower()
        T_des = plan.get("T_ee_des") if plan else None
        if T_des is None:
            log("⑥：规划无 T_ee_des")
            return False

        wps6 = (plan or {}).get("waypoints_6")
        if wps6 and path_mode not in ("cart", "cartesian"):
            log("⑥ 执行=Plan-once 回放 waypoints_6（与预览同轨迹）")
            # ⑥ 要转腕到位再竖直⑦；全局 blend25°/polish15mm 过松 → 带着侧向误差下落（日志 C歪）
            eye_live = self.config.setdefault("eye_in_hand", {})
            blend_save = eye_live.get("pinch_joint_blend_deg")
            auto_save = eye_live.get("pinch_replay_auto_polish_mm")
            eye_live["pinch_joint_blend_deg"] = float(
                eye_live.get("pinch_align_blend_deg", 12.0)
            )
            eye_live["pinch_replay_auto_polish_mm"] = float(
                eye_live.get("pinch_align_auto_polish_mm", 6.0)
            )
            try:
                return self._exec_replay_waypoints(
                    wps6,
                    speed=speed, accel=accel, log=log, should_stop=stop,
                    log_prefix="⑥",
                    seed_joints=(plan or {}).get("seed_joints"),
                    T_des=T_des,
                )
            finally:
                if blend_save is None:
                    eye_live.pop("pinch_joint_blend_deg", None)
                else:
                    eye_live["pinch_joint_blend_deg"] = blend_save
                if auto_save is None:
                    eye_live.pop("pinch_replay_auto_polish_mm", None)
                else:
                    eye_live["pinch_replay_auto_polish_mm"] = auto_save

        # 显式要笛卡尔才走（调试用；捏取几何会偏）
        if path_mode in ("cart", "cartesian"):
            tp = target_pos
            tr = target_rpy
            rik = (plan or {}).get("robot_ik") or {}
            if tp is None:
                tp = rik.get("target_pos")
            if tr is None:
                tr = rik.get("target_rpy")
            if tp is None or tr is None:
                log("⑥ 笛卡尔：无 robot 目标")
                return False
            log("⑥ 执行=纯笛卡尔（pinch_path_mode=cart，几何可能 B≫A）")
            return bool(
                ctrl.move_to_pose_stepped(
                    np.asarray(tp, dtype=np.float64).reshape(3),
                    (float(tr[0]), float(tr[1]), float(tr[2])),
                    speed=speed, accel=accel, log=log, should_stop=stop,
                    max_step_deg=float(eye.get("pinch_align_step_deg", 8)),
                    max_step_m=float(eye.get("pinch_align_step_m", 0.025)),
                    log_prefix="⑥笛卡尔",
                )
            )

        log(
            "⑥ 执行=现场 MJ-IK（无 waypoints；非 Plan-once）"
        )
        return self._exec_mj_ik_smooth_to_T(
            T_des,
            speed=speed, accel=accel, log=log, should_stop=stop,
            log_prefix="⑥",
        )

    def _exec_contact_advance_mj_ik(
        self,
        plan,
        final_m,
        speed=None,
        accel=None,
        log=print,
        should_stop=None,
    ):
        """
        ⑦：⑥ 结束后按「闭合 mid → 青球」现场重规划并回放。

        不用锁物瞬间的旧 waypoints_7：⑥ 补步后姿态/位置已变，旧路点会把 ★C 侧向
        从 ~2mm 拉到 ~8mm；seed_joints=live 还会让种子检查永远 OK。
        """
        stop = should_stop or (lambda: False)
        ctrl = self.controller
        eye = self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {}
        T_ee6 = plan.get("T_ee_des") if plan else None
        obj_mj = plan.get("obj_world") if plan else None
        yellow = plan.get("obj_target") if plan else None
        if T_ee6 is None or obj_mj is None:
            log("⑦ MJ：无 T_ee_des/obj_world")
            return False
        T_ee6 = np.asarray(T_ee6, dtype=np.float64).reshape(4, 4)
        obj_mj = np.asarray(obj_mj, dtype=np.float64).reshape(3)
        final_m = float(final_m)

        path7 = str(eye.get("pinch_path_mode", "mj_ik_smooth")).strip().lower()
        pinch_open = list(
            ctrl.hand_presets.get("pinch_open", DEFAULT_PINCH_OPEN)
        )
        pinch_close = list(
            (plan or {}).get("grasp_close_hand")
            or ctrl.hand_presets.get("pinch_close", DEFAULT_PINCH_CLOSE)
        )

        # —— 优先：⑥ 后实测闭合 mid，法兰同位移把 mid 送到青（闭合 TCP 设计）——
        if path7 not in ("cart", "cartesian") and final_m > 1e-4:
            ctrl.refresh_live_state()
            joints = ctrl.get_live_joints() or ctrl.get_joints()
            if joints is None:
                log("⑦：无当前关节，无法按闭合 mid 重规划")
            else:
                if yellow is not None:
                    yellow = np.asarray(yellow, dtype=np.float64).reshape(3)
                    log(
                        f"⑦ 几何: 黄球 [{yellow[0]:.4f},{yellow[1]:.4f},{yellow[2]:.4f}] "
                        f"青球 [{obj_mj[0]:.4f},{obj_mj[1]:.4f},{obj_mj[2]:.4f}] "
                        f"黄−青 {float(np.linalg.norm(obj_mj-yellow))*1000:.0f}mm"
                    )
                if not self._sync_mj_state(joints, hand_cmd=pinch_close):
                    log("⑦：MJ 同步闭合手失败")
                else:
                    kin = _compute_o6_pinch_kinematics(
                        self.mj_model, self.mj_data, **self._pinch_kin_kwargs()
                    )
                    pose_now = self._mj_tool0_pose(joints, hand_cmd=pinch_close)
                    mid = None
                    if kin is not None and kin.get("pinch_mid_m") is not None:
                        mid = np.asarray(kin["pinch_mid_m"], dtype=np.float64).reshape(3)
                    if mid is None or pose_now is None:
                        log("⑦：无法测闭合 mid / tool0，回退旧路点")
                    else:
                        p0, R0 = pose_now
                        p0 = np.asarray(p0, dtype=np.float64).reshape(3)
                        R0 = np.asarray(R0, dtype=np.float64).reshape(3, 3)
                        # 实指短：MJ mid 要停在青球下方 short_m，实 mid≈青。
                        # 旧逻辑 mid→青 会把 ⑥ 的 pinch_hand_short 抵消干净 → 实机仍偏上。
                        from lbot_grasp_utils import resolve_pinch_hand_short_m
                        wu = np.asarray(
                            eye.get("pinch_world_up", [0.0, 0.0, 1.0]),
                            dtype=np.float64,
                        ).reshape(3)
                        wn = float(np.linalg.norm(wu))
                        wu = wu / wn if wn > 1e-9 else np.array([0.0, 0.0, 1.0])
                        size_info = (
                            (plan or {}).get("size_info")
                            or getattr(self, "_locked_size_info", None)
                            or getattr(self, "_last_size_info", None)
                        )
                        short_m = resolve_pinch_hand_short_m(eye, size_info)
                        mid_tgt = obj_mj - wu * max(short_m, 0.0)
                        delta = mid_tgt - mid
                        d_mm = float(np.linalg.norm(delta)) * 1000.0
                        # 侧向相对黄→青（或世界上）
                        axis = None
                        if yellow is not None:
                            ax = obj_mj - yellow
                            an = float(np.linalg.norm(ax))
                            if an > 1e-9:
                                axis = ax / an
                        if axis is None:
                            axis = -wu
                        along = float(np.dot(delta, axis))
                        lat = float(np.linalg.norm(delta - along * axis))
                        T7 = np.eye(4, dtype=np.float64)
                        T7[:3, :3] = R0  # 锁死 ⑥ 后实机姿态，勿用规划瞬间旧 R
                        T7[:3, 3] = p0 + delta
                        plan["T_ee_contact"] = T7
                        short_s = (
                            f" mid目标=青−{short_m*1000:.0f}mm(手短)"
                            if short_m > 1e-6
                            else " mid目标=青"
                        )
                        log(
                            f"⑦ 按闭合mid重规划{short_s}：|Δ|={d_mm:.1f}mm "
                            f"along={along*1000:.1f}mm 侧向={lat*1000:.1f}mm"
                        )
                        jump7 = float(eye.get("pinch_contact_ik_jump_rad", 0.40))
                        jump7 = float(np.clip(jump7, 0.20, 1.0))
                        step7 = float(eye.get("pinch_contact_step_m", 0.012))
                        cap7 = int(eye.get("pinch_contact_max_steps", 5))
                        n7 = max(
                            int(eye.get("pinch_contact_min_steps", 4)),
                            n_steps_for_travel(
                                max(d_mm / 1000.0, final_m),
                                max(step7, 0.008),
                                max(cap7, 6),
                            ),
                        )
                        wps7 = self._plan_mj_ik_joint_waypoints(
                            T7, joints, pinch_open, n7, jump7,
                            log=log, log_prefix="⑦重规划", taper_power=1.0,
                            fixed_rotation=True,
                            allow_partial=False,
                        )
                        if not wps7:
                            # 与 bak_monolith 一致：失败则回退下方旧路径
                            # （大跳限 ik_max_jump + allow_partial），勿直接中止
                            log("⑦：闭合mid→青 路点失败，回退旧逻辑")
                        else:
                            eye_live = self.config.setdefault("eye_in_hand", {})
                            blend_save = eye_live.get("pinch_joint_blend_deg")
                            auto_save = eye_live.get("pinch_replay_auto_polish_mm")
                            deg_save = eye_live.get("pinch_replay_auto_polish_deg")
                            eye_live["pinch_joint_blend_deg"] = float(
                                eye_live.get("pinch_contact_blend_deg", 10.0)
                            )
                            # 2.5°×手长~130mm≈5–6mm mid 侧向；须严于旧 8mm/5°
                            eye_live["pinch_replay_auto_polish_mm"] = float(
                                eye_live.get("pinch_contact_auto_polish_mm", 4.0)
                            )
                            eye_live["pinch_replay_auto_polish_deg"] = float(
                                eye_live.get("pinch_contact_auto_polish_deg", 1.5)
                            )
                            try:
                                # 种子=重规划时的关节（勿再传「此刻 live」与自身比，恒为 0）
                                return self._exec_replay_waypoints(
                                    wps7,
                                    speed=speed, accel=accel, log=log,
                                    should_stop=stop,
                                    log_prefix="⑦",
                                    seed_joints=list(joints),
                                    T_des=T7,
                                )
                            finally:
                                if blend_save is None:
                                    eye_live.pop("pinch_joint_blend_deg", None)
                                else:
                                    eye_live["pinch_joint_blend_deg"] = blend_save
                                if auto_save is None:
                                    eye_live.pop("pinch_replay_auto_polish_mm", None)
                                else:
                                    eye_live["pinch_replay_auto_polish_mm"] = auto_save
                                if deg_save is None:
                                    eye_live.pop("pinch_replay_auto_polish_deg", None)
                                else:
                                    eye_live["pinch_replay_auto_polish_deg"] = deg_save

        T7 = plan.get("T_ee_contact")
        if T7 is None:
            T7 = compute_contact_T(T_ee6, obj_mj, yellow, final_m)
            if T7 is not None:
                plan["T_ee_contact"] = T7
        else:
            T7 = np.asarray(T7, dtype=np.float64).reshape(4, 4)
            plan["T_ee_contact"] = T7

        wps7 = (plan or {}).get("waypoints_7")
        if wps7 and path7 not in ("cart", "cartesian") and final_m > 1e-4:
            if yellow is not None:
                yellow = np.asarray(yellow, dtype=np.float64).reshape(3)
                log(
                    f"⑦ 几何: 黄球 [{yellow[0]:.4f},{yellow[1]:.4f},{yellow[2]:.4f}] "
                    f"青球 [{obj_mj[0]:.4f},{obj_mj[1]:.4f},{obj_mj[2]:.4f}] "
                    f"黄−青 {float(np.linalg.norm(obj_mj-yellow))*1000:.0f}mm"
                )
            log(
                f"⑦ 回退=Plan-once 旧 waypoints_7 "
                f"（推进 {final_m*1000:.0f}mm）"
            )
            wps6 = plan.get("waypoints_6") or []
            # 对照规划末点，而非 live（live vs live 恒 OK）
            seed7 = list(wps6[-1]) if wps6 else list(wps7[0])
            eye_live = self.config.setdefault("eye_in_hand", {})
            blend_save = eye_live.get("pinch_joint_blend_deg")
            auto_save = eye_live.get("pinch_replay_auto_polish_mm")
            deg_save = eye_live.get("pinch_replay_auto_polish_deg")
            eye_live["pinch_joint_blend_deg"] = float(
                eye_live.get("pinch_contact_blend_deg", 10.0)
            )
            eye_live["pinch_replay_auto_polish_mm"] = float(
                eye_live.get("pinch_contact_auto_polish_mm", 4.0)
            )
            eye_live["pinch_replay_auto_polish_deg"] = float(
                eye_live.get("pinch_contact_auto_polish_deg", 1.5)
            )
            try:
                return self._exec_replay_waypoints(
                    wps7,
                    speed=speed, accel=accel, log=log, should_stop=stop,
                    log_prefix="⑦",
                    seed_joints=seed7,
                    T_des=T7,
                )
            finally:
                if blend_save is None:
                    eye_live.pop("pinch_joint_blend_deg", None)
                else:
                    eye_live["pinch_joint_blend_deg"] = blend_save
                if auto_save is None:
                    eye_live.pop("pinch_replay_auto_polish_mm", None)
                else:
                    eye_live["pinch_replay_auto_polish_mm"] = auto_save
                if deg_save is None:
                    eye_live.pop("pinch_replay_auto_polish_deg", None)
                else:
                    eye_live["pinch_replay_auto_polish_deg"] = deg_save

        # —— 以下：无 waypoints 时的旧现场规划 ——
        pinch_open = list(
            ctrl.hand_presets.get("pinch_open", DEFAULT_PINCH_OPEN)
        )
        ctrl.refresh_live_state()
        joints = ctrl.get_live_joints()
        if joints is None:
            joints = ctrl.get_joints()
        if joints is None:
            log("⑦ MJ：无当前关节")
            return False
        pose_cur = self._mj_tool0_pose(joints, hand_cmd=pinch_open)
        if pose_cur is None:
            log("⑦ MJ：无当前 tool0")
            return False
        p_cur, _R_cur = pose_cur
        p_plan = T_ee6[:3, 3].copy()

        u_mj = None
        axis_src = "tool0→青"
        if yellow is not None:
            yellow = np.asarray(yellow, dtype=np.float64).reshape(3)
            axis = obj_mj - yellow
            an = float(np.linalg.norm(axis))
            if an > 1e-6:
                u_mj = axis / an
                axis_src = "黄→青"
                if abs(an - final_m) > 0.012:
                    log(
                        f"⑦ 黄−青={an*1000:.0f}mm vs 请求推进={final_m*1000:.0f}mm，"
                        f"方向用黄→青、长度用请求"
                    )
        if u_mj is None:
            pinch_z = None
            T_pd = plan.get("T_pinch_des")
            if T_pd is not None:
                pinch_z = np.asarray(T_pd, dtype=np.float64).reshape(4, 4)[:3, 2]
            u_mj = approach_unit_toward_object(
                p_cur, obj_mj, pinch_z_away=pinch_z,
            )
            axis_src = "tool0→青(回退)"
        if u_mj is None:
            log("⑦ MJ：靠近方向无效")
            return False

        drift = float(np.linalg.norm(p_cur - p_plan))
        if drift > 0.018:
            p_start = p_cur
            base_src = f"当前(距规划{drift*1000:.0f}mm)"
        else:
            p_start = p_plan
            base_src = "规划⑥"

        if T7 is None:
            T7 = np.eye(4, dtype=np.float64)
            T7[:3, :3] = T_ee6[:3, :3]
            T7[:3, 3] = p_start + u_mj * final_m
            plan["T_ee_contact"] = T7
        gap = float(np.linalg.norm(T7[:3, 3] - p_cur)) * 1000.0
        p_tgt = T7[:3, 3]
        log(
            f"⑦ 规划: ⑥tool0 [{p_plan[0]:.4f},{p_plan[1]:.4f},{p_plan[2]:.4f}] "
            f"→ 接触tool0 [{p_tgt[0]:.4f},{p_tgt[1]:.4f},{p_tgt[2]:.4f}]"
        )
        log(
            f"⑦ 当前tool0 [{p_cur[0]:.4f},{p_cur[1]:.4f},{p_cur[2]:.4f}] "
            f"基点={base_src} 推进 {final_m * 1000:.0f}mm 行程 {gap:.0f}mm "
            f"轴={axis_src} unit=[{u_mj[0]:+.3f},{u_mj[1]:+.3f},{u_mj[2]:+.3f}]"
        )
        if yellow is not None:
            log(
                f"⑦ 几何: 黄球 [{yellow[0]:.4f},{yellow[1]:.4f},{yellow[2]:.4f}] "
                f"青球 [{obj_mj[0]:.4f},{obj_mj[1]:.4f},{obj_mj[2]:.4f}] "
                f"黄−青 {float(np.linalg.norm(obj_mj-yellow))*1000:.0f}mm"
            )
        if stop():
            return False
        step7 = float(eye.get("pinch_contact_step_m", 0.02))
        return self._exec_mj_ik_smooth_to_T(
            T7,
            speed=speed, accel=accel, log=log, should_stop=stop,
            log_prefix="⑦",
            step_m=max(step7, 0.016),
            max_steps=int(eye.get("pinch_contact_max_steps", 3)),
        )

    def _append_world_translate_keep_rot(
        self, kfs, joints, hand, world_dir, distance_m, label, plan, n_steps, log=print,
    ):
        """
        ⑦ 靠近：沿固定世界方向平移，姿态保持不变（与实机 move_delta_world 一致）。
        """
        world_dir = np.asarray(world_dir, dtype=np.float64).reshape(3)
        dn = float(np.linalg.norm(world_dir))
        dist = float(distance_m)
        n_steps = max(int(n_steps), 1)
        if dn < 1e-9 or dist < 1e-6:
            return joints
        unit = world_dir / dn
        pose0 = self._mj_tool0_pose(joints, hand_cmd=hand)
        if pose0 is None:
            log(f"{label}: 无 tool0，跳过")
            return joints
        step_world = unit * (dist / float(n_steps))
        for i in range(1, n_steps + 1):
            pose = self._mj_tool0_pose(joints, hand_cmd=hand)
            if pose is None:
                log(f"{label}: 中间点 {i}/{n_steps} 无 tool0，中止")
                break
            _p, R = pose
            delta_tool0 = R.T @ step_world
            j_next = self._mj_preview_ik_delta(delta_tool0, joints, hand)
            if j_next is None:
                log(f"{label}: 中间点 {i}/{n_steps} IK 失败，停在此")
                break
            joints = list(j_next)
            kfs.append({
                "label": f"{label} ({i}/{n_steps})",
                "joints": list(joints),
                "hand": list(hand),
                "plan": plan,
                "micro": True,
            })
        return joints

    def _append_joint_path(self, kfs, joints_from, joints_to, hand, label, plan, n_steps):
        """⑥ 姿态对齐：在关节空间插入中间帧，便于看清旋转过程。"""
        j0 = list(joints_from)
        j1 = list(joints_to)
        n_steps = max(int(n_steps), 1)
        for i in range(1, n_steps + 1):
            t = float(i) / float(n_steps)
            joints = _lerp_joints(j0, j1, t)
            kfs.append({
                "label": f"{label} ({i}/{n_steps})",
                "joints": list(joints),
                "hand": list(hand),
                "plan": plan,
                "micro": True,
            })
        return list(joints_to)


    def _mj_sync_fn(self, joints, hand_cmd=None):
        return self._sync_mj_state(joints, hand_cmd=hand_cmd)

    def _compute_pinch_plan_mj(
        self, u, v, depth_mm, joints, hand_cmd, log=print, already_at_pregrasp=False,
        use_live_joints=True,
    ):
        """规划 ⑤⑥（MuJoCo 世界系）；robot IK 单独转换。

        use_live_joints=False：强制用传入 joints 作种子（Plan-once 回放一致）。
        """
        if self.mj_model is None or self.mj_data is None:
            return None
        joints_plan = list(joints)
        robot_pose = None
        # 实机夹取(busy)读最新关节；Plan-once / 预览可冻结种子
        if (
            use_live_joints
            and self.busy
            and self.controller.robot is not None
        ):
            self.controller.refresh_live_state()
            fresh = self.controller.read_joints_fresh(tries=2, pause_s=0.02)
            if fresh is not None:
                joints_plan = fresh
            robot_pose = self.controller.get_live_pose()
            if robot_pose is None:
                robot_pose = self.controller.get_pose()
        elif self.controller.robot is not None:
            robot_pose = self.controller.get_live_pose()
            if robot_pose is None:
                robot_pose = self.controller.get_pose()
        plan = compute_pinch_plan_mj(
            self.mj_model,
            self.mj_data,
            u,
            v,
            depth_mm,
            joints_plan,
            hand_cmd,
            self.controller,
            getattr(self, "_locked_object_mj", None),
            self.mj_cam_optical_in_body,
            self.config,
            self._mj_sync_fn,
            log=log,
            already_at_pregrasp=already_at_pregrasp,
            locked_object_robot=getattr(self, "_locked_object_robot", None),
            robot_pose=robot_pose,
            size_info=getattr(self, "_locked_size_info", None)
            or getattr(self, "_last_size_info", None),
        )
        if plan is not None:
            plan["robot_ik"] = plan_mj_to_robot_ik(
                self.controller.robot,
                plan,
                joints_plan,
                hand_cmd,
                self._mj_sync_fn,
                self.mj_model,
                self.mj_data,
                self.mj_joints,
                self.mj_hand_ranges,
                log=log,
                robot_pose=robot_pose,
                eye=self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {},
                config=self.config if isinstance(self.config, dict) else None,
            )
            if log is self.log:
                self._log_grasp_marker_coords(
                    "⑤⑥规划", u, v, depth_mm, plan=plan,
                )
        return plan

    def build_pick_plan(self, u, v, depth_mm, joints=None, log=print):
        """
        Plan-once：锁物后用当前关节作唯一种子，一次生成几何 + ⑥⑦ 关节路点。
        预览与实机共用返回的 plan dict（含 waypoints_6/7）。
        """
        ctrl = self.controller
        eye = self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {}
        size_info = getattr(self, "_locked_size_info", None) or getattr(
            self, "_last_size_info", None
        )
        ctrl.grasp_size_info = size_info
        pinch_open = list(
            ctrl.hand_presets.get("pinch_open", DEFAULT_PINCH_OPEN)
        )
        pinch_close = grasp_close_preset(ctrl, size_info=size_info, eye=eye)
        # 写回，供 ⑧ / 面板与规划一致
        ctrl.hand_presets["pinch_close"] = list(pinch_close)
        ctrl.hand_presets["grasp"] = list(pinch_close)
        stab = grasp_stabilize_preset(ctrl, size_info=size_info, eye=eye)
        if stab is not None:
            ctrl.hand_presets["pinch_stabilize"] = list(stab)
        if joints is None:
            if self.busy and ctrl.robot is not None:
                ctrl.refresh_live_state()
                joints = ctrl.get_live_joints()
            if joints is None:
                joints = ctrl.get_joints()
        if joints is None:
            log("Plan-once：无当前关节")
            return None
        seed = [float(x) for x in list(joints)[:7]]

        # 锁物后不再模拟 ③④；ee_sim = 当前 tool0
        # 几何用已写入 presets 的尺寸闭合手（与 ⑧ 一致）
        geom = self._compute_pinch_plan_mj(
            u, v, depth_mm, seed, pinch_open,
            log=log, already_at_pregrasp=True, use_live_joints=False,
        )
        if geom is None:
            log("Plan-once：几何规划失败")
            return None
        T_ee = geom.get("T_ee_des")
        if T_ee is None:
            log("Plan-once：无 T_ee_des")
            return None
        T_ee = np.asarray(T_ee, dtype=np.float64).reshape(4, 4)
        final_m = float(ctrl.pick_contact_advance_m())
        T_contact = compute_contact_T(
            T_ee, geom.get("obj_world"), geom.get("obj_target"), final_m,
        )
        if T_contact is None:
            log("Plan-once：无法算 T_ee_contact")
            return None

        # 禁止 max(...,2.4)：那会放宽到 137°，⑥⑦ 易换肘甩臂
        jump6 = float(eye.get("pinch_align_ik_jump_rad", ctrl.ik_max_jump_rad()))
        jump6 = float(np.clip(jump6, 0.35, 2.0))
        taper = float(eye.get("pinch_mj_ik_taper", 0.72))
        step6 = float(eye.get("pinch_align_step_m", 0.045))
        cap6 = int(eye.get("pinch_mj_ik_max_steps", 4))
        pose0 = self._mj_tool0_pose(seed, hand_cmd=pinch_open)
        travel6 = (
            float(np.linalg.norm(pose0[0] - T_ee[:3, 3])) if pose0 is not None else 0.1
        )
        cap6_eff = int(np.clip(max(cap6, int(math.ceil(travel6 / max(step6, 0.02)))), 1, 12))
        n6 = n_steps_for_travel(travel6, step6, cap6_eff)
        wps6 = self._plan_mj_ik_joint_waypoints(
            T_ee, seed, pinch_open, n6, jump6,
            log=log, log_prefix="Plan⑥", taper_power=taper,
            fixed_rotation=False,
            allow_partial=True,
        )
        # 首次失败或末点仍远：加密 + 略放宽关节跳再试（测地~57° 中段易卡）
        def _wps6_end_err_mm(wps):
            if not wps:
                return 1e9
            pe = self._mj_tool0_pose(wps[-1], hand_cmd=pinch_open)
            if pe is None:
                return 1e9
            return float(np.linalg.norm(pe[0] - T_ee[:3, 3])) * 1000.0

        need_retry = (not wps6) or (_wps6_end_err_mm(wps6) > 35.0)
        if need_retry:
            n6b = int(np.clip(max(n6 + 2, int(math.ceil(travel6 / 0.028))), 4, 14))
            jump6b = float(np.clip(jump6 * 1.25, jump6, 2.2))
            log(
                f"Plan⑥：重试加密路点 n={n6}→{n6b} "
                f"jump°={math.degrees(jump6):.0f}→{math.degrees(jump6b):.0f}"
            )
            wps6b = self._plan_mj_ik_joint_waypoints(
                T_ee, seed, pinch_open, n6b, jump6b,
                log=log, log_prefix="Plan⑥重试", taper_power=min(taper, 0.85),
                fixed_rotation=False,
                allow_partial=True,
            )
            if wps6b and (
                not wps6 or _wps6_end_err_mm(wps6b) < _wps6_end_err_mm(wps6)
            ):
                wps6 = wps6b
                n6 = n6b
                jump6 = jump6b
        if not wps6:
            log("Plan-once：⑥ 路点规划失败")
            return None
        end6 = _wps6_end_err_mm(wps6)
        if end6 > 80.0:
            log(
                f"Plan-once：⑥ 末点仍偏 {end6:.0f}mm > 80mm，拒绝残缺计划"
            )
            return None
        if end6 > 35.0:
            log(
                f"Plan-once：⑥ 末点偏 {end6:.0f}mm，仍交路点"
                f"（实机回放会补步/⑦纠偏）"
            )

        j_after6 = list(wps6[-1])
        # ⑦ 实机改为⑥后「闭合mid→青」重规划；此处不再预造 waypoints_7（免双源/白算）
        wps7 = []
        pose6 = self._mj_tool0_pose(j_after6, hand_cmd=pinch_open)
        travel7 = (
            float(np.linalg.norm(pose6[0] - T_contact[:3, 3]))
            if pose6 is not None else final_m
        )
        log(
            f"Plan⑦：跳过预造路点（实机⑥后按闭合mid→青重规划；"
            f"预览行程≈{travel7*1000:.0f}mm）"
        )

        sep_tgt = None
        try:
            from lbot_pinch_plan import grasp_close_sep_target_mm
            sep_tgt = grasp_close_sep_target_mm(size_info, eye)
        except Exception:
            sep_tgt = None
        cls = (size_info or {}).get("class") if isinstance(size_info, dict) else None
        gmode = pinch_grasp_mode(eye=eye)
        log(
            f"⑧闭合档={cls or 'default'} mode={gmode} 主抓L6={pinch_close}"
            + (f" 固定L6={stab}" if stab else "")
            + (f" 目标sep≈{sep_tgt:.0f}mm" if sep_tgt else "")
        )

        metrics = {
            "travel6_mm": travel6 * 1000.0,
            "travel7_mm": travel7 * 1000.0,
            "n6": len(wps6),
            "n7": len(wps7),
            "final_m": final_m,
            "contact_advance_mm": final_m * 1000.0,
            "hand_close": list(pinch_close),
            "hand_stabilize": list(stab) if stab else None,
            "size_class": cls,
            "grasp_mode": gmode,
        }
        pp = pick_plan_from_geom(
            geom, seed, pinch_open, pinch_close,
            wps6, wps7, T_contact, metrics=metrics,
        )
        if pp is None:
            return None
        plan = pp.as_exec_dict()
        plan["grasp_close_hand"] = list(pinch_close)
        plan["grasp_stabilize_hand"] = list(stab) if stab else None
        plan["grasp_mode"] = gmode
        if geom.get("robot_ik") is not None:
            plan["robot_ik"] = geom["robot_ik"]
        log(
            f"Plan-once：seed→⑥ {len(wps6)} 点（{travel6*1000:.0f}mm）→"
            f"⑦ {len(wps7)} 点（{travel7*1000:.0f}mm）；预览/实机同轨迹"
        )
        return plan

