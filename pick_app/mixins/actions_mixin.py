"""Button handlers: pick/align/place/extrinsic"""
from __future__ import annotations

from pick_app.deps import *
from pick_app.constants import *
from pick_app.helpers import *


class ActionsMixin:
    def _begin_motion(self):
        """新动作开始：清停止标志，允许运动。"""
        self._align_stop = False
        try:
            self.controller.clear_motion_abort()
        except Exception:
            pass

    def _run_action(self, name, fn):
        if self.busy:
            self.log(f"忙，忽略: {name}（可点「停止/解锁」）")
            return

        def work():
            self.busy = True
            self._begin_motion()
            try:
                self.log(f"{name}…")
                ok = fn()
                self.log(f"{name}: {'成功' if ok else '失败/超时（看终端；可继续调）'}")
            except Exception as error:
                self.log(f"{name}异常: {error}")
                print(f"[action] {name} exception:", error, flush=True)
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def _sync_model_from_live(self):
        """模型跟实机关节/手，避免沿用上次预览末帧造成「突然跳回」。"""
        joints = self.controller.get_joints()
        if joints is None:
            return
        self._model_joints = list(joints)
        hc = self.controller.hand_cmd
        if hc is not None and len(hc) >= 6:
            self._model_hand_cmd = [int(x) for x in hc[:6]]
        else:
            hp = self.controller.hand_presets
            self._model_hand_cmd = list(
                hp.get("pinch_open", hp.get("open", [255] * 6))
            )

    def on_clear_model_preview(self):
        self._stop_preview_animation()
        self._pinch_plan = None
        self._model_joints = None
        self._model_hand_cmd = None
        self._pinch_plan_time = 0.0
        self._preview_ee_trail = []
        self.log("已清除模型预览（模型恢复跟随实机）")

    def _virtual_object_camera_offset_m(self):
        """UI 虚拟物体偏移 → 相机系米 [右, 下, 前]。"""
        right_m = float(self.virtual_obj_right_cm.get()) / 100.0
        down_m = float(self.virtual_obj_down_cm.get()) / 100.0
        forward_m = max(float(self.virtual_obj_forward_cm.get()) / 100.0, 0.05)
        return right_m, down_m, forward_m

    def on_place_virtual_object(self):
        """在初始关节位姿的相机光心前方放置虚拟物体（无需真机/相机）。"""
        if self.mj_model is None:
            self.log("虚拟物体：模型未加载")
            return
        right_m, down_m, forward_m = self._virtual_object_camera_offset_m()

        # 固定用初始/home 关节 → 相机在「初始位置」
        joints = list(self.controller.default_home_joints)
        self._sync_mj_state(joints)

        depth_mm = forward_m * 1000.0
        intr = self.controller.intrinsics
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr.get("cx", self.args.width / 2))
        cy = float(intr.get("cy", self.args.height / 2))
        # OpenCV 相机系：x右 y下 z前 → 对应像素 (u,v) + 深度
        u = cx + fx * right_m / forward_m
        v = cy + fy * down_m / forward_m
        ray_kw = self.controller._vision_ray_params()

        obj_mj = _object_point_in_mj(
            self.mj_model,
            self.mj_data,
            u,
            v,
            depth_mm,
            intr,
            self.mj_cam_optical_in_body,
            **ray_kw,
        )
        if obj_mj is None:
            self.log("虚拟物体：无法从相机射线计算 3D 点")
            return

        self._locked_object_mj = np.asarray(obj_mj, dtype=np.float64).copy()
        self._virtual_object = True

        obj_robot = None
        if self.controller.robot is not None:
            fk = self.controller.robot.compute_forward_kinematics(
                self.controller.arm, joints
            )
            if fk is not None:
                obj_robot = detection_point_in_world(
                    u,
                    v,
                    depth_mm,
                    intr,
                    self.controller.R_ee_cam,
                    self.controller.t_ee_cam,
                    fk,
                    **ray_kw,
                )
        self._locked_object_robot = (
            np.asarray(obj_robot, dtype=np.float64).copy()
            if obj_robot is not None
            else self._locked_object_mj.copy()
        )
        self._locked_surface_robot = self._locked_object_robot.copy()
        self._locked_detection = (float(u), float(v), float(depth_mm))
        self.log(
            f"已放置虚拟物体 @ [{obj_mj[0]:.3f}, {obj_mj[1]:.3f}, {obj_mj[2]:.3f}] m "
            f"（初始相机：前 {forward_m*100:.0f}cm 下 {down_m*100:.0f}cm 右 {right_m*100:.0f}cm）"
        )
        self.log("可直接点「模型预览夹取」做对准/过程实验")

    def on_clear_virtual_object(self):
        self._virtual_object = False
        self._clear_locked_object()
        self.log("已清除虚拟物体锁定")


    def on_pinch_model_preview(self):
        """分步动画预览：①③④⑥⑦⑧⑨，与理论/实机夹取一致，不发实机运动。"""
        if self.busy:
            self.log("忙，忽略模型预览")
            return
        if getattr(self, "_preview_anim", None) and self._preview_anim.get("playing"):
            self.log("模型预览播放中…")
            return
        self._sync_pinch_z_tilt_to_eye()

        # 虚拟物体：无需相机检测
        using_virtual = bool(
            getattr(self, "_virtual_object", False)
            and self._locked_object_mj is not None
        )
        if not using_virtual:
            if not self._track_colors():
                self.log("模型预览：请先勾选追踪尺寸，或点「放置虚拟物体」")
                return
            sample = self._sample()
            if sample is None and self._locked_detection is None:
                self.log("模型预览：无检测目标（可先「放置虚拟物体」）")
                return
            det = self._detection_for_planning(sample)
            if det is None:
                self.log("模型预览：无检测目标")
                return
            u, v, depth_mm = det
            eye = self.config["eye_in_hand"]
            if depth_mm is None:
                depth_mm = self._depth_tracker.value
            if depth_mm is None:
                depth_mm = float(
                    self.config["eye_in_hand"].get("approach_fallback_depth_mm", 220)
                )
                self.log(f"模型预览：无深度，暂用 {depth_mm/10:.0f}cm")
            if self._locked_object_mj is None:
                # 物体 3D 用实测深度；预抓目标深度仅用于 ③④ 靠近量
                measured_depth = float(depth_mm)
                if not self.lock_object_for_grasp(
                    u, v, measured_depth, log=self.log, source="preview-pregrasp",
                ):
                    self.log("模型预览：物体位置冻结失败")
                    return
                u, v, depth_mm = self._locked_detection
                approach_depth = float(
                    self._depth_tracker.value
                    if self._depth_tracker.value is not None else measured_depth
                )
            else:
                self.log("使用已锁定物体位置")
                approach_depth = depth_mm
        else:
            u, v, depth_mm = self._locked_detection
            approach_depth = depth_mm
            self.log(
                f"使用虚拟物体（深度≈{float(depth_mm)/10:.1f}cm）做模型预览"
            )

        if self.controller.get_pose() is None:
            self.log("模型预览：需要末端位姿（仿真假连接或实机均可）")
            return

        # 从实机姿态起步，避免仍显示上次预览末帧（⑨ 提离位）时一点预览就跳变
        self._sync_model_from_live()
        self._preview_ee_trail = []
        eye = self.config["eye_in_hand"]
        from lbot_grasp_utils import resolve_pinch_standoff_m
        size_info = getattr(self, "_locked_size_info", None) or getattr(
            self, "_last_size_info", None
        )
        standoff = resolve_pinch_standoff_m(eye, size_info)
        frac = float(eye.get("pinch_contact_frac", 1.0))
        extra = float(eye.get("pinch_contact_extra_m", 0.025))
        contact_mm = (standoff * frac + extra) * 1000.0
        mode = str(eye.get("pinch_align_mode", "full_6dof"))
        cls = (size_info or {}).get("class") if isinstance(size_info, dict) else None
        so_note = f" standoff={standoff*1000:.0f}mm" + (f"/{cls}" if cls else "")
        self.log(
            f"▶ 模型预览 Plan-once：①开手→⑥{mode}→黄球→"
            f"⑦朝青球{contact_mm:.0f}mm→⑧闭合→⑨提离（无③④；与实机同轨迹）"
            f"{so_note}"
        )
        keyframes, plan = self._build_model_preview_keyframes(
            u, v, depth_mm, log=self.log, approach_depth_mm=approach_depth,
        )
        if not keyframes or plan is None:
            self.log("模型预览失败")
            return

        self._pinch_plan = plan
        self._pinch_plan_time = time.time()
        self._model_hand_cmd = list(keyframes[0]["hand"])
        self._model_joints = list(keyframes[0]["joints"])
        self._record_preview_ee_trail(self._model_joints, self._model_hand_cmd)
        # 全程显示规划球
        align_kf_i = 0
        micro_n = sum(1 for k in keyframes if k.get("micro"))
        self._preview_anim = {
            "playing": True,
            "keyframes": keyframes,
            "seg_i": 0,
            "seg_t": 0.0,
            "align_kf_i": align_kf_i,
            "seg_duration_s": float(
                self.config["eye_in_hand"].get("model_preview_seg_s", 1.0)
            ),
            "last_t": time.time(),
            "t0": time.time(),
            "_trail_acc": 0.0,
            "_ee_log_acc": 0.0,
        }
        self.log(
            f"▶ {keyframes[0]['label']}（共 {len(keyframes)-1} 段，其中中间点 {micro_n}）"
        )
        met = (plan or {}).get("pick_plan_metrics") or {}
        self.log(
            f"Plan-once 路点：⑥×{int(met.get('n6', 0))} "
            f"⑦×{int(met.get('n7', 0))}（预览=实机将回放的同一序列）"
        )
        self.log("绿线=tool0 轨迹；青球=虚拟物体；播放中可旋转模型")
        eye = self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {}
        if eye_verbose(eye):
            self.log("verbose_logs：预览/实机每周期打印 [preview ee]/[ee @...]")
        else:
            self.log("日志已精简（yaml: verbose_logs / pinch_debug 可开详情）")



    def _plan_grasp_for_execute(self, u, v, depth_mm):
        """
        实机 ⑤：优先回放预览缓存的 Plan-once（seed 未漂且青球未漂）；否则重规划。
        """
        self.controller.refresh_live_state()
        joints = self.controller.get_live_joints()
        if joints is None:
            joints = self.controller.get_joints()
        eye = self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {}
        cached = getattr(self, "_pinch_plan", None)
        locked = getattr(self, "_locked_object_mj", None)
        if (
            isinstance(cached, dict)
            and cached.get("plan_once")
            and cached.get("waypoints_6")
            and joints is not None
        ):
            tol = float(eye.get("plan_replay_seed_tol_deg", 8.0))
            abort = float(eye.get("plan_replay_seed_abort_deg", 25.0))
            ok, ddeg, action = seed_drift_status(
                joints, cached.get("seed_joints"), tol, abort,
            )
            age = time.time() - float(
                cached.get("pick_plan_created_at") or self._pinch_plan_time or 0.0
            )
            obj_ok = True
            obj_dmm = 0.0
            if locked is not None and cached.get("obj_world") is not None:
                obj_dmm = float(np.linalg.norm(
                    np.asarray(locked, dtype=np.float64).reshape(3)
                    - np.asarray(cached["obj_world"], dtype=np.float64).reshape(3)
                )) * 1000.0
                # 对准后重锁若青球漂了 >8mm，禁止回放旧轨迹
                obj_ok = obj_dmm <= 5.0
            if ok and age <= 120.0 and obj_ok:
                self.log(
                    f"⑤ 回放预览 Plan-once（seedΔ={ddeg:.1f}° action={action}"
                    f" 青球Δ={obj_dmm:.1f}mm；不重算 IK）"
                )
                return cached
            if not obj_ok:
                self.log(
                    f"⑤ 预览青球与锁物差 {obj_dmm:.1f}mm>5mm，重新 Plan-once"
                )
            elif action == "abort":
                self.log(
                    f"⑤ 预览 Plan seed 漂移 {ddeg:.1f}°>{abort:.0f}°，重新 Plan-once"
                )
            elif age > 120.0:
                self.log("⑤ 预览 Plan 已过期，重新 Plan-once")
        plan = self.build_pick_plan(
            u, v, depth_mm, joints=joints, log=self.log,
        )
        if plan is not None:
            self._pinch_plan = plan
            self._pinch_plan_time = time.time()
        return plan

    def on_pick_pinch(self):
        if self.busy:
            return
        if self.args.preview:
            self.log("预览模式，不执行夹取")
            return
        self._sync_pinch_z_tilt_to_eye()
        # 实机夹取：取消预览冻结的手/臂，模型跟实机
        self._stop_preview_animation()
        self._model_joints = None
        self._model_hand_cmd = None
        if self.controller.robot is None:
            self.log("夹取失败：未连接机械臂")
            return
        if not self._track_colors():
            self.log("夹取失败：请先勾选追踪尺寸 大/中/小")
            return
        sample = self._sample()
        if sample is None:
            with self.lock:
                status = self.latest.get("status", "")
            if "等待相机" in str(status) or self.latest.get("color") is None:
                self.log("无法夹取：相机尚无画面")
            else:
                self.log("无法夹取：无检测目标")
            return
        if self.args.confirm:
            stale = (
                self._pinch_plan is None
                or (time.time() - self._pinch_plan_time) > 120.0
            )
            msg = (
                "执行夹取到提离后持物停下？\n"
                "放筐/盘请再点：观察 → 对准2 → 放下"
            )
            if stale:
                msg = (
                    "尚未做「模型预览夹取」或预览已过期。\n"
                    "建议先点「模型预览夹取」在模型里核对 ⑤⑥。\n\n"
                    + msg
                )
            if not messagebox.askyesno("确认实机夹取", msg):
                return
        self._apply_hand_grasp_panel()
        self._apply_hand_open_panel()
        self._begin_motion()
        self._clear_locked_object()
        home_pose = None
        if bool(self.pick_return_panel.get()):
            home_joints = self._panel_home_joints()
            if home_joints is None:
                self.log("夹取失败：面板关节角无效")
                return
            # 持物回程需要原位笛卡尔（正上方 10cm）；面板模式原先只传关节导致安全路径被跳过
            home_pose = self.controller.home_pose_from_joints(home_joints)
            if home_pose is None:
                self.log("警告：面板原位 FK 失败，回程将先抬高再关节")
            else:
                hp, _ = home_pose
                self.log(
                    f"回面板原位笛卡尔(FK) xyz=[{hp.x:.3f},{hp.y:.3f},{hp.z:.3f}]"
                )
        else:
            home_joints, home_pose = self.controller.capture_home_state(self.log)
        self._reset_depth_memory()
        self.controller.reset_stability()
        speed = float(np.clip(self.move_speed.get() * 2.5, 0.15, 0.65))
        accel = float(np.clip(self.move_accel.get() * 2.5, 0.35, 1.2))

        def work():
            self.busy = True
            try:
                self.log(
                    f"▶ 夹取流程开始（预抓 ~"
                    f"{self.config['eye_in_hand'].get('pinch_pregrasp_depth_mm', 200)/10:.0f}cm）…"
                )
                if not self._ensure_depth_before_action():
                    self.log("夹取中止：无有效深度")
                    return
                ok = self.controller.execute_pick_pinch(
                    self._sample,
                    self.compute_pinch_frame_for_alignment,
                    self.log,
                    speed=speed,
                    accel=accel,
                    should_stop=lambda: self._align_stop or not self.running,
                    home_joints=home_joints,
                    home_pose=home_pose,
                    soft_prior_mm=self._last_reliable_depth_mm,
                    after_align=lambda u, v, d: self.lock_object_after_align(
                        u, v, d, log=self.log,
                    ),
                    lock_object_fn=(
                        lambda u, v, d: self.lock_object_at_pregrasp(u, v, d, log=self.log)
                    ),
                    get_object_world=lambda: self._locked_object_robot,
                    grasp_plan_fn=(
                        self._plan_grasp_for_execute
                        if self.mj_model is not None
                        else None
                    ),
                    tcp_diag_fn=self._log_pinch_tcp_diag,
                    mj_align_exec_fn=(
                        self._exec_align_mj_ik_joints
                        if self.mj_model is not None
                        else None
                    ),
                    mj_contact_exec_fn=(
                        self._exec_contact_advance_mj_ik
                        if self.mj_model is not None
                        else None
                    ),
                    place_sample_fn=None,
                    place_invalidate_fn=None,
                )
                self.picked = ok or self.picked
                hold = bool(
                    self.config["eye_in_hand"].get("place_hold_after_pick", True)
                ) and bool(
                    self.config["eye_in_hand"].get("place_enabled", True)
                ) and not bool(
                    self.config["eye_in_hand"].get("place_inline", False)
                )
                if ok and hold:
                    self._pipeline_phase = "holding"
                    self._locked_place_uvd = None
                    if (
                        getattr(self, "_plate_frozen_xyz", None) is not None
                        and bool(self.detect_plate_var.get())
                    ):
                        self.log(
                            "夹取成功（持物）。下一步：观察 → 放下"
                            "（盘心已冻结，可跳过对准2）"
                        )
                    else:
                        self.log("夹取成功（持物）。下一步：观察 → 对准2 → 放下")
                else:
                    self._pipeline_phase = "idle" if ok else self._pipeline_phase
                    self.log("夹取成功" if ok else "夹取结束（未完成/已停）")
            except Exception as error:
                self.log(f"夹取异常: {error}")
                print(f"[action] 夹取 exception:", error, flush=True)
            finally:
                hold_now = self._pipeline_phase == "holding"
                self._after_pick_or_align_cleanup(clear_lock=not hold_now)

        threading.Thread(target=work, daemon=True).start()

    def on_pick(self):
        if self.busy:
            return
        if self.args.preview:
            self.log("预览模式，不执行抓取")
            return
        self._stop_preview_animation()
        self._model_joints = None
        self._model_hand_cmd = None
        if self.controller.robot is None:
            self.log("抓取失败：未连接机械臂")
            return
        if not self._track_colors():
            self.log("抓取失败：请先勾选追踪尺寸 大/中/小")
            return
        # 抓取前把面板手参数写入控制器
        self._apply_hand_grasp_panel()
        self._apply_hand_open_panel()
        sample = self._sample()
        if sample is None:
            with self.lock:
                status = self.latest.get("status", "")
            if "等待相机" in str(status) or self.latest.get("color") is None:
                self.log("无法抓取：相机尚无画面（关掉占用 Orbbec 的程序后等首帧）")
            else:
                self.log("无法抓取：无检测目标")
            return
        if self.args.confirm:
            if not messagebox.askyesno("确认", "执行右臂：对准→手偏移→靠近→抓取？"):
                return
        self._begin_motion()
        # 回位目标：默认用面板 J1–J7；取消勾选则记录开抓瞬间实机关节
        home_pose = None
        if bool(self.pick_return_panel.get()):
            home_joints = self._panel_home_joints()
            if home_joints is None:
                self.log("抓取失败：面板关节角无效，请检查 J1–J7")
                return
            self.log(
                "P 回位将用面板关节: "
                + ", ".join(f"{math.degrees(j):.1f}°" for j in home_joints)
            )
        else:
            home_joints, home_pose = self.controller.capture_home_state(self.log)
        # 第二次抓取：清掉近距冻结深度，避免仍按上一次的近距规划
        self._reset_depth_memory()
        self._place_size_class = None
        self._locked_place_xyz = None
        self.controller.reset_stability()
        # 与对准按钮同一套速度限制（面板 ×2.5），对准阶段再 ×align_speed_scale
        speed = float(np.clip(self.move_speed.get() * 2.5, 0.15, 0.65))
        accel = float(np.clip(self.move_accel.get() * 2.5, 0.35, 1.2))

        def work():
            self.busy = True
            try:
                self.log(
                    f"▶ 抓取流程开始（靠近 speed={speed:.2f} accel={accel:.2f}）…"
                )
                if not self._ensure_depth_before_action():
                    self.log("抓取中止：无有效深度")
                    return
                ok = self.controller.execute_pick(
                    self._sample,
                    self.log,
                    speed=speed,
                    accel=accel,
                    should_stop=lambda: self._align_stop or not self.running,
                    home_joints=home_joints,
                    home_pose=home_pose,
                    after_align=lambda u, v, d: self.lock_object_after_align(u, v, d),
                )
                self.picked = ok or self.picked
                self.log("抓取成功" if ok else "抓取结束（未完成/已停）")
            except Exception as error:
                self.log(f"抓取异常: {error}")
                print("[pick]", error, flush=True)
            finally:
                self._after_pick_or_align_cleanup(clear_lock=True)

        threading.Thread(target=work, daemon=True).start()

    def on_stop_align(self):
        self._align_stop = True
        try:
            self.controller.request_motion_abort()
        except Exception:
            pass
        self.log("已请求停止对准")

    def on_stop_motion(self):
        """打断一切等待/轨迹/拧螺丝，并立刻解除 busy。"""
        self._align_stop = True
        self._handeye_seq_stop = True
        self.busy = False
        try:
            self.controller.request_motion_abort()
            self.controller.halt_all_motion("[stop]")
        except Exception as exc:
            print(f"[stop] halt: {exc}", flush=True)
        self.log("已停止一切动作并解锁")

    def on_align_center(self):
        if self.busy:
            self.log("忙，忽略对准")
            return
        if self.controller.robot is None:
            self.log("对准失败：未连接机械臂")
            return
        if not self._track_colors():
            self.log("对准失败：请先勾选追踪尺寸 大/中/小")
            return
        self._begin_motion()
        self._place_aligning = False
        self.on_clear_model_preview()
        self._clear_locked_object()
        speed = float(np.clip(self.move_speed.get() * 2.5, 0.15, 0.65))
        align_speed = speed * float(self.controller.eye.get("align_speed_scale", 0.35))
        align_period = float(self.controller.eye.get("align_step_period_s", 0.04))

        def work():
            self.busy = True
            self._place_aligning = False
            try:
                self.log(
                    f"▶ 连续比例对准（speed={align_speed:.2f}，周期 {align_period*1000:.0f}ms）…"
                )
                if not self._ensure_depth_before_action():
                    self.log("对准中止：无有效深度")
                    return
                eye = self.config["eye_in_hand"]
                depth_ref = [None]
                fusion_out = [None, None, None]
                fusion_buffer = None
                if bool(eye.get("align_fusion_frames", True)):
                    fusion_buffer = AlignFusionBuffer(
                        max_samples=int(eye.get("align_fusion_max_samples", 16)),
                        depth_spread_mm=float(
                            eye.get("align_fusion_depth_spread_mm", 45)
                        ),
                        max_depth_mm=float(eye.get("depth_action_max_mm", 300)),
                        min_depth_mm=float(eye.get("depth_min_valid_mm", 80)),
                    )
                ok = self.controller.align_center_only(
                    self._sample,
                    should_stop=lambda: self._align_stop or not self.running,
                    log=self.log,
                    speed=align_speed,
                    step_period_s=align_period,
                    depth_holder=depth_ref,
                    fusion_buffer=fusion_buffer,
                    fusion_out=fusion_out,
                )
                self.controller.settle_after_follow(log=self.log)
                if ok:
                    sample = self._sample()
                    u = v = d_use = None
                    if fusion_out[0] is not None:
                        u, v, d_use = fusion_out[0], fusion_out[1], fusion_out[2]
                    if sample is not None:
                        if u is None:
                            u, v, d = sample
                            d_use = depth_ref[0] if depth_ref[0] is not None else d
                        elif d_use is None:
                            d_use = depth_ref[0] if depth_ref[0] is not None else sample[2]
                    d_use, why = sanitize_action_depth_mm(
                        d_use, eye, soft_prior_mm=self._last_reliable_depth_mm,
                    )
                    if u is not None and d_use is not None:
                        self.lock_object_after_align(u, v, d_use)
                    else:
                        self.log(f"对准完成但深度不可用，未锁定（{why}）")
                self.log("对准完成" if ok else "对准结束（未完成/已停）")
            except Exception as error:
                self.log(f"对准异常: {error}")
                print("[align]", error, flush=True)
                try:
                    self.controller.settle_after_follow(log=self.log)
                except Exception:
                    pass
            finally:
                # 对准后保留紫球锁定，仅解锁 busy；夹取结束才会清锁
                self._after_pick_or_align_cleanup(clear_lock=False)

        threading.Thread(target=work, daemon=True).start()

    def _sample_screw_board(self):
        """螺丝板中心 (u,v,depth_mm)，供对准板使用。"""
        with self.lock:
            u = self.latest.get("screw_board_u")
            v = self.latest.get("screw_board_v")
            d = self.latest.get("screw_board_depth_mm")
            if u is None or v is None:
                return None
            eye = self.config["eye_in_hand"]
            min_mm = float(eye.get("depth_min_valid_mm", 80))
            max_mm = float(eye.get("depth_action_max_mm", 500))
            if d is not None and np.isfinite(d):
                d = float(d)
                if d < min_mm or d > max_mm:
                    d = None
            return (float(u), float(v), d)

    def on_align_board(self):
        """对准螺丝板中心，并沿光轴推到约 25cm 便于孔识别。"""
        if self.busy:
            self.log("忙，忽略对准板")
            return
        if self.controller.robot is None:
            self.log("对准板失败：未连接机械臂")
            return
        if not self._want_screw_board_detect():
            self.log("对准板失败：请勾选「螺丝板」")
            return
        if int(self.controller.arm) != int(LbotArm.RIGHT_ARM):
            self.log("对准板：请切到「右臂(相机)」")
            return
        sb_cfg = getattr(self, "screw_board_cfg", None) or {}
        target_mm = float(sb_cfg.get("align_distance_mm", 250))
        self._begin_motion()
        self._place_aligning = False
        # 对准板：粗段大步(关节) → 精修 pose_follow（避免俯视微步卡住）
        speed = float(np.clip(self.move_speed.get() * 1.4, 0.12, 0.32))
        align_speed = speed * float(
            self.controller.eye.get("align_approach_speed_scale", 0.70)
        )
        align_period = float(
            self.controller.eye.get(
                "align_approach_step_period_s",
                self.controller.eye.get("align_step_period_s", 0.05),
            )
        )
        coarse_speed = float(np.clip(self.move_speed.get(), 0.14, 0.35))

        def work():
            self.busy = True
            self._place_aligning = False
            stop = lambda: self._align_stop or not self.running
            try:
                if self._sample_screw_board() is None:
                    self.log("对准板失败：未检出板（请先看板位/勾选螺丝板）")
                    return
                self.log(
                    f"▶ 对准板：粗靠近+精修 → {target_mm:.0f}mm "
                    f"（coarse={coarse_speed:.2f} fine={align_speed:.2f}）…"
                )
                depth_ref = [None]
                eye = self.controller.eye
                prev_tol = eye.get("pixel_tolerance_px")
                prev_settle = eye.get("align_settle_err_px")
                prev_depth_tol = eye.get("distance_tolerance_mm")
                eye["pixel_tolerance_px"] = float(
                    sb_cfg.get("align_pixel_tol_px", 20)
                )
                eye["align_settle_err_px"] = float(
                    sb_cfg.get("align_settle_err_px", 16)
                )
                eye["distance_tolerance_mm"] = float(
                    sb_cfg.get("align_depth_tol_mm", 16)
                )
                try:
                    ok_coarse = self.controller.coarse_align_approach(
                        self._sample_screw_board,
                        should_stop=stop,
                        log=self.log,
                        approach_depth_mm=target_mm,
                        speed=coarse_speed,
                        accel=max(0.22, float(self.move_accel.get())),
                    )
                    if not ok_coarse:
                        self.log("对准板：粗段失败")
                        return
                    if stop():
                        self.log("对准板已停止")
                        return
                    ok = self.controller.align_center_only(
                        self._sample_screw_board,
                        should_stop=stop,
                        log=self.log,
                        speed=align_speed,
                        step_period_s=align_period,
                        depth_holder=depth_ref,
                        approach_depth_mm=target_mm,
                    )
                finally:
                    if prev_tol is None:
                        eye.pop("pixel_tolerance_px", None)
                    else:
                        eye["pixel_tolerance_px"] = prev_tol
                    if prev_settle is None:
                        eye.pop("align_settle_err_px", None)
                    else:
                        eye["align_settle_err_px"] = prev_settle
                    if prev_depth_tol is None:
                        eye.pop("distance_tolerance_mm", None)
                    else:
                        eye["distance_tolerance_mm"] = prev_depth_tol
                self.controller.settle_after_follow(log=self.log)
                sample = self._sample_screw_board()
                d_now = depth_ref[0]
                err_now = None
                if sample is not None:
                    u, v, d = sample
                    if d is not None and np.isfinite(d):
                        d_now = d
                    cx = float(self.controller.intrinsics.get("cx", 320))
                    cy = float(self.controller.intrinsics.get("cy", 240))
                    err_now = math.hypot(float(u) - cx, float(v) - cy)
                # 精修超时但已够近：仍算可用（避免「看起来卡死」）
                soft_ok = False
                if not ok and d_now is not None and np.isfinite(d_now):
                    if abs(float(d_now) - target_mm) <= 22.0 and (
                        err_now is None or err_now <= 28.0
                    ):
                        soft_ok = True
                        self.log(
                            f"对准板：精修未稳但已可用 "
                            f"偏心{err_now:.0f}px 深{float(d_now)/10:.1f}cm"
                            if err_now is not None
                            else f"对准板：精修未稳但深度可用 {float(d_now)/10:.1f}cm"
                        )
                if not ok and not soft_ok:
                    self.log("对准板：未完成（中心/距离未同时到位）")
                    return
                if d_now is not None and np.isfinite(d_now):
                    self.log(
                        f"对准板完成：深={float(d_now)/10:.1f}cm "
                        f"（目标 {target_mm/10:.0f}cm）"
                        + (f" 偏心{err_now:.0f}px" if err_now is not None else "")
                    )
                else:
                    self.log("对准板完成")
            except Exception as error:
                self.log(f"对准板异常: {error}")
                print("[align-board]", error, flush=True)
                try:
                    self.controller.settle_after_follow(log=self.log)
                except Exception:
                    pass
            finally:
                self._after_pick_or_align_cleanup(clear_lock=False)

        threading.Thread(target=work, daemon=True).start()

    def on_place_observe(self):
        """夹取后：到观察笛卡尔位姿（关节不固定），停住等人看筐/盘。"""
        if self.busy:
            self.log("忙，忽略观察")
            return
        if self.controller.robot is None:
            self.log("观察失败：未连接机械臂")
            return
        if not (
            self.detect_basket_var.get() or self.detect_plate_var.get()
        ):
            self.log("观察失败：请先勾选「放置检」里的 筐 和/或 盘")
            return
        if self.detect_basket_var.get() and not self.basket_rgbd and self.yolo_basket is None:
            self.log("观察失败：勾了筐但未启用筐子 RGBD/YOLO")
            return
        if self.detect_plate_var.get() and self.yolo_place is None:
            self.log("观察失败：勾了盘但未加载放置 YOLO")
            return
        self._begin_motion()
        speed = float(np.clip(self.move_speed.get() * 2.5, 0.15, 0.65))
        accel = float(np.clip(self.move_accel.get() * 2.5, 0.35, 1.2))

        def work():
            self.busy = True
            try:
                self.log("▶ 观察：到观察位姿…")
                ok = self.controller.execute_place_observe(
                    log=self.log,
                    should_stop=lambda: self._align_stop or not self.running,
                    place_invalidate_fn=self._invalidate_place,
                    speed=speed,
                    accel=accel,
                    size_class=getattr(self, "_place_size_class", None),
                )
                if ok:
                    self._pipeline_phase = "observed"
                    self._locked_place_uvd = None
                    self._locked_place_xyz = None
                    self._place_locked_kind = None
                    self._basket_cache_hit = None
                    self._basket_cache_t = 0.0
                    # _plate_frozen_* 保留：第二颗可跳过对准2
                    if (
                        getattr(self, "_plate_frozen_xyz", None) is not None
                        and bool(self.detect_plate_var.get())
                    ):
                        self.log(
                            "观察完成 → 盘心已冻结，可直接「放下」；"
                            "盘挪了再点「对准2」"
                        )
                    else:
                        bits = []
                        if self.detect_basket_var.get():
                            bits.append("筐")
                        if self.detect_plate_var.get():
                            bits.append("盘")
                        self.log(
                            "观察完成 → 开始检测"
                            + ("/".join(bits) if bits else "")
                            + " → 对准2 → 放下"
                        )
                else:
                    self.log("观察结束（未完成/已停）")
            except Exception as error:
                self.log(f"观察异常: {error}")
                print("[observe]", error, flush=True)
            finally:
                # 保留 _place_size_class，只清检测锁
                self._after_pick_or_align_cleanup(clear_lock=False)

        threading.Thread(target=work, daemon=True).start()

    def on_align_place(self):
        """对准2：把盘心/筐对到黄十字（盘优先圆心精修；否则筐 RGBD）。"""
        if self.busy:
            self.log("忙，忽略对准2")
            return
        if self.controller.robot is None:
            self.log("对准2失败：未连接机械臂")
            return
        if not self._after_observe_place_vision():
            self.log("对准2失败：请先点「观察」到位")
            return
        if not (
            (self.detect_basket_var.get() and (
                self.basket_rgbd or self.yolo_basket is not None
            ))
            or (self.detect_plate_var.get() and self.yolo_place is not None)
        ):
            self.log("对准2失败：请勾选「筐」或「盘」")
            return
        self._begin_motion()
        self._locked_place_uvd = None
        self._locked_place_xyz = None
        self._place_locked_kind = None
        # 再点对准2：暂清冻结用实时盘心；失败则还原
        prev_frozen_xyz = getattr(self, "_plate_frozen_xyz", None)
        prev_frozen_uvd = getattr(self, "_plate_frozen_uvd", None)
        had_freeze = prev_frozen_xyz is not None
        self._plate_frozen_xyz = None
        self._plate_frozen_uvd = None
        self._place_circle_cache = None
        self._place_aligning = True
        speed = float(np.clip(self.move_speed.get() * 2.5, 0.15, 0.65))
        align_speed = speed * float(self.controller.eye.get("align_speed_scale", 0.35))
        align_period = float(self.controller.eye.get("align_step_period_s", 0.04))
        plate_mode = self._place_intent_is_plate()

        def work():
            self.busy = True
            self._place_aligning = True
            ok = False
            try:
                tag = "盘" if plate_mode else "筐"
                self.log(
                    f"▶ 对准2({tag})"
                    + ("（重新对准，已清冻结）" if had_freeze and plate_mode else "")
                    + f"（speed={align_speed:.2f}，"
                    f"周期 {align_period*1000:.0f}ms）…"
                )
                eye = self.config["eye_in_hand"]
                depth_ref = [None]
                fusion_out = [None, None, None]
                fusion_buffer = AlignFusionBuffer(
                    max_samples=int(eye.get("align_fusion_max_samples", 16)),
                    depth_spread_mm=float(
                        eye.get("align_fusion_depth_spread_mm", 80)
                    ),
                    max_depth_mm=float(
                        eye.get("place_depth_max_mm", 800)
                    ),
                    min_depth_mm=float(
                        eye.get("place_depth_min_mm", eye.get("depth_min_valid_mm", 80))
                    ),
                )
                ok = self.controller.align_center_only(
                    self._sample_place,
                    should_stop=lambda: self._align_stop or not self.running,
                    log=self.log,
                    speed=align_speed,
                    step_period_s=align_period,
                    depth_holder=depth_ref,
                    fusion_buffer=fusion_buffer,
                    fusion_out=fusion_out,
                )
                self.controller.settle_after_follow(log=self.log)
                if ok:
                    u = v = d_use = None
                    if fusion_out[0] is not None:
                        u, v, d_use = fusion_out[0], fusion_out[1], fusion_out[2]
                    sample = self._sample_place()
                    if sample is not None:
                        if u is None:
                            u, v, d_use = sample
                        elif d_use is None:
                            d_use = sample[2]
                    if u is not None and d_use is not None:
                        self._lock_place_slot_after_align(
                            float(u), float(v), float(d_use), log=self.log,
                        )
                        self._pipeline_phase = "place_aligned"
                    else:
                        self.log("对准2完成但未锁深度，放下将用当前检测")
                        self._pipeline_phase = "place_aligned"
                self.log("对准2完成" if ok else "对准2结束（未完成/已停）")
            except Exception as error:
                self.log(f"对准2异常: {error}")
                print("[align2]", error, flush=True)
                ok = False
            finally:
                # 未成功重新锁定时，还原旧冻结，避免白丢盘心
                if (
                    plate_mode
                    and getattr(self, "_plate_frozen_xyz", None) is None
                    and prev_frozen_xyz is not None
                ):
                    self._plate_frozen_xyz = prev_frozen_xyz
                    self._plate_frozen_uvd = prev_frozen_uvd
                    self.log("对准2未更新盘心，已还原上次冻结")
                self._after_pick_or_align_cleanup(clear_lock=False)

        threading.Thread(target=work, daemon=True).start()

    def on_place_drop(self):
        """放下：筐=格口沿；盘=mid 15cm→11cm→松手→抬15cm→回面板。"""
        if self.busy:
            self.log("忙，忽略放下")
            return
        if self.controller.robot is None:
            self.log("放下失败：未连接机械臂")
            return
        if (
            not (
                (self.detect_basket_var.get() and (
                    self.basket_rgbd or self.yolo_basket is not None
                ))
                or (self.detect_plate_var.get() and self.yolo_place is not None)
            )
            and self._locked_place_uvd is None
            and self._locked_place_xyz is None
            and getattr(self, "_plate_frozen_xyz", None) is None
        ):
            self.log("放下失败：请勾选筐/盘，或先对准2锁点")
            return
        plate_mode = self._place_target_is_plate()
        if plate_mode and not self._after_observe_place_vision():
            self.log("放下失败：请先点「观察」到位，再「放下」")
            return
        if self.args.confirm:
            if plate_mode:
                msg = (
                    "盘装配：夹取末端(mid)对准盘心，"
                    "先到上方15cm，降到11cm松手，抬回15cm后回面板？"
                )
            else:
                msg = (
                    "夹取末端移到对应格、降到口沿松手（指平面水平）并回面板？"
                )
            if not messagebox.askyesno("确认放下", msg):
                return
        self._begin_motion()
        home_pose = None
        if bool(self.pick_return_panel.get()):
            home_joints = self._panel_home_joints()
            if home_joints is None:
                self.log("放下失败：面板关节角无效")
                return
            home_pose = self.controller.home_pose_from_joints(home_joints)
        else:
            home_joints, home_pose = self.controller.capture_home_state(self.log)
        speed = float(np.clip(self.move_speed.get() * 2.5, 0.15, 0.65))
        accel = float(np.clip(self.move_accel.get() * 2.5, 0.35, 1.2))
        return_home = bool(
            self.config["eye_in_hand"].get("place_return_home", True)
        )

        def work():
            self.busy = True
            try:
                self.log("▶ 放下(盘装配)…" if plate_mode else "▶ 放下…")
                locked_xyz = getattr(self, "_locked_place_xyz", None)
                slot_uvd = self._locked_place_uvd
                if plate_mode:
                    # 盘：用对准2 锁点，或复用本会话冻结盘心（第二颗可跳过对准2）
                    if locked_xyz is None:
                        frozen = getattr(self, "_plate_frozen_xyz", None)
                        if frozen is not None:
                            locked_xyz = np.asarray(
                                frozen, dtype=np.float64
                            ).reshape(3)
                            if getattr(self, "_plate_frozen_uvd", None) is not None:
                                slot_uvd = self._plate_frozen_uvd
                    if locked_xyz is None and slot_uvd is not None:
                        pt = self.controller.object_point_in_world(
                            float(slot_uvd[0]),
                            float(slot_uvd[1]),
                            float(slot_uvd[2]),
                        )
                        if pt is not None:
                            locked_xyz = np.asarray(pt, dtype=np.float64).reshape(3)
                            self._locked_place_xyz = locked_xyz
                    if locked_xyz is None:
                        self.log(
                            "放下失败：请先对准2锁定盘心"
                            "（第一颗必须对准2；之后可复用冻结点）"
                        )
                        return
                    self.log(
                        f"放下：使用冻结盘心 "
                        f"[{locked_xyz[0]:.3f},{locked_xyz[1]:.3f},{locked_xyz[2]:.3f}]"
                    )
                elif locked_xyz is None:
                    slot_uvd = self._resolve_place_slot_uvd(log=self.log)
                    locked_xyz = getattr(self, "_locked_place_xyz", None)
                if plate_mode and locked_xyz is None and slot_uvd is None:
                    self.log("放下失败：无盘心锁点/检测（请先对准2）")
                    return
                cls = getattr(self, "_place_size_class", None)
                if not cls:
                    size_info = getattr(self, "_locked_size_info", None)
                    if isinstance(size_info, dict):
                        cls = size_info.get("class")
                tip_ee = None
                pinch_R = None
                thumb_ee = None
                pack = self._pinch_tip_and_R_in_ee()
                if pack is not None:
                    tip_ee = pack[0]
                    pinch_R = pack[1] if len(pack) > 1 else None
                    thumb_ee = pack[2] if len(pack) > 2 else None
                    if tip_ee is not None and thumb_ee is not None:
                        d_tm = float(np.linalg.norm(
                            np.asarray(thumb_ee) - np.asarray(tip_ee)
                        ))
                        self.log(
                            f"放下 TCP=三指mid(外接圆) 非拇指；"
                            f"|拇−mid|_ee={d_tm*1000:.0f}mm"
                        )
                else:
                    self.log("放下：无 MJ tip/pinchR，控制器将用近似偏移")
                ok = self.controller.execute_place_drop(
                    place_sample_fn=self._sample_place,
                    log=self.log,
                    should_stop=lambda: self._align_stop or not self.running,
                    home_joints=home_joints if return_home else None,
                    home_pose=home_pose if return_home else None,
                    return_home=return_home,
                    speed=speed,
                    accel=accel,
                    locked_uvd=slot_uvd,
                    locked_xyz=locked_xyz,
                    tip_in_ee=tip_ee,
                    pinch_R_in_ee=pinch_R,
                    thumb_in_ee=thumb_ee,
                    size_class=cls,
                    place_kind="plate" if plate_mode else "basket",
                )
                if ok:
                    self._pipeline_phase = "placed"
                    self._locked_place_uvd = None
                    self._locked_place_xyz = None
                    self._place_locked_kind = None
                    self._place_size_class = None
                    self.picked = False
                    # 保留 _plate_frozen_*，下颗观察后可直接放下
                if ok and plate_mode and getattr(self, "_plate_frozen_xyz", None) is not None:
                    self.log("放下成功（盘心仍冻结，下颗：夹取→观察→放下）")
                else:
                    self.log("放下成功" if ok else "放下结束（未完成/已停）")
            except Exception as error:
                self.log(f"放下异常: {error}")
                print("[place]", error, flush=True)
            finally:
                self._after_pick_or_align_cleanup(clear_lock=True)

        threading.Thread(target=work, daemon=True).start()

    def on_teach(self):
        self.log("手动示教已改为「外参验参 + 校准」按钮")

    def on_restore_mount(self):
        path = find_mjcf()
        ok = self.controller.restore_camera_mount(path)
        if not ok:
            self.log("未找到 MJCF，无法读取腕相机")
            return
        self._refresh_extrinsic_labels()
        t = self.controller.t_ee_cam.tolist()
        self.log(f"已从模型加载相机外参 t={np.round(t, 4).tolist()}（已清除手调标记）")
        # 把内存中可能被 yaml 覆盖的相机 body 恢复为当前（已跟 MJCF）外参
        self._sync_mj_camera_mount_from_yaml(write_mjcf=False, path=path, force=True)

    def _middle_tip_in_robot_world(self, joints=None, hand_cmd=None):
        """MuJoCo 中指 tip 相对 tool0 → 实机工作系。"""
        if self.mj_model is None:
            return None
        joints = joints if joints is not None else self.controller.get_joints()
        if joints is None:
            return None
        pose_r = self.controller.get_pose()
        if pose_r is None:
            return None
        pinch = self.compute_pinch_at_joints(joints, hand_cmd=hand_cmd)
        tool0 = self._mj_tool0_pose(joints, hand_cmd=hand_cmd)
        if pinch is None or tool0 is None or pinch.get("middle_tip_m") is None:
            return None
        tip_mj = np.asarray(pinch["middle_tip_m"], dtype=np.float64).reshape(3)
        p_mj, R_mj = tool0
        tip_ee = R_mj.T @ (tip_mj - p_mj)
        pos, eul = pose_r
        R_we = euler_rpy_to_matrix(eul.x, eul.y, eul.z)
        ee = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        return ee + R_we @ tip_ee

    def on_extrinsic_verify(self):
        """
        验参：假定黄十字已对准物体中心 → 把中指 tip 移到锁定物体点。
        优先用对准锁定；全程保持锁定，校准后仍可用同一点。
        """
        if self.busy:
            self.log("忙，忽略: 外参验参（可点「停止/解锁」）")
            return
        if self.mj_model is None:
            self.log("外参验参需要 MuJoCo 模型（中指 tip）")
            return

        def work():
            self.busy = True
            try:
                self.log(
                    "外参验参…（请先对准黄十字≈物体中心；"
                    "将移动中指 tip → 锁定物体点）"
                )
                obj = self._ensure_locked_for_extrinsic(log=self.log)
                if obj is None:
                    return
                if self._locked_detection is not None:
                    u, v, depth_mm = self._locked_detection
                    cx = float(self.controller.intrinsics.get("cx", 320))
                    cy = float(self.controller.intrinsics.get("cy", 240))
                    err_px = math.hypot(float(u) - cx, float(v) - cy)
                    self.log(
                        f"锁定 UV=({u:.0f},{v:.0f}) 相对黄十字偏心 {err_px:.0f}px "
                        f"depth={float(depth_mm)/10:.1f}cm"
                        + ("（建议先对准）" if err_px > 40 else "（已接近中心）")
                    )
                joints = self.controller.get_joints()
                tip = self._middle_tip_in_robot_world(joints=joints)
                if tip is None:
                    self.log("外参验参失败：无法算中指 tip")
                    return
                delta = np.asarray(obj, dtype=np.float64) - np.asarray(tip, dtype=np.float64)
                dist = float(np.linalg.norm(delta))
                fwd_mm = delta[0] * 1000.0
                right_mm = -delta[1] * 1000.0
                up_mm = delta[2] * 1000.0
                self.log(
                    f"锁定物体 [{obj[0]:.4f},{obj[1]:.4f},{obj[2]:.4f}]  "
                    f"中指tip [{tip[0]:.4f},{tip[1]:.4f},{tip[2]:.4f}]"
                )
                self.log(
                    f"中指tip→锁定物体 Δ={dist*1000:.1f}mm "
                    f"(上 {up_mm:.1f} / 右 {right_mm:.1f} / 前 {fwd_mm:.1f})"
                )
                if dist < 1e-4:
                    self.log("中指 tip 已在锁定点，无需移动；目视检查重合情况")
                    return
                # 验参行程常 5–15cm（下探+侧移）；俯视直线易卡死 → 强制关节优先
                speed = float(np.clip(self.move_speed.get(), 0.08, 0.35))
                accel = float(np.clip(self.move_accel.get(), 0.15, 0.8))
                self.log(
                    f"外参验参：平移法兰 {dist*1000:.0f}mm（保持姿态，优先关节 IK）"
                )
                ok = self.controller.move_delta_base_world(
                    delta, speed=speed, accel=accel, prefer_joint=True,
                )
                if not ok:
                    # 大行程一次 PTP 可能越软限位：拆成「先水平、再竖直」两段
                    self.log("外参验参：整段失败，改分两段（先水平再竖直）…")
                    lat = delta.copy()
                    lat[2] = 0.0
                    vert = np.array([0.0, 0.0, float(delta[2])], dtype=np.float64)
                    ok = True
                    if float(np.linalg.norm(lat)) > 1e-4:
                        ok = self.controller.move_delta_base_world(
                            lat, speed=speed, accel=accel, prefer_joint=True,
                        )
                    if ok and abs(float(vert[2])) > 1e-4:
                        ok = self.controller.move_delta_base_world(
                            vert, speed=speed, accel=accel, prefer_joint=True,
                        )
                if not ok:
                    self.log("外参验参：移动失败（物体锁定仍保留）")
                    return
                tip2 = self._middle_tip_in_robot_world()
                if tip2 is not None:
                    left = float(np.linalg.norm(obj - tip2)) * 1000.0
                    err = np.asarray(tip2, dtype=np.float64) - np.asarray(
                        obj, dtype=np.float64
                    )
                    # tip 相对锁点：+Y=偏左，+X=偏前，+Z=偏高
                    tip_right = -err[1] * 1000.0
                    tip_fwd = err[0] * 1000.0
                    tip_up = err[2] * 1000.0
                    self.log(
                        f"到位后 中指tip→锁定点残差 {left:.1f}mm "
                        f"(tip相对锁点 右{tip_right:.1f}/前{tip_fwd:.1f}/"
                        f"上{tip_up:.1f}mm)"
                    )
                    self.log(
                        "目视：若 tip 仍偏左/前/高，在「外参校准」填 "
                        "右+/前−/上−（世界系，已自动映到法兰 t）；"
                        "若 MJ tip≠实指，改用 pinch_grasp_bias，勿硬拧外参"
                    )
                else:
                    self.log(
                        "到位；目视后填「外参校准」右/前/上 cm（世界系）"
                    )
            except Exception as error:
                self.log(f"外参验参异常: {error}")
                print("[extrinsic_verify]", error, flush=True)
            finally:
                # 保留物体锁定，便于连续校准/再验参
                self.busy = False
                self._begin_motion()

        threading.Thread(target=work, daemon=True).start()

    def on_extrinsic_calibrate(self):
        """
        外参校准：UI 填世界系「右/前/上」(cm)，按当前法兰姿态映到 camera_on_ee。
        勿再把目视偏差直接当法兰 X/Y/Z（俯视时轴几乎拧开）。
        """
        if self.busy:
            self.log("忙，忽略: 外参校准")
            return
        try:
            right_m = float(self.calib_right_cm.get()) / 100.0
            forward_m = float(self.calib_forward_cm.get()) / 100.0
            up_m = float(self.calib_up_cm.get()) / 100.0
        except (tk.TclError, ValueError, TypeError, AttributeError):
            # 兼容旧属性名（若 UI 未刷新）
            try:
                right_m = float(self.calib_tx_cm.get()) / 100.0
                forward_m = float(self.calib_ty_cm.get()) / 100.0
                up_m = float(self.calib_tz_cm.get()) / 100.0
                self.log("外参校准：仍用旧 X/Y/Z 控件，已当 右/前/上 解释")
            except (tk.TclError, ValueError, TypeError, AttributeError):
                self.log("外参校准：输入无效")
                return
        if abs(right_m) + abs(forward_m) + abs(up_m) < 1e-6:
            self.log("外参校准：右/前/上均为 0，无修改")
            return
        if self._locked_detection is None:
            self.log("外参校准：尚无锁定物体，请先对准或验参")
            return
        ok = self.controller.apply_camera_translation_calib_world(
            right_m=right_m, forward_m=forward_m, up_m=up_m, log=self.log,
        )
        if ok:
            self._refresh_extrinsic_labels()
            t = np.round(self.controller.t_ee_cam, 4).tolist()
            self.log(
                f"校准后外参 translation={t} 已写入 yaml "
                f"（camera_on_ee_prefer_yaml=true）。"
                "调满意后把该组数发我，可写进默认。"
            )
            self._sync_mj_camera_mount_from_yaml(write_mjcf=True)
            if self._refresh_locked_object_after_extrinsic_calib(log=self.log):
                obj = self._locked_object_robot
                if obj is not None:
                    self.log(
                        f"锁定点已按新外参重算 @ "
                        f"[{obj[0]:.4f},{obj[1]:.4f},{obj[2]:.4f}] "
                        f"（UV/depth 未变，可再验参）"
                    )
            if hasattr(self, "calib_right_cm"):
                self.calib_right_cm.set(0.0)
                self.calib_forward_cm.set(0.0)
                self.calib_up_cm.set(0.0)
            elif hasattr(self, "calib_tx_cm"):
                self.calib_tx_cm.set(0.0)
                self.calib_ty_cm.set(0.0)
                self.calib_tz_cm.set(0.0)

    def _on_vision_close(self):
        # 只藏画面窗，不退出整程序
        self.vision_win.withdraw()
        self.log("已隐藏相机窗口（控制窗仍可用）")

    def close(self):
        if not self.running:
            return
        self.running = False
        stop_process(self.depth_relay)
        stop_process(self.depth_launch)
        if self.camera is not None:
            self.camera.release()
        if self.depth_file is not None:
            self.depth_file.unlink(missing_ok=True)
        if self.color_file is not None:
            self.color_file.unlink(missing_ok=True)
        if self.controller.robot is not None:
            self.controller.robot.disconnect()
        try:
            if self.vision_win.winfo_exists():
                self.vision_win.destroy()
        except tk.TclError:
            pass
        self.root.destroy()


