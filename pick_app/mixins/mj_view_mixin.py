"""Model view orbit/pan/zoom/render + preview animation"""
from __future__ import annotations

from pick_app.deps import *
from pick_app.constants import *
from pick_app.helpers import *


class MjViewMixin:
    def _focus_model_on_markers(self, points):
        """预览后将视角对准 marker 簇，便于看见紫/青/绿/灰球。"""
        if self.mj_cam is None or not points:
            return
        pts = np.array(
            [np.asarray(p, dtype=np.float64).reshape(3) for p in points],
            dtype=np.float64,
        )
        if pts.size == 0:
            return
        mask = np.all(np.isfinite(pts), axis=1)
        pts = pts[mask]
        if len(pts) < 1:
            return
        center = pts.mean(axis=0)
        self.mj_cam.lookat[:] = center
        span = float(np.max(np.linalg.norm(pts - center, axis=1)))
        self.mj_cam.distance = float(np.clip(span * 3.2 + 0.25, 0.35, 2.8))

    def _camera_basis(self):
        az = math.radians(float(self.mj_cam.azimuth))
        el = math.radians(float(self.mj_cam.elevation))
        forward = np.array([
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ], dtype=np.float64)
        forward /= max(np.linalg.norm(forward), 1e-9)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([-math.sin(az), math.cos(az), 0.0], dtype=np.float64)
        else:
            right /= rn
        up = np.cross(right, forward)
        up /= max(np.linalg.norm(up), 1e-9)
        return forward, right, up

    def _orbit_model(self, dx, dy):
        cw, ch = self._model_view_dims()
        az_speed = 360.0 / max(cw, 1)
        el_speed = 170.0 / max(ch, 1)
        self.mj_cam.azimuth = float(self.mj_cam.azimuth) - dx * az_speed * 0.4
        self.mj_cam.elevation = float(
            np.clip(float(self.mj_cam.elevation) - dy * el_speed * 0.4, -89.0, 89.0)
        )

    def _pan_model(self, dx, dy):
        cw, ch = self._model_view_dims()
        _, right, up = self._camera_basis()
        fovy = math.radians(float(self.mj_model.vis.global_.fovy))
        scale = 2.0 * float(self.mj_cam.distance) * math.tan(fovy / 2.0) / max(ch, 1)
        aspect = cw / max(ch, 1)
        delta = (-dx * scale * aspect) * right + (dy * scale) * up
        self.mj_cam.lookat[0] += delta[0]
        self.mj_cam.lookat[1] += delta[1]
        self.mj_cam.lookat[2] += delta[2]

    def _zoom_model_factor(self, factor, mx=None, my=None):
        if self.mj_cam is None:
            return
        old_dist = float(self.mj_cam.distance)
        new_dist = float(np.clip(old_dist * factor, 0.2, 12.0))
        if (
            mx is not None
            and my is not None
            and self.mj_model is not None
            and abs(new_dist - old_dist) > 1e-9
        ):
            cw, ch = self._model_view_dims()
            nx = (mx - cw * 0.5) / max(cw, 1)
            ny = (my - ch * 0.5) / max(ch, 1)
            _, right, up = self._camera_basis()
            fovy = math.radians(float(self.mj_model.vis.global_.fovy))
            pan_scale = old_dist * math.tan(fovy / 2.0) * (1.0 - new_dist / old_dist)
            aspect = cw / max(ch, 1)
            shift = (-nx * pan_scale * aspect) * right + (ny * pan_scale) * up
            self.mj_cam.lookat[0] += shift[0]
            self.mj_cam.lookat[1] += shift[1]
            self.mj_cam.lookat[2] += shift[2]
        self.mj_cam.distance = new_dist

    def _on_model_press(self, event):
        self.model_canvas.focus_set()
        self._drag_last = (event.x, event.y)
        btn = getattr(event, "num", 1)
        shift = bool(getattr(event, "state", 0) & 0x0001)
        if btn in (2, 3) or (btn == 1 and shift):
            self._drag_mode = "pan"
            self.model_canvas.configure(cursor="fleur")
        else:
            self._drag_mode = "orbit"
            self.model_canvas.configure(cursor="crosshair")

    def _on_model_release(self, _event):
        self._drag_last = None
        self.model_canvas.configure(cursor="fleur")

    def _on_model_drag(self, event):
        if self.mj_cam is None or self._drag_last is None:
            return
        dx = event.x - self._drag_last[0]
        dy = event.y - self._drag_last[1]
        self._drag_last = (event.x, event.y)
        if self._drag_mode == "pan":
            self._pan_model(dx, dy)
        else:
            self._orbit_model(dx, dy)
        self._flush_model_canvas()

    def _wheel_steps(self, event):
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return 0.0
        if abs(delta) >= 120:
            return delta / 120.0
        return float(delta)

    def _on_model_wheel(self, event):
        steps = self._wheel_steps(event)
        if abs(steps) < 1e-6:
            return
        factor = 0.9 ** steps
        self._zoom_model_factor(factor, event.x, event.y)
        self._flush_model_canvas()

    def _on_model_wheel_up(self, _event):
        self._zoom_model_factor(0.9, _event.x, _event.y)
        self._flush_model_canvas()

    def _on_model_wheel_down(self, _event):
        self._zoom_model_factor(1.0 / 0.9, _event.x, _event.y)
        self._flush_model_canvas()

    def _on_model_reset_view(self, _event=None):
        if self.mj_cam is None:
            return
        defaults = self._cam_default
        self.mj_cam.distance = float(defaults["distance"])
        self.mj_cam.azimuth = float(defaults["azimuth"])
        self.mj_cam.elevation = float(defaults["elevation"])
        lookat = defaults.get("lookat")
        if lookat is not None:
            self.mj_cam.lookat[:] = lookat
        self._flush_model_canvas()

    def _zoom_model(self, direction):
        """兼容旧调用：direction>0 拉远，<0 拉近。"""
        factor = 1.0 / 0.9 if direction > 0 else 0.9
        self._zoom_model_factor(factor)

    def _render_model(self, joints):
        if self.mj_renderer is None or self.mj_cam is None:
            return None
        block_sync = not self.busy
        if not self._sync_mj_state(joints, block=block_sync):
            return getattr(self, "model_photo", None)
        if self.busy:
            if not self._mj_lock.acquire(blocking=False):
                return getattr(self, "model_photo", None)
            try:
                if self._pinch_info is None:
                    self._pinch_info = _compute_o6_pinch_kinematics(
                        self.mj_model, self.mj_data, **self._pinch_kin_kwargs(),
                    )
            finally:
                self._mj_lock.release()
        else:
            with self._mj_lock:
                self._pinch_info = _compute_o6_pinch_kinematics(
                    self.mj_model, self.mj_data, **self._pinch_kin_kwargs(),
                )

        # 识别物体：对准/预览后锁定世界坐标；预览动画中禁止随动相机重算
        self._vision_obj_world = None
        cam_optical = None
        obj_frozen = self._locked_object_mj is not None
        in_model_preview = (
            not self.busy
            and (
                self._model_joints is not None
                or bool(
                    getattr(self, "_preview_anim", None)
                    and self._preview_anim.get("playing")
                )
            )
        )
        if obj_frozen:
            self._vision_obj_world = np.asarray(self._locked_object_mj, dtype=np.float64)
        else:
            with self.lock:
                lu, lv, ldepth = (
                    self.latest.get("u"), self.latest.get("v"), self.latest.get("depth_mm"),
                )
            if lu is not None and lv is not None:
                d_use = ldepth if ldepth is not None else self._depth_tracker.value
                if d_use is not None:
                    self._vision_obj_world = _object_point_in_mj(
                        self.mj_model, self.mj_data, lu, lv, d_use,
                        self.controller.intrinsics, self.mj_cam_optical_in_body,
                        **self.controller._vision_ray_params(),
                    )
        cam_pose = _mj_object_pose(
            self.mj_model, self.mj_data, "camera_optical", "arm_right_camera",
            optical_in_body=self.mj_cam_optical_in_body,
        )
        if cam_pose is not None:
            cam_optical = cam_pose[0]

        self.mj_renderer.update_scene(
            self.mj_data, camera=self.mj_cam, scene_option=self.mj_vopt
        )

        # 叠加关键坐标系三轴；黄球画在真实原点（相机用光心）
        label_poses = []
        for label, kind, name, meaning, length in getattr(self, "mj_frame_markers", []):
            pose = _mj_object_pose(
                self.mj_model,
                self.mj_data,
                kind,
                name,
                optical_in_body=self.mj_cam_optical_in_body,
            )
            if pose is None:
                continue
            pos, rot = pose
            _add_axis_triad(self.mj_renderer.scene, pos, rot, length=length)
            label_poses.append((label, pos))

        pinch = getattr(self, "_pinch_info", None)
        if pinch is not None:
            _add_scene_sphere(
                self.mj_renderer.scene, pinch["thumb_tip_m"],
                radius=0.004, rgba=MARKER_THUMB,
            )
            if pinch.get("index_tip_m") is not None:
                _add_scene_sphere(
                    self.mj_renderer.scene, pinch["index_tip_m"],
                    radius=0.004, rgba=MARKER_INDEX,
                )
            _add_scene_sphere(
                self.mj_renderer.scene, pinch["middle_tip_m"],
                radius=0.004, rgba=MARKER_MIDDLE,
            )
            if pinch.get("ring_tip_m") is not None:
                _add_scene_sphere(
                    self.mj_renderer.scene, pinch["ring_tip_m"],
                    radius=0.004, rgba=MARKER_MIDDLE,
                )
            _add_scene_sphere(
                self.mj_renderer.scene, pinch["pinch_mid_m"],
                radius=0.003, rgba=MARKER_PINCH_MID,
            )

        plan_marker_labels = []
        if bool(getattr(self, "show_vision_in_model", tk.BooleanVar(value=True)).get()):
            obj_w = getattr(self, "_vision_obj_world", None)
            if obj_w is not None:
                obj_rgba = (
                    MARKER_VIRTUAL_OBJ
                    if getattr(self, "_virtual_object", False)
                    else MARKER_OBJ
                )
                _add_scene_sphere(
                    self.mj_renderer.scene, obj_w,
                    radius=0.018, rgba=obj_rgba,
                )
                plan_marker_labels.append(
                    (
                        "⑤虚拟物" if getattr(self, "_virtual_object", False) else "⑤物体",
                        obj_w,
                        (40, 220, 180)
                        if getattr(self, "_virtual_object", False)
                        else (220, 80, 240),
                    )
                )
                # 锁定/预览时不再画随动射线（相机动会误以为球在跟动）
                if cam_optical is not None and not obj_frozen and not in_model_preview:
                    _add_scene_segment(
                        self.mj_renderer.scene, cam_optical, obj_w,
                        width=0.005, rgba=(0.85, 0.45, 0.95, 0.9),
                    )

        plan = getattr(self, "_pinch_plan", None)
        # 预览全程显示规划球/轴，方便对照每一步是否走偏
        show_plan_targets = True
        anim = getattr(self, "_preview_anim", None)
        plan_viz = plan
        if plan_viz is not None and show_plan_targets:
            _add_scene_sphere(
                self.mj_renderer.scene, plan_viz["obj_target"],
                radius=0.016, rgba=MARKER_STANDOFF,
            )
            plan_marker_labels.append(
                ("预捏合", plan_viz["obj_target"], (40, 220, 255)),
            )
            T_des = plan_viz.get("T_pinch_des")
            if T_des is not None:
                _add_axis_triad(
                    self.mj_renderer.scene,
                    T_des[:3, 3],
                    T_des[:3, :3],
                    length=0.09,
                    width=0.005,
                )
            _add_scene_sphere(
                self.mj_renderer.scene, plan_viz["target_pos"],
                radius=0.017, rgba=MARKER_TOOL0_ALIGN,
            )
            plan_marker_labels.append(
                ("⑥tool0", plan_viz["target_pos"], (80, 80, 255)),
            )
            ee_sim = plan_viz.get("ee_sim")
            if ee_sim is not None:
                _add_scene_sphere(
                    self.mj_renderer.scene, ee_sim,
                    radius=0.015, rgba=MARKER_PREGRASP_EE,
                )
                plan_marker_labels.append(("预抓EE", ee_sim, (200, 200, 200)))
            if not obj_frozen and not in_model_preview:
                obj_w = plan.get("obj_world")
                if obj_w is not None and cam_optical is not None:
                    _add_scene_segment(
                        self.mj_renderer.scene, cam_optical, obj_w,
                        width=0.004, rgba=(0.7, 0.35, 0.85, 0.75),
                    )

        # tool0 预览轨迹（绿折线）
        trail = getattr(self, "_preview_ee_trail", None) or []
        if len(trail) >= 2:
            for a, b in zip(trail[:-1], trail[1:]):
                _add_scene_segment(
                    self.mj_renderer.scene, a, b,
                    width=0.0035, rgba=MARKER_EE_TRAIL,
                )
            _add_scene_sphere(
                self.mj_renderer.scene, trail[-1],
                radius=0.006, rgba=MARKER_EE_TRAIL,
            )

        rgb = np.asarray(self.mj_renderer.render()).copy()
        if in_model_preview and not self.busy:
            cv2.putText(
                rgb, "模型预览姿态(非实机)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 255), 2, cv2.LINE_AA,
            )
            anim = getattr(self, "_preview_anim", None)
            if anim and anim.get("keyframes"):
                seg = int(anim.get("seg_i", 0))
                kfs = anim["keyframes"]
                cur = kfs[min(seg, len(kfs) - 1)]
                cv2.putText(
                    rgb,
                    f"{cur.get('label', '')}  [{seg+1}/{len(kfs)}]",
                    (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    rgb, f"trail={len(getattr(self, '_preview_ee_trail', []) or [])}",
                    (8, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 220, 160), 1, cv2.LINE_AA,
                )
        fovy = float(self.mj_model.vis.global_.fovy)
        glcam = self.mj_renderer.scene.camera[0]
        h, w = rgb.shape[:2]
        for label, pos in label_poses:
            pix = _project_world_to_pixel(pos, glcam, fovy, w, h)
            if pix is None:
                continue
            px = int(np.clip(pix[0], 4, w - 4))
            py = int(np.clip(pix[1], 14, h - 4))
            cv2.putText(
                rgb, label, (px + 6, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 40), 1, cv2.LINE_AA,
            )
        for label, pos, bgr in plan_marker_labels:
            pix = _project_world_to_pixel(pos, glcam, fovy, w, h)
            if pix is None:
                continue
            px = int(np.clip(pix[0], 4, w - 80))
            py = int(np.clip(pix[1], 14, h - 4))
            cv2.putText(
                rgb, label, (px + 8, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, bgr, 2, cv2.LINE_AA,
            )
        legend1 = (
            "紫=⑤物体中心(相机深度)" if not obj_frozen
            else "紫=⑤物体中心(已锁定)"
        )
        legend2 = (
            "黄=预捏合点  红=⑥tool0  灰=③④预抓法兰  |  橙拇指 绿食指 蓝中指  黄=三指中心"
        )
        if in_model_preview and not self.busy:
            legend2 += "  ·  预览(非实机)"
        elif in_model_preview and self.busy:
            legend2 += "  ·  实机运动中"
        cv2.putText(
            rgb, legend1, (8, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA,
        )
        cv2.putText(
            rgb, legend2, (8, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA,
        )
        cw, ch = self._model_view_dims()
        return _rgb_to_photo_fit(rgb, cw, ch)

    def _stop_preview_animation(self):
        self._preview_anim = None

    def _tick_preview_animation(self):
        """分步插值播放模型预览；中间微段更快，并记录 tool0 轨迹。"""
        anim = getattr(self, "_preview_anim", None)
        if not anim or not anim.get("playing"):
            return
        now = time.time()
        dt = float(now - anim.get("last_t", now))
        anim["last_t"] = now
        kfs = anim["keyframes"]
        seg = int(anim["seg_i"])
        if seg >= len(kfs) - 1:
            last = kfs[-1]
            self._model_joints = list(last["joints"])
            self._model_hand_cmd = list(last["hand"])
            anim["seg_i"] = len(kfs) - 1
            anim["playing"] = False
            self.log(
                f"模型预览播放完成 → {last['label']}"
                f"（手/臂仍冻结；开合手或「清除预览」可恢复跟随）"
            )
            return

        nxt = kfs[seg + 1]
        base_dur = float(anim.get("seg_duration_s", 1.0))
        # 微段（中间点）缩短，整体流程仍完整但过程更细
        if nxt.get("micro") or kfs[seg].get("micro"):
            seg_dur = max(base_dur * 0.28, 0.08)
        else:
            seg_dur = max(base_dur, 0.12)
        anim["seg_t"] = float(anim.get("seg_t", 0.0)) + dt / seg_dur
        if anim["seg_t"] >= 1.0:
            nxt_i = seg + 1
            self._model_joints = list(nxt["joints"])
            self._model_hand_cmd = list(nxt["hand"])
            anim["seg_i"] = nxt_i
            anim["seg_t"] = 0.0
            if not nxt.get("micro"):
                self.log(f"▶ {nxt['label']}")
            self._record_preview_ee_trail(self._model_joints, self._model_hand_cmd)
            eye = self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {}
            if eye_verbose(eye):
                pos = self._mj_tool0_pos(self._model_joints, self._model_hand_cmd)
                if pos is not None:
                    self.log(
                        f"[preview ee] {nxt['label']} "
                        f"xyz=[{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}]"
                    )
            return

        t = _smoothstep(anim["seg_t"])
        j0, j1 = kfs[seg]["joints"], kfs[seg + 1]["joints"]
        self._model_joints = _lerp_joints(j0, j1, t)
        h0, h1 = kfs[seg]["hand"], kfs[seg + 1]["hand"]
        self._model_hand_cmd = [
            int(round((1.0 - t) * float(a) + t * float(b)))
            for a, b in zip(h0, h1)
        ]
        # 每隔 0.1s 打印预览末端（仅 verbose_logs）
        eye = self.config.get("eye_in_hand", {}) if isinstance(self.config, dict) else {}
        if eye_verbose(eye):
            anim["_ee_log_acc"] = float(anim.get("_ee_log_acc", 0.0)) + dt
            if anim["_ee_log_acc"] >= 0.1:
                anim["_ee_log_acc"] = 0.0
                pos = self._mj_tool0_pos(self._model_joints, self._model_hand_cmd)
                if pos is not None:
                    lbl = kfs[seg + 1].get("label", "?")
                    self.log(
                        f"[preview ee @{time.time()-anim.get('t0', anim['last_t']):5.1f}s] "
                        f"{lbl} xyz=[{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}]"
                    )
        # 每隔一小段采样轨迹，避免每帧都算
        anim["_trail_acc"] = float(anim.get("_trail_acc", 0.0)) + dt
        if anim["_trail_acc"] >= 0.05:
            anim["_trail_acc"] = 0.0
            self._record_preview_ee_trail(self._model_joints, self._model_hand_cmd)

    def _record_preview_ee_trail(self, joints, hand_cmd=None):
        pos = self._mj_tool0_pos(joints, hand_cmd=hand_cmd)
        if pos is None:
            return
        trail = getattr(self, "_preview_ee_trail", None)
        if trail is None:
            self._preview_ee_trail = []
            trail = self._preview_ee_trail
        if trail and float(np.linalg.norm(pos - trail[-1])) < 0.0015:
            return
        trail.append(pos.copy())
        max_n = int(self.config["eye_in_hand"].get("model_preview_trail_max", 240))
        if len(trail) > max_n:
            del trail[: len(trail) - max_n]

    def _build_model_preview_keyframes(self, u, v, depth_mm, log=print, approach_depth_mm=None):
        """
        Plan-once 预览：①开手 → 回放 build_pick_plan 的 waypoints_6/7 → ⑧闭合 → ⑨提离。
        不再自建 ③④⑥⑦ IK；与实机同一份关节轨迹。
        """
        ctrl = self.controller
        eye = self.config["eye_in_hand"]
        detailed = bool(eye.get("model_preview_detailed", False))
        n_cart = int(eye.get("model_preview_cart_steps", 8)) if detailed else 3
        pinch_open = list(
            ctrl.hand_presets.get(
                "pinch_open", ctrl.hand_presets.get("open", [255] * 6),
            )
        )
        pinch_close = list(
            ctrl.hand_presets.get(
                "grasp",
                ctrl.hand_presets.get("pinch_close", DEFAULT_GRASP_PRESET),
            )
        )
        seed = list(ctrl.get_joints())
        if seed is None or len(seed) < 7:
            log("模型预览：无当前关节")
            return None, None
        seed = [float(x) for x in seed[:7]]
        cur_hand = ctrl.hand_cmd
        if cur_hand is not None and len(cur_hand) >= 6:
            start_hand = [int(x) for x in cur_hand[:6]]
        else:
            start_hand = list(pinch_open)

        plan = self.build_pick_plan(u, v, depth_mm, joints=seed, log=log)
        if plan is None:
            return None, None
        # 与 Plan-once / ⑧ 同一闭合档
        pinch_close = list(
            plan.get("grasp_close_hand")
            or ctrl.hand_presets.get("pinch_close")
            or pinch_close
        )
        kfs = [{
            "label": "① 起点",
            "joints": list(seed),
            "hand": start_hand,
            "plan": plan,
        }]
        if start_hand != list(pinch_open):
            kfs.append({
                "label": "① 张手",
                "joints": list(seed),
                "hand": list(pinch_open),
                "plan": plan,
                "micro": True,
            })

        joints = list(seed)
        wps6 = plan.get("waypoints_6") or []
        for i, w in enumerate(wps6, start=1):
            joints = [float(x) for x in list(w)[:7]]
            kfs.append({
                "label": f"⑥→黄球 ({i}/{len(wps6)})",
                "joints": list(joints),
                "hand": list(pinch_open),
                "plan": plan,
                "micro": True,
            })
        if wps6:
            kfs.append({
                "label": "⑥ 三指 TCP→黄球（Plan-once）",
                "joints": list(joints),
                "hand": list(pinch_open),
                "plan": plan,
            })
            # 闭合手验 mid（与旧预览一致）
            yellow = plan.get("obj_target")
            obj_w = plan.get("obj_world")
            if yellow is not None and obj_w is not None and self._sync_mj_state(
                joints, hand_cmd=pinch_close,
            ):
                info_c = _compute_o6_pinch_kinematics(
                    self.mj_model, self.mj_data, **self._pinch_kin_kwargs()
                )
                if info_c is not None and info_c.get("pinch_mid_m") is not None:
                    mid = np.asarray(info_c["pinch_mid_m"], dtype=np.float64).reshape(3)
                    yellow = np.asarray(yellow, dtype=np.float64).reshape(3)
                    obj_w = np.asarray(obj_w, dtype=np.float64).reshape(3)
                    to_y = float(np.linalg.norm(mid - yellow)) * 1000.0
                    geo = yellow - obj_w
                    gn = float(np.linalg.norm(geo))
                    d = mid - obj_w
                    if gn > 1e-6:
                        uu = geo / gn
                        along = float(np.dot(d, uu)) * 1000.0
                        lat = float(np.linalg.norm(d - uu * (along / 1000.0))) * 1000.0
                    else:
                        along = lat = float("nan")
                    log(
                        f"⑥ 预览验(闭合手) mid→黄球 {to_y:.1f}mm；"
                        f"mid→青球 along={along:.1f}mm 侧向={lat:.1f}mm "
                        f"（侧向>8mm 则模型里会明显歪）"
                    )
                self._sync_mj_state(joints, hand_cmd=pinch_open)

        wps7 = plan.get("waypoints_7") or []
        for i, w in enumerate(wps7, start=1):
            joints = [float(x) for x in list(w)[:7]]
            kfs.append({
                "label": f"⑦→青球 ({i}/{len(wps7)})",
                "joints": list(joints),
                "hand": list(pinch_open),
                "plan": plan,
                "micro": True,
            })
        if wps7:
            final_m = float((plan.get("pick_plan_metrics") or {}).get("final_m", 0.0))
            kfs.append({
                "label": f"⑦ 朝青球补回 {final_m*1000:.0f}mm（Plan-once）",
                "joints": list(joints),
                "hand": list(pinch_open),
                "plan": plan,
            })

        pinch_step1 = list(pinch_open)
        pinch_step1[1] = int(pinch_close[1])
        kfs.append({
            "label": "⑧① 拇指侧摆",
            "joints": list(joints),
            "hand": pinch_step1,
            "plan": plan,
        })
        stab = None
        if isinstance(plan, dict):
            stab = plan.get("grasp_stabilize_hand")
        if stab is None:
            stab = ctrl.hand_presets.get("pinch_stabilize")
        gmode = pinch_grasp_mode(eye=eye)
        if detailed:
            n_pinch = int(eye.get("model_preview_pinch_steps", 4))
            h0 = list(pinch_step1)
            h1 = list(pinch_close)
            for i in range(1, max(n_pinch, 1) + 1):
                t = float(i) / float(max(n_pinch, 1))
                hand = [
                    int(round((1.0 - t) * float(a) + t * float(b)))
                    for a, b in zip(h0, h1)
                ]
                kfs.append({
                    "label": f"⑧② 主抓 ({i}/{n_pinch})",
                    "joints": list(joints),
                    "hand": hand,
                    "plan": plan,
                    "micro": True,
                })
        else:
            kfs.append({
                "label": "⑧② 主抓" if gmode == "power5" else "⑧② pinch_close",
                "joints": list(joints),
                "hand": pinch_close,
                "plan": plan,
            })
        if gmode == "power5" and stab is not None:
            if detailed:
                n_stab = max(2, int(eye.get("model_preview_pinch_steps", 4)) // 2)
                h0 = list(pinch_close)
                h1 = list(stab)
                for i in range(1, n_stab + 1):
                    t = float(i) / float(n_stab)
                    hand = [
                        int(round((1.0 - t) * float(a) + t * float(b)))
                        for a, b in zip(h0, h1)
                    ]
                    kfs.append({
                        "label": f"⑧③ 食小固定 ({i}/{n_stab})",
                        "joints": list(joints),
                        "hand": hand,
                        "plan": plan,
                        "micro": True,
                    })
            else:
                kfs.append({
                    "label": "⑧③ 食小固定",
                    "joints": list(joints),
                    "hand": list(stab),
                    "plan": plan,
                })
            hand_after8 = list(stab)
        else:
            hand_after8 = list(pinch_close)

        world_lift, (lift_up, lift_right, lift_fwd) = ctrl.pick_lift_delta_world()
        lift_n = float(np.linalg.norm(world_lift))
        if lift_n > 1e-4:
            unit9 = world_lift / lift_n
            n9 = n_cart if detailed else int(np.clip(math.ceil(lift_n / 0.01), 2, 5))
            joints = self._append_world_translate_keep_rot(
                kfs, joints, hand_after8, unit9, lift_n,
                f"⑨ 提离 上{lift_up*1000:.0f}/右{lift_right*1000:.0f}/"
                f"前{lift_fwd*1000:.0f}mm（空间）",
                plan, n9, log=log,
            )
        else:
            log("模型预览：⑨ 提离量≈0，跳过")

        log(
            f"预览关键帧共 {len(kfs)} 个（Plan-once 回放："
            f"⑥×{len(wps6)}+⑦×{len(wps7)}+⑧⑨；无③④）"
        )
        return kfs, plan



