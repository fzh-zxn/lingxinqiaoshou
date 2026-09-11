"""UI tick, hand/joint panels, action runner"""
from __future__ import annotations

from pick_app.deps import *
from pick_app.constants import *
from pick_app.helpers import *


class PanelMixin:
    def _show_on_canvas(self, canvas, photo):
        canvas.delete("all")
        if photo is None:
            return
        cw = max(canvas.winfo_width(), PREVIEW_W)
        ch = max(canvas.winfo_height(), PREVIEW_H)
        canvas.create_image(cw // 2, ch // 2, image=photo)

    def _update_pinch_ui(self):
        if not hasattr(self, "pinch_thumb_var"):
            return
        pinch = getattr(self, "_pinch_info", None)
        if pinch is None:
            self.pinch_thumb_var.set("—")
            self.pinch_middle_var.set("—")
            self.pinch_sep_var.set("—")
            self.pinch_thumb_dir_var.set("—")
            self.pinch_middle_dir_var.set("—")
            self.pinch_axis_dir_var.set("—")
            return
        tx, ty, tz = pinch["thumb_tip_m"]
        ix, iy, iz = pinch.get("index_tip_m", pinch["middle_tip_m"])
        mx, my, mz = pinch["middle_tip_m"]
        self.pinch_thumb_var.set(f"{tx:.3f}, {ty:.3f}, {tz:.3f}")
        if hasattr(self, "pinch_index_var"):
            self.pinch_index_var.set(f"{ix:.3f}, {iy:.3f}, {iz:.3f}")
        self.pinch_middle_var.set(f"{mx:.3f}, {my:.3f}, {mz:.3f}")
        self.pinch_sep_var.set(f"三指中心 · X跨度 {pinch['pinch_sep_mm']:.1f} mm")
        zx, zy, zz = pinch["z_approach"]
        xx, xy, xz = pinch["x_close"]
        self.pinch_thumb_dir_var.set(
            f"X闭合 [{xx:.3f}, {xy:.3f}, {xz:.3f}]"
        )
        self.pinch_middle_dir_var.set(
            f"Z接近 [{zx:.3f}, {zy:.3f}, {zz:.3f}]  (⊥连线)"
        )
        rpy = pinch.get("virtual_ee_rpy_deg", [])
        if len(rpy) == 3:
            self.pinch_axis_dir_var.set(
                f"虚拟末端 rpy {rpy[0]:.1f}°, {rpy[1]:.1f}°, {rpy[2]:.1f}°"
            )
        else:
            self.pinch_axis_dir_var.set("—")

    def _ui_tick(self):
        if not self.running:
            return
        self._tick_preview_animation()
        with self.lock:
            color = self.latest["color"]
            depth_view = self.latest["depth_view"]
            status = self.latest["status"]
            depth_mm = self.latest["depth_mm"]

        # 运动/对准时减半刷新深度窗，减轻主线程 PIL 压力
        ui_n = int(getattr(self, "_ui_tick_n", 0)) + 1
        self._ui_tick_n = ui_n
        light_ui = bool(self.busy) or bool(
            getattr(self.controller, "_aligning", False)
        ) or bool(getattr(self, "_place_aligning", False))
        # 画面刷新与机械臂 API 解耦：始终只画最新帧（随窗口画布等比缩放）
        if color is not None and self.color_canvas.winfo_exists():
            cw = max(int(self.color_canvas.winfo_width()), PREVIEW_W)
            ch = max(int(self.color_canvas.winfo_height()), PREVIEW_H)
            self.color_photo = bgr_to_photo(color, cw, ch)
            self._show_on_canvas(self.color_canvas, self.color_photo)
        if (
            depth_view is not None
            and self.depth_canvas.winfo_exists()
            and (not light_ui or (ui_n % 2 == 0))
        ):
            dw = max(int(self.depth_canvas.winfo_width()), PREVIEW_W)
            dh = max(int(self.depth_canvas.winfo_height()), PREVIEW_H)
            self.depth_photo = bgr_to_photo(depth_view, dw, dh)
            self._show_on_canvas(self.depth_canvas, self.depth_photo)

        det_text = status
        if depth_mm is not None:
            det_text += f"  ·  {depth_mm / 10:.1f} cm"
        if self.busy:
            det_text += "  ·  运动中"
        self.det_var.set(det_text)
        if hasattr(self, "vision_status"):
            self.vision_status.set(
                f"{status}"
                + (f" | 深度 {depth_mm/10:.1f}cm" if depth_mm is not None else "")
                + (" | 运动中" if self.busy else "")
            )
        # 孔号按钮：仅有螺母可选
        if (ui_n % 3) == 0:
            try:
                self._refresh_screw_hole_buttons()
            except Exception:
                pass

        # busy 时主线程不抢 TCP；用运动线程维护的 live_joints/pose 刷模型
        if not self.busy:
            joints = self.controller.get_joints()
            pose = self.controller.get_pose()
        else:
            joints = self.controller.get_live_joints()
            pose = self.controller.get_live_pose()
        # 实机运动中始终显示真机关节；仅空闲时才用模型预览关节
        model_preview_joints = (
            self._model_joints is not None and not self.busy
        )
        if model_preview_joints:
            joints = list(self._model_joints)
        self._last_ui_joints = list(joints)
        for index, value in enumerate(joints[:7]):
            self.joint_cell_vars[index].set(f"{math.degrees(value):.1f}")
        for index in range(len(joints), 7):
            self.joint_cell_vars[index].set("—")
        if pose is not None:
            position, euler = pose
            self.pose_xyz_var.set(
                f"xyz  {position.x:.3f}, {position.y:.3f}, {position.z:.3f} m"
            )
            rpy = unwrap_rpy_for_display(
                self._display_rpy, (euler.x, euler.y, euler.z)
            )
            self._display_rpy = rpy
            self.pose_rpy_var.set(
                f"rpy  {math.degrees(rpy[0]):.1f}°, "
                f"{math.degrees(rpy[1]):.1f}°, {math.degrees(rpy[2]):.1f}°"
            )
        else:
            self.pose_xyz_var.set("xyz  未连接机械臂")
            self.pose_rpy_var.set("rpy  —")
            self._display_rpy = None
        if not self.busy:
            self.frame_var.set(
                "角度原点: " + self.controller.frame_info()
                + "  |  rpy=工具相对工作系的姿态；xyz=tool0 在工作系中的位置"
            )

        photo = None
        # busy 时隔帧刷 MuJoCo，避免和相机预览抢主线程
        if self._drag_last is None and (not light_ui or (ui_n % 2 == 1)):
            photo = self._render_model(joints)
        self._update_pinch_ui()
        if photo is not None:
            self.model_photo = photo
            self.model_canvas.delete("all")
            cw, ch = self._model_view_dims()
            self.model_canvas.create_image(
                cw // 2, ch // 2, image=self.model_photo
            )

        tick_ms = self._model_ui_tick_ms()
        if light_ui:
            tick_ms = max(
                tick_ms,
                int(self.config["eye_in_hand"].get("busy_ui_tick_ms", 66)),
            )
        self.root.after(tick_ms, self._ui_tick)

    def _fill_hand_grasp(self, values):
        for var, value in zip(self.hand_grasp_vars, values):
            var.set(int(np.clip(value, 0, 255)))
        self._apply_hand_grasp_panel()

    def _run_left_hand_preset(self, key, label):
        """左手抓握 / 拆螺丝 / 上螺丝。"""
        if int(self.controller.arm) != int(LbotArm.LEFT_ARM):
            self.log(f"左手{label}：请先切到「左臂(批头)」")
            return
        hid = -1
        if key in ("screw_in", "unscrew"):
            try:
                hid = int(self.selected_screw_hole.get())
            except Exception:
                hid = -1
            if hid >= 0:
                self.log(f"左手{label}：孔 #{hid}（仅备注，不校验占用）")
            else:
                self.log(f"左手{label}：未选孔，直接拧（调试）")
        hp_left = ((self.config.get("hand_presets") or {}).get("left") or {})
        vals = hp_left.get(key)
        if vals is None or len(list(vals)) < 6:
            self.log(f"左手{label}：配置缺失 hand_presets.left.{key}")
            return
        vals = [int(np.clip(int(v), 0, 255)) for v in list(vals)[:6]]
        self._release_model_hand_follow()

        # 上/拆螺丝：禁止把拧螺丝手型写入「抓握」面板/预设（否则再点抓握会误按扳机）
        if key in ("screw_in", "unscrew"):
            robot_cfg = self.config.get("robot") or {}
            duration_s = float(robot_cfg.get("screw_drive_hold_s", 2.5))
            dz_mm = float(robot_cfg.get("screw_drive_z_mm", 6.0))
            z_move_s = float(robot_cfg.get("screw_drive_z_move_s", 0.45))
            pre_press_mm = float(robot_cfg.get("screw_drive_pre_press_mm", 2.0))
            dz_m = (-abs(dz_mm) if key == "screw_in" else abs(dz_mm)) / 1000.0
            restore = hp_left.get("grasp")
            if restore is None or len(list(restore)) < 6:
                restore = [255, 0, 100, 100, 45, 50]
            restore = [int(np.clip(int(v), 0, 255)) for v in list(restore)[:6]]
            z_when = "before" if key == "screw_in" else "after"
            pre_press_m = (pre_press_mm / 1000.0) if key == "unscrew" else 0.0
            hole_tag = f"孔{hid} " if hid >= 0 else ""
            if key == "screw_in":
                phase = f"{hole_tag}先Z{dz_m*1000:.0f}mm后拧"
            else:
                phase = f"{hole_tag}下压{pre_press_mm:.0f}→拧→上移{dz_mm:.0f}mm"
            drive_label = f"左手{label}" + (f"#{hid}" if hid >= 0 else "")

            def _do_drive():
                ok = self.controller.execute_screw_hand_with_z(
                    vals,
                    dz_m,
                    duration_s=duration_s,
                    log=self.log,
                    should_stop=lambda: bool(
                        getattr(self.controller, "_motion_abort", False)
                    ),
                    label=drive_label,
                    restore_hand=restore,
                    z_when=z_when,
                    z_move_s=z_move_s,
                    pre_press_m=pre_press_m,
                )
                # 恢复面板抓握显示，避免残留拧螺丝数值
                try:
                    self.root.after(0, lambda: self._fill_hand_grasp(restore))
                except Exception:
                    pass
                return ok

            self._run_action(
                f"左手{label} {phase}/{duration_s:.1f}s",
                _do_drive,
            )
            return

        # 抓握：始终用配置里的 grasp，并写回面板
        self._fill_hand_grasp(vals)
        self._run_action(
            f"左手{label} {vals}",
            lambda: self.controller.hand_grasp(),
        )

    def _select_screw_hole(self, hole_id):
        """UI 点选拧螺丝孔号（仅有螺母的孔可点）。"""
        hid = int(hole_id)
        with self.lock:
            occ = list(self.latest.get("screw_board_occupied_ids") or [])
        if hid not in occ:
            self.log(f"孔{hid}无螺母或未检出，不可选")
            return
        self.selected_screw_hole.set(hid)
        # 日志带上精修像素，便于核对
        xy_r = None
        with self.lock:
            holes = self.latest.get("screw_board_holes") or []
        for h in holes:
            if int(h.get("id", -1)) == hid:
                xy_r = h.get("xy_refined") or h.get("xy")
                break
        if hasattr(self, "screw_hole_sel_var"):
            if xy_r is not None:
                self.screw_hole_sel_var.set(
                    f"#{hid} @({xy_r[0]:.0f},{xy_r[1]:.0f})"
                )
            else:
                self.screw_hole_sel_var.set(f"已选 #{hid}")
        if xy_r is not None:
            self.log(f"已选拧孔 #{hid} 精修=({xy_r[0]:.1f},{xy_r[1]:.1f})")
        else:
            self.log(f"已选拧孔 #{hid}")

    def _sample_selected_screw_uvd(self):
        """选中有螺孔的精修像素 + 板面深度。"""
        try:
            hid = int(self.selected_screw_hole.get())
        except Exception:
            hid = -1
        if hid < 0:
            return None, "未选孔号"
        with self.lock:
            holes = list(self.latest.get("screw_board_holes") or [])
            depth = self.latest.get("screw_board_depth_mm")
            occ = list(self.latest.get("screw_board_occupied_ids") or [])
        if hid not in occ:
            return None, f"孔{hid}当前无螺/未检出"
        hit = None
        for h in holes:
            if int(h.get("id", -1)) == hid:
                hit = h
                break
        if hit is None:
            return None, f"孔{hid}无检测数据"
        xy = hit.get("xy_refined") or hit.get("xy")
        if xy is None or len(xy) < 2:
            return None, f"孔{hid}无像素"
        if depth is None or not np.isfinite(float(depth)):
            return None, "无板面深度（请右臂看板检出板）"
        return (
            {
                "id": hid,
                "u": float(xy[0]),
                "v": float(xy[1]),
                "depth_mm": float(depth),
            },
            None,
        )

    def on_record_screw_for_bit_calib(self):
        """① 用右相机把当前选中螺丝投到工作系，供批头示教。"""
        if self.controller.robot is None:
            self.log("记螺丝失败：未连接机械臂")
            return
        sample, err = self._sample_selected_screw_uvd()
        if sample is None:
            self.log(f"记螺丝失败：{err}")
            return
        # 相机在右臂：必须用右臂位姿投点（即使当前控左臂）
        pt = self.controller.object_point_in_world_arm(
            LbotArm.RIGHT_ARM,
            sample["u"], sample["v"], sample["depth_mm"],
        )
        if pt is None or not np.all(np.isfinite(pt)):
            self.log("记螺丝失败：右臂位姿/外参投点失败（请右臂看板位）")
            return
        self._taught_screw_world = np.asarray(pt, dtype=np.float64).reshape(3)
        self._taught_screw_hole_id = int(sample["id"])
        self._taught_screw_uvd = (
            sample["u"], sample["v"], sample["depth_mm"],
        )
        p = self._taught_screw_world
        self.log(
            f"已记螺丝#{sample['id']} UV=({sample['u']:.0f},{sample['v']:.0f}) "
            f"d={sample['depth_mm']/10:.1f}cm → 世界"
            f"[{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}]"
            f"（下一步：切左臂，手动对准该孔后点「②标定批头」）"
        )
        if hasattr(self, "bit_calib_status_var"):
            self.bit_calib_status_var.set(
                f"已记孔#{sample['id']} 待左臂对准标定"
            )

    def on_teach_bit_tip(self):
        """② 左臂已对准记过的螺丝 → 反算批头在法兰系，写入 yaml。"""
        if self.controller.robot is None:
            self.log("标定批头失败：未连接机械臂")
            return
        if self._taught_screw_world is None:
            self.log("标定批头失败：请先「①记螺丝3D」")
            return
        if int(self.controller.arm) != int(LbotArm.LEFT_ARM):
            self.log("标定批头：请先切到「左臂(批头)」再点标定")
            return
        pose = self.controller.get_pose_for_arm(LbotArm.LEFT_ARM)
        if pose is None:
            self.log("标定批头失败：读不到左臂位姿")
            return
        pos, eul = pose
        ee_xyz = [float(pos.x), float(pos.y), float(pos.z)]
        ee_rpy = [float(eul.x), float(eul.y), float(eul.z)]
        tip = bit_tip_in_ee_from_teach(
            self._taught_screw_world, ee_xyz, ee_rpy,
        )
        tip_n = float(np.linalg.norm(tip))
        if tip_n > 0.45:
            self.log(
                f"标定批头警告：|tip_ee|={tip_n*1000:.0f}mm 过大，"
                f"请确认记点与对准是同一颗螺丝"
            )
        cfg_path = getattr(self.controller, "config_path", None) or getattr(
            self, "config_path", None
        )
        if cfg_path is None and getattr(self, "args", None) is not None:
            cfg_path = getattr(self.args, "config", None)
        if cfg_path is None:
            cfg_path = Path(__file__).resolve().parents[2] / "grasp_config.yaml"
        try:
            save_bit_tip_calib(
                cfg_path,
                tip,
                rpy=ee_rpy,
                hole_id=self._taught_screw_hole_id,
                screw_world=self._taught_screw_world,
            )
        except Exception as exc:
            self.log(f"标定批头写 yaml 失败: {exc}")
            return
        robot = self.config.setdefault("robot", {})
        robot["bit_tip_in_ee"] = [float(tip[0]), float(tip[1]), float(tip[2])]
        robot["bit_calib_rpy"] = list(ee_rpy)
        if self._taught_screw_hole_id is not None:
            robot["bit_calib_hole_id"] = int(self._taught_screw_hole_id)
        msg = (
            f"批头已标定 tip_ee="
            f"[{tip[0]*1000:.1f},{tip[1]*1000:.1f},{tip[2]*1000:.1f}]mm "
            f"|tip|={tip_n*1000:.0f}mm → 已写入 yaml"
        )
        self.log(msg)
        if hasattr(self, "bit_calib_status_var"):
            self.bit_calib_status_var.set(
                f"批头已标定 tip_ee="
                f"[{tip[0]*1000:.1f},{tip[1]*1000:.1f},{tip[2]*1000:.1f}]mm"
            )

    def on_go_selected_screw_hole(self):
        """用已标定批头，把左臂法兰移到选中孔对应拧位（姿态用标定 rpy）。"""
        if self.busy:
            self.log("忙，忽略去拧选中孔")
            return
        if self.controller.robot is None:
            self.log("去拧失败：未连接机械臂")
            return
        loaded = load_bit_tip_from_config(self.config)
        if loaded is None:
            self.log("去拧失败：尚未标定批头（先①②）")
            return
        tip_ee, rpy_calib = loaded
        sample, err = self._sample_selected_screw_uvd()
        if sample is None:
            self.log(f"去拧失败：{err}")
            return
        screw_w = self.controller.object_point_in_world_arm(
            LbotArm.RIGHT_ARM,
            sample["u"], sample["v"], sample["depth_mm"],
        )
        if screw_w is None:
            self.log("去拧失败：螺丝投点失败（右臂位姿？）")
            return
        if rpy_calib is None:
            # 回退拧螺丝示教姿态
            pose_cfg = (self.config.get("robot") or {}).get("screw_drive_pose_left") or {}
            rpy = pose_cfg.get("rpy")
            if rpy is None or len(rpy) < 3:
                self.log("去拧失败：无 bit_calib_rpy / screw_drive_pose_left.rpy")
                return
            rpy_use = [float(rpy[0]), float(rpy[1]), float(rpy[2])]
        else:
            rpy_use = [float(rpy_calib[0]), float(rpy_calib[1]), float(rpy_calib[2])]
        xyz = ee_xyz_for_screw(screw_w, rpy_use, tip_ee)
        # 切到左臂再动
        if int(self.controller.arm) != int(LbotArm.LEFT_ARM):
            if hasattr(self, "control_arm_var"):
                self.control_arm_var.set("left")
                self._on_control_arm_change()
            else:
                self.controller.set_control_arm(LbotArm.LEFT_ARM, log=self.log)

        # 填面板便于核对
        if hasattr(self, "pose_xyz_vars"):
            for var, value in zip(self.pose_xyz_vars, xyz):
                var.set(round(float(value), 4))
            for var, value in zip(self.pose_rpy_vars, rpy_use):
                var.set(round(float(value), 4))

        speed = float(self.move_speed.get()) if hasattr(self, "move_speed") else 0.05
        accel = float(self.move_accel.get()) if hasattr(self, "move_accel") else 0.12
        # 左臂强制慢速（控制器内还会再 cap）
        robot_cfg = self.config.get("robot") or {}
        speed = min(speed, float(robot_cfg.get("left_speed_max", 0.05)))
        accel = min(accel, float(robot_cfg.get("left_accel_max", 0.12)))
        hid = int(sample["id"])
        self.log(
            f"▶ 去拧孔#{hid} 目标 xyz="
            f"[{xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f}] "
            f"speed≤{speed:.3f} "
            f"（螺丝世界[{screw_w[0]:.4f},{screw_w[1]:.4f},{screw_w[2]:.4f}]）"
        )

        def work():
            return self.controller.execute_pose_motion_safe(
                [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                rpy_use,
                speed=speed,
                accel=accel,
                log=self.log,
            )

        self._run_action(f"去拧孔#{hid}", work)

    def _refresh_screw_hole_buttons(self):
        """根据最新孔占用刷新按钮可用状态。"""
        btns = getattr(self, "_screw_hole_btns", None) or {}
        if not btns:
            return
        with self.lock:
            occ = set(self.latest.get("screw_board_occupied_ids") or [])
        try:
            sel = int(self.selected_screw_hole.get())
        except Exception:
            sel = -1
        if sel >= 0 and sel not in occ:
            self.selected_screw_hole.set(-1)
            sel = -1
            if hasattr(self, "screw_hole_sel_var"):
                self.screw_hole_sel_var.set("未选")
        for hid, btn in btns.items():
            enable = hid in occ
            try:
                btn.configure(state=("normal" if enable else "disabled"))
            except tk.TclError:
                pass
        if hasattr(self, "screw_hole_sel_var") and sel >= 0:
            self.screw_hole_sel_var.set(f"已选 #{sel}")
        elif hasattr(self, "screw_hole_sel_var") and not occ:
            self.screw_hole_sel_var.set("无占用孔")

    def _fill_hand_open(self, values):
        for var, value in zip(self.hand_open_vars, values):
            var.set(int(np.clip(value, 0, 255)))
        self._apply_hand_open_panel()

    def _apply_hand_grasp_panel(self):
        vals = []
        for var in self.hand_grasp_vars:
            try:
                vals.append(int(np.clip(int(var.get()), 0, 255)))
            except (tk.TclError, ValueError, TypeError):
                vals.append(50)
        self.controller.hand_presets["grasp"] = vals
        self.controller.hand_presets["pinch_close"] = list(vals)
        self.log(f"闭合/捏合参数已应用: {vals}")

    def _apply_hand_open_panel(self):
        vals = []
        for var in self.hand_open_vars:
            try:
                vals.append(int(np.clip(int(var.get()), 0, 255)))
            except (tk.TclError, ValueError, TypeError):
                vals.append(255)
        self.controller.hand_presets["open"] = vals
        self.controller.hand_presets["pinch_open"] = list(vals)
        self.log(f"张开/预抓参数已应用: {vals}")

    def _release_model_hand_follow(self):
        """实机开合时取消模型手指令覆盖，让 MuJoCo 跟随 hand_cmd。"""
        if self._model_hand_cmd is not None:
            self._model_hand_cmd = None

    def _ui_hand_open(self):
        self._apply_hand_open_panel()
        self._release_model_hand_follow()
        self._run_action("张开", lambda: self.controller.hand_open(force=True))

    def _ui_hand_grasp(self):
        self._apply_hand_grasp_panel()
        self._release_model_hand_follow()

        def _do_grasp():
            ok = self.controller.hand_grasp()
            # power5：面板「闭合」也走⑧b，避免只合三指误以为坏了
            eye = self.config.get("eye_in_hand") or {}
            if ok and pinch_grasp_mode(eye=eye) == "power5":
                size_info = (
                    getattr(self, "_locked_size_info", None)
                    or getattr(self.controller, "grasp_size_info", None)
                )
                stab = grasp_stabilize_preset(
                    self.controller, size_info=size_info, eye=eye,
                )
                if stab is not None:
                    self.controller.hand_presets["pinch_stabilize"] = list(stab)
                    ok_b = self.controller.hand_pinch_stabilize(stab, force=True)
                    self.log(
                        f"面板闭合⑧b 食小固定 L6={list(stab)[:6]}: "
                        f"{'成功' if ok_b else '失败'}"
                    )
                    ok = bool(ok and ok_b)
            return ok

        self._run_action("闭合", _do_grasp)

    def _zero_joint_entries(self):
        for var in self.joint_vars:
            var.set(0.0)

    def _fill_default_home_joints(self):
        # 左臂：示教笛卡尔默认位；右臂：低观察关节
        if int(self.controller.arm) == int(LbotArm.LEFT_ARM):
            robot = self.config.get("robot") or {}
            pose = robot.get("default_home_pose_left")
            if pose is not None:
                self._fill_pose_dict(pose, "左臂默认位")
                return
            self.log("左臂默认位：配置缺失 default_home_pose_left")
            return
        for var, value in zip(
            self.joint_vars, self.controller.default_home_joints
        ):
            var.set(round(float(value), 4))
        self.log(
            "已填低观察位(关节): "
            + ", ".join(
                f"{math.degrees(float(v.get())):.1f}°" for v in self.joint_vars
            )
        )

    def _panel_home_joints(self):
        """界面 J1–J7（弧度）作为抓取回位目标。"""
        joints = []
        for var in self.joint_vars:
            try:
                joints.append(float(var.get()))
            except (tk.TclError, ValueError, TypeError):
                return None
        if len(joints) != 7 or not all(np.isfinite(j) for j in joints):
            return None
        return joints

    def _sync_joint_entries(self):
        joints = self.controller.get_joints()
        for var, value in zip(self.joint_vars, joints):
            var.set(round(float(value), 4))
        self.log(
            "已读取当前关节到面板: "
            + ", ".join(f"{math.degrees(j):.1f}°" for j in joints)
        )

    def _sync_pose_entries(self):
        pose = self.controller.get_pose()
        if pose is None:
            return
        position, euler = pose
        for var, value in zip(
            self.pose_xyz_vars, (position.x, position.y, position.z)
        ):
            var.set(round(float(value), 4))
        for var, value in zip(
            self.pose_rpy_vars, (euler.x, euler.y, euler.z)
        ):
            var.set(round(float(value), 4))

    def _fill_demo_pose_entries(self):
        for var, value in zip(self.pose_xyz_vars, (0.3, -0.3, -0.3)):
            var.set(value)
        for var, value in zip(self.pose_rpy_vars, (0.0, -math.pi / 2, 0.0)):
            var.set(value)

    def _fill_pose_dict(self, pose_cfg, label):
        if not isinstance(pose_cfg, dict):
            self.log(f"{label}：配置缺失")
            return False
        xyz = pose_cfg.get("xyz")
        rpy = pose_cfg.get("rpy")
        if xyz is None or rpy is None or len(xyz) < 3 or len(rpy) < 3:
            self.log(f"{label}：xyz/rpy 不完整")
            return False
        if not hasattr(self, "pose_xyz_vars") or not hasattr(self, "pose_rpy_vars"):
            self.log(f"{label}：位姿面板未就绪")
            return False
        xyz_f = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        rpy_f = [float(rpy[0]), float(rpy[1]), float(rpy[2])]
        for var, value in zip(self.pose_xyz_vars, xyz_f):
            var.set(round(value, 4))
        for var, value in zip(self.pose_rpy_vars, rpy_f):
            var.set(round(value, 4))
        self.log(
            f"已填{label}位姿 xyz=[{xyz_f[0]:.4f},{xyz_f[1]:.4f},{xyz_f[2]:.4f}] "
            f"rpy=[{rpy_f[0]:.4f},{rpy_f[1]:.4f},{rpy_f[2]:.4f}]"
        )
        # 关节页点这些按钮后常接着「执行关节」：同步 IK 到关节框，避免仍走低观察位
        joints = None
        if self.controller.robot is not None:
            try:
                joints = self.controller.ik_to_pose(
                    xyz_f, rpy_f, max_jump_rad=3.5,
                )
            except Exception as exc:
                self.log(f"{label} IK 异常: {exc}")
                joints = None
        if joints is not None and len(joints) >= 7:
            for var, value in zip(self.joint_vars, list(joints)[:7]):
                var.set(round(float(value), 4))
            self.log(
                f"已同步{label}关节(IK): "
                + ", ".join(
                    f"{math.degrees(float(v.get())):.1f}°" for v in self.joint_vars
                )
            )
        else:
            self.log(f"{label}：请到「位姿控制」点▶执行位姿（关节 IK 未解出）")
        return True

    def _fill_screw_board_view_pose(self):
        robot = self.config.get("robot") or {}
        self._fill_pose_dict(robot.get("screw_board_view_pose"), "看板位")

    def _fill_screw_driver_view_pose(self):
        robot = self.config.get("robot") or {}
        self._fill_pose_dict(robot.get("screw_driver_view_pose"), "看批头位")

    def _fill_screw_drive_pose_left(self):
        """左臂拧螺丝示教位。"""
        robot = self.config.get("robot") or {}
        if int(self.controller.arm) != int(LbotArm.LEFT_ARM):
            self.log("拧螺丝位：建议先切到「左臂(批头)」")
        self._fill_pose_dict(robot.get("screw_drive_pose_left"), "拧螺丝位")

    def _fill_high_observe_pose(self):
        """放筐/盘时的高观察位（eye_in_hand.place_view_pose）。"""
        eye = self.config.get("eye_in_hand") or {}
        self._fill_pose_dict(eye.get("place_view_pose"), "高观察位")

    def _on_control_arm_change(self):
        # 写回当前臂开/合参数，但不得整表覆盖（会丢掉 unscrew/screw_in），
        # 也不得把临时拧螺丝手型当成 grasp 持久化。
        old_key = (
            "left" if self.controller.arm == LbotArm.LEFT_ARM else "right"
        )
        hp_all = self.controller.config.setdefault("hand_presets", {})
        if not isinstance(hp_all, dict):
            hp_all = {}
            self.controller.config["hand_presets"] = hp_all
        side_cfg = hp_all.setdefault(old_key, {})
        if not isinstance(side_cfg, dict):
            side_cfg = {}
            hp_all[old_key] = side_cfg
        if hasattr(self, "hand_grasp_vars") and hasattr(self, "hand_open_vars"):
            try:
                grasp = [
                    int(np.clip(int(v.get()), 0, 255)) for v in self.hand_grasp_vars
                ]
                open0 = [
                    int(np.clip(int(v.get()), 0, 255)) for v in self.hand_open_vars
                ]
                yaml_grasp = side_cfg.get("grasp")
                if (
                    old_key == "left"
                    and yaml_grasp is not None
                    and len(list(yaml_grasp)) >= 6
                ):
                    yg = [int(v) for v in list(yaml_grasp)[:6]]
                    # 面板若已是拧螺丝扳机姿，拒绝污染 grasp
                    if (grasp[2] < yg[2] - 20) or (grasp[3] < yg[3] - 20):
                        grasp = yg
                self.controller.hand_presets["grasp"] = list(grasp)
                self.controller.hand_presets["pinch_close"] = list(grasp)
                self.controller.hand_presets["open"] = list(open0)
                self.controller.hand_presets["pinch_open"] = list(open0)
                side_cfg["grasp"] = list(grasp)
                side_cfg["pinch_close"] = list(grasp)
                side_cfg["open"] = list(open0)
                side_cfg["pinch_open"] = list(open0)
            except (tk.TclError, ValueError, TypeError):
                pass

        which = str(self.control_arm_var.get() or "right").strip().lower()
        arm = LbotArm.LEFT_ARM if which == "left" else LbotArm.RIGHT_ARM
        self.controller.set_control_arm(arm, log=self.log)
        # 切臂后模型预览跟随新臂实机关节，勿沿用旧臂的预览覆盖
        self._model_joints = None
        side = "左" if which == "left" else "右"
        if hasattr(self, "info_card"):
            try:
                self.info_card.configure(text=f"臂状态（{side}）")
            except tk.TclError:
                pass
        self._fill_default_home_joints()
        hp = self.controller.hand_presets or {}
        grasp = list(hp.get("grasp", DEFAULT_GRASP_PRESET))
        while len(grasp) < 6:
            grasp.append(DEFAULT_GRASP_PRESET[len(grasp) % 6])
        open0 = list(hp.get("open", [255] * 6))
        while len(open0) < 6:
            open0.append(255)
        if hasattr(self, "hand_grasp_vars"):
            for var, value in zip(self.hand_grasp_vars, grasp[:6]):
                var.set(int(np.clip(value, 0, 255)))
        if hasattr(self, "hand_open_vars"):
            for var, value in zip(self.hand_open_vars, open0[:6]):
                var.set(int(np.clip(value, 0, 255)))
        if self.controller.robot is not None:
            try:
                self._sync_joint_entries()
                self._sync_pose_entries()
            except Exception as exc:
                self.log(f"切换臂后读取状态失败: {exc}")

    def _exec_joint_motion(self):
        joints = [float(var.get()) for var in self.joint_vars]
        # 立刻刷新模型，避免「点了没反应」的观感
        self._model_joints = list(joints)
        speed = float(self.move_speed.get())
        accel = float(self.move_accel.get())
        if self.controller.robot is None:
            self.log(f"未连臂：仅更新模型预览 J={['%.3f' % j for j in joints]}")
            return

        def do_move():
            # 左右臂面板关节均走安全路径（抬高→上方→下降），防蹭桌
            ok = self.controller.execute_joint_motion_safe(
                joints,
                speed=speed,
                accel=accel,
                log=self.log,
                force_direct=False,
            )
            # 到位才改回跟随实机；超时/中止保留指令姿态，便于对照符号
            if ok:
                self._model_joints = None
            return ok

        self._run_action(
            f"关节运动 {['%.3f' % j for j in joints]}",
            do_move,
        )

    def _exec_pose_motion(self):
        if self.controller.robot is None:
            self.log("位姿运动失败：未连接机械臂")
            return
        xyz = [float(var.get()) for var in self.pose_xyz_vars]
        rpy = [float(var.get()) for var in self.pose_rpy_vars]
        speed = float(self.move_speed.get())
        accel = float(self.move_accel.get())
        # 一律走抬高→横移→下降，避免直线/PTP 蹭桌；勾选「直线运动」仅作末段微调偏好
        self._run_action(
            f"安全位姿 xyz={xyz} rpy={rpy}",
            lambda: self.controller.execute_pose_motion_safe(
                xyz, rpy, speed=speed, accel=accel, log=self.log,
            ),
        )

