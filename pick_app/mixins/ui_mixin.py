"""UI build + parameter panel callbacks"""
from __future__ import annotations

from pick_app.deps import *
from pick_app.constants import *
from pick_app.helpers import resolve_basket_mode
from pick_app.mixins.handeye_mixin import _HANDEYE_SAMPLE_DELTAS


class UiMixin:
    def _model_ui_tick_ms(self):
        # 默认 50ms≈20fps；12ms 会把主线程拖死（PIL+MuJoCo）
        return max(int(self.config["eye_in_hand"].get("model_ui_tick_ms", 50)), 16)

    @staticmethod
    def _section_label(parent, text):
        ui.label(parent, text=text, size=FONT_SM, bold=True, muted=True).pack(
            anchor="w", pady=(6, 1)
        )

    def _mount_summary(self):
        t = np.round(self.controller.t_ee_cam, 3).tolist()
        prefer = bool(
            self.controller.eye.get("camera_on_ee_prefer_yaml", False)
        )
        tag = "手调" if prefer else "模型"
        mount = self.controller.eye.get("camera_on_ee", {})
        eul = mount.get("euler_rpy", [0.0, 0.0, 0.0])
        eul_deg = [round(math.degrees(float(x)), 1) for x in eul]
        fx = bool(mount.get("flip_x", False))
        fy = bool(mount.get("flip_y", False))
        return (
            f"t={t} [{tag}]  ·  euler°={eul_deg}  "
            f"flip_x={fx} flip_y={fy}  （已并入 R_ee_cam）"
        )

    def _extrinsic_summary(self):
        t = np.round(
            np.asarray(self.controller.t_ee_cam, dtype=np.float64), 4
        ).tolist()
        prefer = bool(
            self.controller.eye.get("camera_on_ee_prefer_yaml", False)
        )
        return f"当前外参 t={t}  ({'yaml手调' if prefer else '模型/yaml'})"

    def _refresh_extrinsic_labels(self):
        self.mount_var.set(self._mount_summary())
        if hasattr(self, "extrinsic_var"):
            self.extrinsic_var.set(self._extrinsic_summary())

    def _build_ui(self):
        # —— 窗口1：仅相机画面（不碰机械臂 API）——
        self.vision_win = ui.create_toplevel(
            self.root,
            title="相机画面（彩色 / 深度）",
            geometry="1040x560",
            minsize=(800, 420),
        )
        vis = ui.frame(self.vision_win)
        vis.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        vis.columnconfigure(0, weight=1)
        vis.columnconfigure(1, weight=1)
        vis.rowconfigure(0, weight=1)

        left_wrap = ui.frame(vis)
        left_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left_wrap.rowconfigure(0, weight=1)
        left_wrap.columnconfigure(0, weight=1)
        left_cam_card = ui.card(left_wrap, title="彩色 · 黄十字=光轴", padding=4)
        left_cam_card.pack(fill=tk.BOTH, expand=True)
        left_cam = ui.card_body(left_cam_card)
        self.color_canvas = ui.canvas(left_cam, PREVIEW_W, PREVIEW_H)
        self.color_canvas.pack(fill=tk.BOTH, expand=True)

        right_wrap = ui.frame(vis)
        right_wrap.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_wrap.rowconfigure(0, weight=1)
        right_wrap.columnconfigure(0, weight=1)
        right_cam_card = ui.card(right_wrap, title="深度", padding=4)
        right_cam_card.pack(fill=tk.BOTH, expand=True)
        right_cam = ui.card_body(right_cam_card)
        self.depth_canvas = ui.canvas(right_cam, PREVIEW_W, PREVIEW_H)
        self.depth_canvas.pack(fill=tk.BOTH, expand=True)

        self.vision_status = tk.StringVar(value="画面窗口：与控制窗口分离，运动时也应流畅")
        ui.label(vis, textvariable=self.vision_status, size=FONT_SM, wraplength=1000).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self.vision_win.protocol("WM_DELETE_WINDOW", self._on_vision_close)

        # —— 窗口2：控制 + 模型（可调用机械臂）——
        # grid：底栏固定高度，模型区吃剩余空间，避免底行被裁切
        self.root.title("LBOT 控制 / 模型 / 运动")
        self.root.geometry("1280x860")
        self.root.minsize(1000, 680)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        if getattr(self.args, "sim", False):
            mode = "仿真(假连接)"
        elif self.args.preview:
            mode = "预览"
        elif self.args.dry_run:
            mode = "Dry-run"
        else:
            mode = "实机"
        rot = float(self.controller.xy_rotate_deg)
        header = ui.status_header(
            self.root,
            "控制窗口 · 对准 / 关节 / 位姿 · 模型",
            f"模式: {mode}  轴向修正: {rot:.0f}°",
        )
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 2))

        top = ui.frame(self.root)
        top.grid(row=1, column=0, sticky="nsew", padx=6, pady=2)
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=2, minsize=320)
        top.columnconfigure(1, weight=5, minsize=560)

        self.info_card = ui.card(top, title="臂状态（右）", padding=4)
        self.info_card.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        info = ui.card_body(self.info_card)

        self._section_label(info, "视觉检测")
        self.det_var = tk.StringVar(value="等待检测…")
        ui.label(info, textvariable=self.det_var, size=FONT_MD, wraplength=500).pack(
            anchor="w"
        )

        self._section_label(info, "关节 (°)")
        self.joint_cell_vars = [tk.StringVar(value="—") for _ in range(7)]
        joint_row = ui.frame(info)
        joint_row.pack(fill=tk.X)
        for index, var in enumerate(self.joint_cell_vars, start=1):
            cell = ui.frame(joint_row)
            cell.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            ui.label(cell, text=f"J{index}", size=FONT_SM, muted=True).pack()
            ui.label(cell, textvariable=var, size=FONT_MD, bold=True).pack()

        self._section_label(info, "末端位姿")
        self.pose_xyz_var = tk.StringVar(value="xyz  —")
        self.pose_rpy_var = tk.StringVar(value="rpy  —")
        ui.label(info, textvariable=self.pose_xyz_var, size=FONT_MD).pack(anchor="w")
        ui.label(info, textvariable=self.pose_rpy_var, size=FONT_MD).pack(anchor="w")

        self._section_label(info, "相机 / 伺服")
        self.mount_var = tk.StringVar(value=self._mount_summary())
        ui.label(
            info, textvariable=self.mount_var, size=FONT_SM, muted=True, wraplength=500,
        ).pack(anchor="w")

        model_wrap = ui.frame(top)
        model_wrap.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        model_wrap.rowconfigure(0, weight=1)
        model_card = ui.card(model_wrap, title="机器人模型", padding=2)
        model_card.pack(fill=tk.BOTH, expand=True)
        model_frame = ui.card_body(model_card)
        self.model_canvas = ui.canvas(
            model_frame, MODEL_W, MODEL_H, cursor="fleur"
        )
        self.model_canvas.pack(fill=tk.BOTH, expand=True)
        self.model_canvas.bind("<Enter>", lambda _e: self.model_canvas.focus_set())
        self.model_canvas.bind("<ButtonPress-1>", self._on_model_press)
        self.model_canvas.bind("<B1-Motion>", self._on_model_drag)
        self.model_canvas.bind("<ButtonRelease-1>", self._on_model_release)
        self.model_canvas.bind("<Double-Button-1>", self._on_model_reset_view)
        self.model_canvas.bind("<ButtonPress-2>", self._on_model_press)
        self.model_canvas.bind("<B2-Motion>", self._on_model_drag)
        self.model_canvas.bind("<ButtonRelease-2>", self._on_model_release)
        self.model_canvas.bind("<ButtonPress-3>", self._on_model_press)
        self.model_canvas.bind("<B3-Motion>", self._on_model_drag)
        self.model_canvas.bind("<ButtonRelease-3>", self._on_model_release)
        self.model_canvas.bind("<MouseWheel>", self._on_model_wheel)
        self.model_canvas.bind("<Button-4>", self._on_model_wheel_up)
        self.model_canvas.bind("<Button-5>", self._on_model_wheel_down)

        # —— 顶栏操作（两行：主流程 / 外参·手眼）——
        bar_wrap = ui.frame(self.root)
        bar_wrap.grid(row=2, column=0, sticky="ew", padx=6, pady=2)

        def _sep(parent):
            ui.separator(parent, orient=tk.VERTICAL).pack(
                side=tk.LEFT, fill=tk.Y, padx=5, pady=2
            )

        def _grp(parent, title):
            f = ui.frame(parent)
            f.pack(side=tk.LEFT, padx=(0, 2))
            if title:
                ui.label(f, text=title, size=FONT_SM, muted=True).pack(
                    side=tk.LEFT, padx=(0, 4)
                )
            return f

        # 行1：对准 · 抓取 · 放置 · 手/预览
        row1 = ui.frame(bar_wrap)
        row1.pack(fill=tk.X, pady=(0, 2))

        g = _grp(row1, "对准")
        ui.button(
            g, "▶ 黄十字", style="warning", size=FONT_MD,
            command=self.on_align_center,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "停止对准", style="secondary", size=FONT_SM,
            command=self.on_stop_align,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "停止/解锁", style="danger", size=FONT_SM,
            command=self.on_stop_motion,
        ).pack(side=tk.LEFT, padx=1)

        _sep(row1)
        g = _grp(row1, "抓取")
        ui.button(
            g, "抓取(P)", style="accent", size=FONT_MD, command=self.on_pick,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "夹取(O)", style="accent", size=FONT_MD, command=self.on_pick_pinch,
        ).pack(side=tk.LEFT, padx=1)

        _sep(row1)
        g = _grp(row1, "放置")
        ui.button(
            g, "观察", style="warning", size=FONT_MD, command=self.on_place_observe,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "对准2", style="warning", size=FONT_MD, command=self.on_align_place,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "对准板", style="warning", size=FONT_SM, command=self.on_align_board,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "放下", style="success", size=FONT_MD, command=self.on_place_drop,
        ).pack(side=tk.LEFT, padx=1)

        _sep(row1)
        g = _grp(row1, "手/预览")
        ui.button(
            g, "模型预览", style="primary", size=FONT_SM,
            command=self.on_pinch_model_preview,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "清预览", style="secondary", size=FONT_SM,
            command=self.on_clear_model_preview,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "张开手", style="success", size=FONT_SM, command=self._ui_hand_open,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "闭合手", style="danger", size=FONT_SM, command=self._ui_hand_grasp,
        ).pack(side=tk.LEFT, padx=1)

        ui.button(
            row1, "退出", style="neutral", size=FONT_MD, command=self.close,
        ).pack(side=tk.RIGHT, padx=2)

        # 行2：外参微调 · 手眼标定
        row2 = ui.frame(bar_wrap)
        row2.pack(fill=tk.X)

        g = _grp(row2, "外参")
        ui.button(
            g, "←模型", style="primary", size=FONT_SM,
            command=self.on_restore_mount,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "验参", style="warning", size=FONT_SM,
            command=self.on_extrinsic_verify,
        ).pack(side=tk.LEFT, padx=1)

        _sep(row2)
        g = _grp(row2, "手眼")
        ui.checkbox(g, "标定模式", self.handeye_mode_var).pack(side=tk.LEFT, padx=2)
        self.handeye_mode_var.trace_add(
            "write", lambda *_: self.on_handeye_toggle()
        )
        ui.button(
            g, "记录姿态", style="accent", size=FONT_SM,
            command=self.on_handeye_record,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "清空", style="secondary", size=FONT_SM,
            command=self.on_handeye_clear,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "求解", style="warning", size=FONT_SM,
            command=self.on_handeye_solve,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            g, "写回外参", style="danger", size=FONT_SM,
            command=self.on_handeye_write_yaml,
        ).pack(side=tk.LEFT, padx=1)
        ui.label(
            row2, textvariable=self.handeye_status_var, size=FONT_SM, muted=True,
        ).pack(side=tk.LEFT, padx=(8, 2))

        # 手眼采样位（标定模式开启后显示）
        self.handeye_sample_bar = ui.frame(bar_wrap)
        he_ctrl = ui.frame(self.handeye_sample_bar)
        he_ctrl.pack(fill=tk.X)
        ui.label(
            he_ctrl,
            text="手眼采样位(保守·慢速≤0.06)",
            size=FONT_SM,
            muted=True,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ui.button(
            he_ctrl, "去下一位并记", style="accent", size=FONT_SM,
            command=lambda: self.on_handeye_next_sample(and_record=True),
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            he_ctrl, "自动采12组", style="warning", size=FONT_SM,
            command=self.on_handeye_auto_sequence,
        ).pack(side=tk.LEFT, padx=1)
        ui.button(
            he_ctrl, "停序列", style="danger", size=FONT_SM,
            command=self.on_handeye_stop_sequence,
        ).pack(side=tk.LEFT, padx=1)
        ui.label(
            he_ctrl,
            text="单击=去该位 · 右键=去并记 · 偏近会跳过",
            size=FONT_SM,
            muted=True,
        ).pack(side=tk.LEFT, padx=(8, 2))

        he_btns = ui.frame(self.handeye_sample_bar)
        he_btns.pack(fill=tk.X, pady=(2, 0))
        for i, (name, _dxyz, _drpy) in enumerate(_HANDEYE_SAMPLE_DELTAS):
            btn = ui.button(
                he_btns,
                name,
                style="secondary",
                size=FONT_SM,
                command=lambda idx=i: self.on_handeye_goto_sample(idx, and_record=False),
            )
            btn.pack(side=tk.LEFT, padx=1)
            try:
                btn.bind(
                    "<Button-3>",
                    lambda _e, idx=i: self.on_handeye_goto_sample(idx, and_record=True),
                )
            except Exception:
                pass
        # 默认隐藏；勾选「标定模式」后由 on_handeye_toggle pack

        # —— 底栏：参数 + 运动 + 日志（不 expand，保证整块可见）——
        bottom = ui.frame(self.root)
        bottom.grid(row=3, column=0, sticky="ew", padx=4, pady=(0, 2))

        params_card = ui.card(bottom, title="追踪 / 夹取 / 外参", padding=3)
        params_card.pack(fill=tk.X, padx=2, pady=(0, 2))
        params = ui.card_body(params_card)

        # 行1：追踪 + 倾角 + 提离
        track_bar = ui.frame(params)
        track_bar.pack(fill=tk.X, pady=(0, 1))
        ui.label(track_bar, text="追踪", size=FONT_SM, bold=True).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        if getattr(self, "_track_is_size", True):
            ui.checkbox(track_bar, "大", self.track_big).pack(side=tk.LEFT, padx=2)
            ui.checkbox(track_bar, "中", self.track_medium).pack(side=tk.LEFT, padx=2)
            ui.checkbox(track_bar, "小", self.track_small).pack(side=tk.LEFT, padx=2)
        else:
            ui.checkbox(track_bar, "red", self.track_red).pack(side=tk.LEFT, padx=2)
            ui.checkbox(track_bar, "green", self.track_green).pack(side=tk.LEFT, padx=2)
            ui.checkbox(track_bar, "blue", self.track_blue).pack(side=tk.LEFT, padx=2)
        self.show_vision_in_model = tk.BooleanVar(value=True)
        ui.checkbox(
            track_bar, "模型物体", self.show_vision_in_model,
        ).pack(side=tk.LEFT, padx=(6, 2))
        ui.separator(track_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ui.label(track_bar, text="放置检", size=FONT_SM, bold=True).pack(
            side=tk.LEFT, padx=(0, 2)
        )
        ui.checkbox(track_bar, "筐", self.detect_basket_var).pack(side=tk.LEFT, padx=2)
        ui.checkbox(track_bar, "盘", self.detect_plate_var).pack(side=tk.LEFT, padx=2)
        ui.checkbox(track_bar, "螺丝板", self.detect_screw_board_var).pack(
            side=tk.LEFT, padx=2
        )
        ui.checkbox(track_bar, "电批", self.detect_driver_var).pack(
            side=tk.LEFT, padx=2
        )
        ui.label(
            track_bar, text="(观察后/板·电批实时)", size=FONT_SM, muted=True,
        ).pack(side=tk.LEFT, padx=(0, 2))
        ui.label(track_bar, text="阈值", size=FONT_SM).pack(side=tk.LEFT, padx=(6, 2))
        ui.spinbox(track_bar, self.det_conf, 0.1, 0.95, 0.05, 4, "%.2f").pack(
            side=tk.LEFT
        )
        self.det_conf.trace_add("write", self._on_det_conf_change)
        tilt0 = float(
            self.config["eye_in_hand"].get("pinch_des_z_tilt_deg", 0.0)
        )
        self.pinch_z_tilt_deg = tk.DoubleVar(value=round(tilt0, 1))
        ui.label(track_bar, text="Z倾角°", size=FONT_SM).pack(side=tk.LEFT, padx=(8, 2))
        ui.spinbox(
            track_bar, self.pinch_z_tilt_deg, 0.0, 90.0, 5.0, 4, "%.0f"
        ).pack(side=tk.LEFT)
        self.pinch_z_tilt_deg.trace_add("write", self._on_pinch_z_tilt_change)

        eye_lift = self.config["eye_in_hand"]
        self.pinch_lift_up_cm = tk.DoubleVar(
            value=round(float(eye_lift.get("pinch_lift_up_m", 0.05)) * 100.0, 1)
        )
        self.pinch_lift_right_cm = tk.DoubleVar(
            value=round(float(eye_lift.get("pinch_lift_right_m", 0.0)) * 100.0, 1)
        )
        self.pinch_lift_forward_cm = tk.DoubleVar(
            value=round(float(eye_lift.get("pinch_lift_forward_m", 0.0)) * 100.0, 1)
        )
        ui.separator(track_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ui.label(track_bar, text="⑨提离cm", size=FONT_SM, bold=True).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        for label, var in (
            ("上", self.pinch_lift_up_cm),
            ("右", self.pinch_lift_right_cm),
            ("前", self.pinch_lift_forward_cm),
        ):
            ui.label(track_bar, text=label, size=FONT_SM).pack(side=tk.LEFT)
            ui.spinbox(track_bar, var, -20.0, 20.0, 0.5, 4, "%.1f").pack(
                side=tk.LEFT, padx=(1, 6)
            )
            var.trace_add("write", self._on_pinch_lift_change)

        # 行2：抓取侧偏(世界右/前) + ⑦补回（standoff×frac + extra）
        eye_pinch = self.config["eye_in_hand"]
        self.pinch_bias_right_cm = tk.DoubleVar(
            value=round(float(eye_pinch.get("pinch_grasp_bias_right_m", 0.0)) * 100.0, 1)
        )
        self.pinch_bias_forward_cm = tk.DoubleVar(
            value=round(
                float(eye_pinch.get("pinch_grasp_bias_forward_m", 0.0)) * 100.0, 1
            )
        )
        self.pinch_standoff_cm = tk.DoubleVar(
            value=round(float(eye_pinch.get("pinch_standoff_m", 0.100)) * 100.0, 1)
        )
        self.pinch_contact_frac = tk.DoubleVar(
            value=round(float(eye_pinch.get("pinch_contact_frac", 1.0)), 2)
        )
        self.pinch_contact_extra_cm = tk.DoubleVar(
            value=round(float(eye_pinch.get("pinch_contact_extra_m", 0.0)) * 100.0, 1)
        )
        self.pinch_contact_sum_var = tk.StringVar(value="")
        row_pinch = ui.frame(params)
        row_pinch.pack(fill=tk.X, pady=(1, 0))
        ui.label(row_pinch, text="抓偏cm", size=FONT_SM, bold=True).pack(
            side=tk.LEFT, padx=(0, 2)
        )
        ui.label(row_pinch, text="右", size=FONT_SM).pack(side=tk.LEFT)
        ui.spinbox(
            row_pinch, self.pinch_bias_right_cm, -10.0, 10.0, 0.5, 4, "%.1f"
        ).pack(side=tk.LEFT, padx=(0, 4))
        ui.label(row_pinch, text="前", size=FONT_SM).pack(side=tk.LEFT)
        ui.spinbox(
            row_pinch, self.pinch_bias_forward_cm, -10.0, 10.0, 0.5, 4, "%.1f"
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.pinch_bias_right_cm.trace_add("write", self._on_pinch_bias_change)
        self.pinch_bias_forward_cm.trace_add("write", self._on_pinch_bias_change)
        ui.separator(row_pinch, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ui.label(row_pinch, text="⑦补回", size=FONT_SM, bold=True).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ui.label(row_pinch, text="距cm", size=FONT_SM).pack(side=tk.LEFT)
        ui.spinbox(
            row_pinch, self.pinch_standoff_cm, 0.0, 20.0, 0.5, 4, "%.1f"
        ).pack(side=tk.LEFT, padx=(1, 6))
        ui.label(row_pinch, text="×系数", size=FONT_SM).pack(side=tk.LEFT)
        ui.spinbox(
            row_pinch, self.pinch_contact_frac, 0.0, 2.0, 0.05, 4, "%.2f"
        ).pack(side=tk.LEFT, padx=(1, 6))
        ui.label(row_pinch, text="+额外cm", size=FONT_SM).pack(side=tk.LEFT)
        ui.spinbox(
            row_pinch, self.pinch_contact_extra_cm, 0.0, 10.0, 0.5, 4, "%.1f"
        ).pack(side=tk.LEFT, padx=(1, 6))
        for var in (
            self.pinch_standoff_cm,
            self.pinch_contact_frac,
            self.pinch_contact_extra_cm,
        ):
            var.trace_add("write", self._on_pinch_contact_change)
        ui.label(
            row_pinch, textvariable=self.pinch_contact_sum_var, size=FONT_SM, muted=True,
        ).pack(side=tk.LEFT, padx=(2, 0))
        self._on_pinch_bias_change()
        self._on_pinch_contact_change()

        # 行3：外参校准（世界系右/前/上）
        self.calib_right_cm = tk.DoubleVar(value=0.0)
        self.calib_forward_cm = tk.DoubleVar(value=0.0)
        self.calib_up_cm = tk.DoubleVar(value=0.0)
        # 旧名别名：避免其它代码引用崩
        self.calib_tx_cm = self.calib_right_cm
        self.calib_ty_cm = self.calib_forward_cm
        self.calib_tz_cm = self.calib_up_cm
        self.extrinsic_var = tk.StringVar(value=self._extrinsic_summary())
        cam_off0 = list(
            self.config["eye_in_hand"].get("virtual_object_in_camera_m", [0.02, 0.05, 0.25])
        )
        while len(cam_off0) < 3:
            cam_off0.append(0.0)
        self.virtual_obj_right_cm = tk.DoubleVar(value=round(float(cam_off0[0]) * 100.0, 1))
        self.virtual_obj_down_cm = tk.DoubleVar(value=round(float(cam_off0[1]) * 100.0, 1))
        self.virtual_obj_forward_cm = tk.DoubleVar(value=round(float(cam_off0[2]) * 100.0, 1))

        row_ext = ui.frame(params)
        row_ext.pack(fill=tk.X, pady=(1, 0))
        ui.label(row_ext, text="外参cm(世界)", size=FONT_SM, bold=True).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        for label, var in (
            ("右", self.calib_right_cm),
            ("前", self.calib_forward_cm),
            ("上", self.calib_up_cm),
        ):
            ui.label(row_ext, text=label, size=FONT_SM).pack(side=tk.LEFT)
            ui.spinbox(row_ext, var, -10.0, 10.0, 0.1, 4, "%.1f").pack(
                side=tk.LEFT, padx=(1, 6)
            )
        ui.button(
            row_ext, "校准", style="accent", size=FONT_SM,
            command=self.on_extrinsic_calibrate,
        ).pack(side=tk.LEFT, padx=(2, 6))
        ui.label(
            row_ext, textvariable=self.extrinsic_var, size=FONT_SM, muted=True,
        ).pack(side=tk.LEFT, padx=(0, 4))

        # 行4：虚拟物体（单独一行，避免和外参挤在一起）
        row_virt = ui.frame(params)
        row_virt.pack(fill=tk.X, pady=(1, 0))
        ui.label(row_virt, text="虚拟cm", size=FONT_SM, bold=True).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        for label, var, lo, hi in (
            ("右", self.virtual_obj_right_cm, -30.0, 30.0),
            ("下", self.virtual_obj_down_cm, -30.0, 30.0),
            ("前", self.virtual_obj_forward_cm, 5.0, 80.0),
        ):
            ui.label(row_virt, text=label, size=FONT_SM).pack(side=tk.LEFT)
            ui.spinbox(row_virt, var, lo, hi, 0.5, 4, "%.1f").pack(
                side=tk.LEFT, padx=(1, 6)
            )
        ui.button(
            row_virt, "放置", style="warning", size=FONT_SM,
            command=self.on_place_virtual_object,
        ).pack(side=tk.LEFT, padx=(4, 2))
        ui.button(
            row_virt, "清除", style="secondary", size=FONT_SM,
            command=self.on_clear_virtual_object,
        ).pack(side=tk.LEFT, padx=2)

        self.move_card = ui.card(bottom, title="机械臂运动", padding=3)
        self.move_card.pack(fill=tk.X, padx=2, pady=2)
        move = ui.card_body(self.move_card)

        self.move_speed = tk.DoubleVar(value=float(self.controller.speed))
        self.move_accel = tk.DoubleVar(value=float(self.controller.accel))
        self.move_block = tk.BooleanVar(value=True)
        self.pose_linear = tk.BooleanVar(value=False)
        self.control_arm_var = tk.StringVar(value="right")

        arm_row = ui.frame(move)
        arm_row.pack(fill=tk.X, pady=(0, 2))
        ui.label(arm_row, text="控制臂", size=FONT_SM, bold=True).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        for text, val in (("右臂(相机)", "right"), ("左臂(批头)", "left")):
            ttk.Radiobutton(
                arm_row,
                text=text,
                value=val,
                variable=self.control_arm_var,
                command=self._on_control_arm_change,
            ).pack(side=tk.LEFT, padx=4)
        ui.label(
            arm_row,
            text="关节/位姿/手参数作用在所选臂",
            size=FONT_SM,
            muted=True,
        ).pack(side=tk.LEFT, padx=(8, 0))

        notebook = ui.tabview(move)
        notebook.pack(fill=tk.X)
        joint_tab = ui.add_tab(notebook, "关节控制")
        pose_tab = ui.add_tab(notebook, "位姿控制")
        hand_tab = ui.add_tab(notebook, "手参数")

        home0 = list(self.controller.default_home_joints)
        self.joint_vars = [tk.DoubleVar(value=round(float(j), 4)) for j in home0]
        self.pick_return_panel = tk.BooleanVar(value=True)
        joint_row = ui.frame(joint_tab)
        joint_row.pack(fill=tk.X)
        for i, var in enumerate(self.joint_vars, start=1):
            cell = ui.frame(joint_row)
            cell.pack(side=tk.LEFT, padx=2)
            ui.label(cell, text=f"J{i}", size=FONT_SM).pack()
            ui.spinbox(cell, var, -3.2, 3.2, 0.05, 5, "%.3f").pack()
        ui.button(joint_row, "置零", style="secondary", size=FONT_SM, command=self._zero_joint_entries).pack(
            side=tk.LEFT, padx=(6, 2)
        )
        ui.button(joint_row, "读取", style="secondary", size=FONT_SM, command=self._sync_joint_entries).pack(
            side=tk.LEFT, padx=2
        )
        ui.button(
            joint_row, "低观察位", style="secondary", size=FONT_SM,
            command=self._fill_default_home_joints,
        ).pack(side=tk.LEFT, padx=2)
        ui.button(
            joint_row, "高观察位", style="secondary", size=FONT_SM,
            command=self._fill_high_observe_pose,
        ).pack(side=tk.LEFT, padx=2)
        ui.button(
            joint_row, "看板位", style="secondary", size=FONT_SM,
            command=self._fill_screw_board_view_pose,
        ).pack(side=tk.LEFT, padx=2)
        ui.button(
            joint_row, "看批头位", style="secondary", size=FONT_SM,
            command=self._fill_screw_driver_view_pose,
        ).pack(side=tk.LEFT, padx=2)
        ui.button(
            joint_row, "拧螺丝位", style="secondary", size=FONT_SM,
            command=self._fill_screw_drive_pose_left,
        ).pack(side=tk.LEFT, padx=2)

        joint_footer = ui.frame(joint_tab)
        joint_footer.pack(fill=tk.X, pady=(4, 0))
        ui.checkbox(joint_footer, "抓取(P)后回上方关节", self.pick_return_panel).pack(
            side=tk.LEFT
        )
        self._pack_move_params(joint_footer)
        ui.button(
            joint_footer, "▶ 执行关节", style="success", size=FONT_SM,
            command=self._exec_joint_motion,
        ).pack(side=tk.RIGHT, padx=4)

        self.pose_xyz_vars = [tk.DoubleVar(value=v) for v in (0.3, -0.3, -0.3)]
        self.pose_rpy_vars = [
            tk.DoubleVar(value=0.0),
            tk.DoubleVar(value=-math.pi / 2),
            tk.DoubleVar(value=0.0),
        ]
        pose_row = ui.frame(pose_tab)
        pose_row.pack(fill=tk.X)
        for axis_label, var, lo, hi, step in (
            ("X(m)", self.pose_xyz_vars[0], -1.0, 1.0, 0.01),
            ("Y(m)", self.pose_xyz_vars[1], -1.0, 1.0, 0.01),
            ("Z(m)", self.pose_xyz_vars[2], -1.0, 1.0, 0.01),
            ("Roll", self.pose_rpy_vars[0], -3.2, 3.2, 0.05),
            ("Pitch", self.pose_rpy_vars[1], -3.2, 3.2, 0.05),
            ("Yaw", self.pose_rpy_vars[2], -3.2, 3.2, 0.05),
        ):
            cell = ui.frame(pose_row)
            cell.pack(side=tk.LEFT, padx=2)
            ui.label(cell, text=axis_label, size=FONT_SM).pack()
            ui.spinbox(cell, var, lo, hi, step, 6, "%.3f").pack()
        ui.button(pose_row, "读取", style="secondary", size=FONT_SM, command=self._sync_pose_entries).pack(
            side=tk.LEFT, padx=6
        )
        ui.button(pose_row, "演示位", style="secondary", size=FONT_SM, command=self._fill_demo_pose_entries).pack(
            side=tk.LEFT, padx=2
        )
        ui.button(
            pose_row, "高观察位", style="secondary", size=FONT_SM,
            command=self._fill_high_observe_pose,
        ).pack(side=tk.LEFT, padx=2)
        ui.button(
            pose_row, "看板位", style="secondary", size=FONT_SM,
            command=self._fill_screw_board_view_pose,
        ).pack(side=tk.LEFT, padx=2)
        ui.button(
            pose_row, "看批头位", style="secondary", size=FONT_SM,
            command=self._fill_screw_driver_view_pose,
        ).pack(side=tk.LEFT, padx=2)
        ui.button(
            pose_row, "拧螺丝位", style="secondary", size=FONT_SM,
            command=self._fill_screw_drive_pose_left,
        ).pack(side=tk.LEFT, padx=2)

        pose_footer = ui.frame(pose_tab)
        pose_footer.pack(fill=tk.X, pady=(4, 0))
        self._pack_move_params(pose_footer)
        ui.checkbox(pose_footer, "直线运动", self.pose_linear).pack(side=tk.LEFT, padx=8)
        ui.button(
            pose_footer, "▶ 执行位姿", style="primary", size=FONT_SM,
            command=self._exec_pose_motion,
        ).pack(side=tk.RIGHT, padx=4)

        grasp0 = list(self.controller.hand_presets.get("grasp", DEFAULT_GRASP_PRESET))
        while len(grasp0) < 6:
            grasp0.append(DEFAULT_GRASP_PRESET[len(grasp0) % 6])
        self.hand_grasp_vars = [
            tk.IntVar(value=int(np.clip(v, 0, 255))) for v in grasp0[:6]
        ]
        open0 = list(self.controller.hand_presets.get("open", [255] * 6))
        while len(open0) < 6:
            open0.append(255)
        self.hand_open_vars = [
            tk.IntVar(value=int(np.clip(v, 0, 255))) for v in open0[:6]
        ]
        ui.label(
            hand_tab,
            text="闭合 H1–H6（拇弯/横摆/食/中/无名/小）",
            size=FONT_SM,
        ).pack(anchor="w")
        grasp_row = ui.frame(hand_tab)
        grasp_row.pack(fill=tk.X, pady=(2, 2))
        for i, var in enumerate(self.hand_grasp_vars, start=1):
            cell = ui.frame(grasp_row)
            cell.pack(side=tk.LEFT, padx=2)
            ui.label(cell, text=f"H{i}", size=FONT_SM).pack()
            ui.spinbox(cell, var, 0, 255, 5, 4, "%d").pack()
        ui.button(grasp_row, "应用闭合", style="danger", size=FONT_SM, command=self._apply_hand_grasp_panel).pack(
            side=tk.LEFT, padx=6
        )
        ui.button(
            grasp_row, "复位", style="secondary", size=FONT_SM,
            command=lambda: self._fill_hand_grasp(list(DEFAULT_GRASP_PRESET)),
        ).pack(side=tk.LEFT, padx=2)

        left_hand_row = ui.frame(hand_tab)
        left_hand_row.pack(fill=tk.X, pady=(4, 2))
        ui.label(
            left_hand_row, text="左手任务", size=FONT_SM, bold=True,
        ).pack(side=tk.LEFT, padx=(0, 6))
        for text, key in (
            ("抓握", "grasp"),
            ("拆螺丝", "unscrew"),
            ("上螺丝", "screw_in"),
        ):
            ui.button(
                left_hand_row,
                text,
                style="accent",
                size=FONT_SM,
                command=lambda k=key, t=text: self._run_left_hand_preset(k, t),
            ).pack(side=tk.LEFT, padx=2)
        ui.label(
            left_hand_row,
            text="上/拆：当前位置直接拧（可选孔号仅备注）",
            size=FONT_SM,
            muted=True,
        ).pack(side=tk.LEFT, padx=(8, 0))

        hole_row = ui.frame(hand_tab)
        hole_row.pack(fill=tk.X, pady=(2, 2))
        ui.label(
            hole_row, text="拧孔号", size=FONT_SM, bold=True,
        ).pack(side=tk.LEFT, padx=(0, 6))
        self._screw_hole_btns = {}
        n_h = len(getattr(self, "screw_board_holes_cfg", None) or []) or 8
        for hid in range(n_h):
            btn = ui.button(
                hole_row,
                str(hid),
                style="secondary",
                size=FONT_SM,
                command=lambda i=hid: self._select_screw_hole(i),
            )
            btn.pack(side=tk.LEFT, padx=1)
            try:
                btn.configure(state="disabled")
            except tk.TclError:
                pass
            self._screw_hole_btns[hid] = btn
        self.screw_hole_sel_var = tk.StringVar(value="未选")
        ui.label(
            hole_row, textvariable=self.screw_hole_sel_var, size=FONT_SM, muted=True,
        ).pack(side=tk.LEFT, padx=(8, 0))

        bit_row = ui.frame(hand_tab)
        bit_row.pack(fill=tk.X, pady=(2, 2))
        ui.label(
            bit_row, text="批头示教", size=FONT_SM, bold=True,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ui.button(
            bit_row, "①记螺丝3D", style="secondary", size=FONT_SM,
            command=self.on_record_screw_for_bit_calib,
        ).pack(side=tk.LEFT, padx=2)
        ui.button(
            bit_row, "②标定批头", style="accent", size=FONT_SM,
            command=self.on_teach_bit_tip,
        ).pack(side=tk.LEFT, padx=2)
        ui.button(
            bit_row, "去拧选中孔", style="success", size=FONT_SM,
            command=self.on_go_selected_screw_hole,
        ).pack(side=tk.LEFT, padx=2)
        ui.label(
            bit_row, textvariable=self.bit_calib_status_var, size=FONT_SM, muted=True,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ui.label(
            hand_tab,
            text="流程：右臂看板→选有螺孔→①记螺丝3D→切左臂手动对准该孔→②标定批头→再选孔「去拧」",
            size=FONT_SM,
            muted=True,
        ).pack(anchor="w", pady=(0, 2))

        ui.label(hand_tab, text="张开 H1–H6", size=FONT_SM).pack(anchor="w")
        open_row = ui.frame(hand_tab)
        open_row.pack(fill=tk.X, pady=(2, 2))
        for i, var in enumerate(self.hand_open_vars, start=1):
            cell = ui.frame(open_row)
            cell.pack(side=tk.LEFT, padx=2)
            ui.label(cell, text=f"H{i}", size=FONT_SM).pack()
            ui.spinbox(cell, var, 0, 255, 5, 4, "%d").pack()
        ui.button(open_row, "应用张开", style="success", size=FONT_SM, command=self._apply_hand_open_panel).pack(
            side=tk.LEFT, padx=6
        )
        ui.button(
            open_row, "255", style="secondary", size=FONT_SM,
            command=lambda: self._fill_hand_open(list(DEFAULT_OPEN_PRESET)),
        ).pack(side=tk.LEFT, padx=2)

        self.frame_var = tk.StringVar(value="")
        self.log_var = tk.StringVar(value="就绪")
        log_row = ui.frame(bottom)
        log_row.pack(fill=tk.X, padx=2, pady=(2, 0))
        ui.log_bar(log_row, self.log_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ui.label(
            log_row, textvariable=self.frame_var, size=FONT_SM, muted=True,
        ).pack(side=tk.RIGHT, padx=(8, 2))

        ready_msg = "就绪"
        if self.args.dry_run:
            ready_msg = (
                "Dry-run：臂不动；可「模型预览夹取」+ 点动调参 | "
                + ready_msg
            )
        self.log_var.set(ready_msg)

        self.root.bind("p", lambda _e: self.on_pick())
        self.root.bind("o", lambda _e: self.on_pick_pinch())
        self.root.bind("a", lambda _e: self.on_align_center())
        self.root.bind("v", lambda _e: self.on_place_observe())
        self.root.bind("b", lambda _e: self.on_align_place())
        self.root.bind("d", lambda _e: self.on_place_drop())
        self.root.bind(
            "<Escape>",
            lambda _e: self.on_stop_align() if self.busy else self.close(),
        )
        self.root.after(300, self._sync_pose_entries)

    def _on_det_conf_change(self, *_args):
        try:
            value = float(self.det_conf.get())
            self.yolo.model.conf = max(0.05, min(0.99, value))
        except (ValueError, tk.TclError, AttributeError):
            pass

    def _on_pinch_z_tilt_change(self, *_args):
        """UI 捏取 Z 倾角 → eye_in_hand（夹取规划即时生效）。"""
        try:
            value = float(self.pinch_z_tilt_deg.get())
        except (ValueError, tk.TclError, AttributeError):
            return
        value = float(np.clip(value, 0.0, 90.0))
        eye = self.config["eye_in_hand"]
        eye["pinch_des_z_tilt_deg"] = value
        if getattr(self, "controller", None) is not None:
            self.controller.eye["pinch_des_z_tilt_deg"] = value

    def _sync_pinch_z_tilt_to_eye(self):
        """夹取/预览前强制同步一次（防止 trace 未触发）。"""
        self._on_pinch_z_tilt_change()
        self._on_pinch_lift_change()
        self._on_pinch_bias_change()
        self._on_pinch_contact_change()

    def _on_pinch_lift_change(self, *_args):
        """UI ⑨ 提离 上/右/前(cm) → eye（空间系，米）。"""
        if not hasattr(self, "pinch_lift_up_cm"):
            return
        try:
            up = float(self.pinch_lift_up_cm.get()) / 100.0
            right = float(self.pinch_lift_right_cm.get()) / 100.0
            forward = float(self.pinch_lift_forward_cm.get()) / 100.0
        except (ValueError, tk.TclError, AttributeError):
            return
        up = float(np.clip(up, -0.20, 0.20))
        right = float(np.clip(right, -0.20, 0.20))
        forward = float(np.clip(forward, -0.20, 0.20))
        eye = self.config["eye_in_hand"]
        eye["pinch_lift_up_m"] = up
        eye["pinch_lift_right_m"] = right
        eye["pinch_lift_forward_m"] = forward
        if getattr(self, "controller", None) is not None:
            self.controller.eye["pinch_lift_up_m"] = up
            self.controller.eye["pinch_lift_right_m"] = right
            self.controller.eye["pinch_lift_forward_m"] = forward

    def _on_pinch_bias_change(self, *_args):
        """UI 抓取侧偏(右/前, cm) → pinch_grasp_bias_*_m（世界系）。"""
        if not hasattr(self, "pinch_bias_right_cm"):
            return
        try:
            right_m = float(self.pinch_bias_right_cm.get()) / 100.0
            forward_m = (
                float(self.pinch_bias_forward_cm.get()) / 100.0
                if hasattr(self, "pinch_bias_forward_cm")
                else 0.0
            )
        except (ValueError, tk.TclError, AttributeError):
            return
        right_m = float(np.clip(right_m, -0.10, 0.10))
        forward_m = float(np.clip(forward_m, -0.10, 0.10))
        eye = self.config["eye_in_hand"]
        eye["pinch_grasp_bias_right_m"] = right_m
        eye["pinch_grasp_bias_forward_m"] = forward_m
        # by_size 优先于 base；UI 一改则三档同步，避免改面板不生效
        for key, val in (
            ("pinch_grasp_bias_right_by_size", right_m),
            ("pinch_grasp_bias_forward_by_size", forward_m),
        ):
            by = dict(eye.get(key) or {})
            for cls in ("small", "medium", "big"):
                by[cls] = val
            eye[key] = by
        if getattr(self, "controller", None) is not None:
            self.controller.eye["pinch_grasp_bias_right_m"] = right_m
            self.controller.eye["pinch_grasp_bias_forward_m"] = forward_m
            self.controller.eye["pinch_grasp_bias_right_by_size"] = dict(
                eye["pinch_grasp_bias_right_by_size"]
            )
            self.controller.eye["pinch_grasp_bias_forward_by_size"] = dict(
                eye["pinch_grasp_bias_forward_by_size"]
            )

    def _on_pinch_contact_change(self, *_args):
        """UI ⑦ 补回：standoff(cm) × frac + extra(cm)。"""
        if not hasattr(self, "pinch_standoff_cm"):
            return
        try:
            standoff_m = float(self.pinch_standoff_cm.get()) / 100.0
            frac = float(self.pinch_contact_frac.get())
            extra_m = float(self.pinch_contact_extra_cm.get()) / 100.0
        except (ValueError, tk.TclError, AttributeError):
            return
        standoff_m = float(np.clip(standoff_m, 0.0, 0.20))
        frac = float(np.clip(frac, 0.0, 2.0))
        extra_m = float(np.clip(extra_m, 0.0, 0.10))
        eye = self.config["eye_in_hand"]
        eye["pinch_standoff_m"] = standoff_m
        eye["pinch_contact_frac"] = frac
        eye["pinch_contact_extra_m"] = extra_m
        if getattr(self, "controller", None) is not None:
            self.controller.eye["pinch_standoff_m"] = standoff_m
            self.controller.eye["pinch_contact_frac"] = frac
            self.controller.eye["pinch_contact_extra_m"] = extra_m
        if hasattr(self, "pinch_contact_sum_var"):
            from lbot_grasp_utils import resolve_pinch_standoff_m
            size_info = getattr(self, "_locked_size_info", None) or getattr(
                self, "_last_size_info", None
            )
            so_eff = resolve_pinch_standoff_m(eye, size_info)
            total_cm = (so_eff * frac + extra_m) * 100.0
            cls = (size_info or {}).get("class") if isinstance(size_info, dict) else None
            tag = f"/{cls}" if cls else ""
            self.pinch_contact_sum_var.set(f"= {total_cm:.1f}cm{tag}")
    def _pack_move_params(self, parent):
        ui.label(parent, text="速度", size=FONT_SM).pack(side=tk.LEFT, padx=(8, 0))
        ui.spinbox(
            parent, self.move_speed, 0.05, 5.0, 0.05, 5, "%.2f"
        ).pack(side=tk.LEFT, padx=(4, 8))
        ui.label(parent, text="加速度", size=FONT_SM).pack(side=tk.LEFT)
        ui.spinbox(
            parent, self.move_accel, 0.05, 5.0, 0.05, 5, "%.2f"
        ).pack(side=tk.LEFT, padx=(4, 8))
        ui.checkbox(parent, "阻塞", self.move_block).pack(side=tk.LEFT, padx=4)

