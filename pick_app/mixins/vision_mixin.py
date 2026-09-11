"""Sensors, vision loop, basket detect"""
from __future__ import annotations

from pick_app.deps import *
from pick_app.constants import *
from pick_app.helpers import bgr_to_photo, resolve_basket_mode


class VisionMixin:
    def _start_sensors(self):
        if not self.args.no_depth:
            color_topic = getattr(self.args, "color_topic", None) or "/camera/color/image_raw"
            (
                self.depth_launch,
                self.depth_relay,
                self.depth_file,
                self.color_file,
                self._cam_log_path,
            ) = start_depth_processes(self.args.depth_topic, color_topic=color_topic)
            self._cam_wait_t0 = time.time()
            self._cam_wait_last_log = 0.0
            reused = self.depth_launch is None
            self.log(
                f"深度相机启动中: {self.args.depth_topic}"
                + ("（复用已有 Orbbec 话题，未再 launch）" if reused else "")
            )
            self.log(
                "等待 Orbbec 彩色首帧（勿同时开 demo_camera）…"
                + (f" 日志 {self._cam_log_path}" if self._cam_log_path else "")
            )
        else:
            self.camera = cv2.VideoCapture(
                camera_source(self.args.camera), cv2.CAP_V4L2
            )
            if not self.camera.isOpened():
                if self.args.preview or getattr(self.args, "sim", False):
                    self.log(f"未打开摄像头 ({self.args.camera})：仅模型可用")
                    self.camera = None
                    return
                raise RuntimeError(f"无法打开摄像头: {self.args.camera}")
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)

    @staticmethod
    def _safe_imread(path, flags=cv2.IMREAD_COLOR):
        """文件未就绪时不调 imread，避免 OpenCV 刷屏 WARN。"""
        if path is None:
            return None
        try:
            if not path.is_file() or path.stat().st_size < 100:
                return None
        except OSError:
            return None
        return cv2.imread(str(path), flags)

    def _read_frames(self):
        if self.color_file is not None:
            # ROS 进程挂了就别空转
            if self.depth_launch is not None and self.depth_launch.poll() is not None:
                if not self._cam_fail_logged:
                    self._cam_fail_logged = True
                    self.log(
                        "Orbbec launch 已退出（可能被占用）。"
                        "请关掉 demo_camera 等程序后重启。"
                    )
                return None, None
            if self.depth_relay is not None and self.depth_relay.poll() is not None:
                if not self._cam_fail_logged:
                    self._cam_fail_logged = True
                    self.log("ros_depth_relay 已退出，检查 ROS2 / orbbec_camera")
                return None, None

            frame = self._safe_imread(self.color_file, cv2.IMREAD_COLOR)
            if frame is None:
                now = time.time()
                if self._cam_wait_t0 is None:
                    self._cam_wait_t0 = now
                waited = now - float(self._cam_wait_t0)
                if not self._cam_wait_logged:
                    self._cam_wait_logged = True
                    with self.lock:
                        self.latest["status"] = "等待相机画面…"
                # 每 5s 刷一次进度，避免「卡住却无声」
                if now - float(self._cam_wait_last_log or 0.0) >= 5.0:
                    self._cam_wait_last_log = now
                    launch_s = (
                        "无launch(复用)"
                        if self.depth_launch is None
                        else (
                            "launch运行中"
                            if self.depth_launch.poll() is None
                            else f"launch已退code={self.depth_launch.returncode}"
                        )
                    )
                    relay_s = (
                        "relay运行中"
                        if self.depth_relay is not None
                        and self.depth_relay.poll() is None
                        else (
                            f"relay已退code={getattr(self.depth_relay, 'returncode', '?')}"
                            if self.depth_relay is not None
                            else "无relay"
                        )
                    )
                    tip = ""
                    if waited >= 20.0:
                        tip = (
                            "；仍无图：查是否占用设备，或看 "
                            f"{self._cam_log_path or '/tmp/lbot_cam_*.log'}"
                        )
                    self.log(
                        f"仍在等彩色首帧 {waited:.0f}s（{launch_s}；{relay_s}）{tip}"
                    )
                return None, None
            if self._cam_wait_logged:
                self._cam_wait_logged = False
                waited = (
                    time.time() - float(self._cam_wait_t0)
                    if self._cam_wait_t0 is not None
                    else 0.0
                )
                self.log(f"相机彩色画面已就绪（等了 {waited:.1f}s）")
                self._cam_wait_t0 = None
            depth = None
            if self.depth_file is not None:
                depth = self._safe_imread(self.depth_file, cv2.IMREAD_UNCHANGED)
                if depth is not None and depth.ndim == 3:
                    depth = depth[:, :, 0]
            return frame, depth
        if self.camera is None:
            return None, None
        ok, frame = self.camera.read()
        return (frame, None) if ok else (None, None)

    def _track_colors(self):
        """当前勾选的追踪类别集合（螺母尺寸或颜色）。"""
        colors = set()
        if getattr(self, "track_big", None) is not None and self.track_big.get():
            colors.add("big")
        if getattr(self, "track_medium", None) is not None and self.track_medium.get():
            colors.add("medium")
        if getattr(self, "track_small", None) is not None and self.track_small.get():
            colors.add("small")
        if self.track_red.get():
            colors.add("red")
        if self.track_green.get():
            colors.add("green")
        if self.track_blue.get():
            colors.add("blue")
        return colors

    def _after_observe_place_vision(self):
        """已到观察位（或对准2/放下阶段）才允许跑筐/盘检测。"""
        return getattr(self, "_pipeline_phase", "idle") in (
            "observed", "place_aligned", "placed",
        )

    def _want_basket_detect(self):
        has_basket = bool(getattr(self, "basket_rgbd", False)) or (
            self.yolo_basket is not None
        )
        return (
            has_basket
            and bool(self.detect_basket_var.get())
            and self._after_observe_place_vision()
        )

    def _want_plate_detect(self):
        # 对准2 已锁盘心：不再跑盘 YOLO/深度（持物螺母易污染框与深度）
        if (
            getattr(self, "_place_locked_kind", None) == "plate"
            and getattr(self, "_locked_place_xyz", None) is not None
        ):
            return False
        return (
            self.yolo_place is not None
            and bool(self.detect_plate_var.get())
            and self._after_observe_place_vision()
        )

    def _want_screw_board_detect(self):
        cfg = getattr(self, "screw_board_cfg", None) or {}
        if not bool(cfg.get("enabled", True)):
            return False
        return bool(self.detect_screw_board_var.get())

    def _screw_board_detect_kwargs(self):
        cfg = getattr(self, "screw_board_cfg", None) or {}
        hsv_low = cfg.get("hsv_low", [0, 0, 90])
        hsv_high = cfg.get("hsv_high", [180, 70, 255])
        return dict(
            min_area_frac=float(cfg.get("min_area_frac", 0.02)),
            max_area_frac=float(cfg.get("max_area_frac", 0.65)),
            aspect_min=float(cfg.get("aspect_min", 1.15)),
            aspect_max=float(cfg.get("aspect_max", 3.8)),
            min_rect_fill=float(cfg.get("min_rect_fill", 0.55)),
            min_depth_mm=float(cfg.get("min_depth_mm", 80.0)),
            max_depth_mm=float(cfg.get("max_depth_mm", 900.0)),
            hsv_low=tuple(hsv_low),
            hsv_high=tuple(hsv_high),
            dark_v_max=float(cfg.get("dark_v_max", 95)),
            dark_s_max=float(cfg.get("dark_s_max", 90)),
            depth_band_mm=float(cfg.get("depth_band_mm", 40.0)),
            prefer_depth_plane=bool(cfg.get("prefer_depth_plane", False)),
            prefer_rgb=bool(cfg.get("prefer_rgb", True)),
            max_border_touch=float(cfg.get("max_border_touch", 0.08)),
        )

    def _detect_screw_board_scaled(self, frame, depth_aligned):
        """降采样跑螺丝板检测，再缩回原图坐标。"""
        h, w = frame.shape[:2]
        max_w = max(160, int(getattr(self, "screw_board_proc_width", 640)))
        scale = 1.0
        small = frame
        dsmall = depth_aligned
        if w > max_w:
            scale = max_w / float(w)
            nh = max(1, int(round(h * scale)))
            small = cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_AREA)
            if depth_aligned is not None:
                dsmall = cv2.resize(
                    depth_aligned, (max_w, nh), interpolation=cv2.INTER_NEAREST,
                )
        hit = find_screw_board_rgbd(
            small, dsmall, **self._screw_board_detect_kwargs(),
        )
        if hit is None or scale == 1.0:
            return hit
        inv = 1.0 / scale
        hit.corners = np.asarray(hit.corners, dtype=np.float32) * inv
        hit.center_uv = (float(hit.center_uv[0]) * inv, float(hit.center_uv[1]) * inv)
        x1, y1, x2, y2 = hit.box_xyxy
        hit.box_xyxy = (
            int(round(x1 * inv)),
            int(round(y1 * inv)),
            int(round(x2 * inv)),
            int(round(y2 * inv)),
        )
        hit.size_wh = (float(hit.size_wh[0]) * inv, float(hit.size_wh[1]) * inv)
        hit.area_px = float(hit.area_px) * (inv * inv)
        return hit

    def _screw_board_refine_kwargs(self):
        cfg = getattr(self, "screw_board_cfg", None) or {}
        return dict(
            pad_frac=float(cfg.get("refine_pad_frac", 0.15)),
            max_gray=float(cfg.get("black_max_gray", 75)),
            min_area_frac_roi=float(cfg.get("refine_min_area_frac_roi", 0.08)),
            min_dark_coverage=float(cfg.get("refine_min_dark_coverage", 0.24)),
            min_edge_support=float(cfg.get("refine_min_edge_support", 0.10)),
            aspect_min=float(cfg.get("aspect_min", 1.15)),
            aspect_max=float(cfg.get("aspect_max", 3.8)),
            min_rect_fill=float(cfg.get("refine_min_rect_fill", 0.34)),
            min_depth_mm=float(cfg.get("min_depth_mm", 80.0)),
            max_depth_mm=float(cfg.get("max_depth_mm", 900.0)),
            occlusion_hold_frames=int(cfg.get("occlusion_hold_frames", 24)),
        )

    def _screw_board_from_yolo(
        self, frame, last_board_raw, last_board_names, depth_aligned,
    ):
        """YOLO 最高分框 → 可选 ROI RGB 精修旋转框。"""
        if last_board_raw is None or len(last_board_raw) < 1:
            return None
        det = max(list(last_board_raw), key=lambda d: float(d[4]))
        cls = detection_class_name(last_board_names, det[5]) or "板"
        hit = screw_board_hit_from_yolo(
            det, depth_mm=depth_aligned, reason=f"yolo:{cls}",
        )
        if hit is None:
            return None
        cfg = getattr(self, "screw_board_cfg", None) or {}
        if not bool(cfg.get("yolo_rgb_refine", True)) or frame is None:
            return hit
        try:
            refined = refine_screw_board_from_yolo_box(
                frame,
                hit.box_xyxy,
                depth_mm=depth_aligned,
                yolo_conf=float(det[4]),
                state=getattr(self, "_screw_board_refine_state", None),
                reason=f"yolo+rgb:{cls}",
                **self._screw_board_refine_kwargs(),
            )
        except Exception as exc:
            if not getattr(self, "_screw_board_refine_err_logged", False):
                self._screw_board_refine_err_logged = True
                self.log(f"螺丝板 RGB 精修失败（之后静默）: {exc}")
                print(f"[screw-board-refine] {exc}", flush=True)
            refined = None
        return refined if refined is not None else hit

    def _detect_basket_rgbd_scaled(self, frame, depth_aligned):
        """降采样跑筐检测，再把框缩回原图坐标（全分辨率约 240ms，640 宽约 60ms）。"""
        h, w = frame.shape[:2]
        max_w = max(160, int(self.basket_proc_width))
        scale = 1.0
        small = frame
        dsmall = depth_aligned
        if w > max_w:
            scale = max_w / float(w)
            nh = max(1, int(round(h * scale)))
            small = cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_AREA)
            if depth_aligned is not None:
                dsmall = cv2.resize(
                    depth_aligned, (max_w, nh), interpolation=cv2.INTER_NEAREST,
                )
        hit, _mask, _cands = find_blue_basket_rgbd(
            small, dsmall, **self.basket_rgbd_kwargs,
        )
        if hit is None or scale == 1.0:
            return hit
        inv = 1.0 / scale
        hit.corners = np.asarray(hit.corners, dtype=np.float32) * inv
        hit.center_uv = (float(hit.center_uv[0]) * inv, float(hit.center_uv[1]) * inv)
        x1, y1, x2, y2 = hit.box_xyxy
        hit.box_xyxy = (
            int(round(x1 * inv)),
            int(round(y1 * inv)),
            int(round(x2 * inv)),
            int(round(y2 * inv)),
        )
        hit.size_wh = (float(hit.size_wh[0]) * inv, float(hit.size_wh[1]) * inv)
        hit.area_px = float(hit.area_px) * (inv * inv)
        return hit

    def _vision_loop(self):
        last_raw = None
        last_names = {}
        last_yolo_t = 0.0
        last_place_raw = None
        last_place_names = {}
        last_basket_raw = None
        last_basket_names = {}
        last_driver_raw = None
        last_driver_names = {}
        last_board_raw = None
        last_board_names = {}
        last_perf_log = 0.0
        while self.running:
            frame, depth = self._read_frames()
            if frame is None:
                time.sleep(0.02)
                continue

            loop_t0 = time.perf_counter()
            # 对准阶段：检测与伺服同频，避免「等 YOLO 帧才动一下」的顿挫
            aligning = bool(getattr(self.controller, "_aligning", False))
            place_align = bool(getattr(self, "_place_aligning", False))
            # 对准时不要每帧硬跑 YOLO（画面会跟着卡死）；给视觉留余量
            eye_cfg = self.config["eye_in_hand"]
            if place_align:
                # 对准2：盘 YOLO+精修更重，默认比螺母对准更稀
                yolo_period = float(
                    eye_cfg.get(
                        "place_align_yolo_period_s",
                        eye_cfg.get("align_yolo_period_s", 0.12),
                    )
                )
            elif aligning:
                yolo_period = float(eye_cfg.get("align_yolo_period_s", 0.06))
            elif self.busy:
                yolo_period = float(eye_cfg.get("busy_yolo_period_s", 0.20))
            else:
                yolo_period = float(eye_cfg.get("idle_yolo_period_s", 0.08))
            now = time.time()
            want_plate = self._want_plate_detect()
            want_basket = self._want_basket_detect()
            want_driver = (
                self.yolo_driver is not None
                and bool(self.detect_driver_var.get())
            )
            want_screw = self._want_screw_board_detect()
            want_board_yolo = (
                want_screw
                and self.yolo_board is not None
                and bool((getattr(self, "screw_board_cfg", None) or {}).get(
                    "prefer_yolo", True,
                ))
            )
            # 夹取对准时不跑盘 YOLO/圆心；对准2 时不跑螺母 YOLO（省算力）
            run_place_yolo = want_plate and (place_align or not aligning)
            run_nut_yolo = not place_align
            yolo_refreshed = False
            if now - last_yolo_t >= yolo_period:
                if run_nut_yolo:
                    results = self.yolo.predict([frame], size=self.args.imgsz)
                    last_raw = results.pred[0].detach().cpu().numpy()
                    last_names = self.yolo.model.names
                if run_place_yolo:
                    try:
                        pr = self.yolo_place.predict([frame], size=self.args.imgsz)
                        last_place_raw = pr.pred[0].detach().cpu().numpy()
                        last_place_names = self.yolo_place.model.names
                    except Exception as place_exc:
                        if not getattr(self, "_place_pred_err_logged", False):
                            self._place_pred_err_logged = True
                            self.log(f"放置 YOLO 推理失败（之后静默）: {place_exc}")
                            print(f"[place] predict error: {place_exc}", flush=True)
                elif not want_plate:
                    last_place_raw = None
                if want_driver:
                    try:
                        dr = self.yolo_driver.predict(
                            [frame], size=self.args.imgsz,
                        )
                        last_driver_raw = dr.pred[0].detach().cpu().numpy()
                        last_driver_names = self.yolo_driver.model.names
                    except Exception as driver_exc:
                        if not getattr(self, "_driver_pred_err_logged", False):
                            self._driver_pred_err_logged = True
                            self.log(f"电批 YOLO 推理失败（之后静默）: {driver_exc}")
                            print(f"[driver] predict error: {driver_exc}", flush=True)
                elif not want_driver:
                    last_driver_raw = None
                if want_board_yolo:
                    try:
                        brd = self.yolo_board.predict(
                            [frame], size=self.args.imgsz,
                        )
                        last_board_raw = brd.pred[0].detach().cpu().numpy()
                        last_board_names = self.yolo_board.model.names
                    except Exception as board_exc:
                        if not getattr(self, "_board_pred_err_logged", False):
                            self._board_pred_err_logged = True
                            self.log(f"螺丝板 YOLO 推理失败（之后静默）: {board_exc}")
                            print(f"[board] predict error: {board_exc}", flush=True)
                elif not want_screw:
                    last_board_raw = None
                if self.yolo_basket is not None and want_basket and not (
                    place_align and want_plate and bool(
                        eye_cfg.get("place_prefer_plate", True)
                    )
                ):
                    try:
                        br = self.yolo_basket.predict([frame], size=self.args.imgsz)
                        last_basket_raw = br.pred[0].detach().cpu().numpy()
                        last_basket_names = self.yolo_basket.model.names
                    except Exception as basket_exc:
                        if not getattr(self, "_basket_pred_err_logged", False):
                            self._basket_pred_err_logged = True
                            self.log(f"筐子 YOLO 推理失败（之后静默）: {basket_exc}")
                            print(f"[basket] predict error: {basket_exc}", flush=True)
                last_yolo_t = now
                yolo_refreshed = True
                self.controller._vision_det_seq = (
                    int(getattr(self.controller, "_vision_det_seq", 0)) + 1
                )

            names = last_names
            allowed = self._track_colors()
            det_cfg = self.config.get("detection") or {}
            size_metric_on = bool(
                (det_cfg.get("size_metric") or {}).get("enabled", False)
            )
            eye_cfg = self.config["eye_in_hand"]
            # 对准2：不用旧螺母框（本阶段不跑 nut YOLO）
            pick_raw = None if place_align else last_raw
            # 深度定档：候选框可能比「仅 YOLO 勾选」更宽，后面再按尺寸过滤
            prior_for_pick = self._depth_tracker.mm
            if prior_for_pick is None:
                prior_for_pick = self._last_reliable_depth_mm
            # 只对齐深度（不做伪彩）；伪彩后面做一次即可
            h_pre, w_pre = frame.shape[:2]
            depth_aligned_pre = align_depth_to_size(depth, w_pre, h_pre)
            if size_metric_on and depth_aligned_pre is not None:
                self.controller.intrinsics = scale_camera_intrinsics(
                    self.controller.intrinsics_base, w_pre, h_pre
                )
                detection, size_info_pre = best_detection_with_size_metric(
                    pick_raw,
                    names,
                    allowed,
                    depth_aligned_pre,
                    self.controller.intrinsics,
                    det_cfg,
                    eye_cfg,
                    prior_mm=prior_for_pick,
                )
                filtered = filter_detections_by_colors(
                    pick_raw, names, list(allowed) if allowed else []
                )
                if size_metric_on and bool(
                    (det_cfg.get("size_metric") or {}).get("yolo_all_then_metric", True)
                ):
                    # 画面上画出所有螺母框（single.pt=nut / best.pt=大中小）
                    filtered = filter_nut_detections(pick_raw, names) or filtered
            else:
                filtered = filter_detections_by_colors(pick_raw, names, allowed)
                # nut-only：勾选尺寸无法匹配 YOLO 类名，改用全部螺母框
                if not filtered:
                    filtered = filter_nut_detections(pick_raw, names)
                detection = best_detection(
                    pick_raw, names=names, allowed_colors=allowed
                )
                if detection is None and filtered:
                    detection = max(filtered, key=lambda d: float(d[4]))
                size_info_pre = None

            # YOLO 短丢框：沿用上一帧 nut 框（避免对准「短暂丢框」）
            # 深度定档开启时：过滤为空不要沿用旧框（可能是其它尺寸）
            pick_mismatch = False
            if detection is not None:
                self._det_hold_box = [float(x) for x in detection[:4]]
                self._det_hold_n = 0
            elif size_metric_on:
                self._det_hold_box = None
                self._det_hold_n = 0
            elif (
                self._det_hold_box is not None
                and self._det_hold_n < self._det_hold_max
            ):
                self._det_hold_n += 1
                hb = self._det_hold_box
                detection = [hb[0], hb[1], hb[2], hb[3], 0.45, 0]
            else:
                self._det_hold_box = None
                self._det_hold_n = 0

            annotated = frame.copy()

            h, w = annotated.shape[:2]
            # 复用已对齐深度；伪彩只做一次（原先会双重 resize+colormap，很卡）
            if (
                depth_aligned_pre is not None
                and depth_aligned_pre.shape[1] == w
                and depth_aligned_pre.shape[0] == h
            ):
                depth_aligned = depth_aligned_pre
            else:
                depth_aligned = align_depth_to_size(depth, w, h)
            depth_view, _ = depth_colormap_view(depth_aligned, w, h)
            self.controller.intrinsics = scale_camera_intrinsics(
                self.controller.intrinsics_base, w, h
            )
            cx = float(self.controller.intrinsics["cx"])
            cy = float(self.controller.intrinsics["cy"])
            if not getattr(self, "_intrinsics_logged", False):
                self._intrinsics_logged = True
                kin = self.controller.intrinsics
                ref = self.controller.intrinsics_base
                rw = ref.get("width") or ref.get("image_width") or w
                rh = ref.get("height") or ref.get("image_height") or h
                self.log(
                    f"相机内参: 画面 {w}×{h} yaml参考 {rw}×{rh} → "
                    f"fx={kin['fx']:.1f} fy={kin['fy']:.1f} "
                    f"cx={kin['cx']:.1f} cy={kin['cy']:.1f}"
                )

            status = "未检测到目标"
            u = v = depth_mm = None
            size_info = size_info_pre
            stable = False
            allowed = self._track_colors()
            pick_mismatch = False

            # 放置目标（盘 YOLO）：仅观察后且 UI 勾选「盘」
            # 圆心精修只跟 YOLO 同频；帧间刷新深度（勿冻深度），禁止每帧轮廓
            place_det = None
            place_u = place_v = place_depth_mm = None
            place_cls = None
            place_circle = None
            plate_frozen = (
                getattr(self, "_place_locked_kind", None) == "plate"
                and self._locked_place_uvd is not None
            )
            # 本会话长期冻结（第二颗观察后、未再对准2）：画面标 freeze
            # 对准2 进行中必须实时 YOLO，勿用旧冻结 UV
            plate_reuse = (
                not plate_frozen
                and not getattr(self, "_place_aligning", False)
                and getattr(self, "_plate_frozen_uvd", None) is not None
                and bool(self.detect_plate_var.get())
                and self._after_observe_place_vision()
            )

            def _plate_depth_at_uv(det_box, gu, gv):
                if depth_aligned is None or gu is None or gv is None:
                    return None
                try:
                    _, _, dmm = box_center_and_depth(
                        depth_aligned,
                        det_box,
                        prior_mm=None,
                        freeze_below_mm=0.0,
                        grasp_u=float(gu),
                        grasp_v=float(gv),
                        roi_mode=str(
                            eye_cfg.get("place_depth_roi_mode", "foreground")
                            or "foreground"
                        ),
                        outlier_mm=float(eye_cfg.get("depth_outlier_mm", 20.0)),
                        eye=eye_cfg,
                        return_meta=False,
                    )
                    return float(dmm) if dmm is not None and np.isfinite(dmm) else None
                except Exception:
                    return None

            # 对准2 已锁盘心：只画锁点，不再 YOLO/采深（持物螺母会污染）
            if plate_frozen:
                lu, lv, ld = self._locked_place_uvd[:3]
                place_u, place_v = float(lu), float(lv)
                place_depth_mm = float(ld)
                place_cls = "plate"
                cv2.drawMarker(
                    annotated,
                    (int(round(place_u)), int(round(place_v))),
                    (0, 255, 180), cv2.MARKER_TILTED_CROSS, 18, 2,
                )
                cv2.putText(
                    annotated,
                    "plate lock",
                    (int(round(place_u)) + 8, int(round(place_v)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 180),
                    1,
                    cv2.LINE_AA,
                )
            elif plate_reuse:
                fu, fv, fd = self._plate_frozen_uvd[:3]
                place_u, place_v = float(fu), float(fv)
                place_depth_mm = float(fd)
                place_cls = "plate"
                cv2.drawMarker(
                    annotated,
                    (int(round(place_u)), int(round(place_v))),
                    (0, 220, 255), cv2.MARKER_TILTED_CROSS, 18, 2,
                )
                cv2.putText(
                    annotated,
                    "plate freeze",
                    (int(round(place_u)) + 8, int(round(place_v)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
            elif want_plate and last_place_raw is not None and run_place_yolo:
                place_det = best_place_detection(last_place_raw, last_place_names)
                if place_det is not None:
                    place_cls = (
                        logical_color_from_detection(last_place_names, place_det[5])
                        or detection_class_name(last_place_names, place_det[5])
                    )
                    cache = getattr(self, "_place_circle_cache", None)
                    box_p = place_det[:4]
                    if yolo_refreshed or cache is None:
                        place_circle = refine_plate_disk_center(
                            frame, box_p, eye=eye_cfg,
                        )
                        if place_circle is not None:
                            place_u = float(place_circle["u"])
                            place_v = float(place_circle["v"])
                        else:
                            place_u = float((box_p[0] + box_p[2]) * 0.5)
                            place_v = float((box_p[1] + box_p[3]) * 0.5)
                        # 深度必须在精修 UV 上采，勿用框心深度再改 UV
                        place_depth_mm = _plate_depth_at_uv(box_p, place_u, place_v)
                        self._place_circle_cache = {
                            "circle": place_circle,
                            "u": place_u,
                            "v": place_v,
                            "depth_mm": place_depth_mm,
                            "cls": place_cls,
                            "det": place_det,
                        }
                    else:
                        place_circle = cache.get("circle")
                        place_u = cache.get("u")
                        place_v = cache.get("v")
                        place_cls = cache.get("cls") or place_cls
                        cached_det = cache.get("det")
                        if cached_det is not None:
                            place_det = cached_det
                        # 帧间：UV/圆心可缓存，深度每帧重采（臂在动）
                        box_c = (
                            place_det[:4] if place_det is not None else box_p
                        )
                        place_depth_mm = _plate_depth_at_uv(
                            box_c, place_u, place_v,
                        )
                        if place_depth_mm is None:
                            place_depth_mm = cache.get("depth_mm")
                        else:
                            cache["depth_mm"] = place_depth_mm
                    draw_boxes_p = bool(det_cfg.get("draw_yolo_boxes", True))
                    if draw_boxes_p:
                        draw_detections(
                            annotated, [place_det], last_place_names,
                            depth_aligned,
                        )
                        draw_detections(
                            depth_view, [place_det], last_place_names,
                            depth_aligned,
                        )
                    if place_u is not None and place_v is not None:
                        cv2.drawMarker(
                            annotated,
                            (int(round(place_u)), int(round(place_v))),
                            (40, 200, 255), cv2.MARKER_TILTED_CROSS, 16, 2,
                        )
                        if place_circle is not None:
                            pr = int(round(float(place_circle.get("radius_px", 0))))
                            if pr > 4:
                                cv2.circle(
                                    annotated,
                                    (int(round(place_u)), int(round(place_v))),
                                    pr, (0, 220, 120), 2, cv2.LINE_AA,
                                )
                            cv2.circle(
                                annotated,
                                (int(round(place_u)), int(round(place_v))),
                                4, (0, 255, 180), -1, cv2.LINE_AA,
                            )
            elif not want_plate and not plate_frozen:
                self._place_circle_cache = None

            # 筐子 RGBD：仅观察后且 UI 勾选「筐」；螺母对准时跳过
            basket_det = None
            basket_n = 0
            basket_hit = None
            if want_basket and bool(getattr(self, "basket_rgbd", False)):
                # 盘优先对准2：不每帧烧 ~100ms RGBD（用观察位缓存即可）
                plate_first_align = (
                    place_align
                    and want_plate
                    and bool(eye_cfg.get("place_prefer_plate", True))
                )
                skip_basket = (aligning and not place_align) or plate_first_align
                if skip_basket:
                    basket_hit = self._basket_cache_hit
                else:
                    period = (
                        0.05 if place_align else float(self.basket_period_s)
                    )
                    need_refresh = (now - self._basket_cache_t) >= period
                    if need_refresh or self._basket_cache_hit is None:
                        try:
                            t_b0 = time.perf_counter()
                            basket_hit = self._detect_basket_rgbd_scaled(
                                frame, depth_aligned,
                            )
                            dt_b = (time.perf_counter() - t_b0) * 1000.0
                            if now - last_perf_log > 2.5:
                                last_perf_log = now
                                print(
                                    f"[perf] basket_rgbd={dt_b:.0f}ms "
                                    f"loop≈{(time.perf_counter()-loop_t0)*1000:.0f}ms "
                                    f"align={aligning} placeA={place_align}",
                                    flush=True,
                                )
                        except Exception as basket_exc:
                            if not getattr(self, "_basket_rgbd_err_logged", False):
                                self._basket_rgbd_err_logged = True
                                self.log(f"筐子 RGBD 失败（之后静默）: {basket_exc}")
                                print(f"[basket-rgbd] {basket_exc}", flush=True)
                            basket_hit = None
                        self._basket_cache_hit = basket_hit
                        self._basket_cache_t = now
                    else:
                        basket_hit = self._basket_cache_hit
                if basket_hit is not None:
                    basket_n = 1
                    x1, y1, x2, y2 = basket_hit.box_xyxy
                    bconf = float(np.clip(basket_hit.score / 150.0, 0.05, 0.99))
                    basket_det = np.array(
                        [x1, y1, x2, y2, bconf, 0], dtype=np.float32
                    )
                    draw_boxes_b = bool(det_cfg.get("draw_yolo_boxes", True))
                    if draw_boxes_b and not skip_basket:
                        corners = np.asarray(
                            basket_hit.corners, dtype=np.int32
                        ).reshape(-1, 1, 2)
                        cv2.polylines(
                            annotated, [corners], True, (255, 80, 255), 2, cv2.LINE_AA,
                        )
                        if depth_view is not None:
                            cv2.polylines(
                                depth_view, [corners], True, (255, 80, 255), 2,
                            )
                        # 沿长边三格：世界前=大、中、近=小
                        eye_b = self.config.get("eye_in_hand") or {}
                        if bool(eye_b.get("place_draw_slots", True)):
                            slots = self._compute_basket_slots(basket_hit)
                            self._basket_slots = slots
                            hl = None
                            si = getattr(self, "_locked_size_info", None)
                            if isinstance(si, dict):
                                hl = str(si.get("class") or "").strip().lower()
                                if hl == "large":
                                    hl = "big"
                            draw_basket_slots(annotated, slots, highlight_label=hl)
                            if depth_view is not None:
                                draw_basket_slots(
                                    depth_view, slots, highlight_label=hl,
                                )
                        else:
                            self._basket_slots = None
                    bu = float(basket_hit.center_uv[0])
                    bv = float(basket_hit.center_uv[1])
                    if not skip_basket:
                        cv2.drawMarker(
                            annotated,
                            (int(round(bu)), int(round(bv))),
                            (255, 80, 255), cv2.MARKER_DIAMOND, 18, 2,
                        )
                    if self.basket_use_for_place:
                        # 盘装配优先 / 已锁盘心：不要被筐心覆盖
                        prefer_plate = plate_frozen or (
                            want_plate
                            and place_u is not None
                            and place_v is not None
                            and place_depth_mm is not None
                            and np.isfinite(float(place_depth_mm))
                            and bool(eye_cfg.get("place_prefer_plate", True))
                        )
                        if not prefer_plate:
                            place_u, place_v = bu, bv
                            pd = basket_hit.rim_mm
                            if pd is None and depth_aligned is not None:
                                _, _, pd = box_center_and_depth(
                                    depth_aligned, (x1, y1, x2, y2),
                                )
                            if pd is not None and np.isfinite(pd):
                                place_depth_mm = float(pd)
                            place_cls = "basket"
                            place_det = basket_det
                            place_circle = None
                            if not skip_basket:
                                cv2.drawMarker(
                                    annotated,
                                    (int(round(place_u)), int(round(place_v))),
                                    (40, 200, 255), cv2.MARKER_TILTED_CROSS, 16, 2,
                                )
            elif want_basket and self.yolo_basket is not None and last_basket_raw is not None:
                basket_pool = filter_place_detections(
                    last_basket_raw, last_basket_names,
                )
                if not basket_pool and last_basket_raw is not None and len(last_basket_raw):
                    n_bn = (
                        len(last_basket_names)
                        if isinstance(last_basket_names, dict)
                        else (len(last_basket_names) if last_basket_names else 0)
                    )
                    if n_bn <= 1:
                        basket_pool = list(last_basket_raw)
                basket_n = len(basket_pool) if basket_pool else 0
                if basket_pool:
                    basket_det = max(basket_pool, key=lambda d: float(d[4]))
                    draw_boxes_b = bool(det_cfg.get("draw_yolo_boxes", True))
                    if draw_boxes_b:
                        overrides_b = ["basket"] * len(basket_pool)
                        draw_detections(
                            annotated, basket_pool, last_basket_names,
                            depth_aligned, class_overrides=overrides_b,
                        )
                        draw_detections(
                            depth_view, basket_pool, last_basket_names,
                            depth_aligned, class_overrides=overrides_b,
                        )
                    bx = basket_det[:4]
                    bu = float((bx[0] + bx[2]) * 0.5)
                    bv = float((bx[1] + bx[3]) * 0.5)
                    cv2.drawMarker(
                        annotated,
                        (int(round(bu)), int(round(bv))),
                        (255, 80, 255), cv2.MARKER_DIAMOND, 18, 2,
                    )

            # 矩形螺丝板：UI 勾选即可实时检（不依赖观察位）
            # 优先 YOLO；无框且允许时回退经典 RGB
            screw_hit = None
            want_screw = self._want_screw_board_detect()
            if want_screw:
                sb_cfg = getattr(self, "screw_board_cfg", None) or {}
                period = float(getattr(self, "screw_board_period_s", 0.15))
                need_sb = (now - self._screw_board_cache_t) >= period
                if need_sb or self._screw_board_cache_hit is None:
                    screw_hit = None
                    if (
                        self.yolo_board is not None
                        and bool(sb_cfg.get("prefer_yolo", True))
                    ):
                        try:
                            screw_hit = self._screw_board_from_yolo(
                                frame,
                                last_board_raw,
                                last_board_names,
                                depth_aligned,
                            )
                        except Exception as sb_exc:
                            if not getattr(self, "_screw_board_err_logged", False):
                                self._screw_board_err_logged = True
                                self.log(f"螺丝板 YOLO 解析失败（之后静默）: {sb_exc}")
                                print(f"[screw-board-yolo] {sb_exc}", flush=True)
                            screw_hit = None
                    if screw_hit is None and (
                        self.yolo_board is None
                        or bool(sb_cfg.get("yolo_fallback_rgbd", True))
                    ):
                        try:
                            screw_hit = self._detect_screw_board_scaled(
                                frame, depth_aligned,
                            )
                        except Exception as sb_exc:
                            if not getattr(self, "_screw_board_err_logged", False):
                                self._screw_board_err_logged = True
                                self.log(f"螺丝板检测失败（之后静默）: {sb_exc}")
                                print(f"[screw-board] {sb_exc}", flush=True)
                            screw_hit = None
                    self._screw_board_cache_hit = screw_hit
                    self._screw_board_cache_t = now
                else:
                    screw_hit = self._screw_board_cache_hit
                if screw_hit is not None:
                    draw_screw_board(annotated, screw_hit)
                    if depth_view is not None:
                        draw_screw_board(depth_view, screw_hit)
                    # 标定孔投影 + RGB 有无螺母
                    hole_hits = []
                    holes_cfg = getattr(self, "screw_board_holes_cfg", None) or []
                    if holes_cfg:
                        try:
                            hole_hits = evaluate_board_holes(
                                frame,
                                screw_hit.corners,
                                holes_cfg,
                                radius_frac=float(sb_cfg.get("hole_radius_frac", 0.048)),
                                min_delta=float(sb_cfg.get("hole_min_delta", 5.0)),
                                min_contrast=float(sb_cfg.get("hole_min_contrast", 5.0)),
                                max_sat=float(sb_cfg.get("hole_max_sat", 140.0)),
                                score_min=float(sb_cfg.get("hole_score_min", 0.30)),
                            )
                        except Exception as hole_exc:
                            if not getattr(self, "_hole_eval_err_logged", False):
                                self._hole_eval_err_logged = True
                                self.log(f"孔位判空失败（之后静默）: {hole_exc}")
                                print(f"[board-holes] {hole_exc}", flush=True)
                            hole_hits = []
                        # 连续帧确认，减少闪烁
                        need_n = max(1, int(sb_cfg.get("hole_occupy_frames", 2)))
                        streak = getattr(self, "_hole_occ_streak", None)
                        if streak is None:
                            streak = {}
                            self._hole_occ_streak = streak
                        for hh in hole_hits:
                            if hh.occupied:
                                streak[hh.id] = int(streak.get(hh.id, 0)) + 1
                            else:
                                streak[hh.id] = 0
                            hh.occupied = streak.get(hh.id, 0) >= need_n
                        # 有螺后精修螺头中心（RGB + 可选深度）+ 时序 EMA
                        try:
                            hole_hits = refine_occupied_hole_centers(
                                frame,
                                depth_aligned if depth_aligned is not None else None,
                                hole_hits,
                            )
                            alpha = float(sb_cfg.get("hole_refine_ema", 0.35))
                            alpha = float(np.clip(alpha, 0.05, 1.0))
                            ema = getattr(self, "_hole_refine_ema", None)
                            if ema is None:
                                ema = {}
                                self._hole_refine_ema = ema
                            live_ids = set()
                            for hh in hole_hits:
                                live_ids.add(int(hh.id))
                                if not hh.occupied or hh.xy_refined is None:
                                    ema.pop(int(hh.id), None)
                                    continue
                                rx, ry = float(hh.xy_refined[0]), float(hh.xy_refined[1])
                                prev = ema.get(int(hh.id))
                                if prev is None:
                                    ema[int(hh.id)] = (rx, ry)
                                else:
                                    sx = alpha * rx + (1.0 - alpha) * float(prev[0])
                                    sy = alpha * ry + (1.0 - alpha) * float(prev[1])
                                    ema[int(hh.id)] = (sx, sy)
                                    hh.xy_refined = (sx, sy)
                            for kid in list(ema.keys()):
                                if kid not in live_ids:
                                    ema.pop(kid, None)
                        except Exception as ref_exc:
                            if not getattr(self, "_hole_refine_err_logged", False):
                                self._hole_refine_err_logged = True
                                self.log(f"螺头精修失败（之后静默）: {ref_exc}")
                        sel = -1
                        try:
                            sel = int(self.selected_screw_hole.get())
                        except Exception:
                            sel = -1
                        draw_board_holes(annotated, hole_hits, selected_id=sel)
                        if depth_view is not None:
                            draw_board_holes(depth_view, hole_hits, selected_id=sel)
                    self._screw_board_hole_hits = hole_hits
                else:
                    self._screw_board_hole_hits = []
            else:
                self._screw_board_cache_hit = None
                self._screw_board_hole_hits = []
                st = getattr(self, "_screw_board_refine_state", None)
                if st is not None:
                    st.last_corners = None
                    st.occlusion_frames = 0
                self._hole_occ_streak = {}
                self._hole_refine_ema = {}

            # 电动螺丝刀 YOLO：勾选「电批」实时检（批头用示教标定，不做 RGB 尖端）
            driver_det = None
            driver_u = driver_v = None
            driver_cls = None
            if want_driver and last_driver_raw is not None and len(last_driver_raw):
                driver_det = max(
                    list(last_driver_raw), key=lambda d: float(d[4]),
                )
                if bool(det_cfg.get("draw_yolo_boxes", True)):
                    draw_detections(
                        annotated,
                        [driver_det],
                        last_driver_names,
                        depth_aligned if depth_aligned is not None else None,
                    )
                    if depth_view is not None:
                        draw_detections(
                            depth_view,
                            [driver_det],
                            last_driver_names,
                            depth_aligned if depth_aligned is not None else None,
                        )
                bx = driver_det[:4]
                driver_u = float((bx[0] + bx[2]) * 0.5)
                driver_v = float((bx[1] + bx[3]) * 0.5)
                driver_cls = detection_class_name(
                    last_driver_names, driver_det[5],
                ) or "刀"
                cv2.drawMarker(
                    annotated,
                    (int(round(driver_u)), int(round(driver_v))),
                    (0, 165, 255), cv2.MARKER_DIAMOND, 16, 2,
                )

            if not allowed:
                status = "请勾选追踪尺寸 大/中/小" if self._track_is_size else "请勾选追踪颜色 red / green / blue"
                self.controller.reset_stability()
            elif detection is not None:
                # 跟踪仍用检测；画框时优先显示深度定档
                to_draw = filtered if filtered else [detection]
                pick_mismatch = bool(
                    size_metric_on
                    and size_info_pre is not None
                    and size_info_pre.get("class") not in (allowed or set())
                )
                R_wc = self.controller.camera_rotation_world()
                img_up = None
                if R_wc is not None:
                    img_up = world_up_in_image_uv(
                        R_wc,
                        eye_cfg.get("pinch_world_up", [0.0, 0.0, 1.0]),
                    )
                u, v = grasp_pixel_from_box(
                    detection[:4],
                    eye_cfg.get("grasp_point_mode", "bbox_center"),
                    float(eye_cfg.get("grasp_point_top_frac", 0.35)),
                    side=eye_cfg.get("grasp_point_side", "right"),
                    image_up_uv=img_up,
                )
                # 每个检测框：nut_surface 精修 + 定档（含跨帧跟踪）
                # 原先只对主目标画平面精修；现对 to_draw 内每个框都算/画
                box_overrides = []
                box_size_infos = []
                draw_boxes = bool(det_cfg.get("draw_yolo_boxes", True))
                draw_surf_all = bool(eye_cfg.get("draw_nut_surface_all", True))
                # 对准也算全部螺母框：定档/跟踪更稳，少丢目标、少被邻框抢走
                dets_for_size = list(to_draw) if to_draw else [detection]
                if (
                    size_metric_on
                    and bool((det_cfg.get("size_metric") or {}).get("temporal_filter", True))
                    and getattr(self, "_size_bank", None) is not None
                ):
                    self._size_bank.configure(det_cfg)
                    self._size_bank.begin_frame()
                for det_i in dets_for_size:
                    info_i = None
                    d_i = None
                    meta_i = None
                    is_pri = bool(
                        np.allclose(det_i[:4], detection[:4], atol=1.0)
                    )
                    if depth_aligned is not None:
                        # 非主目标勿共用主目标 prior，否则小框层分离/半径会被带偏
                        _ui, _vi, d_i, meta_i = measure_detection_uvd(
                            depth_aligned,
                            det_i,
                            eye_cfg,
                            prior_mm=(prior_for_pick if is_pri else None),
                            freeze_below_mm=0.0,
                            R_world_cam=R_wc,
                            image_up_uv=img_up,
                            return_meta=True,
                        )
                        if draw_surf_all and meta_i is not None:
                            draw_nut_surface_depth_debug(
                                depth_view, meta_i, box=det_i[:4],
                            )
                            draw_nut_surface_depth_debug(
                                annotated, meta_i, box=det_i[:4],
                            )
                    if depth_aligned is not None and size_metric_on:
                        yolo_i = yolo_size_prior_class(names, det_i[5])
                        info_i = measure_and_classify_nut(
                            det_i[:4],
                            d_i,
                            self.controller.intrinsics,
                            det_cfg,
                            yolo_cls=yolo_i,
                            yolo_conf=float(det_i[4]),
                        )
                        # 主目标尺寸轨留给后面 fused depth，避免单帧坏深度把大占成小
                        if (
                            info_i is not None
                            and (not is_pri)
                            and bool(
                                (det_cfg.get("size_metric") or {}).get(
                                    "temporal_filter", True
                                )
                            )
                            and getattr(self, "_size_bank", None) is not None
                        ):
                            info_i = self._size_bank.update(det_i[:4], info_i)
                    if info_i is not None and info_i.get("class"):
                        box_overrides.append(info_i["class"])
                        box_size_infos.append(info_i)
                    else:
                        box_overrides.append(None)
                        box_size_infos.append(None)
                if size_metric_on and any(box_size_infos):
                    box_overrides = refine_size_classes_relative(
                        box_size_infos, det_cfg,
                    )
                # 全框定档后：主目标必须落在勾选尺寸内（禁止对准其它档）
                track_ok = True
                if size_metric_on and allowed and dets_for_size:
                    matched = []
                    for di, (det_i, info_i) in enumerate(
                        zip(dets_for_size, box_size_infos)
                    ):
                        cls_i = None
                        if box_overrides and di < len(box_overrides):
                            cls_i = box_overrides[di]
                        if cls_i is None and info_i is not None:
                            cls_i = info_i.get("class")
                        if cls_i in allowed:
                            matched.append((det_i, info_i, float(det_i[4]), di))
                    if matched:
                        aligning_now = bool(
                            getattr(self.controller, "_aligning", False)
                        ) or bool(getattr(self, "_place_aligning", False))
                        track_uv = getattr(self, "_servo_track_uv", None)
                        track_max = float(
                            eye_cfg.get("align_track_max_px", 220)
                        )

                        def _box_cxy(det):
                            return (
                                0.5 * (float(det[0]) + float(det[2])),
                                0.5 * (float(det[1]) + float(det[3])),
                            )

                        keep_pri = False
                        if (
                            aligning_now
                            and track_uv is not None
                            and track_max > 1.0
                        ):
                            # 对准中：优先跟上一目标（防偏心时高分邻框抢走 → UV/深度跳变）
                            ranked = []
                            for det_i, info_i, sc, di in matched:
                                cu, cv_ = _box_cxy(det_i)
                                dist = math.hypot(
                                    cu - float(track_uv[0]),
                                    cv_ - float(track_uv[1]),
                                )
                                ranked.append((dist, -sc, det_i, info_i, di))
                            ranked.sort()
                            if ranked[0][0] <= track_max:
                                detection = ranked[0][2]
                                size_info_pre = ranked[0][3]
                                keep_pri = True
                        if not keep_pri:
                            for det_i, info_i, _sc, _di in matched:
                                if np.allclose(det_i[:4], detection[:4], atol=1.0):
                                    detection = det_i
                                    size_info_pre = info_i
                                    keep_pri = True
                                    break
                        if not keep_pri:
                            matched.sort(key=lambda t: t[2], reverse=True)
                            detection, size_info_pre, _sc, _di = matched[0]
                        pick_mismatch = False
                    else:
                        track_ok = False
                        pick_mismatch = True
                if draw_boxes:
                    draw_detections(
                        annotated, to_draw, names,
                        depth_aligned if depth_aligned is not None else None,
                        class_overrides=box_overrides,
                        size_infos=box_size_infos,
                    )
                if not track_ok:
                    self.controller.reset_stability()
                    status = f"未检测到勾选尺寸 {sorted(allowed)}"
                    u = v = None
                    depth_mm = None
                    size_info = None
                else:
                    u, v = grasp_pixel_from_box(
                        detection[:4],
                        eye_cfg.get("grasp_point_mode", "bbox_center"),
                        float(eye_cfg.get("grasp_point_top_frac", 0.35)),
                        side=eye_cfg.get("grasp_point_side", "right"),
                        image_up_uv=img_up,
                    )
                    yolo_cls = logical_color_from_detection(names, detection[5]) or detection_class_name(
                        names, detection[5]
                    )
                    cls_name = yolo_cls
                    size_info = size_info_pre
                    held = False
                    if depth_aligned is not None:
                        if draw_boxes:
                            draw_detections(
                                depth_view, to_draw, names, depth_aligned,
                                class_overrides=box_overrides,
                                size_infos=box_size_infos,
                            )
                        prior = self._depth_tracker.mm
                        if prior is None:
                            prior = self._last_reliable_depth_mm
                        u, v, raw_mm, depth_meta = measure_detection_uvd(
                            depth_aligned,
                            detection,
                            eye_cfg,
                            prior_mm=prior,
                            freeze_below_mm=0.0,
                            R_world_cam=R_wc,
                            image_up_uv=img_up,
                            return_meta=True,
                        )
                        depth_conf = float(
                            (depth_meta or {}).get("confidence", 0.55)
                        )
                        if depth_meta is not None and not draw_surf_all:
                            draw_nut_surface_depth_debug(
                                depth_view, depth_meta, box=detection[:4],
                            )
                            if bool(eye_cfg.get("pinch_debug", False)) or str(
                                eye_cfg.get("depth_roi_mode", "")
                            ).startswith("nut"):
                                draw_nut_surface_depth_debug(
                                    annotated, depth_meta, box=detection[:4],
                                )
                        if u is None or v is None:
                            u = float((detection[0] + detection[2]) * 0.5)
                            v = float((detection[1] + detection[3]) * 0.5)
                        # 偏心对准：用框心伺服（环心/孔核离轴易抖）；近中心再信环心
                        aligning_now = bool(
                            getattr(self.controller, "_aligning", False)
                        ) or bool(getattr(self, "_place_aligning", False))
                        bbox_until = float(
                            eye_cfg.get("align_bbox_servo_err_px", 70)
                        )
                        if aligning_now and bbox_until > 1.0:
                            err_px = math.hypot(float(u) - cx, float(v) - cy)
                            if err_px > bbox_until:
                                u = float((detection[0] + detection[2]) * 0.5)
                                v = float((detection[1] + detection[3]) * 0.5)
                        if raw_mm is None:
                            _, _, raw_mm = box_center_and_depth(
                                depth_aligned,
                                detection,
                                prior_mm=prior,
                                freeze_below_mm=0.0,
                                eye=eye_cfg,
                            )
                            depth_conf = 0.45
                        fused_raw = self._ring_depth_buf.push(raw_mm, depth_conf)
                        if (
                            fused_raw is not None
                            and fused_raw >= float(eye_cfg.get("depth_min_valid_mm", 80))
                        ):
                            raw_mm = fused_raw
                        depth_mm, held = self._depth_tracker.update(
                            raw_mm, confidence=depth_conf,
                        )
                        if depth_mm is None:
                            depth_mm = prior
                        if depth_mm is None:
                            depth_mm = self._last_reliable_depth_mm
                        if size_metric_on and depth_mm is not None:
                            size_info = measure_and_classify_nut(
                                detection[:4],
                                depth_mm,
                                self.controller.intrinsics,
                                det_cfg,
                                yolo_cls=yolo_size_prior_class(names, detection[5]),
                                yolo_conf=float(detection[4]),
                            )
                            if size_info is not None and box_size_infos:
                                try:
                                    idx_pri = next(
                                        i for i, d in enumerate(to_draw)
                                        if np.allclose(d[:4], detection[:4], atol=1.0)
                                    )
                                except StopIteration:
                                    idx_pri = 0
                                if idx_pri < len(box_size_infos):
                                    box_size_infos[idx_pri] = size_info
                                    box_overrides = refine_size_classes_relative(
                                        box_size_infos, det_cfg,
                                    )
                                    abs_cls = size_info.get("class")
                                    if abs_cls:
                                        box_overrides[idx_pri] = abs_cls
                                    if draw_boxes:
                                        draw_detections(
                                            annotated, to_draw, names,
                                            depth_aligned,
                                            class_overrides=box_overrides,
                                            size_infos=box_size_infos,
                                        )
                                        draw_detections(
                                            depth_view, to_draw, names,
                                            depth_aligned,
                                            class_overrides=box_overrides,
                                            size_infos=box_size_infos,
                                        )
                            if (
                                size_info is not None
                                and bool(
                                    (det_cfg.get("size_metric") or {}).get(
                                        "temporal_filter", True
                                    )
                                )
                                and getattr(self, "_size_bank", None) is not None
                            ):
                                self._size_bank.configure(det_cfg)
                                size_info = self._size_bank.update(
                                    detection[:4], size_info
                                )
                            # 融合定档后若飘出勾选，放弃本帧对准目标
                            if (
                                size_info is not None
                                and size_info.get("class") not in allowed
                            ):
                                self.controller.reset_stability()
                                status = (
                                    f"定档[{size_info.get('class')}]∉勾选"
                                    f"{sorted(allowed)}，不对准"
                                )
                                u = v = None
                                depth_mm = None
                                size_info = None
                                pick_mismatch = True
                        if size_info is not None and size_info.get("class"):
                            cls_name = size_info["class"]
                        if u is not None and depth_mm is not None:
                            tag = ""
                            if held and self._depth_tracker.frozen:
                                tag = " ·近距保持"
                            elif held:
                                tag = " ·沿用"
                            size_tag = ""
                            if size_info is not None:
                                hex_mm = size_info.get(
                                    "hex_mm", size_info.get("long_mm", 0)
                                )
                                size_tag = f" | 对边{hex_mm:.0f}mm"
                                raw_l = size_info.get(
                                    "hex_raw_mm", size_info.get("long_raw_mm")
                                )
                                if (
                                    raw_l is not None
                                    and abs(float(raw_l) - float(hex_mm)) > 0.5
                                ):
                                    size_tag += f"(raw{float(raw_l):.0f})"
                                if yolo_cls and yolo_cls != size_info.get("class"):
                                    size_tag += f" YOLO={yolo_cls}"
                                if size_info.get("ambiguous"):
                                    size_tag += " ?"
                                if size_info.get("size_held"):
                                    size_tag += "·稳"
                            ready = self.controller.is_ready(u, v, depth_mm)
                            stable = self.controller.update_stability(u, v, depth_mm)
                            status = (
                                f"[{cls_name}] {depth_mm / 10:.1f}cm / "
                                f"目标≤{self.target_cm:.0f}cm | "
                                f"偏心({u - cx:+.0f},{v - cy:+.0f})px | "
                                f"{'已对准' if ready else '对准中'} "
                                f"{self.controller.stable_count}/"
                                f"{self.controller.stable_required}"
                                f"{tag}{size_tag}"
                            )
                            draw_grasp_point_marker(
                                annotated, u, v,
                                mode=eye_cfg.get("grasp_point_mode", "bbox_center"),
                            )
                            self._last_size_info = size_info
                        elif u is not None:
                            self.controller.reset_stability()
                            status = (
                                f"[{cls_name}] 偏心({u - cx:+.0f},{v - cy:+.0f})px "
                                f"（框内无有效深度）"
                            )
                    else:
                        self.controller.reset_stability()
                        status = (
                            f"[{cls_name}] 偏心({u - cx:+.0f},{v - cy:+.0f})px "
                            f"（等待深度）"
                        )
            else:
                self.controller.reset_stability()
                # 仅 YOLO 真丢框时才清深度记忆；尺寸过滤失败不应每帧 reset
                if self._det_hold_box is None:
                    self._depth_tracker.reset()
                    if getattr(self, "_size_bank", None) is not None:
                        self._size_bank.reset()
                size_info = None
                if allowed:
                    status = f"未检测到勾选尺寸 {sorted(allowed)}"
                # 仍画出画面上的螺母框（定档标签），只是不伺服其它档
                if filtered and bool(det_cfg.get("draw_yolo_boxes", True)):
                    draw_detections(
                        annotated, filtered, names,
                        depth_aligned if depth_aligned is not None else None,
                    )
            held_depth = self._depth_tracker.value
            # 状态行：抓 / 盘 / 筐（观察前不跑，只提示）
            after_obs = self._after_observe_place_vision()
            if self.detect_plate_var.get():
                if not after_obs:
                    status = f"{status} | 盘=待观察"
                elif self.yolo_place is None:
                    status = f"{status} | 盘=未加载"
                elif place_det is not None and place_cls != "basket":
                    pc = track_class_display_name(place_cls or "bin")
                    pconf = float(place_det[4])
                    pd = (
                        f" {place_depth_mm/10:.0f}cm"
                        if place_depth_mm is not None and np.isfinite(place_depth_mm)
                        else ""
                    )
                    status = f"{status} | 盘={pc} {pconf:.2f}{pd}"
                    if place_circle is not None:
                        status += f"⊙{place_circle.get('method', '?')}"
                else:
                    status = f"{status} | 盘=无"
            if self.detect_basket_var.get():
                if not after_obs:
                    status = f"{status} | 筐=待观察"
                elif not self.basket_rgbd and self.yolo_basket is None:
                    status = f"{status} | 筐=未启用"
                elif basket_det is not None:
                    if self.basket_rgbd and basket_hit is not None:
                        rd = basket_hit.rim_delta_mm
                        rd_s = f" Δ{rd:.0f}mm" if rd is not None else ""
                        pd = (
                            f" {place_depth_mm/10:.0f}cm"
                            if (
                                self.basket_use_for_place
                                and place_depth_mm is not None
                                and np.isfinite(place_depth_mm)
                            )
                            else ""
                        )
                        status = (
                            f"{status} | 筐=rgbd{rd_s}{pd} "
                            f"{basket_hit.fit_mode}"
                        )
                    else:
                        bconf = float(basket_det[4])
                        status = f"{status} | 筐=basket {bconf:.2f}×{basket_n}"
                else:
                    status = f"{status} | 筐=无"
            if want_screw:
                if screw_hit is not None:
                    dtxt = (
                        f" {screw_hit.surface_mm/10:.0f}cm"
                        if screw_hit.surface_mm is not None
                        and np.isfinite(screw_hit.surface_mm)
                        else ""
                    )
                    tag = str(screw_hit.reason or "")
                    status = (
                        f"{status} | 板={screw_hit.size_wh[0]:.0f}x"
                        f"{screw_hit.size_wh[1]:.0f}{dtxt}"
                    )
                    if tag:
                        status = f"{status} [{tag}]"
                    hole_hits = getattr(self, "_screw_board_hole_hits", None) or []
                    if hole_hits:
                        occ_ids = [h.id for h in hole_hits if h.occupied]
                        status = (
                            f"{status} | 孔有螺={occ_ids if occ_ids else '无'}"
                        )
                else:
                    status = f"{status} | 板=无"
            if want_driver:
                if self.yolo_driver is None:
                    status = f"{status} | 电批=未加载"
                elif driver_det is not None:
                    status = (
                        f"{status} | 电批={driver_cls or '刀'} "
                        f"{float(driver_det[4]):.2f}"
                    )
                else:
                    status = f"{status} | 电批=无"
            draw_axis_overlay(
                annotated,
                self.controller.intrinsics,
                self.controller.R_ee_cam,
                self.controller.t_ee_cam,
                depth_mm if depth_mm is not None else held_depth,
            )
            if detection is not None and allowed and self._locked_detection is None:
                draw_servo_direction_hint(
                    annotated,
                    u,
                    v,
                    depth_mm if depth_mm is not None else held_depth,
                    self.controller.intrinsics,
                    self.controller.R_ee_cam,
                    xy_rotate_deg=self.controller.xy_rotate_deg,
                    flip_x=self.controller.servo_flip_x,
                    flip_y=self.controller.servo_flip_y,
                    gain=float(self.controller.eye.get("servo_gain", 0.45)),
                )
            locked_obj = getattr(self, "_locked_object_robot", None)
            locked_surf = getattr(self, "_locked_surface_robot", None)
            if locked_obj is not None or locked_surf is not None:
                pose = (
                    self.controller.get_live_pose()
                    if self.busy
                    else self.controller.get_pose()
                )
                if pose is None:
                    pose = self.controller.get_pose()
                if pose is not None:
                    # 紫色：上表面锁点（应对准通孔）；勿用含下偏青球（透视会偏几十 px）
                    overlay_pt = locked_surf if locked_surf is not None else locked_obj
                    proj = world_point_to_image_pixel(
                        overlay_pt,
                        pose,
                        self.controller.intrinsics,
                        self.controller.R_ee_cam,
                        self.controller.t_ee_cam,
                        **self.controller._vision_ray_params(),
                    )
                    if proj is not None:
                        draw_model_object_overlay(
                            annotated, proj[0], proj[1], u_det=u, v_det=v,
                        )
            draw_axis_overlay(
                depth_view,
                self.controller.intrinsics,
                self.controller.R_ee_cam,
                self.controller.t_ee_cam,
                depth_mm if depth_mm is not None else held_depth,
            )
            # 手眼标定：叠 ChArUco + 刷新 READY 状态（板固定，主界面调姿后点「记录姿态」）
            if bool(getattr(self, "handeye_mode_var", None) and self.handeye_mode_var.get()):
                try:
                    annotated = self._handeye_process_frame(annotated)
                except Exception as he_exc:
                    if not getattr(self, "_handeye_err_logged", False):
                        self._handeye_err_logged = True
                        self.log(f"手眼叠加失败（之后静默）: {he_exc}")
            cv2.putText(
                annotated, status, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )

            # UI 只需要低分辨率预览；全分辨率 PIL 缩放会卡主线程
            ui_scale = float(eye_cfg.get("vision_ui_scale", 0.5))
            color_ui, depth_ui = annotated, depth_view
            if ui_scale < 0.99 and annotated is not None:
                nw = max(1, int(round(w * ui_scale)))
                nh = max(1, int(round(h * ui_scale)))
                if (nw, nh) != (w, h):
                    color_ui = cv2.resize(
                        annotated, (nw, nh), interpolation=cv2.INTER_AREA
                    )
                    if depth_view is not None:
                        depth_ui = cv2.resize(
                            depth_view, (nw, nh), interpolation=cv2.INTER_AREA
                        )

            with self.lock:
                yolo_cls_store = None
                if detection is not None:
                    yolo_cls_store = logical_color_from_detection(
                        names, detection[5]
                    ) or detection_class_name(names, detection[5])
                aligning_now = bool(
                    getattr(self.controller, "_aligning", False)
                ) or bool(getattr(self, "_place_aligning", False))
                if aligning_now and u is not None and v is not None:
                    self._servo_track_uv = (float(u), float(v))
                elif not aligning_now:
                    self._servo_track_uv = None
                self.latest.update(
                    {
                        "color": color_ui,
                        "depth_view": depth_ui,
                        "depth_aligned": depth_aligned,
                        "box": (
                            [float(detection[0]), float(detection[1]),
                             float(detection[2]), float(detection[3])]
                            if detection is not None else None
                        ),
                        "u": u,
                        "v": v,
                        "depth_mm": depth_mm,
                        "status": status,
                        "stable": stable,
                        "size_info": size_info if detection is not None else None,
                        "yolo_class": yolo_cls_store,
                        "confidence": (
                            float(detection[4]) if detection is not None else None
                        ),
                        "metric_class": (
                            (size_info or {}).get("class")
                            if detection is not None and size_info
                            else None
                        ),
                        "place_box": (
                            [float(place_det[0]), float(place_det[1]),
                             float(place_det[2]), float(place_det[3])]
                            if place_det is not None else None
                        ),
                        "place_u": place_u,
                        "place_v": place_v,
                        "place_depth_mm": place_depth_mm,
                        "place_class": place_cls,
                        "screw_board_box": (
                            list(screw_hit.box_xyxy) if screw_hit is not None else None
                        ),
                        "screw_board_u": (
                            float(screw_hit.center_uv[0])
                            if screw_hit is not None else None
                        ),
                        "screw_board_v": (
                            float(screw_hit.center_uv[1])
                            if screw_hit is not None else None
                        ),
                        "screw_board_depth_mm": (
                            float(screw_hit.surface_mm)
                            if screw_hit is not None
                            and screw_hit.surface_mm is not None
                            else None
                        ),
                        "screw_board_corners": (
                            np.asarray(screw_hit.corners, dtype=np.float32).tolist()
                            if screw_hit is not None else None
                        ),
                        "screw_board_holes": (
                            [
                                {
                                    "id": int(h.id),
                                    "u": float(h.u),
                                    "v": float(h.v),
                                    "xy": [float(h.xy[0]), float(h.xy[1])],
                                    "xy_refined": (
                                        [float(h.xy_refined[0]), float(h.xy_refined[1])]
                                        if getattr(h, "xy_refined", None) is not None
                                        else [float(h.xy[0]), float(h.xy[1])]
                                    ),
                                    "refine_score": float(
                                        getattr(h, "refine_score", 0.0) or 0.0
                                    ),
                                    "occupied": bool(h.occupied),
                                    "score": float(h.score),
                                    "corner": str(h.corner),
                                    "local": int(h.local),
                                }
                                for h in (getattr(self, "_screw_board_hole_hits", None) or [])
                            ]
                            if want_screw else None
                        ),
                        "screw_board_occupied_ids": (
                            [
                                int(h.id)
                                for h in (getattr(self, "_screw_board_hole_hits", None) or [])
                                if h.occupied
                            ]
                            if want_screw else None
                        ),
                        "driver_box": (
                            [float(driver_det[0]), float(driver_det[1]),
                             float(driver_det[2]), float(driver_det[3])]
                            if driver_det is not None else None
                        ),
                        "driver_u": driver_u,
                        "driver_v": driver_v,
                        "driver_conf": (
                            float(driver_det[4]) if driver_det is not None else None
                        ),
                        "driver_class": driver_cls,
                    }
                )

            if (
                self.args.pick
                and not self.picked
                and not self.busy
                and stable
                and depth_mm is not None
                and not pick_mismatch
            ):
                self.root.after(0, self.on_pick)

            time.sleep(0.001)

