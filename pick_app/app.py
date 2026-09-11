"""PickMonitorApp: composed from mixins (was ~7k-line monolith)."""
from __future__ import annotations

from pick_app.deps import *
from pick_app.constants import *
from pick_app.helpers import resolve_basket_mode
from pick_app.mixins import (
    ActionsMixin,
    HandeyeMixin,
    LockMixin,
    MjPlanMixin,
    MjViewMixin,
    PanelMixin,
    PlaceMixin,
    UiMixin,
    VisionMixin,
)


class PickMonitorApp(
    UiMixin,
    LockMixin,
    MjPlanMixin,
    MjViewMixin,
    VisionMixin,
    PlaceMixin,
    PanelMixin,
    HandeyeMixin,
    ActionsMixin,
):
    def __init__(
        self,
        root,
        args,
        config,
        controller,
        yolo,
        yolo_place=None,
        yolo_basket=None,
        yolo_driver=None,
        yolo_board=None,
        basket_rgbd=False,
    ):
        self.root = root
        self.args = args
        self.config = config
        self.controller = controller
        self.yolo = yolo
        self.yolo_place = yolo_place
        self.yolo_basket = yolo_basket
        self.yolo_driver = yolo_driver
        self.yolo_board = yolo_board
        self.basket_rgbd = bool(basket_rgbd)
        det0 = config.get("detection") or {}
        br = det0.get("basket_rgbd") or {}
        self.basket_use_for_place = bool(br.get("use_for_place", True))
        self.basket_rgbd_kwargs = {
            "min_rim_delta_mm": float(br.get("min_rim_delta_mm", 20.0)),
            "require_depth": bool(br.get("require_depth", False)),
            "elevate_mm": float(br.get("elevate_mm", 22.0)),
            "aspect_prior": float(br.get("aspect_prior", 1.45)),
        }
        # 性能：全分辨率每帧 ~240ms 会把对准拖死；降采样 + 节流
        self.basket_proc_width = int(br.get("proc_width", 640))
        self.basket_period_s = float(br.get("period_s", 0.12))
        self.basket_also_place_yolo = bool(br.get("also_run_place_yolo", False))
        self._basket_cache_hit = None
        self._basket_cache_t = 0.0
        self._basket_slots = None
        self._place_aligning = False
        self._place_locked_kind = None  # "plate" | "basket" | None（对准2 后锁定）
        self._place_circle_cache = None
        # 盘心世界点长期冻结：多颗螺母复用，观察/放下不清；仅对准2 覆盖
        self._plate_frozen_xyz = None
        self._plate_frozen_uvd = None
        # 观察到位后才检筐/盘；UI 勾选控制（默认：有 RGBD 则勾筐，盘默认关省算力）
        self.detect_basket_var = tk.BooleanVar(
            value=bool(self.basket_rgbd) and bool(br.get("ui_default_on", True))
        )
        self.detect_plate_var = tk.BooleanVar(
            value=bool(yolo_place is not None)
            and bool(det0.get("place_ui_default_on", False))
        )
        sb0 = det0.get("screw_board") or {}
        self.screw_board_cfg = dict(sb0) if isinstance(sb0, dict) else {}
        self.detect_screw_board_var = tk.BooleanVar(
            value=bool(self.screw_board_cfg.get("enabled", True))
            and bool(self.screw_board_cfg.get("ui_default_on", True))
        )
        self.detect_driver_var = tk.BooleanVar(
            value=bool(yolo_driver is not None)
            and bool(det0.get("driver_ui_default_on", True))
        )
        self._screw_board_cache_hit = None
        self._screw_board_cache_t = 0.0
        self.screw_board_proc_width = int(self.screw_board_cfg.get("proc_width", 640))
        self.screw_board_period_s = float(self.screw_board_cfg.get("period_s", 0.15))
        self._screw_board_refine_state = ScrewBoardRefineState()
        self._screw_board_hole_hits = []
        self._hole_occ_streak = {}  # id -> consecutive occupied frames
        holes_cfg = list(self.screw_board_cfg.get("holes") or [])
        self.screw_board_holes_cfg = holes_cfg
        self.selected_screw_hole = tk.IntVar(value=-1)
        self._screw_hole_btns = {}
        self._taught_screw_world = None  # np.array(3,) 示教用螺丝点
        self._taught_screw_hole_id = None
        self._taught_screw_uvd = None  # (u,v,depth_mm)
        self.bit_calib_status_var = tk.StringVar(value="批头未标定")
        tip_loaded = load_bit_tip_from_config(config)
        if tip_loaded is not None:
            tip, rpy = tip_loaded
            self.bit_calib_status_var.set(
                f"批头已标定 tip_ee="
                f"[{tip[0]*1000:.1f},{tip[1]*1000:.1f},{tip[2]*1000:.1f}]mm"
            )
        self.running = True
        self.busy = False
        self.picked = False
        self.target_cm = config["eye_in_hand"]["target_distance_mm"] / 10.0

        self.latest = {
            "color": None,
            "depth_view": None,
            "depth_aligned": None,
            "box": None,
            "u": None,
            "v": None,
            "depth_mm": None,
            "status": "启动中…",
            "stable": False,
            "place_box": None,
            "place_u": None,
            "place_v": None,
            "place_depth_mm": None,
            "place_class": None,
            "screw_board_box": None,
            "screw_board_u": None,
            "screw_board_v": None,
            "screw_board_depth_mm": None,
            "screw_board_corners": None,
            "screw_board_holes": None,  # list[{id,u,v,xy,xy_refined,occupied,...}]
            "screw_board_occupied_ids": None,
            "driver_box": None,
            "driver_u": None,
            "driver_v": None,
            "driver_conf": None,
            "driver_class": None,
        }
        self.lock = threading.Lock()
        self._mj_lock = threading.RLock()

        self.depth_launch = None
        self.depth_relay = None
        self.depth_file = None
        self.color_file = None
        self.camera = None
        self._cam_log_path = None
        self._cam_wait_t0 = None
        self._cam_wait_last_log = 0.0
        self._cam_wait_logged = False
        self._cam_fail_logged = False

        self.mj_model = None
        self.mj_data = None
        self.mj_renderer = None
        self.mj_joints = {}
        self.mj_cam = None
        self.mj_vopt = None
        self.mj_frame_markers = []
        self.mj_cam_optical_in_body = np.array([0.0, 0.0, -0.019], dtype=np.float64)
        self._drag_last = None
        self._drag_mode = "orbit"
        self._cam_default = {"distance": 1.6, "azimuth": 140.0, "elevation": -20.0}
        self._model_joints = None  # 关节指令预览（执行后立刻刷新模型）
        self._model_hand_cmd = None  # 模型预览时的手指令
        self._align_stop = False
        self._pipeline_phase = "idle"  # idle|holding|observed|place_aligned|placed
        self._locked_place_uvd = None  # 对准2 后锁定的筐/格 (u,v,depth_mm)
        self._locked_place_xyz = None  # 对准2 锁格世界坐标（放下优先）
        self._place_size_class = None  # 持物尺寸档，观察/对准2 不清
        self._ui_tick_n = 0
        self._display_rpy = None
        self._pinch_info = None
        self._vision_obj_world = None
        self._locked_object_mj = None
        self._locked_object_robot = None
        self._locked_surface_robot = None
        self._locked_detection = None
        self._locked_size_info = None
        self._last_size_info = None
        self._pinch_plan = None
        self._pinch_plan_time = 0.0
        self._mj_render_size = (0, 0)
        self._preview_anim = None
        self._virtual_object = False  # True=模型空间手动放置的实验物体
        self._preview_ee_trail = []  # 预览播放时 tool0 轨迹（模型世界系）
        eye0 = config["eye_in_hand"]
        self._depth_tracker = DepthTracker(
            freeze_below_mm=float(eye0.get("depth_freeze_below_mm", 150)),
            smooth=float(eye0.get("depth_smooth", 0.40)),
            max_up_jump_mm=float(eye0.get("depth_max_up_jump_mm", 70)),
            max_down_jump_mm=float(eye0.get("depth_max_down_jump_mm", 120)),
            min_confidence=float(eye0.get("depth_min_confidence", 0.35)),
            min_valid_mm=float(eye0.get("depth_min_valid_mm", 80)),
            max_valid_mm=float(eye0.get("depth_max_valid_mm", 900)),
        )
        self._ring_depth_buf = RingDepthBuffer(
            maxlen=int(eye0.get("depth_temporal_median_frames", 5)),
            spread_tol_mm=float(eye0.get("depth_temporal_spread_mm", 16.0)),
            min_samples=int(eye0.get("depth_temporal_min_samples", 3)),
            min_confidence=float(eye0.get("depth_min_confidence", 0.35)),
        )
        sm0 = (config.get("detection") or {}).get("size_metric") or {}
        # 仅多目标尺寸轨（一对一）；不再并行维护单目标 SizeClassTracker
        self._size_bank = MultiSizeClassTracker(
            detection_cfg=config.get("detection"),
            grid_px=float(sm0.get("temporal_track_grid_px", 48)),
            max_tracks=int(sm0.get("temporal_max_tracks", 12)),
            stale_frames=int(sm0.get("temporal_stale_frames", 18)),
            maxlen=int(sm0.get("temporal_frames", 7)),
            ema=float(sm0.get("temporal_ema", 0.35)),
            jump_reject_mm=float(sm0.get("temporal_jump_reject_mm", 8.0)),
            hysteresis_mm=float(sm0.get("temporal_hysteresis_mm", 2.0)),
            vote_frames=int(sm0.get("temporal_vote_frames", 3)),
            min_samples=int(sm0.get("temporal_min_samples", 3)),
        )
        # 上次可靠物体深度（mm），用于拒背景飙远；不对齐 freeze 门槛
        self._last_reliable_depth_mm = None
        self._det_hold_box = None
        self._det_hold_n = 0
        self._det_hold_max = int(eye0.get("detection_hold_frames", 10))
        self._servo_track_uv = None  # 对准粘滞：跟上一目标，防邻框抢走
        # 手眼标定（ChArUco）：主界面调姿 + 记录/求解
        self.handeye_mode_var = tk.BooleanVar(value=False)
        self.handeye_status_var = tk.StringVar(value="手眼OFF")
        self._handeye_samples = []
        self._handeye_live = None
        self._handeye_detector = None
        self._handeye_board = None
        self._handeye_min_corners = 40
        self._handeye_last_mount = None
        self._handeye_last_stats = None
        # 追踪类别勾选（深度定档：big/medium/small；旧 three_color: red/green/blue）
        cfg_colors = {
            str(c).strip().lower()
            for c in config.get("detection", {}).get(
                "track_colors", ["big", "medium", "small"]
            )
        }
        self.track_big = tk.BooleanVar(value="big" in cfg_colors or "large" in cfg_colors)
        self.track_medium = tk.BooleanVar(value="medium" in cfg_colors or "mid" in cfg_colors)
        self.track_small = tk.BooleanVar(value="small" in cfg_colors)
        # 兼容旧三色勾选（若 yaml 仍写 red/green/blue）
        self.track_red = tk.BooleanVar(value="red" in cfg_colors)
        self.track_green = tk.BooleanVar(value="green" in cfg_colors)
        self.track_blue = tk.BooleanVar(value="blue" in cfg_colors)
        self.det_conf = tk.DoubleVar(value=float(config["detection"]["conf"]))
        self._track_is_size = bool(
            cfg_colors & {"big", "medium", "small", "large", "mid"}
        ) or not bool(cfg_colors & {"red", "green", "blue"})

        self._build_ui()
        self._load_mujoco()
        self._start_sensors()
        self.worker = threading.Thread(target=self._vision_loop, daemon=True)
        self.worker.start()
        self.root.after(self._model_ui_tick_ms(), self._ui_tick)
        self.root.protocol("WM_DELETE_WINDOW", self.close)


