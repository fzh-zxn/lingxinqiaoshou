"""CLI helpers and small pure utilities."""
from __future__ import annotations

from pick_app.deps import *
from pick_app.constants import (
    PROJECT_ROOT,
    MJCF_RELATIVE,
    PREVIEW_W,
    PREVIEW_H,
    DEFAULT_CAMERA,
    MODEL_MIN_W,
    MODEL_MAX_W,
    MODEL_MIN_H,
    MODEL_MAX_H,
)

def parse_args():
    parser = argparse.ArgumentParser(description="右臂视觉抓取单窗口监控")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--place-model",
        type=str,
        default=None,
        help="放置目标 YOLO（默认读 yaml，现为 pan_best.pt；传 none/- 关闭）",
    )
    parser.add_argument(
        "--basket-model",
        type=str,
        default=None,
        help="筐子检测：rgbd（默认，蓝+深度）/ none / 或 YOLO 权重路径",
    )
    parser.add_argument(
        "--driver-model",
        type=str,
        default=None,
        help="电动螺丝刀 YOLO（默认读 yaml 刀_best(2).pt；传 none/- 关闭）",
    )
    parser.add_argument(
        "--board-model",
        type=str,
        default=None,
        help="螺丝板 YOLO（默认读 yaml 板_best.pt；传 none/- 关闭，仍可用 RGB）",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--preview", action="store_true", help="不连接机械臂")
    parser.add_argument("--dry-run", action="store_true", help="连接但不运动")
    parser.add_argument("--pick", action="store_true", help="稳定对准后自动抓取一次")
    parser.add_argument("--confirm", action="store_true")
    return parser.parse_args()


def find_mjcf():
    candidates = [
        PROJECT_ROOT / MJCF_RELATIVE,
        Path.cwd() / MJCF_RELATIVE,
        PROJECT_ROOT / "robot_model_assets/workstations/lkls73_i1_o6_bimanual/workstation.mjcf",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def bgr_to_photo(bgr, canvas_w, canvas_h):
    """按画布尺寸等比缩放（可放大/缩小），居中贴图用。"""
    if Image is None or ImageTk is None or bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    cw = max(int(canvas_w), 1)
    ch = max(int(canvas_h), 1)
    iw, ih = image.size
    if iw < 1 or ih < 1:
        return None
    scale = min(cw / iw, ch / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    if (nw, nh) != (iw, ih):
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
    return ImageTk.PhotoImage(image=image)


def _lerp_joints(joints_a, joints_b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return [
        (1.0 - t) * float(a) + t * float(b)
        for a, b in zip(joints_a, joints_b)
    ]


def _slerp_rot3(R0, R1, t):
    """SO(3) 球面插值（模型预览 ⑥ 姿态过渡）。"""
    R0 = np.asarray(R0, dtype=np.float64).reshape(3, 3)
    R1 = np.asarray(R1, dtype=np.float64).reshape(3, 3)
    t = float(np.clip(t, 0.0, 1.0))
    if mujoco is not None:
        q0 = np.zeros(4, dtype=np.float64)
        q1 = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q0, R0.flatten())
        mujoco.mju_mat2Quat(q1, R1.flatten())
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        q = (1.0 - t) * q0 + t * q1
        qn = float(np.linalg.norm(q))
        if qn < 1e-12:
            return R0.copy()
        q /= qn
        R = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3)
    M = (1.0 - t) * R0 + t * R1
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _tapered_path_ts(n_steps, power=0.72):
    """
    路点进度 t∈(0,1]。power<1 前粗后细（末段略加密压终姿）。
    末点恒为 1；不把首段拉得过大（power≈0.7 温和）。
    """
    n = int(max(int(n_steps), 1))
    p = float(np.clip(power, 0.45, 1.0))
    if n == 1:
        return [1.0]
    if p >= 0.999:
        return [_smoothstep(float(i) / float(n)) for i in range(1, n + 1)]
    return [_smoothstep((float(i) / float(n)) ** p) for i in range(1, n + 1)]


def _mj_prepare_offscreen(model, width, height):
    """加载时一次性扩大 MuJoCo 离屏 framebuffer（默认仅 480px 高）。"""
    width = int(np.clip(width, MODEL_MIN_W, MODEL_MAX_W))
    height = int(np.clip(height, MODEL_MIN_H, MODEL_MAX_H))
    g = model.vis.global_
    g.offwidth = max(int(g.offwidth), width)
    g.offheight = max(int(g.offheight), height)
    return width, height


def _rgb_to_photo_fit(rgb, canvas_w, canvas_h):
    """固定分辨率渲染后，缩放贴到任意大小的画布（避免每帧重建 Renderer）。"""
    if Image is None or ImageTk is None or rgb is None:
        return None
    cw = max(int(canvas_w), MODEL_MIN_W)
    ch = max(int(canvas_h), MODEL_MIN_H)
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    iw, ih = image.size
    if iw < 1 or ih < 1:
        return None
    scale = min(cw / iw, ch / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    if (nw, nh) != (iw, ih):
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
    return ImageTk.PhotoImage(image=image)


def resolve_basket_mode(config=None, cli_path=None):
    """
    筐子检测模式：rgbd | yolo | none。
    CLI --basket-model 优先：rgbd / none / 权重路径。
    yaml：detection.basket_mode，或 basket_model=rgbd|none|*.pt。
    """
    raw = cli_path
    if raw is None and isinstance(config, dict):
        det = config.get("detection") or {}
        mode = str(det.get("basket_mode", "") or "").strip().lower()
        if mode in ("rgbd", "yolo", "none", "off", "-"):
            return "none" if mode in ("none", "off", "-") else mode
        raw = det.get("basket_model", "rgbd")
    if raw is None:
        return "rgbd"
    s = str(raw).strip().lower()
    if s in ("", "none", "off", "-", "null", "false", "0"):
        return "none"
    if s in ("rgbd", "color", "depth", "hsv"):
        return "rgbd"
    if s == "yolo":
        return "yolo"
    return "yolo"


# `from pick_app.helpers import *` 默认跳过 _ 前缀工具函数
__all__ = [name for name in globals() if not name.startswith("__")]


