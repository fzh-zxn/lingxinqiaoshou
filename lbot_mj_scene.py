"""MuJoCo 场景：位姿查询、物体射线、渲染叠加。"""
from __future__ import annotations

import math

import cv2
import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

from lbot_grasp_utils import detection_point_from_camera_pose

MODEL_FRAME_MARKERS = (
    ("world", "world", "world", "世界/默认工作系原点", 0.18),
    ("R基座", "site", "arm_right_base_mount", "右臂基座", 0.12),
    ("tool0", "site", "arm_right_tool0", "右法兰=默认工具原点", 0.12),
    ("相机", "camera_optical", "arm_right_camera", "右腕相机光心", 0.10),
    ("手", "site", "hand_right_wrist_mount", "右手腕安装", 0.10),
    ("L tool0", "site", "arm_left_tool0", "左法兰(对照无相机)", 0.08),
)


def mj_object_pose(model, data, kind, name, optical_in_body=None):
    if kind == "world":
        return np.zeros(3), np.eye(3)
    if kind == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            return None
        return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()
    if kind == "body":
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return None
        return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()
    if kind == "camera_optical":
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return None
        rot = data.xmat[bid].reshape(3, 3).copy()
        pos = data.xpos[bid].copy()
        if optical_in_body is not None:
            pos = pos + rot @ np.asarray(optical_in_body, dtype=np.float64)
        return pos, rot
    return None


def _init_scene_geom(scene, geom_type, size, pos, rgba):
    """MuJoCo mjv_initGeom 要求 mat 为 9 元向量、rgba 为 float32。"""
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        geom_type,
        np.asarray(size, dtype=np.float64).reshape(3),
        np.asarray(pos, dtype=np.float64).reshape(3),
        np.eye(3, dtype=np.float64).ravel(),
        np.asarray(rgba, dtype=np.float32).reshape(4),
    )


def add_scene_sphere(scene, pos, radius=0.006, rgba=(1.0, 0.85, 0.2, 0.95)):
    if scene.ngeom >= scene.maxgeom:
        return
    _init_scene_geom(
        scene,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        [float(radius), 0.0, 0.0],
        pos,
        rgba,
    )
    scene.ngeom += 1


def object_point_in_mj(
    model, data, u, v, depth_mm, intrinsics, optical_in_body,
    xy_rotate_deg=90.0, flip_x=True, flip_y=False, depth_sign=-1.0,
):
    pose = mj_object_pose(
        model, data, "camera_optical", "arm_right_camera",
        optical_in_body=optical_in_body,
    )
    if pose is None:
        return None
    cam_pos, cam_rot = pose
    return detection_point_from_camera_pose(
        cam_pos, cam_rot, u, v, depth_mm, intrinsics,
        xy_rotate_deg=xy_rotate_deg, flip_x=flip_x, flip_y=flip_y,
        depth_sign=depth_sign,
    )


def add_axis_triad(scene, pos, rot, length=0.12, width=0.006):
    """在 MuJoCo scene 上画 RGB=XYZ 三轴。"""
    colors = (
        np.array([1.0, 0.15, 0.15, 1.0], dtype=np.float32),
        np.array([0.15, 0.9, 0.2, 1.0], dtype=np.float32),
        np.array([0.2, 0.45, 1.0, 1.0], dtype=np.float32),
    )
    for axis, rgba in enumerate(colors):
        if scene.ngeom >= scene.maxgeom:
            return
        end = pos + length * rot[:, axis]
        _init_scene_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            rgba,
        )
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            width,
            pos,
            end,
        )
        scene.ngeom += 1
    if scene.ngeom < scene.maxgeom:
        _init_scene_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            [width * 1.8, 0, 0],
            pos,
            [1.0, 1.0, 0.2, 1.0],
        )
        scene.ngeom += 1


def add_scene_segment(scene, p0, p1, width=0.004, rgba=(0.9, 0.9, 0.9, 0.85)):
    if scene.ngeom >= scene.maxgeom:
        return
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
        return
    _init_scene_geom(
        scene,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        rgba,
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        p0,
        p1,
    )
    scene.ngeom += 1


def project_world_to_pixel(world_pos, glcam, fovy_deg, width, height):
    pos = np.asarray(glcam.pos, dtype=float)
    forward = np.asarray(glcam.forward, dtype=float)
    up = np.asarray(glcam.up, dtype=float)
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-9:
        return None
    right /= norm
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-12
    rel = np.asarray(world_pos, dtype=float) - pos
    z = float(np.dot(rel, forward))
    if z <= 1e-6:
        return None
    x = float(np.dot(rel, right))
    y = float(np.dot(rel, up))
    half_h = z * math.tan(math.radians(fovy_deg) * 0.5)
    half_w = half_h * width / max(height, 1)
    u = (x / half_w) * 0.5 + 0.5
    v = 0.5 - (y / half_h) * 0.5
    px, py = int(round(u * width)), int(round(v * height))
    if px < -20 or py < -20 or px > width + 20 or py > height + 20:
        return None
    return px, py


def mj_prepare_offscreen(model, width, height):
    need_w = max(int(model.vis.global_.offwidth), int(width))
    need_h = max(int(model.vis.global_.offheight), int(height))
    if need_w > model.vis.global_.offwidth or need_h > model.vis.global_.offheight:
        model.vis.global_.offwidth = need_w
        model.vis.global_.offheight = need_h
    return need_w, need_h


def rgb_to_photo_fit(rgb, canvas_w, canvas_h, image_tk_factory):
    """PIL ImageTk 缩放；image_tk_factory 为 ImageTk.PhotoImage。"""
    from PIL import Image

    h, w = rgb.shape[:2]
    scale = min(canvas_w / max(w, 1), canvas_h / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = Image.fromarray(rgb)
    img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    return image_tk_factory(img)


def filter_frame_markers(mj_model):
    markers = []
    for label, kind, name, meaning, length in MODEL_FRAME_MARKERS:
        if kind in ("world", "camera_optical"):
            markers.append((label, kind, name, meaning, length))
            continue
        obj = mujoco.mjtObj.mjOBJ_SITE if kind == "site" else mujoco.mjtObj.mjOBJ_BODY
        if mujoco.mj_name2id(mj_model, obj, name) >= 0:
            markers.append((label, kind, name, meaning, length))
    return markers
