"""Shared third-party / project imports for pick_app mixins."""
from __future__ import annotations

import argparse
import math
import threading
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import mujoco
    from PIL import Image, ImageTk
except ImportError:
    mujoco = None
    Image = ImageTk = None

from lbot_grasp_utils import (
    DEFAULT_CONFIG,
    DEFAULT_MODEL,
    DEFAULT_PLACE_MODEL,
    AlignFusionBuffer,
    DepthTracker,
    RingDepthBuffer,
    MultiSizeClassTracker,
    euler_rpy_to_matrix,
    apply_mjcf_camera_to_config,
    apply_camera_on_ee_to_mj_model,
    camera_body_pos_for_optical,
    write_mjcf_camera_body_pos,
    best_detection,
    best_detection_with_size_metric,
    best_place_detection,
    box_center_and_depth,
    consensus_depth_mm,
    depth_acceptable_for_action,
    grasp_pixel_from_box,
    measure_and_classify_nut,
    measure_detection_uvd,
    measure_place_uvd,
    refine_plate_disk_center,
    refine_size_classes_relative,
    resolve_place_model_path,
    resolve_basket_model_path,
    resolve_driver_model_path,
    resolve_board_model_path,
    yolo_class_names_brief,
    resolve_pregrasp_depth_mm,
    sanitize_action_depth_mm,
    scale_camera_intrinsics,
    build_camera_on_ee,
    camera_delta_to_ee,
    camera_on_ee_from_mjcf,
    camera_optical_center,
    camera_source,
    center_only_correction_in_ee,
    depth_colormap_view,
    align_depth_to_size,
    detection_class_name,
    draw_axis_overlay,
    draw_detections,
    draw_grasp_point_marker,
    draw_model_object_overlay,
    draw_nut_surface_depth_debug,
    draw_servo_direction_hint,
    filter_detections_by_colors,
    filter_nut_detections,
    filter_place_detections,
    is_grasp_aligned,
    nut_size_classes,
    yolo_size_prior_class,
    load_grasp_config,
    load_yolo_model,
    logical_color_from_detection,
    track_class_display_name,
    joints_soft_limit_violation,
    resolve_soft_joint_limits_rad,
    clamp_joints_to_soft_limits,
    soft_limit_inward_seed,
    save_camera_on_ee_translation,
    start_depth_processes,
    stop_process,
    unwrap_rpy_for_display,
    O6_L6_MASTER_JOINTS,
    O6_L6_MIMIC_JOINTS,
    approach_unit_toward_object,
    apply_pinch_grasp_bias,
    build_pinch_frame_at_object,
    detection_point_in_world,
    diagnose_level_pinch_top_rub,
    ee_target_from_pinch,
    object_up_direction_world,
    pinch_grasp_down_m,
    resolve_pinch_level_finger_plane,
    world_point_to_image_pixel,
    world_up_in_image_uv,
    matrix_to_rpy_near,
    pinch_dict_to_matrix,
    pose_to_matrix,
    rotation_matrix_to_rpy,
    rotation_geodesic_deg,
)
from lbot.lbot_robot import LbotArm, LbotEuler, LbotPosition, LbotRobot
from lbot_basket_rgbd import (
    find_blue_basket_rgbd,
    split_basket_slots_from_hit,
    slot_for_size_class,
    draw_basket_slots,
)
import lbot_ui_font as uifont
import lbot_ui_theme as ui
from lbot_ui_font import font_status_line
from lbot_ui_theme import FONT_MD, FONT_SM, create_root
from lbot_logutil import dbg, eye_debug, eye_verbose, vrb
from lbot_mj_hand import (
    apply_o6_hand_to_mjcf as _apply_o6_hand_to_mjcf,
    apply_hand_mount_rpy,
    clear_pad_plane_cache,
    collect_hand_joint_maps,
    compute_o6_pinch_kinematics as _compute_o6_pinch_kinematics,
    log_hand_preset_mj_compare,
    resolve_arm_joint_signs,
    resolve_hand_l6_cmd,
    robot_joints_to_mj_qpos,
    summarize_hand_l6_geometry,
)
from lbot_mj_scene import (
    MODEL_FRAME_MARKERS,
    add_axis_triad as _add_axis_triad,
    add_scene_segment as _add_scene_segment,
    add_scene_sphere as _add_scene_sphere,
    filter_frame_markers,
    mj_object_pose as _mj_object_pose,
    object_point_in_mj as _object_point_in_mj,
    project_world_to_pixel as _project_world_to_pixel,
)
from lbot_pinch_plan import (
    compute_pinch_plan_mj,
    grasp_close_preset,
    grasp_stabilize_preset,
    mj_tool0_delta_to_robot_ik,
    mj_tool0_pose_to_robot_ik,
    pinch_grasp_mode,
    plan_mj_to_robot_ik,
    point_mj_to_robot,
    point_robot_to_mj,
    tool0_matrices_from_poses,
)
from lbot_pick_plan import (
    compute_contact_T,
    n_steps_for_travel,
    pick_plan_from_geom,
    seed_drift_status,
)
from lbot_screw_board import (
    draw_screw_board,
    find_screw_board_rgbd,
    screw_board_hit_from_yolo,
    refine_screw_board_from_yolo_box,
    ScrewBoardRefineState,
    evaluate_board_holes,
    refine_occupied_hole_centers,
    draw_board_holes,
)
from lbot_bit_calib import (
    bit_tip_in_ee_from_teach,
    ee_xyz_for_screw,
    save_bit_tip_calib,
    load_bit_tip_from_config,
)

from lbot_arm_controller import (
    RightArmController,
    DEFAULT_GRASP_PRESET,
    DEFAULT_OPEN_PRESET,
    DEFAULT_PINCH_CLOSE,
    DEFAULT_PINCH_OPEN,
)

# `from pick_app.deps import *` 默认跳过 _ 前缀；别名必须显式进 __all__
__all__ = [name for name in globals() if not name.startswith("__")]

