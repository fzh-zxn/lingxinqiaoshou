"""UI / marker / path constants for the pick demo."""
from __future__ import annotations

from pathlib import Path

from lbot.lbot_robot import LbotArm

ARM = LbotArm.RIGHT_ARM
# pick_app/ is under lbot_pick_demo/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MJCF_RELATIVE = Path(
    "robot_model_assets/workstations/"
    "lkls73_i1_o6_bimanual/workstation.mjcf"
)
DEFAULT_CAMERA = (
    "/dev/v4l/by-id/"
    "usb-Orbbec_R__Orbbec_R__Gemini_TM__AY6R4"
    "6301JR-video-index0"
)

PREVIEW_W, PREVIEW_H = 480, 360
MODEL_W, MODEL_H = 640, 520
MODEL_MIN_W, MODEL_MIN_H = 480, 360
MODEL_MAX_W, MODEL_MAX_H = 1920, 1200

MARKER_OBJ = (0.95, 0.25, 0.85, 0.95)
MARKER_STANDOFF = (1.0, 0.78, 0.0, 0.98)
MARKER_TOOL0_ALIGN = (1.0, 0.12, 0.12, 0.95)
MARKER_PREGRASP_EE = (0.82, 0.82, 0.82, 0.92)
MARKER_THUMB = (1.0, 0.55, 0.15, 0.95)
MARKER_INDEX = (0.55, 1.0, 0.35, 0.95)
MARKER_MIDDLE = (0.35, 0.75, 1.0, 0.95)
MARKER_PINCH_MID = (0.95, 0.95, 0.35, 0.9)
MARKER_VIRTUAL_OBJ = (0.15, 0.95, 0.75, 0.98)
MARKER_EE_TRAIL = (0.2, 0.95, 0.35, 0.85)
