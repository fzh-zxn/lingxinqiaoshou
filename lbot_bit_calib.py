#!/usr/bin/env python3
"""
批头示教标定：已知螺丝世界点 + 左臂拧到位姿 → 批头在左法兰系下的偏移。

用法：
1) 右相机看到板，选有螺孔，记螺丝 3D（像素精修 + 板面深度）
2) 切左臂，手动把批头对准该螺丝
3) 标定：p_bit_ee = R^T (p_screw - t_ee)
4) 其它孔：t_ee' = p_screw' - R p_bit_ee（姿态沿用标定时 rpy）
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from lbot_grasp_utils import euler_rpy_to_matrix


def bit_tip_in_ee_from_teach(
    screw_world: Sequence[float],
    ee_xyz: Sequence[float],
    ee_rpy: Sequence[float],
) -> np.ndarray:
    """
    拧到位时批头 ≈ 螺丝点 → 批头在法兰系坐标。
    p_w = t + R @ p_ee  ⇒  p_ee = R^T (p_w - t)
    """
    p = np.asarray(screw_world, dtype=np.float64).reshape(3)
    t = np.asarray(ee_xyz, dtype=np.float64).reshape(3)
    R = euler_rpy_to_matrix(float(ee_rpy[0]), float(ee_rpy[1]), float(ee_rpy[2]))
    return R.T @ (p - t)


def ee_xyz_for_screw(
    screw_world: Sequence[float],
    ee_rpy: Sequence[float],
    bit_tip_in_ee: Sequence[float],
) -> np.ndarray:
    """目标：批头落到螺丝 → t = p_screw - R @ p_bit_ee。"""
    p = np.asarray(screw_world, dtype=np.float64).reshape(3)
    tip = np.asarray(bit_tip_in_ee, dtype=np.float64).reshape(3)
    R = euler_rpy_to_matrix(float(ee_rpy[0]), float(ee_rpy[1]), float(ee_rpy[2]))
    return p - R @ tip


def save_bit_tip_calib(
    config_path,
    bit_tip_in_ee: Sequence[float],
    *,
    rpy: Optional[Sequence[float]] = None,
    hole_id: Optional[int] = None,
    screw_world: Optional[Sequence[float]] = None,
):
    """写回 robot.bit_tip_in_ee 及可选示教元数据。"""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("需要 PyYAML: pip install pyyaml") from exc
    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    robot = config.setdefault("robot", {})
    tip = [float(bit_tip_in_ee[0]), float(bit_tip_in_ee[1]), float(bit_tip_in_ee[2])]
    robot["bit_tip_in_ee"] = tip
    if rpy is not None and len(rpy) >= 3:
        robot["bit_calib_rpy"] = [float(rpy[0]), float(rpy[1]), float(rpy[2])]
    if hole_id is not None:
        robot["bit_calib_hole_id"] = int(hole_id)
    if screw_world is not None and len(screw_world) >= 3:
        robot["bit_calib_screw_xyz"] = [
            float(screw_world[0]), float(screw_world[1]), float(screw_world[2]),
        ]
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return tip


def load_bit_tip_from_config(config) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """返回 (bit_tip_in_ee, bit_calib_rpy|None)。"""
    robot = (config or {}).get("robot") or {}
    tip = robot.get("bit_tip_in_ee")
    if tip is None or len(tip) < 3:
        return None
    rpy = robot.get("bit_calib_rpy")
    rpy_arr = (
        np.asarray(rpy, dtype=np.float64).reshape(3)
        if rpy is not None and len(rpy) >= 3 else None
    )
    return np.asarray(tip, dtype=np.float64).reshape(3), rpy_arr
