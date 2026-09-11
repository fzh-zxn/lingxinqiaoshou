"""
Plan-once / Replay：锁物后一次规划，预览与实机共用同一份关节路点。

PickPlan 是一等公民：几何目标 + seed_joints + waypoints_6/7。
执行端只回放；禁止在 Execute 里重新 _plan_mj_ik（seed 漂移时仅 1 段 bridge）。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PickPlan:
    """锁物后产出的不可变抓取计划（几何 + 可回放关节轨迹）。"""

    seed_joints: List[float]
    obj_world: np.ndarray
    obj_target: np.ndarray
    T_ee_des: np.ndarray
    T_ee_contact: np.ndarray
    waypoints_6: List[List[float]]
    waypoints_7: List[List[float]]
    hand_open: List[int]
    hand_close: List[int]
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    # 兼容旧 execute_pick_pinch / TCP 诊断：完整几何 dict
    geom: Dict[str, Any] = field(default_factory=dict)

    def as_exec_dict(self) -> Dict[str, Any]:
        """供 controller / 诊断使用的 plan dict（含 waypoints）。"""
        d = dict(self.geom) if self.geom else {}
        d.update(
            {
                "obj_world": np.asarray(self.obj_world, dtype=np.float64).reshape(3),
                "obj_target": np.asarray(self.obj_target, dtype=np.float64).reshape(3),
                "T_ee_des": np.asarray(self.T_ee_des, dtype=np.float64).reshape(4, 4),
                "T_ee_contact": np.asarray(
                    self.T_ee_contact, dtype=np.float64
                ).reshape(4, 4),
                "seed_joints": [float(x) for x in self.seed_joints[:7]],
                "waypoints_6": [list(w) for w in self.waypoints_6],
                "waypoints_7": [list(w) for w in self.waypoints_7],
                "hand_open": list(self.hand_open),
                "hand_close": list(self.hand_close),
                "grasp_close_hand": list(self.hand_close),
                "pick_plan_metrics": dict(self.metrics),
                "pick_plan_created_at": float(self.created_at),
                "plan_once": True,
            }
        )
        return d


def joint_max_delta_deg(a, b) -> float:
    """两组 7 关节最大角差（度）。"""
    if a is None or b is None:
        return float("inf")
    aa = list(a)[:7]
    bb = list(b)[:7]
    if len(aa) < 7 or len(bb) < 7:
        return float("inf")
    return max(
        abs(math.degrees(float(x) - float(y))) for x, y in zip(aa, bb)
    )


def seed_drift_status(live_joints, seed_joints, tol_deg: float, abort_deg: float):
    """
    回放前种子检查。
    返回 (ok_or_bridge, delta_deg, action)：
      action = "ok" | "bridge" | "abort"
    """
    d = joint_max_delta_deg(live_joints, seed_joints)
    if d <= float(tol_deg):
        return True, d, "ok"
    if d <= float(abort_deg):
        return True, d, "bridge"
    return False, d, "abort"


def compute_contact_T(
    T_ee_des,
    obj_world,
    obj_target,
    final_m: float,
) -> Optional[np.ndarray]:
    """
    ⑦ 接触位：保持 ⑥ 姿态，沿黄→青轴推进 final_m。
    基点用规划 ⑥ tool0（Plan-once 时无 drift）。
    """
    if T_ee_des is None or obj_world is None:
        return None
    T6 = np.asarray(T_ee_des, dtype=np.float64).reshape(4, 4)
    obj = np.asarray(obj_world, dtype=np.float64).reshape(3)
    final_m = float(final_m)
    if final_m < 1e-6:
        return T6.copy()

    u = None
    if obj_target is not None:
        yellow = np.asarray(obj_target, dtype=np.float64).reshape(3)
        axis = obj - yellow
        an = float(np.linalg.norm(axis))
        if an > 1e-6:
            u = axis / an
    if u is None:
        # 回退：从 tool0 指向青球
        p0 = T6[:3, 3]
        axis = obj - p0
        an = float(np.linalg.norm(axis))
        if an < 1e-6:
            return T6.copy()
        u = axis / an

    T7 = np.eye(4, dtype=np.float64)
    T7[:3, :3] = T6[:3, :3]
    T7[:3, 3] = T6[:3, 3] + u * final_m
    return T7


def n_steps_for_travel(travel_m: float, step_m: float, cap: int) -> int:
    """与 _exec_mj_ik_smooth_to_T 相同的路点密度策略。"""
    travel = float(travel_m)
    step = max(float(step_m), 1e-4)
    cap = int(np.clip(int(cap), 1, 16))
    if travel < 0.012:
        return 1
    if travel < 0.04:
        return min(2, cap)
    return int(np.clip(math.ceil(travel / max(step, 0.02)), 2, cap))


def pick_plan_from_geom(
    geom: dict,
    seed_joints,
    hand_open,
    hand_close,
    waypoints_6,
    waypoints_7,
    T_ee_contact,
    metrics=None,
) -> Optional[PickPlan]:
    """由 compute_pinch_plan_mj 几何 + 路点组装 PickPlan。"""
    if geom is None:
        return None
    T_ee = geom.get("T_ee_des")
    obj_w = geom.get("obj_world")
    obj_t = geom.get("obj_target")
    if T_ee is None or obj_w is None or obj_t is None:
        return None
    if T_ee_contact is None:
        return None
    return PickPlan(
        seed_joints=[float(x) for x in list(seed_joints)[:7]],
        obj_world=np.asarray(obj_w, dtype=np.float64).reshape(3).copy(),
        obj_target=np.asarray(obj_t, dtype=np.float64).reshape(3).copy(),
        T_ee_des=np.asarray(T_ee, dtype=np.float64).reshape(4, 4).copy(),
        T_ee_contact=np.asarray(T_ee_contact, dtype=np.float64).reshape(4, 4).copy(),
        waypoints_6=[list(w) for w in (waypoints_6 or [])],
        waypoints_7=[list(w) for w in (waypoints_7 or [])],
        hand_open=[int(x) for x in list(hand_open)[:6]],
        hand_close=[int(x) for x in list(hand_close)[:6]],
        metrics=dict(metrics or {}),
        created_at=time.time(),
        geom=dict(geom),
    )
