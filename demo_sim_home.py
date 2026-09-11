#!/usr/bin/env python3
"""
在家离线仿真：假装机械臂已连接，打开与 demo_pick_square 相同的监控界面。

不连真机；关节/位姿由 MuJoCo 假机械臂提供。
适合看模型、点动关节；（有摄像头时）也可试模型预览夹取。

用法（在本目录）：
  python demo_sim_home.py
  python demo_sim_home.py --device cpu
  python demo_sim_home.py --camera /dev/video0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_pick_square import (  # noqa: E402
    PickMonitorApp,
    RightArmController,
    find_mjcf,
    parse_args,
)
from lbot_grasp_utils import (  # noqa: E402
    apply_mjcf_camera_to_config,
    load_grasp_config,
    load_yolo_model,
    save_camera_on_ee_translation,
)
from lbot_mock_robot import MockLbotRobot  # noqa: E402
from lbot_ui_font import font_status_line  # noqa: E402
from lbot_ui_theme import create_root  # noqa: E402


def main():
    if "--no-depth" not in sys.argv:
        sys.argv.append("--no-depth")

    args = parse_args()
    args.preview = False
    args.dry_run = False
    args.sim = True

    config = load_grasp_config(args.config)
    mjcf = find_mjcf()
    if mjcf is None:
        raise FileNotFoundError("找不到 workstation.mjcf，请确认 robot_model_assets 完整")

    prefer_yaml = bool(
        (config.get("eye_in_hand") or {}).get("camera_on_ee_prefer_yaml", False)
    )
    if prefer_yaml:
        t = (config.get("eye_in_hand") or {}).get("camera_on_ee", {}).get(
            "translation"
        )
        print(
            f"相机外参沿用手调 yaml（camera_on_ee_prefer_yaml=true）→ {t}",
            flush=True,
        )
    else:
        try:
            apply_mjcf_camera_to_config(config, mjcf)
            save_camera_on_ee_translation(
                args.config,
                np.array(config["eye_in_hand"]["camera_on_ee"]["translation"]),
                prefer_yaml=False,
            )
            print(
                f"相机外参已从模型加载: {mjcf.name} → "
                f"{config['eye_in_hand']['camera_on_ee']['translation']}",
                flush=True,
            )
        except Exception as error:
            print(f"警告: 未能从 MJCF 读相机外参 ({error})，沿用 yaml", flush=True)

    if not args.model.is_file():
        raise FileNotFoundError(f"找不到 YOLO 模型: {args.model}")

    print("正在加载 YOLO（可能需要几秒）…", flush=True)
    yolo = load_yolo_model(args.model, args.device, config["detection"]["conf"])
    print(f"YOLO 就绪: {args.model.resolve()}", flush=True)

    print("正在创建假机械臂（MuJoCo IK/FK，不连实机）…", flush=True)
    robot = MockLbotRobot(config, mjcf)
    if not robot.connect():
        raise RuntimeError("假机械臂初始化失败")
    print("假机械臂已就绪（仿真模式）", flush=True)

    controller = RightArmController(
        config,
        robot,
        dry_run=False,
        config_path=args.config,
    )

    print("正在打开监控窗口…", flush=True)
    root = create_root(
        title="LBOT 仿真（假连接）",
        geometry="1320x980",
        minsize=(1100, 800),
    )
    print(font_status_line(), flush=True)
    PickMonitorApp(root, args, config, controller, yolo)
    root.mainloop()


if __name__ == "__main__":
    main()
