"""Application entry: load models, open PickMonitorApp."""
from __future__ import annotations

from pick_app.deps import *
from pick_app.constants import *
from pick_app.helpers import parse_args, find_mjcf, resolve_basket_mode
from pick_app.app import PickMonitorApp

def main():
    args = parse_args()
    config = load_grasp_config(args.config)
    mjcf = find_mjcf()
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
    elif mjcf is not None:
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
    else:
        print("警告: 未找到 workstation.mjcf，相机外参沿用 yaml", flush=True)

    if not args.model.is_file():
        raise FileNotFoundError(f"找不到 YOLO 模型: {args.model}")

    print("正在加载 YOLO（可能需要几秒）…", flush=True)
    yolo = load_yolo_model(args.model, args.device, config["detection"]["conf"])
    print(
        f"YOLO 抓取就绪: {args.model.resolve()}  "
        f"classes={yolo_class_names_brief(yolo)}  "
        f"conf={float(config['detection']['conf']):.2f}",
        flush=True,
    )

    place_path = resolve_place_model_path(config, args.place_model)
    yolo_place = None
    if place_path is not None:
        if not place_path.is_file():
            print(f"警告: 放置模型不存在 {place_path}，关闭放筐", flush=True)
        else:
            place_conf = float(
                config["detection"].get(
                    "place_conf", config["detection"]["conf"]
                )
            )
            print(f"正在加载放置 YOLO: {place_path} …", flush=True)
            yolo_place = load_yolo_model(
                place_path, args.device, place_conf,
            )
            print(
                f"YOLO 放置就绪: {place_path.resolve()}  "
                f"classes={yolo_class_names_brief(yolo_place)}  "
                f"conf={place_conf:.2f}",
                flush=True,
            )
            config.setdefault("eye_in_hand", {})
            if "place_enabled" not in config["eye_in_hand"]:
                config["eye_in_hand"]["place_enabled"] = True
    else:
        print("放置模型关闭（place_model=none）", flush=True)
        config.setdefault("eye_in_hand", {})["place_enabled"] = False

    driver_path = resolve_driver_model_path(
        config, getattr(args, "driver_model", None),
    )
    yolo_driver = None
    if driver_path is not None:
        if not driver_path.is_file():
            print(f"警告: 电批模型不存在 {driver_path}，已关闭", flush=True)
        else:
            driver_conf = float(
                config["detection"].get(
                    "driver_conf",
                    config["detection"].get("place_conf", 0.35),
                )
            )
            print(f"正在加载电批 YOLO: {driver_path} …", flush=True)
            yolo_driver = load_yolo_model(
                driver_path, args.device, driver_conf,
            )
            print(
                f"YOLO 电批就绪: {driver_path.resolve()}  "
                f"classes={yolo_class_names_brief(yolo_driver)}  "
                f"conf={driver_conf:.2f}",
                flush=True,
            )
    else:
        print("电批检测关闭（driver_model=none）", flush=True)

    board_path = resolve_board_model_path(
        config, getattr(args, "board_model", None),
    )
    yolo_board = None
    if board_path is not None:
        if not board_path.is_file():
            print(f"警告: 螺丝板模型不存在 {board_path}，回退 RGB", flush=True)
        else:
            board_conf = float(
                config["detection"].get(
                    "board_conf",
                    config["detection"].get("driver_conf", 0.35),
                )
            )
            print(f"正在加载螺丝板 YOLO: {board_path} …", flush=True)
            yolo_board = load_yolo_model(
                board_path, args.device, board_conf,
            )
            print(
                f"YOLO 螺丝板就绪: {board_path.resolve()}  "
                f"classes={yolo_class_names_brief(yolo_board)}  "
                f"conf={board_conf:.2f}",
                flush=True,
            )
    else:
        print("螺丝板 YOLO 关闭（board_model=none）；可用 RGB 检测", flush=True)

    basket_mode = resolve_basket_mode(config, args.basket_model)
    yolo_basket = None
    basket_rgbd = False
    if basket_mode == "rgbd":
        basket_rgbd = True
        br = (config.get("detection") or {}).get("basket_rgbd") or {}
        print(
            "筐子检测: RGB+深度 OBB（替代 YOLO）  "
            f"rimΔ≥{float(br.get('min_rim_delta_mm', 20)):.0f}mm  "
            f"use_for_place={bool(br.get('use_for_place', True))}",
            flush=True,
        )
        config.setdefault("eye_in_hand", {})
        if "place_enabled" not in config["eye_in_hand"]:
            config["eye_in_hand"]["place_enabled"] = True
        elif not config["eye_in_hand"].get("place_enabled", False):
            # 仅 RGBD 筐时也要开放置管线
            if bool(br.get("use_for_place", True)):
                config["eye_in_hand"]["place_enabled"] = True
    elif basket_mode == "yolo":
        basket_path = resolve_basket_model_path(config, args.basket_model)
        if basket_path is not None and basket_path.is_file():
            basket_conf = float(
                config["detection"].get(
                    "basket_conf",
                    config["detection"].get("place_conf", 0.30),
                )
            )
            print(f"正在加载筐子 YOLO: {basket_path} …", flush=True)
            yolo_basket = load_yolo_model(
                basket_path, args.device, basket_conf,
            )
            print(
                f"YOLO 筐子就绪: {basket_path.resolve()}  "
                f"classes={yolo_class_names_brief(yolo_basket)}  "
                f"conf={basket_conf:.2f}",
                flush=True,
            )
        else:
            print(f"警告: 筐子 YOLO 不可用 ({basket_path})，已关闭", flush=True)
    else:
        print("筐子检测关闭（basket_mode=none）", flush=True)

    robot = None
    if not args.preview:
        print(f"正在连接机械臂 {config['robot']['host']} …", flush=True)
        robot = LbotRobot(config["robot"]["host"])
        if not robot.connect():
            print("连接失败：将以离线预览继续（无末端位姿）", flush=True)
            robot = None
        else:
            print("机械臂已连接", flush=True)
    else:
        print("预览模式：不连接机械臂", flush=True)

    controller = RightArmController(
        config,
        robot,
        dry_run=args.dry_run or args.preview or robot is None,
        config_path=args.config,
    )

    print("正在打开监控窗口…", flush=True)
    root = create_root(
        title="LBOT 控制 / 模型 / 运动",
        geometry="1280x860",
        minsize=(1000, 680),
    )
    print(font_status_line(), flush=True)
    PickMonitorApp(
        root, args, config, controller, yolo,
        yolo_place=yolo_place, yolo_basket=yolo_basket,
        yolo_driver=yolo_driver, yolo_board=yolo_board,
        basket_rgbd=basket_rgbd,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
