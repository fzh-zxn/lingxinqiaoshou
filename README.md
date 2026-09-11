# LBOT 双臂眼在手抓取 / 放置 / 拧螺丝 Demo

面向 **LBOT 双臂工作站** 的 Python 演示与工程化应用：右臂腕部 Orbbec 相机（眼在手上）完成螺母检测与夹取，配合 RGB-D 找筐 / YOLO 找盘完成放置，并支持右臂看板、左臂批头拧螺丝。UI、视觉、规划与实机运动分层实现，配置统一落在 `grasp_config.yaml`。

| 项目 | 说明 |
|------|------|
| 仓库根目录 | 本目录（`lbot_pick_demo/`） |
| 主入口 | `demo_pick_square.py` → `pick_app/` |
| 配置中心 | `grasp_config.yaml`（**修改后必须重启程序**） |
| 主操作臂 | 右臂（夹取 / 观察 / 看板）；左臂（批头拧螺丝） |
| 灵巧手 | LinkerHand O6 |
| 感知 | YOLOv5 + ROS2 深度话题 `/camera/depth/image_raw` |
| 仿真预览 | MuJoCo MJCF（`robot_model_assets/`） |
| 机器人 SDK | `lbot/` + `lbot/libs/**/liblbot_api.so*` |

---

## 1. 系统架构

```text
┌─────────────────────────────────────────────────────────────┐
│  Orbbec 腕部相机                                              │
│    RGB ──► YOLOv5（螺母 / 盘 / 批头 / 螺丝板）                  │
│    Depth ─► ROS2 `/camera/depth/image_raw`                    │
│              ├─ 螺母表面深度 → 大/中/小分档                     │
│              └─ 筐沿 RGB-D 定位（basket_mode=rgbd）             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  pick_app/（单窗口 UI + Mixin）                                │
│    vision / lock / place / handeye / mj_plan / mj_view / …   │
└───────┬─────────────────────┬───────────────────┬───────────┘
        │                     │                   │
        ▼                     ▼                   ▼
  Plan-once 路点        手眼标定采样/求解      面板与确认交互
  （MuJoCo + IK）              │
        │                     │
        └──────────┬──────────┘
                   ▼
        lbot_arm_controller.py
        （对准 / 夹取 / 观察 / 放下 / 拧螺丝 FSM）
                   │
                   ▼
              lbot/ → liblbot_api（实机）
```

### 1.1 抓放主流水线

```text
Sense（检测）
  → Lock（锁目标 + 深度定档）
  → Plan-once（一次规划出夹取路点）
  → Replay（实机按相同路点夹取）
  → Observe（抬离 → 转腕 → 高观察位 place_view_pose）
  → Align-2（放置目标二次对准）
  → Place（放筐 / 放盘）
  → Return（回位）
```

**Plan-once**：MuJoCo「模型预览」与实机「抓取」共用同一组 waypoints，预览通过即表示实机将按相同路径执行（中间不再重新规划）。

### 1.2 拧螺丝流水线（概要）

```text
右臂看板 / 看批头 → 记录孔位（板系 / 基座系）
  → 左臂对准孔 → 批头尖端标定（bit tip）
  → 下压 / 拧紧保持 → 回退
```

---

## 2. 功能说明

| 能力 | 说明 |
|------|------|
| 眼在手夹取 | 像素伺服对准 → 稳定锁框 → Plan-once → 五指包络（`pinch_grasp_mode: power5`）夹取 |
| 尺寸分档 | 深度量测六角对边 / 标定折线，分为 **大 / 中 / 小**；可按尺寸切换指尖开合与放置格位 |
| 放筐 | 默认 `basket_mode: rgbd`（蓝筐 + 深度沿），格位按尺寸映射（大偏前 / 中居中 / 小偏近，具体见 yaml） |
| 放盘 | YOLO `pan_best.pt`；侧偏与高度与放筐 **分开配置** |
| 拧螺丝 | 板 YOLO / RGB 孔位；批头检测；左臂示教位 + IK；尖端外参 `bit_tip_in_ee` |
| 手眼标定 | UI 内 ChArUco：S1–S12 采样、自动序列、多求解器对比、可选写回 `camera_on_ee` |
| 仿真模式 | `--preview` 不连臂；`--dry-run` 连臂但不下发运动 |

---

## 3. 硬件与运行环境

### 3.1 硬件

- **机械臂**：LBOT 双臂（默认控制器 IP `192.168.10.21`，见 `robot.host`）
- **末端**：LinkerHand O6（右夹取 / 左批头，按配置）
- **相机**：腕部 Orbbec（如 Gemini2），RGB + 深度；深度经 ROS2 发布
- **标定板**：手眼用 ChArUco；螺丝板孔位可按 `detection.screw_board.holes` 布局

### 3.2 软件依赖

- Python **3.10+**（团队常用 conda 环境名 `lxqs`，Python 3.12）
- UI：`requirements-ui.txt`（`customtkinter`、`Pillow`）
- 核心：`numpy`、`opencv-python`、`PyYAML`、`torch`、YOLOv5、`mujoco`
- 原生库：仓库已携带 `lbot/libs/linux/{linux_x64,linux_arm64}/liblbot_api.so*`

### 3.3 运行注意

- **不要**与 `demo_camera` 等其它深度发布节点同时抢同一话题。
- 软限位、左臂到位公差等安全相关项请勿随意改大；现场已调参数以 yaml 为准。
- 改 `grasp_config.yaml` 后必须 **退出并重新启动** demo，热加载不保证完整生效。

---

## 4. 安装与准备

```bash
cd lbot_pick_demo

# 建议使用独立环境
conda create -n lxqs python=3.12 -y
conda activate lxqs

pip install -r requirements-ui.txt
pip install numpy opencv-python pyyaml pillow mujoco torch
# YOLOv5 按团队现有方式安装 / 链接到 PYTHONPATH
```

### 4.1 检测权重（不进 Git）

`*.pt` 体积较大，已由 `.gitignore` 排除。请将权重放在工程根目录（或与 yaml 中路径一致）：

| 文件 | 用途 | 配置键 |
|------|------|--------|
| `single.pt` | 螺母（主夹取） | CLI `--model` / 代码默认 |
| `pan_best.pt` | 盘（放置） | `detection.place_model` |
| `刀_best(2).pt` | 电动批头 | `detection.driver_model` |
| `板_best.pt` | 螺丝板 | `detection.board_model` |

筐默认走 RGB-D，不依赖筐 YOLO（`basket_model: none`）。

---

## 5. 启动与命令行参数

### 5.1 常用启动方式

```bash
# 实机（推荐：带确认）
python demo_pick_square.py --pick --confirm

# 连接机器人但不发运动指令（联调 / 验 UI）
python demo_pick_square.py --dry-run

# 不连接机械臂，仅 UI + MuJoCo
python demo_pick_square.py --preview

# 无深度（能力受限，仅排障用）
python demo_pick_square.py --no-depth
```

### 5.2 参数一览

| 参数 | 默认 | 说明 |
|------|------|------|
| `--config` | `grasp_config.yaml` | 配置文件路径 |
| `--model` | `single.pt` | 螺母 YOLO 权重 |
| `--place-model` | 读 yaml | 放置盘 YOLO；`none` / `-` 关闭 |
| `--basket-model` | 读 yaml | `rgbd` / `none` / 或 YOLO 路径 |
| `--driver-model` | 读 yaml | 批头 YOLO；`none` / `-` 关闭 |
| `--board-model` | 读 yaml | 螺丝板 YOLO；`none` / `-` 关闭 |
| `--device` | `cuda:0` | 推理设备 |
| `--camera` | 工程默认相机 id | RGB 设备 |
| `--depth-topic` | `/camera/depth/image_raw` | ROS2 深度话题 |
| `--no-depth` | off | 禁用深度 |
| `--width` / `--height` | 640 / 480 | 采集分辨率 |
| `--imgsz` | 640 | YOLO 输入边长 |
| `--preview` | off | 不连臂 |
| `--dry-run` | off | 连臂但不运动 |
| `--pick` | off | 稳定对准后自动抓取一次 |
| `--confirm` | off | 关键运动前确认交互 |

其它辅助脚本：

```bash
python demo_sim_home.py                 # 仿真回零 / 模型检查
python demo_find_basket_rgbd.py         # 单独调试找筐
python calibrate_handeye_charuco.py     # 手眼独立求解 / 写回
python calibrate_board_holes.py         # 螺丝板孔位相关标定
```

---

## 6. 操作流程（现场）

### 6.1 抓取 → 放置

1. 启动深度节点与本程序；确认画面有螺母框与深度读数。
2. 勾选追踪尺寸（大 / 中 / 小）与放置检（筐 / 盘）。
3. **对准**（或流程内自动对准）→ **模型预览** → **抓取**。
4. **观察**（抬到 `place_view_pose`）→ **对准2** → **放下**。
5. 若落点系统性偏右/偏左，只调对应 bias，不要混用筐/盘两套参数。

### 6.2 拧螺丝

1. 右臂切到 **看板** / **看批头** 位姿，确认板与孔可见。
2. 记录目标孔（UI / 标定孔 id，见 `bit_calib_hole_id` 等）。
3. 左臂对准 → 批头尖端标定 → **去拧**（下压量、保持时间见 `robot.screw_drive_*`）。
4. 底栏状态与日志用于判断到位公差与 IK 是否被软限位拒绝。

### 6.3 UI 分区（概念）

- **顶栏第一行**：模式切换、停止/解锁、常用运动与确认。
- **顶栏第二行**：手眼标定开关与采样位 S1–S12、自动采、求解、写回。
- **主区**：相机画面 + MuJoCo 预览；忙碌时 YOLO 周期会降频以保 UI。

---

## 7. 配置要点（`grasp_config.yaml`）

配置体量大，下列为日常最常改、且影响现场精度的项。完整键名以文件为准。

### 7.1 机器人与安全

| 键 | 含义 |
|----|------|
| `robot.host` | 控制器 IP |
| `robot.speed` / `accel` | 全局速度、加速度 |
| `robot.soft_joint_limits_deg` | 软关节限位（**勿擅自放宽**） |
| `robot.left_arrive_tol_m` | 左臂笛卡尔到位公差（现场常用极紧） |
| `robot.left_speed_max` / `left_accel_max` | 左臂速度上限 |
| `robot.default_home_joints` | 右臂默认关节回零 |
| `robot.*_pose` / `screw_*` | 观察、看板、拧螺丝等笛卡尔位姿 |

### 7.2 眼在手对准与深度

| 键 | 含义 |
|----|------|
| `eye_in_hand.pixel_tolerance_px` | 对准像素容差 |
| `eye_in_hand.target_distance_mm` | 目标工作距离 |
| `eye_in_hand.depth_roi_mode` | 深度 ROI（如 `nut_surface`） |
| `eye_in_hand.pinch_skip_pregrasp` | 是否跳过预抓取段 |
| `eye_in_hand.pinch_path_mode` | 夹取路径模式（如 `mj_ik_smooth`） |
| `eye_in_hand.camera_on_ee` | 相机相对法兰外参（手眼结果写回此处） |

### 7.3 观察与放置

| 键 | 含义 |
|----|------|
| `eye_in_hand.place_view_pose` | 高观察位（相机朝向与夹取腕不同） |
| `eye_in_hand.place_view_orient_steps` | 转腕分段数 |
| `eye_in_hand.place_tip_standoff_m` / `place_tip_clearance_m` | **放筐** 接近高 / 松手高 |
| `eye_in_hand.plate_tip_standoff_m` / `plate_tip_clearance_m` | **放盘** 接近高 / 松手高 |
| `eye_in_hand.place_bias_right_m` / `place_bias_forward_m` | **放筐** 侧偏（正=右 / 前） |
| `eye_in_hand.plate_bias_right_m` / `plate_bias_forward_m` | **放盘** 侧偏（与筐独立） |
| `eye_in_hand.pinch_hand_short_m` | 夹取沿接近方向“手短”补偿 |

若观察位已低于放置接近高度，控制器会避免无意义的先抬升再下降（以当前 `lbot_arm_controller` 逻辑为准）。

### 7.4 检测与分档

| 键 | 含义 |
|----|------|
| `detection.conf` / `place_conf` | 螺母 / 盘置信度阈值 |
| `detection.basket_mode` | 通常为 `rgbd` |
| `detection.basket_rgbd.*` | 筐沿高度差、长宽先验、处理周期等 |
| `detection.screw_board.*` | 板检测、孔布局、对准距离等 |
| `detection.size_metric.*` | 大中小阈值、时序滤波、标定折线 |

### 7.5 相机内参

`camera_intrinsics`（`fx/fy/cx/cy`、分辨率）须与实机标定一致；内参流程见 `calib_intrinsics/`。

---

## 8. 目录结构

```text
lbot_pick_demo/
├── README.md                      # 本说明
├── grasp_config.yaml              # 唯一主配置
├── demo_pick_square.py            # 薄入口
├── requirements-ui.txt
├── pick_app/                      # 应用层
│   ├── main.py / app.py
│   ├── helpers.py                 # CLI
│   ├── constants.py / deps.py
│   └── mixins/                    # 按职责拆分
│       ├── vision_mixin.py        # 检测、深度、画面
│       ├── lock_mixin.py          # 锁目标
│       ├── place_mixin.py         # 放置
│       ├── handeye_mixin.py       # 手眼 UI 与采样
│       ├── mj_plan_mixin.py       # Plan-once
│       ├── mj_view_mixin.py       # MuJoCo 显示
│       ├── actions_mixin.py       # 按钮动作
│       ├── panel_mixin.py / ui_mixin.py
│       └── …
├── lbot_arm_controller.py         # 实机运动与抓放 FSM（大文件）
├── lbot_grasp_utils.py            # 深度、定档、外参工具函数
├── lbot_pinch_plan.py             # 夹取路点
├── lbot_pick_plan.py
├── lbot_basket_rgbd.py            # 找筐
├── lbot_screw_board.py            # 板 / 孔
├── lbot_bit_calib.py / lbot_driver_bit.py
├── lbot_mj_scene.py / lbot_mj_hand.py
├── lbot_mock_robot.py             # 预览用假机器人
├── calibrate_handeye_charuco.py
├── calibrate_board_holes.py
├── calib_handeye/                 # 手眼样本与说明
├── calib_intrinsics/              # 内参结果
├── lbot/                          # API 封装 + .so
├── robot_model_assets/            # MJCF 与网格
└── docs/                          # 运行说明、报告、交接材料
```

| 要改什么 | 优先看 |
|----------|--------|
| IP、位姿、侧偏、高度 | `grasp_config.yaml` |
| 按钮与顶栏布局 | `pick_app/mixins/ui_mixin.py` |
| 手眼采样逻辑 | `pick_app/mixins/handeye_mixin.py` |
| 视觉 / 筐 RGBD | `pick_app/mixins/vision_mixin.py` |
| 夹取与放下轨迹 | `lbot_arm_controller.py` |
| 深度定档公式 | `lbot_grasp_utils.py` |

`_archive/` 为调试与历史备份，默认不纳入版本库日常使用路径。

---

## 9. 标定

### 9.1 相机内参

步骤见 `calib_intrinsics/README.txt`。结果写入 `calib_intrinsics/`，再同步到 yaml 的 `camera_intrinsics`。

### 9.2 手眼外参（推荐在主程序内完成）

```bash
python demo_pick_square.py --pick --confirm
```

1. 标定板固定于桌面；切到右臂（相机）模式。  
2. 勾选「手眼标定」，出现 S1–S12。  
3. **单击** S*：运动到该位，再点「记录姿态」。  
   **右键** S*：到位后自动记录。  
   「自动采12组」可一键跑完（可用停止/解锁中断）。  
4. 速度自动封顶、走安全路径；到位后静置约 **0.5 s** 再读实机法兰位姿。  
5. 「求解手眼」会跑多组求解器并对比：  
   - `tsai` / `park`：经典 AX=XB  
   - `fixR_yaml+t_*`：固定 yaml 旋转，只解平移  
   - `park+board_refine`：Park 初值 + 板系一致性精修（通常优先）  
6. 以 **板在基座系重合 RMS** 与 AX 残差为准；仅当残差可接受时再「写回外参」（只改 `camera_on_ee`，不动 `plate_bias`）。  
7. 经验阈值：板 RMS &lt; 5 mm 且 AX 旋转 &lt; 1°、平移 &lt; 5 mm 较理想；**&gt; 10 mm 不要轻易写回**。  

独立脚本：

```bash
python calibrate_handeye_charuco.py --solve-only
python calibrate_handeye_charuco.py --solve-only --write-yaml   # 确认残差后再用
```

### 9.3 批头尖端 / 板孔

- 批头：`robot.bit_tip_in_ee`、`bit_calib_*`  
- 板孔布局：`detection.screw_board.holes`；可用 `calibrate_board_holes.py` 辅助  

---

## 10. 常见问题

| 现象 | 排查方向 |
|------|----------|
| 无深度 / 深度乱跳 | ROS2 话题是否唯一发布；`depth_roi_mode` 与 ROI 是否对准螺母表面 |
| 对准抖动或不收敛 | `pixel_tolerance_px`、工作距离、外参是否过期；灯光与反光 |
| 预览与实机差很大 | 是否未走 Plan-once；关节符号 `model_arm_joint_signs`；外参与内参 |
| 放偏一侧 | 分清筐 `place_bias_*` 与盘 `plate_bias_*`；先小步改侧偏 |
| 放下先莫名抬高 | 观察位 Z 是否已低于 standoff；对照控制器放置段逻辑 |
| IK / 软限位报警 | 查 `soft_joint_limits_deg` 与目标位；左臂用更低速与更紧到位公差 |
| 手眼写回后更差 | 样本姿态多样性不足；对比 board RMS，必要时只保留更好的求解器结果 |
| GPU / YOLO 失败 | `--device cpu` 试验；权重路径与类名是否匹配 |

排障时可临时打开 yaml 中的 `verbose_logs` / `pinch_debug`，用完关闭。

---

## 11. 版本管理说明

- 本仓库默认忽略：`*.pt`、`_archive/`、缓存与大量标定中间图。  
- 勿提交内网密码、Token；`robot.host` 可按发布需要改成示例 IP。  
- 厂商 API（`lbot/`）遵循灵心巧手 / LBOT 授权；YOLO 权重版权归训练方，需自行保管与分发。

---

## 12. 相关文档

| 文档 | 内容 |
|------|------|
| `docs/运行说明.txt` | 精简现场 checklist |
| `docs/目录整理说明.txt` | 目录归档约定 |
| `calib_handeye/README.txt` | 手眼标定步骤与求解器说明 |
| `calib_intrinsics/README.txt` | 内参标定 |
| `docs/` 下报告 / 心得 | 设计说明与交接材料 |

---

## 13. 许可证与归属

- 机械臂控制库与 `.so`：遵循 LBOT / 灵心巧手官方授权与文档。  
- 本目录业务代码（`pick_app/`、`lbot_*.py` 等）：按团队约定标注作者与用途。  
- 检测权重：不随本仓库分发，版权与使用权由训练/提供方负责。  

