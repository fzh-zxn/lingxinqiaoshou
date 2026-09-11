# LBOT 双臂抓取 / 放置 / 拧螺丝 Demo

基于 **LBOT 机械臂** + **LinkerHand O6** + **腕部 Orbbec（眼在手上）** 的 Python 演示工程：YOLO 检螺母/盘/批头/板，RGB+深度找筐，MuJoCo 规划与 UI 监控。

> 主入口目录：本仓库根目录（`lbot_pick_demo/`）  
> 配置中心：`grasp_config.yaml`（改完需重启程序）

---

## 功能概览

| 能力 | 说明 |
|------|------|
| 眼在手抓取 | 右臂相机对准 → 锁物 → Plan-once → 五指包络夹取 |
| 放置 | 观察 → 对准2 → 放盘 / 放筐（可人工侧偏） |
| 拧螺丝 | 右臂看板 / 看批头；左臂批头去拧（示教位 + IK） |
| 手眼标定 | UI 内 ChArUco 采样 + Tsai/Park / 板系精修 |
| 仿真预览 | MuJoCo 模型预览轨迹；`--preview` / `--dry-run` |

流水线（抓放）：

```text
Sense → Lock → Plan-once → Replay(夹取) → 观察 → 对准2 → 放下 → 回位
```

---

## 硬件与环境

- **机械臂**：LBOT（默认 `robot.host: 192.168.10.21`，见 yaml）
- **手**：LinkerHand O6（右夹取 / 左批头按配置）
- **相机**：Orbbec Gemini2，ROS2 深度话题 `/camera/depth/image_raw`
- **软件**：Python 3.10+ 建议；`numpy` / `opencv-python` / `pillow` / `mujoco` / `yolov5` / `torch`；UI 见 `requirements-ui.txt`
- **注意**：运行本程序时 **不要同时开** `demo_camera`（抢相机话题）

---

## 快速开始

```bash
# 1. 进入工程
cd lbot_pick_demo

# 2. （建议）虚拟环境
conda create -n lxqs python=3.12 -y   # 或 venv
conda activate lxqs
pip install -r requirements-ui.txt
# 另按环境安装：numpy opencv-python pillow mujoco torch yolov5 pyyaml …

# 3. 准备权重（默认不进 Git，见下文）
#    将 single.pt / pan_best.pt / 刀_best(2).pt / 板_best.pt 等放到本目录
#    路径与类名以 grasp_config.yaml → detection 为准

# 4. 启动
python demo_pick_square.py --pick --confirm   # 实机常用
python demo_pick_square.py --dry-run          # 连臂不发运动
python demo_pick_square.py --preview          # 不连臂，只看 UI/模型
```

**建议操作顺序（抓放）**

1. 勾选追踪尺寸（大/中/小）与放置检（筐/盘）  
2. 「对准」或夹取流程内对准 → 「模型预览」→ 「抓取」  
3. 「观察」→ 「对准2」→ 「放下」  

**拧螺丝**：右臂「看板」→ 记螺丝 → 左臂对准孔 → 标定批头 → 「去拧」。详见 UI 底栏提示与 `docs/`。

---

## 目录结构

```text
lbot_pick_demo/
├── README.md                 # 本说明
├── grasp_config.yaml         # 主配置（IP、外参、偏置、位姿…）
├── demo_pick_square.py       # 薄入口 → pick_app
├── pick_app/                 # UI / 视觉 / 规划 Mixin
├── lbot_arm_controller.py    # 实机运动、夹取/放下 FSM
├── lbot_grasp_utils.py       # 深度、定档、外参工具
├── lbot_pinch_plan.py / lbot_pick_plan.py / lbot_basket_rgbd.py
├── calibrate_handeye_charuco.py
├── calib_handeye/            # 手眼样本与说明
├── calib_intrinsics/         # 内参标定结果
├── lbot/                     # liblbot_api Python 封装
├── robot_model_assets/       # MuJoCo MJCF
├── requirements-ui.txt
└── docs/                     # 补充文档
```

| 要改什么 | 去哪 |
|----------|------|
| IP / 观察位 / 侧偏 / 松手高 | `grasp_config.yaml` |
| UI 布局、手眼按钮 | `pick_app/mixins/ui_mixin.py` / `handeye_mixin.py` |
| 视觉、筐 RGBD | `pick_app/mixins/vision_mixin.py` |
| 夹取/放下运动 | `lbot_arm_controller.py` |

---

## 配置要点（摘录）

改 yaml 后必须 **重启** demo。

| 项 | 配置键 | 说明 |
|----|--------|------|
| 机器人 IP | `robot.host` | 局域网地址 |
| 放筐侧偏 | `eye_in_hand.place_bias_right_m` 等 | 正=往右 |
| 放盘侧偏 | `eye_in_hand.plate_bias_right_m` 等 | 与筐分开调 |
| 放盘高度 | `plate_tip_standoff_m` / `plate_tip_clearance_m` | 接近高 / 松手高 |
| 相机外参 | `eye_in_hand.camera_on_ee` | 手调 yaml；手眼标定可写回 |
| 低/高观察、看板等 | `robot.*_pose` / `place_view_pose` | 笛卡尔或关节 |

手眼标定：UI 勾选「标定模式」→ S1–S12 或自动采 →「求解」（勿轻易「写回」除非板 RMS 小）。说明见 `calib_handeye/README.txt`。

---

## 权重与大文件

`*.pt` 体积大（每个约十余 MB），**默认已加入 `.gitignore`，不会推上 GitHub**。

请自行保留本地权重，或使用 [Git LFS](https://git-lfs.com/) / 网盘另传，并在本 README 注明获取方式。

上库前请确认：

- 不要提交内网密码、私人 token  
- `grasp_config.yaml` 里的 `robot.host` 可按需改成示例 IP  
- `_archive/`、标定大量 `npz`/图片建议忽略或精简后再推  

---

## 上传到 GitHub（简明）

详细逐步说明见下文「完整上传教程」。精简版：

```bash
cd lbot_pick_demo
git init
git add .
git status          # 确认没有 *.pt / 密钥
git commit -m "Initial commit: LBOT pick/place/screw demo"
# 在 GitHub 新建空仓库后：
git remote add origin git@github.com:<你的用户名>/<仓库名>.git
git branch -M main
git push -u origin main
```

---

## 许可证与归属

- 机械臂 API（`lbot/`、厂商 `api_lk73_*`）遵循灵心巧手 / LBOT 官方授权与文档。  
- 本 demo 业务代码请按团队约定标注作者与用途。  
- YOLO 权重版权归训练方，分发需自行负责。

---

## 完整上传教程（GitHub）

### 0. 准备

1. 安装 [Git](https://git-scm.com/)：`git --version`  
2. 注册 [GitHub](https://github.com/) 账号  
3. 推荐配置 SSH 密钥（一次配置，以后免密推送）：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# 一路回车即可；然后把 ~/.ssh/id_ed25519.pub 内容粘到
# GitHub → Settings → SSH and GPG keys → New SSH key
ssh -T git@github.com   # 成功会提示 Hi <username>!
```

也可用 HTTPS + [Personal Access Token](https://github.com/settings/tokens)（推送时密码处填 token）。

### 1. 只上传本 demo 目录（推荐）

本说明默认仓库根目录 = `lbot_pick_demo/`（不要把整个 `lbot_arm_api-master` 里相机录像 zip、多余 `.pt` 一并推上去）。

```bash
cd /path/to/lbot_pick_demo
```

若该目录还没有 Git：

```bash
git init
```

确认已有 `.gitignore`（本仓库已提供）。再检查将要提交的内容：

```bash
git status
git check-ignore -v single.pt   # 应显示被忽略
```

**不要** `git add -f *.pt`，除非你已启用 Git LFS 且清楚配额。

### 2. 首次提交

```bash
git add .
git status                      # 再扫一眼：无 .pt、无 __pycache__、无巨大 zip
git commit -m "Initial commit: LBOT eye-in-hand pick/place/screw demo"
```

若提示配置用户名邮箱（仅本机一次）：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

### 3. 在 GitHub 建空仓库

1. 打开 https://github.com/new  
2. Repository name 例如：`lbot-pick-demo`  
3. Public 或 Private 自选  
4. **不要**勾选 “Add a README”（本地已有，避免冲突）  
5. Create repository  

### 4. 关联远程并推送

SSH 示例：

```bash
git remote add origin git@github.com:<你的用户名>/lbot-pick-demo.git
git branch -M main
git push -u origin main
```

HTTPS 示例：

```bash
git remote add origin https://github.com/<你的用户名>/lbot-pick-demo.git
git branch -M main
git push -u origin main
```

浏览器打开仓库页，确认文件已在。

### 5. 以后改代码再推

```bash
git add -A
git status
git commit -m "说明这次改了什么"
git push
```

### 6. 常见问题

| 问题 | 处理 |
|------|------|
| `File too large`（单文件 >100MB） | 从提交中移除；用 Git LFS 或网盘；本仓库应忽略 `.pt` |
| 推送被拒、要先 pull | `git pull --rebase origin main` 后再 `git push` |
| 已误提交大文件 | 用 `git rm --cached <文件>` 后重新 commit；若已 push，需历史清理（谨慎） |
| 想把上级 `api_lk73` 一起开源 | 另建仓库或子模块；注意厂商许可与体积 |
| 私有代码不想公开 | 建 **Private** 仓库，或只推脱敏后的分支 |

### 7. （可选）Git LFS 上传权重

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
# 再 git add 某个.pt 并 commit / push
```

免费 LFS 有额度，团队权重大时更推荐对象存储 / 网盘。

---

## 相关文档

- `calib_handeye/README.txt` — 手眼标定步骤  
- `calib_intrinsics/README.txt` — 内参标定  
- `docs/` — 其他笔记与交接材料  
