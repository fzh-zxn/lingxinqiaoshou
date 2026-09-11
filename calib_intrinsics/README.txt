内参标定采集目录（彩色图 1280x720）
--------------------------------
板参数：DFOPTIX CC300-20-15
  CharucoBoard (14列 x 9行), square=20mm, marker=15mm, DICT_5X5_100

采集（推荐）：
  1. 先退出占用相机的 demo_pick_square
  2. cd api_lk73_v1.0.5/Demo/demo_python
  3. python demo_camera.py
     （默认存到本目录 calib_XX.png；角点>=20 才保存）
  4. 采 15～25 张，多角度/多位置；画面看 corners 数
  5. 标定结果写入 ../grasp_config.yaml 的 camera_intrinsics
  6. 不要改 plate_bias_right_m / plate_bias_forward_m

按键：s/空格=保存  d=叠加开关  q=退出
角点不足仍要存：python demo_camera.py --force
V4L 直连：python demo_camera.py --no-depth --width 1280 --height 720
旧颜色采集：python demo_camera.py --legacy --color 三
