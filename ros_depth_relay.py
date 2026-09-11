#!/usr/bin/env python3
"""ROS2 深度/彩色中继：写到 /tmp 供 demo 用 OpenCV 读。

Orbbec 图像话题多为 BEST_EFFORT（sensor_data）；默认 RELIABLE 订阅会对不上，
表现为「永远等不到彩色首帧」。
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class DepthRelay(Node):
    def __init__(self, depth_topic, depth_output, color_output, color_topic):
        super().__init__("lbot_depth_relay")
        self.bridge = CvBridge()
        self.depth_output = Path(depth_output)
        self.color_output = Path(color_output)
        self.depth_output.parent.mkdir(parents=True, exist_ok=True)
        self.color_output.parent.mkdir(parents=True, exist_ok=True)
        self._n_color = 0
        self._n_depth = 0
        self._t0 = time.time()
        # 相机流：必须与 Orbbec 发布端兼容（BEST_EFFORT）
        qos = qos_profile_sensor_data
        self.depth_subscription = self.create_subscription(
            Image, depth_topic, self.on_depth, qos
        )
        self.color_subscription = self.create_subscription(
            Image, color_topic, self.on_color, qos
        )
        self.create_timer(5.0, self._heartbeat)
        self.get_logger().info(
            f"订阅 depth={depth_topic} color={color_topic} "
            f"(QoS=sensor_data/BEST_EFFORT) → {depth_output} / {color_output}"
        )

    def _heartbeat(self):
        dt = time.time() - self._t0
        if self._n_color == 0 or self._n_depth == 0:
            self.get_logger().warning(
                f"等待话题中… {dt:.0f}s  color帧={self._n_color} depth帧={self._n_depth} "
                f"（若一直为0：查 orbbec launch / 设备占用 / topic 名）"
            )
        elif self._n_color < 3:
            self.get_logger().info(
                f"已收 color={self._n_color} depth={self._n_depth} ({dt:.0f}s)"
            )

    def on_color(self, message):
        try:
            color = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            temporary = self.color_output.with_suffix(".tmp.jpg")
            if cv2.imwrite(str(temporary), color, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
                os.replace(temporary, self.color_output)
            self._n_color += 1
            if self._n_color == 1:
                self.get_logger().info(
                    f"彩色首帧已写 {self.color_output} "
                    f"{color.shape[1]}x{color.shape[0]}"
                )
        except Exception as error:
            self.get_logger().warning(f"彩色图转换失败: {error}")

    def on_depth(self, message):
        try:
            depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            depth = np.asarray(depth)
            if np.issubdtype(depth.dtype, np.floating):
                depth = depth * 1000.0
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.uint16
            )
            temporary = self.depth_output.with_suffix(".tmp.png")
            if cv2.imwrite(str(temporary), depth):
                os.replace(temporary, self.depth_output)
            self._n_depth += 1
            if self._n_depth == 1:
                self.get_logger().info(f"深度首帧已写 {self.depth_output}")
        except Exception as error:
            self.get_logger().warning(f"深度图转换失败: {error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/depth/image_raw")
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-output", required=True, type=Path)
    parser.add_argument("--color-output", required=True, type=Path)
    args = parser.parse_args()

    rclpy.init()
    node = DepthRelay(
        args.topic, args.depth_output, args.color_output, args.color_topic,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
