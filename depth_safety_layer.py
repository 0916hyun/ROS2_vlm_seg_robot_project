#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class DepthSafetyLayer(Node):
    """
    Depth 기반 Stop/Slow 게이팅 노드.
    - /cmd_vel_in  : 상위 제어(Nav2/teleop) 속도
    - /depth_topic : depth image (16UC1 or 32FC1)
    - /cmd_vel_out : 안전 게이팅된 속도 (stop=0 clamp / slow=ratio scaling)
    """

    def __init__(self):
        super().__init__('depth_safety_layer')

        self.declare_parameter('cmd_vel_in', '/cmd_vel')
        self.declare_parameter('cmd_vel_out', '/cmd_vel_safe')
        self.declare_parameter('depth_topic', '/ascamera/camera_publisher/depth0/image_raw')

        # ROI (pixel)
        self.declare_parameter('roi_x', 240)   # left
        self.declare_parameter('roi_y', 200)   # top
        self.declare_parameter('roi_w', 160)
        self.declare_parameter('roi_h', 160)

        # thresholds (meters)
        self.declare_parameter('d_stop', 0.60)
        self.declare_parameter('d_slow', 1.50)

        # robust depth statistic
        self.declare_parameter('percentile', 20.0)

        self.cmd_vel_in = self.get_parameter('cmd_vel_in').value
        self.cmd_vel_out = self.get_parameter('cmd_vel_out').value
        self.depth_topic = self.get_parameter('depth_topic').value

        self.roi_x = int(self.get_parameter('roi_x').value)
        self.roi_y = int(self.get_parameter('roi_y').value)
        self.roi_w = int(self.get_parameter('roi_w').value)
        self.roi_h = int(self.get_parameter('roi_h').value)

        self.d_stop = float(self.get_parameter('d_stop').value)
        self.d_slow = float(self.get_parameter('d_slow').value)
        self.pctl = float(self.get_parameter('percentile').value)

        self.bridge = CvBridge()

        self.latest_depth = None
        self.latest_encoding = None

        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(Image, self.depth_topic, self.depth_cb, qos_img)
        self.create_subscription(Twist, self.cmd_vel_in, self.cmd_cb, 10)
        self.pub = self.create_publisher(Twist, self.cmd_vel_out, 10)

        self.get_logger().info(f"[DepthSafety] depth={self.depth_topic}, in={self.cmd_vel_in}, out={self.cmd_vel_out}")
        self.get_logger().info(f"[DepthSafety] ROI=({self.roi_x},{self.roi_y},{self.roi_w},{self.roi_h}), d_stop={self.d_stop}, d_slow={self.d_slow}")

    def depth_cb(self, msg: Image):
        self.latest_depth = msg
        self.latest_encoding = msg.encoding

    def _depth_to_meters(self, depth_img: np.ndarray, encoding: str) -> np.ndarray:
        # Common cases:
        # - 16UC1 : usually millimeters
        # - 32FC1 : meters
        if encoding == '16UC1':
            return depth_img.astype(np.float32) * 0.001
        if encoding == '32FC1':
            return depth_img.astype(np.float32)
        # fallback: try as float meters
        return depth_img.astype(np.float32)

    def _compute_distance(self) -> float | None:
        if self.latest_depth is None:
            return None

        # Convert to numpy
        try:
            depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
        except Exception:
            return None

        depth_m = self._depth_to_meters(depth, self.latest_encoding or '')

        h, w = depth_m.shape[:2]
        x1 = max(0, min(w - 1, self.roi_x))
        y1 = max(0, min(h - 1, self.roi_y))
        x2 = max(0, min(w, x1 + self.roi_w))
        y2 = max(0, min(h, y1 + self.roi_h))
        roi = depth_m[y1:y2, x1:x2]

        valid = roi[np.isfinite(roi)]
        valid = valid[(valid > 0.05) & (valid < 10.0)]
        if valid.size < 50:
            return None

        d = float(np.percentile(valid, self.pctl))
        return d

    def _slowdown_ratio(self, d: float) -> float:
        if d <= self.d_stop:
            return 0.0
        if d >= self.d_slow:
            return 1.0
        return (d - self.d_stop) / (self.d_slow - self.d_stop)

    def cmd_cb(self, msg: Twist):
        d = self._compute_distance()

        out = Twist()
        out.linear = msg.linear
        out.angular = msg.angular

        # If no depth, pass-through (보수적으로 stop을 원하면 여기서 0으로 바꿀 수도 있음)
        if d is None:
            self.pub.publish(out)
            return

        ratio = self._slowdown_ratio(d)

        # Stop clamp: set velocities to 0
        if ratio <= 0.0:
            out.linear.x = 0.0
            out.linear.y = 0.0
            out.linear.z = 0.0
            out.angular.x = 0.0
            out.angular.y = 0.0
            out.angular.z = 0.0
        else:
            # Slow: scale
            out.linear.x *= ratio
            out.linear.y *= ratio
            out.linear.z *= ratio
            out.angular.z *= ratio

        self.pub.publish(out)


def main():
    rclpy.init()
    node = DepthSafetyLayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
