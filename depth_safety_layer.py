#!/usr/bin/env python3
# encoding: utf-8

import base64
import json
import os
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Empty

try:
    from vision_msgs.msg import Detection2DArray
except Exception:
    Detection2DArray = None

# Load OpenRouter config from source directory
config_path = '/home/ubuntu/ros2_ws/src/large_models/large_models/openrouter/config.json'
with open(config_path, 'r') as f:
    openrouter_config = json.load(f)

OPENROUTER_API_KEY = openrouter_config.get('api_key', '')
OPENROUTER_BASE_URL = openrouter_config.get('base_url', 'https://openrouter.ai/api/v1')
OPENROUTER_VLM_MODEL = openrouter_config.get('vlm_model', openrouter_config.get('llm_model', ''))
GENERATION_CONFIG = openrouter_config.get('generation', {})

PROMPT = r'''
# Role
You are a vision assistant for a robot camera.

# Task
Given the current camera image and the user's text command, respond concisely.

# Extra Context (Perception Summary)
- If a Perception Summary is provided, use it as the primary evidence for object presence (classes/counts).
- The image may be used for additional context, but do not contradict the Perception Summary unless it is obviously inconsistent.

# Requirements
- If the user asks a question about what is visible, describe it.
- If the user asks to find an object, say whether it is visible.
- Output JSON only:
{
  "response": "..."
}
'''


class VLLMWithCameraText(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)

        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)

        self.latest_frame_lock = threading.Lock()
        self.latest_rgb = None

        self.processing = False
        self.last_response_text = ''

        timer_cb_group = ReentrantCallbackGroup()
        self.callback_group = ReentrantCallbackGroup()

        self.result_pub = self.create_publisher(String, '~/result', 1)

        # --- YOLO summary cache ---
        self.latest_det_lock = threading.Lock()
        self.latest_det_text = ""
        self.latest_det_stamp = 0.0

        self.declare_parameter('yolo_topic', '/yolo/detections')
        self.declare_parameter('max_dets', 10)
        self.declare_parameter('min_score', 0.25)

        self.max_dets = int(self.get_parameter('max_dets').value)
        self.min_score = float(self.get_parameter('min_score').value)
        yolo_topic = self.get_parameter('yolo_topic').get_parameter_value().string_value

        # Camera subscription
        self.create_subscription(
            Image,
            'ascamera/camera_publisher/rgb0/image',
            self.image_callback,
            1,
        )

        # Text command subscription
        self.create_subscription(
            String,
            '/text_command',
            self.text_command_callback,
            10,
            callback_group=self.callback_group,
        )

        # YOLO subscription (optional)
        if Detection2DArray is None:
            self.get_logger().warn('vision_msgs not available. YOLO summary will be disabled.')
        else:
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(
                Detection2DArray,
                yolo_topic,
                self.yolo_callback,
                qos,
            )
            self.get_logger().info(f'[YOLO] Subscribed detections: {yolo_topic} (vision_msgs/Detection2DArray)')

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def _truncate_for_log(self, text, limit=3000):
        if text is None:
            return ''
        text = str(text)
        if len(text) <= limit:
            return text
        return text[:limit] + '...<truncated>'

    def _bgr_to_data_url(self, bgr_image):
        ok, buf = cv2.imencode('.jpg', bgr_image)
        if not ok:
            raise RuntimeError('Failed to encode image')
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        return f'data:image/jpeg;base64,{b64}'

    def init_process(self):
        self.timer.cancel()
        self.create_service(Empty, '~/init_finish', self.get_node_state)

        self.get_logger().info('\033[1;32m%s\033[0m' % '========================================')
        self.get_logger().info('\033[1;32m%s\033[0m' % 'VLLM With Camera (Text Input Mode)')
        self.get_logger().info('\033[1;32m%s\033[0m' % f'Using OpenRouter VLM Model: {OPENROUTER_VLM_MODEL}')
        self.get_logger().info('\033[1;32m%s\033[0m' % '========================================')
        self.get_logger().info('\033[1;33m%s\033[0m' % 'Waiting for commands on /text_command topic...')
        self.get_logger().info('\033[1;33m%s\033[0m' % 'Example: ros2 topic pub -1 /text_command std_msgs/String "data: \'what do you see?\'"')

        threading.Thread(target=self.process, daemon=True).start()

    def get_node_state(self, request, response):
        return response

    def image_callback(self, ros_image):
        try:
            bgr_image = np.array(self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8'), dtype=np.uint8)

            with self.latest_frame_lock:
                self.latest_rgb = bgr_image

            if self.image_queue.full():
                self.image_queue.get()
            self.image_queue.put(bgr_image)
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    # --- YOLO parsing helpers ---
    def _safe_label(self, det):
        try:
            if det.results and len(det.results) > 0:
                hyp = det.results[0].hypothesis
                cid = getattr(hyp, 'class_id', '')
                return str(cid)
        except Exception:
            pass
        return 'unknown'

    def _safe_score(self, det):
        try:
            if det.results and len(det.results) > 0:
                hyp = det.results[0].hypothesis
                score = getattr(hyp, 'score', None)
                if score is not None:
                    return float(score)
        except Exception:
            pass
        return 0.0

    def _safe_bbox(self, det):
        try:
            bb = det.bbox
            cx, cy = float(bb.center.x), float(bb.center.y)
            sx, sy = float(bb.size_x), float(bb.size_y)
            x1 = int(cx - sx / 2.0)
            y1 = int(cy - sy / 2.0)
            x2 = int(cx + sx / 2.0)
            y2 = int(cy + sy / 2.0)
            return (x1, y1, x2, y2)
        except Exception:
            return None

    def yolo_callback(self, msg: 'Detection2DArray'):
        try:
            dets = []
            for det in msg.detections:
                label = self._safe_label(det)
                score = self._safe_score(det)
                if score < self.min_score:
                    continue
                bbox = self._safe_bbox(det)
                dets.append((label, score, bbox))

            dets.sort(key=lambda x: x[1], reverse=True)
            dets = dets[: self.max_dets]

            lines = []
            if len(dets) == 0:
                lines.append('- (no detections)')
            else:
                for i, (label, score, bbox) in enumerate(dets, 1):
                    if bbox is None:
                        lines.append(f'- {i}. {label}, score={score:.2f}')
                    else:
                        x1, y1, x2, y2 = bbox
                        lines.append(f'- {i}. {label}, score={score:.2f}, bbox=({x1},{y1},{x2},{y2})')

            text = '[Perception Summary: YOLO Detections]\n' + '\n'.join(lines)

            with self.latest_det_lock:
                self.latest_det_text = text
                self.latest_det_stamp = time.time()

        except Exception as e:
            self.get_logger().error(f'YOLO callback error: {e}')

    def call_openrouter_vlm(self, user_input, bgr_image):
        if not OPENROUTER_API_KEY:
            raise RuntimeError('OpenRouter api_key is empty. Please set it in config.json')
        if not OPENROUTER_VLM_MODEL:
            raise RuntimeError('OpenRouter vlm_model/llm_model is empty. Please set it in config.json')

        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
        }

        image_url = self._bgr_to_data_url(bgr_image)

        data = {
            'model': OPENROUTER_VLM_MODEL,
            'messages': [
                {'role': 'system', 'content': PROMPT},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': user_input},
                        {'type': 'image_url', 'image_url': {'url': image_url}},
                    ],
                },
            ],
            'temperature': GENERATION_CONFIG.get('temperature', 0.2),
            'top_p': GENERATION_CONFIG.get('top_p', 0.9),
            'max_tokens': GENERATION_CONFIG.get('max_tokens', 512),
        }

        response = requests.post(
            f'{OPENROUTER_BASE_URL}/chat/completions',
            headers=headers,
            json=data,
            timeout=openrouter_config.get('http', {}).get('timeout_sec', 60),
        )

        self.get_logger().info(
            f'OpenRouter HTTP {response.status_code} raw response: {self._truncate_for_log(response.text)}'
        )

        response.raise_for_status()
        result = response.json()
        self.get_logger().info(
            'OpenRouter JSON response: '
            f'{self._truncate_for_log(json.dumps(result, ensure_ascii=False))}'
        )

        return result['choices'][0]['message']['content']

    def parse_llm_response(self, llm_result):
        try:
            start_idx = llm_result.find('{')
            end_idx = llm_result.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_result[start_idx:end_idx]
                return json.loads(json_str)
        except Exception as e:
            self.get_logger().error(f'JSON parse error: {e}')
        return None

    def text_command_callback(self, msg):
        user_input = msg.data.strip()
        if not user_input:
            return

        self.get_logger().info(f'Received text command: {user_input}')

        if self.processing:
            self.get_logger().warn('Already processing a command, please wait...')
            return

        self.processing = True
        threading.Thread(target=self.process_command, args=(user_input,), daemon=True).start()

    def process_command(self, user_input):
        try:
            with self.latest_frame_lock:
                bgr = None if self.latest_rgb is None else self.latest_rgb.copy()

            if bgr is None:
                self.get_logger().warn('No camera image received yet. Please wait...')
                return

            with self.latest_det_lock:
                det_text = self.latest_det_text

            aug_user_input = user_input
            if det_text:
                aug_user_input += "\n\n" + det_text + "\n"

            llm_result = self.call_openrouter_vlm(aug_user_input, bgr)
            if not llm_result:
                self.get_logger().error('Failed to get response from VLM')
                return

            self.get_logger().info(f'LLM Response: {llm_result}')

            parsed = self.parse_llm_response(llm_result)
            if parsed and isinstance(parsed, dict):
                response_text = str(parsed.get('response', '')).strip()
            else:
                response_text = llm_result

            self.last_response_text = response_text

            out = String()
            out.data = response_text
            self.result_pub.publish(out)

            self.get_logger().info(f'\033[1;32mResponse: {response_text}\033[0m')
        except Exception as e:
            self.get_logger().error(f'process_command failed: {e}')
        finally:
            self.processing = False
            self.get_logger().info('\033[1;33mReady for next command on /text_command\033[0m')

    def process(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.waitKey(10)

        while rclpy.ok():
            image = self.image_queue.get(block=True)
            height, width = image.shape[:2]
            cv2.resizeWindow('image', width, height)

            if self.last_response_text:
                cv2.putText(
                    image,
                    self.last_response_text[:60],
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow('image', image)
            cv2.waitKey(1)

        cv2.destroyAllWindows()


def main():
    node = VLLMWithCameraText('vllm_with_camera_text')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
