#!/usr/bin/env python3
# encoding: utf-8

import base64
import json
import os
import re
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
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool

from interfaces.srv import SetPose2D

# -----------------------------
# OpenRouter configuration
# -----------------------------
config_path = '/home/ubuntu/ros2_ws/src/large_models/large_models/openrouter/config.json'
with open(config_path, 'r') as f:
    openrouter_config = json.load(f)

OPENROUTER_API_KEY = openrouter_config.get('api_key', '')
OPENROUTER_BASE_URL = openrouter_config.get('base_url', 'https://openrouter.ai/api/v1')
OPENROUTER_LLM_MODEL = openrouter_config.get('llm_model', '')
OPENROUTER_VLM_MODEL = openrouter_config.get('vlm_model', '')
GENERATION_CONFIG = openrouter_config.get('generation', {})

# -----------------------------
# Navigation targets
# -----------------------------
position_dict = {  # x, y, roll, pitch, yaw
    'kitchen': [3.56647, -2.01839, 0.0, 0.0, 0.981751],
    'front desk': [1.93939, 1.06236, 0.0, 0.0, 0.399843],
    'bedroom': [3.14153, -0.321892, 0.0, 0.0, 0.0733996],
    'zoo': [1.13, 0.0179, 0.0, 0.0, 0.9790],
    'space base': [1.58, -0.74, 0.0, 0.0, -48.0],
    'football field': [0.32, -0.65, 0.0, 0.0, -90.0],
    'origin': [0.0, 0.0, 0.0, 0.0, 0.0],
    'home': [1.0, 0.0, 0.0, 0.0, 0.0],
}

# -----------------------------
# Prompts
# -----------------------------
LLM_PROMPT = f'''
## Role
You are a robot navigation planner.

## Task
Convert the user's instruction into a JSON plan using the action function library.

## Requirements
1. Output JSON only.
2. Output format:
{{
  "action": ["...", "..."],
  "response": "..."
}}
3. "response" must be Korean.
4. If you cannot map the instruction to actions, return "action": [] and explain in "response".

## Action Function Library
- Move to a named place: move('<place>')
- Analyze current camera view with a query: vision('<question>')
- Speak/announce the result: play_audio()

## Available places
{', '.join(sorted(position_dict.keys()))}

## Examples
Input: Go to the front desk and check if the door is closed
Output: {{"action": ["move('front desk')", "vision('Is the door closed?')", "play_audio()"], "response": "On it."}}
'''

VLM_PROMPT = '''
# Role
You are a helpful robot butler.

# Task
Given an image and a user query, answer directly.

# Requirements
- Do not ask questions back.
- 20 to 40 Korean words.
- Output JSON only:
{
  "response": "..."
}
'''


class VLLMNavigationText(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)

        # Runtime state
        self.processing = False
        self.interrupt = False
        self.reach_goal = False

        self.latest_frame_lock = threading.Lock()
        self.latest_bgr = None
        self.last_vision_response = ''

        # ROS interfaces
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()
        timer_cb_group = ReentrantCallbackGroup()

        self.result_pub = self.create_publisher(String, '~/result', 1)

        self.create_subscription(
            Image,
            'ascamera/camera_publisher/rgb0/image',
            self.image_callback,
            1,
        )

        self.create_subscription(
            String,
            '/text_command',
            self.text_command_callback,
            10,
            callback_group=self.callback_group,
        )

        self.create_subscription(
            Bool,
            'navigation_controller/reach_goal',
            self.reach_goal_callback,
            1,
        )

        self.set_pose_client = self.create_client(SetPose2D, 'navigation_controller/set_pose')
        self.set_pose_client.wait_for_service()

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

    def _publish_result(self, text):
        msg = String()
        msg.data = text
        self.result_pub.publish(msg)

    def init_process(self):
        self.timer.cancel()
        self.get_logger().info('\033[1;32m%s\033[0m' % '========================================')
        self.get_logger().info('\033[1;32m%s\033[0m' % 'VLLM Navigation (Text Input Mode)')
        self.get_logger().info('\033[1;32m%s\033[0m' % f'LLM Model: {OPENROUTER_LLM_MODEL}')
        self.get_logger().info('\033[1;32m%s\033[0m' % f'VLM Model: {OPENROUTER_VLM_MODEL}')
        self.get_logger().info('\033[1;32m%s\033[0m' % '========================================')
        self.get_logger().info('\033[1;33m%s\033[0m' % 'Waiting for commands on /text_command ...')

    def image_callback(self, ros_image):
        try:
            bgr = np.array(self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8'), dtype=np.uint8)
            with self.latest_frame_lock:
                self.latest_bgr = bgr
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def reach_goal_callback(self, msg):
        self.reach_goal = msg.data

    def text_command_callback(self, msg):
        user_input = msg.data.strip()
        if not user_input:
            return

        if user_input.lower() in ('stop', 'cancel', 'abort'):
            self.interrupt = True
            self.get_logger().warn('Interrupt requested by user')
            return

        self.get_logger().info(f'Received text command: {user_input}')

        if self.processing:
            self.get_logger().warn('Already processing a command, please wait...')
            return

        self.processing = True
        self.interrupt = False
        threading.Thread(target=self.process_command, args=(user_input,), daemon=True).start()

    def call_openrouter_llm(self, user_input):
        if not OPENROUTER_API_KEY:
            raise RuntimeError('OpenRouter api_key is empty. Please set it in config.json')
        if not OPENROUTER_LLM_MODEL:
            raise RuntimeError('OpenRouter llm_model is empty. Please set it in config.json')

        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
        }

        data = {
            'model': OPENROUTER_LLM_MODEL,
            'messages': [
                {'role': 'system', 'content': LLM_PROMPT},
                {'role': 'user', 'content': user_input},
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

        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']

    def call_openrouter_vlm(self, query, bgr_image):
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
                {'role': 'system', 'content': VLM_PROMPT},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': query},
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

        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']

    def parse_json(self, llm_result):
        try:
            start_idx = llm_result.find('{')
            end_idx = llm_result.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                return json.loads(llm_result[start_idx:end_idx])
        except Exception as e:
            self.get_logger().error(f'JSON parse error: {e}')
        return None

    def parse_action(self, action_str):
        m = re.fullmatch(r"move\('([^']+)'\)", action_str)
        if m:
            return ('move', m.group(1))

        m = re.fullmatch(r"vision\('(.+)'\)", action_str)
        if m:
            return ('vision', m.group(1))

        if re.fullmatch(r'play_audio\(\)', action_str):
            return ('play_audio', None)

        return (None, None)

    def move(self, place):
        if place not in position_dict:
            raise ValueError(f'Unknown place: {place}')

        p = position_dict[place]
        msg = SetPose2D.Request()
        msg.data.x = float(p[0])
        msg.data.y = float(p[1])
        msg.data.roll = float(p[2])
        msg.data.pitch = float(p[3])
        msg.data.yaw = float(p[4])

        self.set_pose_client.call_async(msg)
        self.get_logger().info(f"Sent goal: {place} -> x={p[0]:.3f}, y={p[1]:.3f}, yaw={p[4]:.3f}")

    def vision(self, query):
        with self.latest_frame_lock:
            bgr = None if self.latest_bgr is None else self.latest_bgr.copy()

        if bgr is None:
            raise RuntimeError('No camera image received yet. Please wait...')

        raw = self.call_openrouter_vlm(query, bgr)
        parsed = self.parse_json(raw)
        if parsed and isinstance(parsed, dict) and 'response' in parsed:
            return str(parsed['response']).strip()
        return raw

    def play_audio(self):
        to_say = self.last_vision_response
        if not to_say:
            to_say = 'No vision result available.'
        self._publish_result(to_say)
        self.get_logger().info(f'Published: {to_say}')

    def process_command(self, user_input):
        try:
            llm_raw = self.call_openrouter_llm(user_input)
            self.get_logger().info(f'LLM Response: {llm_raw}')

            parsed = self.parse_json(llm_raw)
            if not parsed or not isinstance(parsed, dict):
                self._publish_result('Failed to parse LLM response.')
                return

            response_text = str(parsed.get('response', '')).strip()
            if response_text:
                self._publish_result(response_text)
                self.get_logger().info(f'Response: {response_text}')

            actions = parsed.get('action', [])
            if not isinstance(actions, list):
                actions = []

            for action_str in actions:
                if self.interrupt:
                    self.get_logger().warn('Interrupted')
                    break

                if not isinstance(action_str, str):
                    continue

                kind, arg = self.parse_action(action_str)
                if kind is None:
                    self.get_logger().warn(f'Unsupported action: {action_str}')
                    continue

                if kind == 'move':
                    self.reach_goal = False
                    self.move(arg)

                    while rclpy.ok() and not self.reach_goal:
                        if self.interrupt:
                            self.get_logger().warn('Interrupted while moving')
                            break
                        time.sleep(0.05)

                elif kind == 'vision':
                    res = self.vision(arg)
                    self.last_vision_response = res
                    self._publish_result(res)
                    self.get_logger().info(f'Vision result: {res}')

                elif kind == 'play_audio':
                    self.play_audio()

        except Exception as e:
            self.get_logger().error(f'process_command failed: {e}')
        finally:
            self.processing = False
            self.get_logger().info('\033[1;33mReady for next command on /text_command\033[0m')


def main():
    node = VLLMNavigationText('vllm_navigation_text')
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
