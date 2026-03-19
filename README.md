# 🏠🤖 ROS2 & VLM Autonomous Robot with VD-MKDF

> ROS2 기반 자율주행(**Nav2**)과 대규모 멀티모달 모델(**LMM: LLM/VLM**)을 결합해, **자연어 명령 → 행동 계획 → 주행 → 장면 분석**까지 수행하는 로봇입니다.

---

## 📌 1. Project Overview

사용자는 자연어로

* “현관으로 가서 문이 닫혀있는지 확인해줘”
* “주방으로 가서 뭐가 있는지 설명해줘”

같은 명령을 내릴 수 있고, 로봇은 이를 **행동 계획(Action Plan)** 으로 변환한 뒤 순차 실행합니다.

핵심은 다음 4가지입니다.

1. **LLM 기반 계획 수립(Planner)**: 자연어 → 실행 가능한 action 리스트(JSON)
2. **Nav2 기반 자율주행(Navigation)**: 지정 좌표(장소)로 이동
3. **객체 인지(Perception: OD/Seg)**: VD-MKDF 또는 YOLO 기반 세그멘테이션/객체 탐지 결과를 생성
4. **VLM 기반 상황 이해(Vision)**: 목적지 도착 후 카메라 화면 분석 및 응답

---

## ✨ 2. Key Features

### 🧠 Natural Language → Action Plan (LLM)

* 사용자의 텍스트 명령을 받아
* `move('place')`, `vision('question')`, `play_audio()` 형태의 action sequence로 변환
* JSON만 출력하도록 프롬프트 설계

### 🧭 Autonomous Navigation (Nav2)

* `navigation.launch.py`로 Nav2 bringup
* `navigation_controller/set_pose` 서비스로 목표 좌표 전달
* `navigation_controller/reach_goal` 피드백으로 도착 여부 확인

### 👁️ Scene Understanding (VLM)

* 목적지 도착 후 최신 카메라 프레임을 OpenRouter VLM으로 전달
* “문이 닫혀있는가?”, “뭐가 보이는가?” 같은 질의응답 수행
* 결과를 `~/result` 토픽으로 publish

### 🎯 Perception (Object Detection / Segmentation)

본 프로젝트는 **주행 + 상황 이해**뿐 아니라, 카메라 기반 **객체 인지(세그멘테이션/탐지)** 결과를 활용해

* 사용자에게 더 구체적인 설명을 제공하고
* (추후) Depth 기반 Safety Layer/리스크 판단으로 확장
  할 수 있도록 설계되었습니다.

인지 모듈은 환경/성능 요구에 따라 **두 가지 중 하나를 선택**해 구성합니다.

* **Option A — VD-MKDF (자체 개발 모델)**

  * 악천후/도메인 특화 등 목적에 맞춘 인지 성능을 목표로 설계
  * ROS2 노드로 구동하며 결과 토픽을 publish

* **Option B — YOLO (Ultralytics 기반 세그멘테이션/탐지)**

  * 빠른 프로토타이핑과 안정적인 baseline 제공
  * `/yolo/detections`, `/yolo/dbg_image` 등 토픽 활용

> **현재 단계 권장:** 기능 검증/데모는 YOLO로 빠르게 완성 → 이후 VD‑MKDF로 교체/비교

### 🔎 Perception → LMM 결합 (Optional)

* 인지 결과(VD‑MKDF 또는 YOLO)를 텍스트 요약으로 변환하여
* VLM 프롬프트에 함께 넣어 **응답 일관성/정확도**를 높일 수 있습니다.

---

## ❓ 3. Why This Project?

실내 로봇이 실제로 “도와주는” 형태로 동작하려면,
단순한 원격 조종이나 고정된 시나리오보다 **상황에 따른 행동 계획 + 실행 + 피드백**이 필요합니다.

이 프로젝트는 Nav2를 통한 **현실적인 주행**과, LMM을 통한 **자연어 이해/시각적 판단**을 연결하여
사용자가 로봇을 보다 자연스럽게 사용할 수 있도록 설계했습니다.

---

## 🧠 4. Core Idea

### 1) 🧾 Action Function Library 기반 계획

LLM이 아래 action 라이브러리를 사용해 계획을 생성합니다.

* `move('<place>')` : 장소로 이동
* `vision('<question>')` : 현재 화면 기반 질의응답
* `play_audio()` : 결과를 사용자에게 전달(텍스트 모드에서는 `~/result` publish)

### 2) 🧩 Planner / Executor 분리

* **Planner(LLM)**: 무엇을 할지 계획(JSON)
* **Executor(ROS2 Node)**: 계획을 안전하게 파싱하고, 순차 실행

### 3) 🗺️ 좌표 기반 장소 관리

목표 장소는 `position_dict`에 사전 정의합니다.

* `place -> (x, y, yaw)` 형태
* RViz에서 목표점을 찍어 좌표를 확보한 뒤 코드에 반영

---

## 🔄 5. System Pipeline

```text
/text_command (User)
        ↓
LLM Planner (OpenRouter)
        ↓
JSON Plan { action: [...], response: "..." }
        ↓
Action Executor
  ├─ move('place')  ──> navigation_controller/set_pose
  │                     (Nav2 navigation)
  │                     ↓
  │                 reach_goal feedback
  ├─ vision('query') ──> VLM(OpenRouter) with latest RGB frame
  │                     ↓
  │                 JSON { response }
  └─ play_audio()   ──> ~/result publish (text mode)
```

---

## 🧩 6. ROS Graph

### 주요 토픽/서비스

* **Input**

  * `/text_command` : 사용자 명령 입력
  * `/ascamera/camera_publisher/rgb0/image` : 카메라 RGB 프레임

* **Navigation**

  * `navigation_controller/set_pose` (**service**, `interfaces/SetPose2D`) : 목표 좌표 전달
  * `navigation_controller/reach_goal` (**topic**, `std_msgs/Bool`) : 목표 도달 여부

* **Output**

  * `~/result` : 사용자 피드백(텍스트)

* **(Optional) YOLO**

  * `/yolo/detections` : 탐지 결과
  * `/yolo/dbg_image` : 디버그 이미지

---

## 🗂️ 7. Project Structure

```bash
ros2_ws/src/
├── large_models/
│   ├── large_models/
│   │   ├── openrouter/
│   │   │   └── config.json
│   │   ├── vllm_with_camera_text.py
│   │   ├── smart_home_assistant_text.py
│   │   └── (optional) yolo_prompt_bridge.py
│   └── launch/
│       ├── vllm_with_camera_text.launch.py
│       └── smart_home_assistant_text.launch.py
├── navigation/
│   └── launch/navigation.launch.py
└── (other packages...)
```

---

## ⚙️ 8. Tech Stack

* **Robot / Middleware**: ROS2 Humble, MentorPi
* **Navigation**: Nav2, AMCL / Map Server (or SLAM)
* **Perception**: ASCamera (RGB/Depth), (Optional) YOLO
* **LMM**: OpenRouter (LLM + VLM)
* **Language**: Python (rclpy)

---

## 🚀 9. Getting Started

### 1) 준비: OpenRouter 설정

`config.json`에서 아래 항목이 정의되어 있어야 합니다.

* `api_key`
* `base_url`
* `llm_model`
* `vlm_model`

```bash
/home/ubuntu/ros2_ws/src/large_models/large_models/openrouter/config.json
```

### 2) 빌드

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select large_models
source install/local_setup.zsh
```

### 3) (권장) 맵 준비

* SLAM으로 맵을 만든 뒤 저장하거나
* 기존 맵(`my_map`)을 준비합니다.

### 4) 실행 (Final Project)

```bash
ros2 launch large_models smart_home_assistant_text.launch.py
```

### 5) RViz (Remote PC/VM)

```bash
ros2 launch navigation rviz_navigation.launch.py
```

### 6) 명령 예시

```bash
# 1) 현관으로 이동 후 문 상태 확인
ros2 topic pub -1 /text_command std_msgs/String "data: 'Go to the front desk and check if the door is closed'"

# 2) 주방으로 이동 후 상황 요약
ros2 topic pub -1 /text_command std_msgs/String "data: 'Go to the kitchen and tell me what you see'"

# 3) 집(home)으로 복귀
ros2 topic pub -1 /text_command std_msgs/String "data: 'Go back home'"
```

---

## 🧭 10. How to Add / Calibrate Places (position_dict)

1. RViz에서 **2D Nav Goal**로 목표점을 찍습니다.
2. goal 토픽(`PoseStamped`)에서 `x`, `y`, `yaw`를 추출합니다.
3. `position_dict`에 장소 키와 함께 추가합니다.

예시:

```python
position_dict = {
  'kitchen': [3.56, -2.02, 0.0, 0.0, 0.98],
  'front desk': [1.94, 1.06, 0.0, 0.0, 0.40],
}
```

---

## 🛡️ 11. Safety Layer (Depth-based Stop/Slow) — Implemented

본 프로젝트는 주행 파이프라인에 **Depth 기반 Safety Layer(감속/정지)** 를 실제로 결합했습니다.

### 핵심 아이디어

* Depth 이미지에서 전방 위험 구간(ROI)의 거리를 추정한 뒤
* 위험도가 높으면 **속도 명령(cmd_vel)을 감속/정지**시키는 방식으로 충돌 위험을 낮춥니다.

### 적용 방식 (cmd_vel 게이팅)

* Nav2/Teleop 등 상위 모듈이 publish 하는 속도 명령을 입력으로 받고
* Safety Layer가 다음 중 하나로 출력합니다.

  * **Stop:** `cmd_vel`을 **0으로 클램프**(선형/각속도 모두 0)
  * **Slow:** `cmd_vel`에 **slowdown_ratio(0~1)** 를 곱해 감속

> 구현상 “컨트롤러 쪽 cmd_vel을 0으로 만드는” 게이팅을 포함하며, 감속은 비율 스케일링으로 수행합니다.

### 동작 예시

* `d < d_stop` → 정지(Stop)
* `d_stop ≤ d < d_slow` → 감속(Slow, ratio 선형 스케일)
* `d ≥ d_slow` → 원래 속도 유지

---

## ⚠️ 12. Limitations

* 맵/로컬라이제이션(AMCL)이 불안정하면 goal 수행이 실패할 수 있습니다.
* LLM이 생성한 action이 예상과 다를 수 있어, 프롬프트/예외처리 고도화가 필요합니다.
* 네트워크 상태(OpenRouter API 지연)에 따라 응답 시간이 변동될 수 있습니다.
* Depth 기반 Safety Layer는 ROI/임계값 설정에 따라 오탐/미탐이 발생할 수 있어 환경별 튜닝이 필요합니다.

---

## 🔮 12. Future Work

* ✅ Depth 기반 Safety Layer(Stop/Slow) 결합
* ✅ YOLO 탐지 결과를 VLM 프롬프트에 결합(설명 정확도/일관성 향상)
* ✅ 장소 자동 등록(지도에서 클릭 → position_dict 자동 생성)
* ✅ 멀티스텝 플래닝(조건 분기/재시도/타임아웃)
* ✅ 주행 실패 원인 분석/복구(Recovery behaviors)

---

## ✅ 13. Summary

이 프로젝트는
**자연어 명령 → 행동 계획(LLM) → 자율주행(Nav2) → 시각적 확인(VLM) → 결과 피드백**
까지 이어지는 스마트 홈 어시스턴트 로봇 파이프라인을 ROS2에서 구현한 것입니다.

> Nav2 기반의 “실제 주행”과 LMM 기반의 “상황 이해/대화”를 결합해, 사용자가 집 안에서 로봇에게 자연어로 일을 시킬 수 있도록 만들었습니다.
