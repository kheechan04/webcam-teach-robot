"""손 관측 → 로봇 목표(집게 끝 위치 + 집게 열림)로 바꾸는 기본 처리.

모든 웹캠 조건(②, ③)에 공통으로 들어가는 "상식적인 기본 처리"만 여기에 둔다.
    - 물리적으로 불가능한 깊이 튐 무시 (게이트)
    - 지수평활
    - 클러치 기준점을 직전 몇 프레임의 중앙값으로
    - 집게는 히스테리시스 스위치
자세(기울기·집는 손모양)에 따른 깊이 치우침 보정은 조건 ③의 몫이라 여기서 하지 않는다.
"""

from collections import deque

import numpy as np

from webcam_teach_robot.hand_tracking import focal_length_px

GRIPPER_CLOSED, GRIPPER_OPEN = -0.17, 1.0  # 집게 관절 각도(rad)
# 엄지-검지 거리 / 손바닥 너비. 두 번째 조종 기록에서 붙였을 때 0.4~0.9, 폈을 때 1.1~1.5에 몰려 있었다.
PINCH_CLOSE_BELOW = 0.8
PINCH_OPEN_ABOVE = 1.0
# 30 fps에서 한 프레임에 10 cm = 초속 3 m. 사람 손이 조종 중에 이렇게 움직일 일은 없다고 보고 무시한다.
MAX_DEPTH_JUMP_M = 0.10
MAX_GATED_FRAMES = 10  # 이만큼(약 0.33초) 연속으로 무시했으면 새 값을 받아들인다. 영원히 막히지 않게
REF_WINDOW = 10  # 클러치 기준점: 직전 10프레임(약 0.33초) 중앙값


def hand_point_camera(palm_center_px: np.ndarray, depth_m: float, frame_w: int, frame_h: int,
                      fov_deg: float) -> np.ndarray:
    """손바닥 중심을 카메라 좌표(m)로: x 오른쪽, y 아래, z 카메라에서 멀어지는 쪽.
    x, y도 깊이 z를 곱해서 구하므로 깊이 오차가 옆 방향으로 번진다."""
    f = focal_length_px(frame_w, fov_deg)
    u, v = palm_center_px
    return np.array([(u - frame_w / 2) * depth_m / f, (v - frame_h / 2) * depth_m / f, depth_m])


def camera_delta_to_robot(d: np.ndarray) -> np.ndarray:
    """카메라 좌표의 손 이동량 → 로봇 좌표(x 앞, y 왼쪽, z 위)의 이동량. 화면은 좌우 반전된 상태."""
    dx_cam, dy_cam, dz_cam = d
    return np.array([-dz_cam, -dx_cam, -dy_cam])


class GripperSwitch:
    """집게 딸깍 스위치. 닫는 기준(0.8)과 여는 기준(1.0)이 달라서 경계에서 손이 떨려도 왔다 갔다 하지 않는다."""

    def __init__(self):
        self.closed = False

    def update(self, pinch: float) -> float:
        if self.closed and pinch > PINCH_OPEN_ABOVE:
            self.closed = False
        elif not self.closed and pinch < PINCH_CLOSE_BELOW:
            self.closed = True
        return GRIPPER_CLOSED if self.closed else GRIPPER_OPEN


class HandFilter:
    """깊이 게이트 + 지수평활 + 클러치 기준점용 기록."""

    def __init__(self, smooth: float = 0.5):
        self.smooth = smooth
        self.value: np.ndarray | None = None
        self.gated_in_a_row = 0
        self.gated_total = 0
        self.recent: deque[np.ndarray] = deque(maxlen=REF_WINDOW)

    def update(self, hand: np.ndarray) -> tuple[np.ndarray, bool]:
        """새 손 위치를 넣고 (걸러진 위치, 이번 프레임을 무시했는지)를 돌려준다."""
        if self.value is None:
            self.value = hand.copy()
            self.recent.append(hand)
            return self.value, False
        if abs(hand[2] - self.value[2]) > MAX_DEPTH_JUMP_M and self.gated_in_a_row < MAX_GATED_FRAMES:
            self.gated_in_a_row += 1
            self.gated_total += 1
            return self.value, True
        if self.gated_in_a_row >= MAX_GATED_FRAMES:
            self.value = hand.copy()  # 오래 막혔으면 새 위치가 진짜라고 보고 바로 옮긴다
        else:
            self.value = self.smooth * hand + (1 - self.smooth) * self.value
        self.gated_in_a_row = 0
        self.recent.append(hand)
        return self.value, False

    def lost(self) -> None:
        """손이 안 보인 프레임. 0.5초 넘게 안 보이면 예전 값을 버린다(다시 나타났을 때 옛 위치에 끌려가지 않게)."""
        self._lost = getattr(self, "_lost", 0) + 1
        if self._lost > 15:
            self.value = None
            self.recent.clear()
            self.gated_in_a_row = 0

    def found(self) -> None:
        self._lost = 0

    def reference(self) -> np.ndarray | None:
        """클러치를 거는 순간의 기준점. 한 프레임만 쓰면 그 프레임이 튀었을 때 기준 전체가 틀어진다."""
        if len(self.recent) < REF_WINDOW // 2:
            return None
        return np.median(np.array(self.recent), axis=0)
