"""조건 ①: 프로그램이 만드는 큐브 옮기기 시범 (스크립트 전문가).

정답 위치(큐브·목표)를 알고 있으므로 경유점을 순서대로 지나간다.
    큐브 위 → 내려가기 → 집게 닫기 → 들기 → 목표 위 → 내려놓기 → 집게 열기 → 들기
각 제어 주기(30 Hz, 웹캠 조종과 같게)마다 다음 집게 끝 목표를 정하고, 역기구학으로 관절 명령을 만든다.
"""

from dataclasses import dataclass

import numpy as np

from webcam_teach_robot.scene import CUBE_HALF
from webcam_teach_robot.teleop_mapping import GRIPPER_CLOSED, GRIPPER_OPEN

CONTROL_HZ = 30
# 고정 턱 안쪽 면이 집게 기준점(gripperframe)보다 로봇 쪽으로 약 2.2 cm 떨어져 있다(재 본 값).
# 처음엔 기준점을 큐브 중심보다 0.7 cm 바깥에 둬서 고정 턱이 큐브 면에 딱 붙게 했는데, 내려갈 때 고정 턱이
# 큐브 모서리 위에 올라타는 실패가 나왔다(20개 중 2개). 기준점을 큐브 중심에 두면 고정 턱이 큐브 면에서
# 약 0.7 cm 떨어져 내려가고, 닫힐 때 움직이는 턱이 큐브를 고정 턱 쪽으로 밀어 준다.
# 바꾼 뒤 무작위 배치 30개 × (오프셋 0, −0.4 cm) × (높이 1.2, 1.8 cm) 전부 성공.
GRASP_RADIAL_OFFSET = 0.0
GRASP_Z = 0.012  # 집을 때 기준점 높이. 집게 끝이 바닥에서 약 1 cm 위
HOVER_Z = 0.07  # 옮길 때 높이
PLACE_Z = GRASP_Z + 0.003  # 내려놓을 때는 조금 높게(큐브가 바닥에 끼지 않게)
SPEED = 0.12  # 집게 끝 이동 속도 (m/s)
GRIP_WAIT_S = 0.5  # 집게 여닫고 기다리는 시간


@dataclass
class Waypoint:
    pos: np.ndarray
    gripper: float
    wait_s: float = 0.0  # 도착 후 기다리기


def radial(xy: np.ndarray) -> np.ndarray:
    return np.array([*xy, 0.0]) / np.linalg.norm(xy)


def plan(cube_xy: np.ndarray, target_xy: np.ndarray, start_pos: np.ndarray) -> list[Waypoint]:
    c = np.array([*cube_xy, 0.0]) + GRASP_RADIAL_OFFSET * radial(cube_xy)
    t = np.array([*target_xy, 0.0]) + GRASP_RADIAL_OFFSET * radial(target_xy)
    up = lambda p, z: np.array([p[0], p[1], z])
    return [
        Waypoint(start_pos, GRIPPER_OPEN),
        Waypoint(up(c, HOVER_Z), GRIPPER_OPEN),
        Waypoint(up(c, GRASP_Z), GRIPPER_OPEN),
        Waypoint(up(c, GRASP_Z), GRIPPER_CLOSED, GRIP_WAIT_S),
        Waypoint(up(c, HOVER_Z), GRIPPER_CLOSED),
        Waypoint(up(t, HOVER_Z), GRIPPER_CLOSED),
        Waypoint(up(t, PLACE_Z), GRIPPER_CLOSED),
        Waypoint(up(t, PLACE_Z), GRIPPER_OPEN, GRIP_WAIT_S),
        Waypoint(up(t, HOVER_Z), GRIPPER_OPEN),
    ]


def trajectory(waypoints: list[Waypoint]) -> list[tuple[np.ndarray, float]]:
    """경유점 사이를 일정한 속도로 이어 제어 주기마다의 (집게 끝 목표, 집게 명령) 목록을 만든다."""
    dt = 1.0 / CONTROL_HZ
    out: list[tuple[np.ndarray, float]] = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b.pos - a.pos) / (SPEED * dt))))
        for k in range(1, n + 1):
            out.append((a.pos + (b.pos - a.pos) * k / n, b.gripper))
        out += [(b.pos.copy(), b.gripper)] * int(round(b.wait_s / dt))
    return out


def cube_on_table(cube_z: float) -> bool:
    return abs(cube_z - CUBE_HALF) < 0.005
