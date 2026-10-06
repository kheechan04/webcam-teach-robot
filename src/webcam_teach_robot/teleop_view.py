"""조종을 돕는 2D 안내 그림: 위에서 본 지도 + 옆에서 본 단면 + 거리 숫자.

로봇 창(뒤에서 보는 3D 시점)으로는 집게와 큐브의 앞뒤 거리·높이를 판단하기 어려웠다
(첫 큐브 옮기기 시도에서 목표 위라고 생각했을 때 실제로는 앞으로 약 6 cm 벗어나 있었다).
처음엔 MuJoCo로 작은 3D 화면 두 개를 더 그리려 했지만 한 프레임에 약 110 ms가 걸려서, 2D로 직접 그린다.
이 그림은 조종하는 사람에게만 보이고, 기록되는 데이터에는 영향이 없다.
"""

import cv2
import numpy as np

from webcam_teach_robot.scene import BASE_HALF, CUBE_HALF, TARGET_HALF, yaw_facing_robot

PANEL_W, PANEL_H = 210, 160
PX_PER_M = PANEL_W / 0.32  # 위·옆 그림 모두 같은 축척(가로 32 cm). 축척이 다르면 정사각형 큐브가 찌그러져 보인다
Y_RANGE = (-0.16, 0.16)  # 로봇 좌우 (m), +y가 로봇(=사람)의 왼쪽
Z_RANGE = (-0.01, -0.01 + PANEL_H / PX_PER_M)  # 옆 그림 세로 = 높이

RED = (60, 60, 220)
GREEN = (90, 190, 90)
WHITE = (255, 255, 255)
GRAY = (130, 130, 130)
YELLOW = (40, 210, 240)


def _square(center, half, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[half, half], [-half, half], [-half, -half], [half, -half]])
    return center + corners @ np.array([[c, s], [-s, c]])


TOP_X_RANGE = (0.10, 0.10 + PANEL_H / PX_PER_M)  # 위 그림 세로(로봇 앞뒤) 범위
SIDE_X_RANGE = (0.10, 0.10 + PANEL_W / PX_PER_M)  # 옆 그림 가로(로봇 앞뒤) 범위


def _top_px(xy):
    """위에서 본 지도: 로봇 앞쪽이 그림 위, 사람 왼쪽(+y)이 그림 왼쪽."""
    u = (Y_RANGE[1] - xy[..., 1]) * PX_PER_M
    v = (TOP_X_RANGE[1] - xy[..., 0]) * PX_PER_M
    return np.stack([u, v], -1).astype(int)


def _side_px(x, z):
    """옆에서 본 단면: 로봇 앞쪽이 그림 오른쪽, 위가 위."""
    return int((x - SIDE_X_RANGE[0]) * PX_PER_M), int((Z_RANGE[1] - z) * PX_PER_M)


def draw_guides(frame: np.ndarray, tip: np.ndarray, gripper_closed: bool, cube: np.ndarray,
                cube_quat: np.ndarray, goal_xy: np.ndarray, workspace_lo: np.ndarray, workspace_hi: np.ndarray,
                base: np.ndarray | None = None) -> None:
    """base: 쌓기 받침 블록 위치(쌓기 과제일 때만). 있으면 목표 사각형 대신 받침 블록을 그리고, 집은 뒤엔 받침 윗면까지 거리를 보여 준다."""
    h, w = frame.shape[:2]
    y0 = h - PANEL_H - 8
    top = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    side = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)

    # 위에서 본 지도
    ws = np.array([[workspace_lo[0], workspace_lo[1]], [workspace_lo[0], workspace_hi[1]],
                   [workspace_hi[0], workspace_hi[1]], [workspace_hi[0], workspace_lo[1]]])
    cv2.polylines(top, [_top_px(ws)], True, GRAY, 1)
    if base is None:
        cv2.fillPoly(top, [_top_px(_square(goal_xy, TARGET_HALF, yaw_facing_robot(goal_xy)))], (40, 90, 40))
    else:
        cv2.fillPoly(top, [_top_px(_square(base[:2], BASE_HALF, yaw_facing_robot(base[:2])))], GREEN)
    cube_yaw = 2 * np.arctan2(cube_quat[3], cube_quat[0])
    cv2.fillPoly(top, [_top_px(_square(cube[:2], CUBE_HALF, cube_yaw))], RED)
    tp = tuple(_top_px(tip[:2]))
    cv2.circle(top, tp, 6, YELLOW, -1 if gripper_closed else 2)

    # 옆에서 본 단면
    cv2.line(side, _side_px(SIDE_X_RANGE[0], 0), _side_px(SIDE_X_RANGE[1], 0), GRAY, 1)
    if base is None:
        cv2.line(side, _side_px(goal_xy[0] - TARGET_HALF, 0.001), _side_px(goal_xy[0] + TARGET_HALF, 0.001), GREEN, 4)
    else:
        cv2.rectangle(side, _side_px(base[0] - BASE_HALF, base[2] + BASE_HALF),
                      _side_px(base[0] + BASE_HALF, base[2] - BASE_HALF), GREEN, -1)
    cv2.rectangle(side, _side_px(cube[0] - CUBE_HALF, cube[2] + CUBE_HALF),
                  _side_px(cube[0] + CUBE_HALF, cube[2] - CUBE_HALF), RED, -1)
    sp = _side_px(tip[0], tip[2])
    cv2.line(side, sp, (sp[0], sp[1] - 25), YELLOW, 2)  # 집게 몸통 방향(위로)
    cv2.circle(side, sp, 6, YELLOW, -1 if gripper_closed else 2)

    for img, label in ((top, "TOP (up = forward)"), (side, "SIDE (right = forward)")):
        cv2.putText(img, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
    frame[y0:y0 + PANEL_H, 8:8 + PANEL_W] = cv2.addWeighted(frame[y0:y0 + PANEL_H, 8:8 + PANEL_W], 0.25, top, 0.75, 0)
    x1 = 16 + PANEL_W
    frame[y0:y0 + PANEL_H, x1:x1 + PANEL_W] = cv2.addWeighted(frame[y0:y0 + PANEL_H, x1:x1 + PANEL_W], 0.25, side, 0.75, 0)

    # 지금 노리는 것(집기 전엔 큐브, 집은 뒤엔 목표)까지의 거리. 집게 기준점이 큐브 중심에 오면 0
    goal_ref = (np.array([goal_xy[0], goal_xy[1], CUBE_HALF]) if base is None
                else np.array([base[0], base[1], base[2] + BASE_HALF + CUBE_HALF]))  # 쌓기: 받침 윗면에 놓인 큐브 중심
    ref = goal_ref if gripper_closed else cube
    d = (ref - tip) * 100
    name = "to GOAL" if gripper_closed else "to CUBE"
    words = [("FWD" if d[0] > 0 else "BACK", abs(d[0])), ("LEFT" if d[1] > 0 else "RIGHT", abs(d[1])),
             ("UP" if d[2] > 0 else "DOWN", abs(d[2]))]
    text = f"{name}: " + "  ".join(f"{w} {v:4.1f}" for w, v in words) + " cm"
    cv2.putText(frame, text, (8, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(frame, text, (8, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
