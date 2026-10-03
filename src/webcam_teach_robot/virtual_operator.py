"""조건 ④: 깊이 오차만 넣은 시범을 만드는 "가상 조작자".

사람 대신 프로그램이 손을 움직인다. 손 위치는 정확히 알고, 웹캠 조종과 **같은 코드**(`teleop_rig.TeleopRig`:
깊이 게이트·평활·클러치·집게 스위치·집기 잠금·작업 범위·IK)를 거쳐 로봇을 움직인다. 웹캠과 다른 점은 손 관측을
만들어 넣는다는 것뿐이고, 그 관측의 깊이에 오차를 넣느냐 빼느냐로 두 조건을 만든다.

    ④0 (depth_error=False): 손 깊이를 정확히 넣는다 → 조종 처리·조작 방식만의 영향
    ④  (depth_error=True) : 손 깊이에 실측 오차를 넣는다 → ④0과의 차이가 깊이 오차의 몫

넣는 오차 (모두 실측에서 가져옴):
    1. 자세에 따른 치우침: 실제 ② 시범 하나를 골라, 그 시범의 단계(집기 전 / 집고 있는 동안 / 놓은 뒤)별
       손 자세 특징을 같은 단계에 맞춰 가져오고, M5 측정으로 맞춘 자세-치우침 모델(depth_correction.json)로
       "그 자세일 때 웹캠이 읽는 거리 / 실제 거리"를 구한다.
    2. 프레임마다 흔들림: 같은 ② 시범의 손 깊이(보정 전)에서 9프레임 중앙값을 뺀 나머지(로그).
깊이 오차는 웹캠과 같은 계산(hand_point_camera)을 거쳐 좌우·위아래로도 번진다.

가상 조작자는 화면의 로봇을 보고 손을 움직인다(되먹임): 다음 경유점까지 남은 거리를 보고 손을 옮기고, 오차 때문에
로봇이 엇나가면 그만큼 다시 고친다. 경유점은 조건 ①과 같은 `scripted.plan`.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from webcam_teach_robot.depth_correction import DepthCorrection, posture_features
from webcam_teach_robot.hand_tracking import HandObservation, focal_length_px
from webcam_teach_robot.scripted import GRIP_WAIT_S, HOVER_Z, SPEED, Waypoint, plan
from webcam_teach_robot.teleop_mapping import GRASP_LOCK_AFTER_S
from webcam_teach_robot.teleop_rig import RigSettings, TeleopRig

FPS = 30
W, H = 640, 480
HAND_START = np.array([0.0, 0.03, 0.50])  # 카메라 좌표(m): 화면 가운데 조금 아래, 50 cm (녹화 안내와 같은 거리)
PINCH_OPEN, PINCH_CLOSED = 1.3, 0.45  # ② 녹화의 편 손·집은 손 중앙값 근처
PINCH_RAMP_FRAMES = 6  # 손가락을 붙이거나 펴는 데 0.2초
TOL_GRASP = 0.004  # 집기·놓기 경유점 도착 판정 (m)
TOL_MOVE = 0.01
MAX_S = 60.0
LAG_HOLD = 0.015


@dataclass
class DemoErrorSource:
    """실제 ② 시범 하나에서 뽑은 오차 재료."""
    name: str
    phases: dict[str, np.ndarray]  # "approach" / "carry" / "after" → 프레임별 (웹캠이 읽는 거리 / 실제 거리)
    jitter: np.ndarray  # 프레임별 log 깊이 흔들림


def load_error_source(csv_path: Path, correction: DepthCorrection) -> DemoErrorSource:
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r["engaged"] == "1" and r["px0_u"]]
    P = np.array([[[float(r[f"px{i}_u"]), float(r[f"px{i}_v"])] for i in range(21)] for r in rows])
    Wl = np.array([[[float(r[f"w{i}_{a}"]) for a in "xyz"] for i in range(21)] for r in rows])
    d = np.array([min(float(r["depth_len_m"]), float(r["depth_width_m"])) for r in rows])
    closed = np.array([r["gripper_closed"] == "1" for r in rows])
    ratio = np.array([1.0 / correction.factor(P[k], Wl[k], d[k], W, H) for k in range(len(rows))])
    first_close = int(np.argmax(closed)) if closed.any() else len(rows)
    last_close = len(rows) - int(np.argmax(closed[::-1])) if closed.any() else len(rows)
    phases = {"approach": ratio[:first_close], "carry": ratio[first_close:last_close], "after": ratio[last_close:]}
    phases = {k: (v if len(v) else ratio) for k, v in phases.items()}
    ld = np.log(d)
    med = np.array([np.median(ld[max(0, k - 4):k + 5]) for k in range(len(ld))])
    return DemoErrorSource(csv_path.name, phases, ld - med)


def fake_obs(hand: np.ndarray, depth_seen: float, pinch: float) -> HandObservation:
    """손 위치(카메라 좌표, 실제)를 웹캠이 본 것처럼: 화면 위치는 실제 투영, 깊이만 depth_seen."""
    f = focal_length_px(W, 60.0)
    c = np.array([W / 2 + f * hand[0] / hand[2], H / 2 + f * hand[1] / hand[2]])
    z = np.zeros((21, 3))
    return HandObservation(pixels=np.tile(c, (21, 1)), rel_z=np.zeros(21), world=z, handedness="virtual", palm_px=0.0,
                           palm_m=0.0, depth_len_m=depth_seen, depth_width_m=depth_seen, depth_m=depth_seen,
                           palm_center_px=c, pinch=pinch)


def robot_to_hand_delta(d_robot: np.ndarray, s: RigSettings) -> np.ndarray:
    """camera_delta_to_robot의 역: 로봇을 d_robot만큼 움직이려면 손을 카메라 좌표로 얼마나 옮기나."""
    return np.array([-d_robot[1] / s.scale, -d_robot[2] / s.scale, -d_robot[0] / s.scale_depth])


def run_episode(rig: TeleopRig, cube_xy, goal_xy, source: DemoErrorSource | None, writer, layout_id: int) -> dict:
    """한 배치를 가상 조작자로 끝까지. writer(csv.writer)에 웹캠 녹화와 같은 형식으로 기록한다."""
    s = rig.s
    rig.reset(np.asarray(cube_xy), np.asarray(goal_xy), 0.0)
    rig.hand_filter.__init__(smooth=s.smooth)
    hand = HAND_START.copy()
    pinch = PINCH_OPEN
    phase = "approach"
    phase_k = {"approach": 0, "carry": 0, "after": 0}
    waypoints = plan(np.asarray(cube_xy), np.asarray(goal_xy), rig.tip())[1:]
    wi, wait_left, ramp = 0, 0, None
    first_success_t = None
    done_k = None
    k = 0
    while True:
        t = k / FPS
        if source is not None:
            seq = source.phases[phase]
            ratio = seq[min(phase_k[phase], len(seq) - 1)]
            depth_seen = hand[2] * ratio * np.exp(source.jitter[k % len(source.jitter)])
        else:
            depth_seen = hand[2]
        rig.update_hand(fake_obs(hand, depth_seen, pinch), W, H, t, lambda: 0.0)
        if k == 12:
            rig.toggle_clutch()  # 기준점(10프레임 중앙값)이 찬 뒤 조종 시작
        if rig.step_sim(t):
            first_success_t = t
        if writer is not None:
            writer.writerow(rig.log_row(t, fake_obs(hand, depth_seen, pinch), 0.0, 1000 / FPS, layout_id))
        phase_k[phase] += 1
        k += 1
        if first_success_t is not None and t - first_success_t > 1.5:
            return {"success": True, "seconds": round(first_success_t, 2), "frames": k}
        if t > MAX_S or (done_k is not None and k > done_k + 2 * FPS and first_success_t is None):
            return {"success": False, "seconds": round(t, 2), "frames": k}
        if not rig.engaged:
            continue

        # ---- 손가락 여닫기 (손은 가만히) ----
        if ramp is not None:
            a, b, n = ramp
            pinch = a + (b - a) * min(1.0, n / PINCH_RAMP_FRAMES)
            ramp = (a, b, n + 1)
            if n >= PINCH_RAMP_FRAMES:
                ramp = None
                wait_left = int((GRIP_WAIT_S + GRASP_LOCK_AFTER_S) * FPS)
                if b == PINCH_CLOSED:
                    phase = "carry"
                else:
                    phase = "after"
            continue
        if wait_left > 0:
            wait_left -= 1
            continue
        if wi >= len(waypoints):
            continue
        wp: Waypoint = waypoints[wi]
        want_closed = wp.gripper < 0.5
        if want_closed != rig.gripper_switch.closed and np.linalg.norm(rig.tip() - wp.pos) < TOL_GRASP * 2:
            ramp = (pinch, PINCH_CLOSED if want_closed else PINCH_OPEN, 1)
            continue

        # ---- 화면의 로봇 집게 끝을 보고 손을 옮긴다 ----
        if np.linalg.norm(rig.target - rig.tip()) > LAG_HOLD:
            continue  # 로봇이 아직 따라오는 중: 손을 더 밀지 않고 기다린다(사람도 로봇이 따라오길 본다)
        err = wp.pos - rig.tip()
        tol = TOL_GRASP if wp.pos[2] < HOVER_Z - 1e-6 else TOL_MOVE
        if np.linalg.norm(err) < tol and want_closed == rig.gripper_switch.closed:
            wi += 1
            if wi >= len(waypoints):
                done_k = k
            continue
        step = err if np.linalg.norm(err) < SPEED / FPS else err / np.linalg.norm(err) * SPEED / FPS
        hand = hand + robot_to_hand_delta(step, s)


def write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
