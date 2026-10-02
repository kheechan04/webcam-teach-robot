"""웹캠 조종 한 판의 핵심: 손 관측 → 목표 위치 → IK → 시뮬레이션 → 성공 판정 → 기록 한 줄.

scripts/teleop.py(자유 조종)와 scripts/record_demos.py(M8 시범 녹화)가 같이 쓴다. 조건 ②와 ③이 완전히 같은 코드로
돌고, 차이는 깊이 보정(DepthCorrection)을 켜느냐뿐이다.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from webcam_teach_robot.depth_correction import DepthCorrection
from webcam_teach_robot.hand_tracking import HAND_CONNECTIONS, HandObservation
from webcam_teach_robot.ik import SO101IK
from webcam_teach_robot.scene import PlacementTracker, build_model, cube_pos, place, task_ids
from webcam_teach_robot.teleop_mapping import (GraspLock, GripperSwitch, HandFilter, camera_delta_to_robot,
                                               hand_point_camera)
from webcam_teach_robot.teleop_view import draw_guides

ROOT = Path(__file__).resolve().parents[2]

# 처음 자세: 집게 끝이 작업 상자 앞뒤 가운데(x=22, z=6 cm)에서 아래를 향하게 IK로 구함.
# 처음엔 x=19 cm라서 몸 쪽 여유가 5 cm뿐이었고, 세 번째 조종에서 몸 쪽 한계에 붙은 시간이 45%였다.
HOME_Q = np.array([0.0, -0.21, 0.346, 1.434, 0.0, 1.0])  # 집게는 열린 채로 시작
# 목표 위치를 이 상자 안으로 제한 (m): 집게를 아래로 향한 채(방향 오차 <10°) 위치 오차 2 mm 안에 닿는 영역.
#   1차: x 10~35, z 1~25 cm → 팔이 안 닿아 x 상한에 막힌 프레임 36%, IK 오차 >10 mm 46%
#   2차: x 14~30, z 1~15 cm (y=0 단면만 봄) → 재생해 보니 몸 가까이·높은 곳에서 어깨가 한계(-100°)까지 젖혀지고
#        집게가 43° 기울거나 모터가 못 버텨 4 cm 처짐
#   3차(지금): 3D 격자로 확인한 x 16~28, y ±15, z 1~8 cm
WORKSPACE_LO = np.array([0.16, -0.15, 0.01])
WORKSPACE_HI = np.array([0.28, 0.15, 0.08])

# MediaPipe가 찾은 손 마디 21개 원래 값. 조건 ③ 보정·조건 ④ 오차 패턴을 만들 때 손 자세를 다시 분석하려고 남긴다.
# (화면 픽셀 u·v, 손목 기준 상대 z, 손 중심 기준 월드 좌표 x·y·z) — measure_depth.py와 같은 열 이름.
LANDMARK_COLUMNS = ([f"px{i}_{a}" for i in range(21) for a in ("u", "v")] + [f"relz{i}" for i in range(21)]
                    + [f"w{i}_{a}" for i in range(21) for a in ("x", "y", "z")])
LOG_COLUMNS = (["t_s", "engaged", "hand_found", "detect_ms", "ik_ms", "loop_ms",
                "hand_x", "hand_y", "hand_z", "depth_len_m", "depth_width_m", "pinch", "gated", "gripper_closed",
                "target_x", "target_y", "target_z", "tip_x", "tip_y", "tip_z",
                "ik_pos_err_mm", "ik_dir_err_deg", "depth_used_m"] + [f"q{i}" for i in range(6)]
               + [f"ctrl{i}" for i in range(6)]
               + ["layout", "cube_x", "cube_y", "cube_z", "goal_x", "goal_y", "placed_success"]
               + ["cube_qw", "cube_qx", "cube_qy", "cube_qz", "handedness", "grasp_locked"] + LANDMARK_COLUMNS)


@dataclass
class RigSettings:
    fov: float = 60.0
    # 0.7이었는데 큐브(오른쪽)→목표(왼쪽) 최대 24 cm를 옮기려면 손을 옆으로 약 34 cm 움직여야 해서
    # 45 cm 거리에서도 손이 웹캠 화면 밖으로 나갔다(2026-10-02 사용자 소감). 1.0이면 약 24 cm.
    scale: float = 1.0
    scale_depth: float = 0.5  # 앞뒤(카메라 쪽). 깊이는 범위가 넓고 흔들려서 좌우·위아래보다 작게
    smooth: float = 0.5


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          cwd=ROOT).stdout.strip()


def run_meta(settings: RigSettings, correction: DepthCorrection | None, frame_size, extra: dict) -> dict:
    """기록 CSV 옆에 남기는 실행 설정(나중에 같은 조건인지 확인하려고). 영상은 저장하지 않는다."""
    return {**extra, "settings": vars(settings), "git_commit": git_commit(), "workspace_lo": WORKSPACE_LO.tolist(),
            "workspace_hi": WORKSPACE_HI.tolist(), "home_q": HOME_Q.tolist(), "frame_size": list(frame_size),
            "mirror": True, "condition": 3 if correction else 2,
            "depth_correction": json.loads(DepthCorrection.coef_text()) if correction else None}


def add_marker(viewer, pos: np.ndarray, rgba) -> None:
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.008, 0, 0]),
                        pos, np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    scn.ngeom += 1


def setup_viewer_camera(viewer) -> None:
    viewer.cam.lookat[:] = [0.2, 0.0, 0.08]
    viewer.cam.distance = 0.75
    viewer.cam.azimuth = 0  # 로봇 뒤에서 앞(+x)을 보는 시점: 너와 같은 방향. (처음에 180으로 해서 좌우가 거울처럼 뒤집혀 보였다)
    viewer.cam.elevation = -25


class TeleopRig:
    def __init__(self, settings: RigSettings, correction: DepthCorrection | None = None):
        self.s = settings
        self.correction = correction
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.ids = task_ids(self.model)
        self.ik = SO101IK(self.model)
        self.hand_filter = HandFilter(smooth=settings.smooth)

    # ---- 한 판 시작 ----
    def reset(self, cube_xy: np.ndarray, goal_xy: np.ndarray, now: float) -> None:
        """로봇을 처음 자세로, 큐브·목표를 놓고, 조종 상태를 새로. 손 추적 평활 기록(hand_filter)은 이어 간다."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:6] = HOME_Q
        d.ctrl[:6] = HOME_Q
        place(m, d, self.ids, cube_xy, goal_xy)
        self.goal_xy = np.asarray(goal_xy, float)
        self.placement = PlacementTracker()
        self.success = False
        self.engaged = False
        self.gripper_switch = GripperSwitch()
        self.grasp_lock = GraspLock()
        self.locked = False
        self.gated = False
        self.hand_ref = self.tip_ref = None
        self.target = self.ik.tip(d.qpos.copy())[0]
        self.q_ik = d.qpos.copy()
        self.gripper = HOME_Q[5]
        self.t_sim0 = now  # 시뮬레이션 시각 0에 해당하는 실제 시각
        self.ik_res = None
        self.ik_ms = 0.0
        self.depth_used = float("nan")
        self.hand_smooth = None

    # ---- 매 프레임 ----
    def update_hand(self, obs: HandObservation | None, frame_w: int, frame_h: int, now: float, perf) -> None:
        """손 관측을 받아 목표·IK·관절 명령까지. perf는 time.perf_counter (IK 시간 재기용)."""
        self.ik_res, self.ik_ms, self.gated, self.depth_used = None, 0.0, False, float("nan")
        if obs is None or not np.isfinite(obs.depth_m):
            self.hand_filter.lost()
            return
        self.hand_filter.found()
        self.depth_used = (self.correction(obs.pixels, obs.world, obs.depth_m, frame_w, frame_h)
                           if self.correction else obs.depth_m)
        hand = hand_point_camera(obs.palm_center_px, self.depth_used, frame_w, frame_h, self.s.fov)
        self.hand_smooth, self.gated = self.hand_filter.update(hand)
        if not self.engaged:
            return
        was_closed = self.gripper_switch.closed
        self.gripper = self.gripper_switch.update(obs.pinch)
        self.locked, released = self.grasp_lock.update(obs.pinch, self.gripper_switch.closed != was_closed, now)
        if released:  # 집기 잠금이 풀리는 순간: 지금 손 위치 = 지금 목표로 기준을 다시 잡는다
            self.hand_ref = self.hand_smooth.copy()
            self.tip_ref = self.target.copy()
        scale = np.array([self.s.scale_depth, self.s.scale, self.s.scale])  # 로봇 x(앞뒤), y, z
        raw = self.tip_ref + scale * camera_delta_to_robot(self.hand_smooth - self.hand_ref)
        if self.locked:  # 손가락을 붙이거나 벌리는 중: 팔은 그대로
            raw = self.target.copy()
        self.target = np.clip(raw, WORKSPACE_LO, WORKSPACE_HI)
        # 작업 범위 밖으로 넘친 만큼은 버린다(기준점을 같이 밀어 줌). 안 그러면 손을 되돌려도
        # 넘친 만큼 돌아올 때까지 로봇이 안 움직인다 — 큐브 옮기기 첫 시도에서 높이 상한(8 cm)에
        # 막힌 프레임이 27%였고 "내리는 게 인식이 잘 안 된다"는 소감이 나왔다.
        self.tip_ref = self.tip_ref + (self.target - raw)
        t_ik = perf()
        self.ik_res = self.ik.solve(self.target, self.q_ik)
        self.ik_ms = (perf() - t_ik) * 1000
        self.q_ik = self.ik_res.q
        self.data.ctrl[:5] = self.q_ik[:5]
        self.data.ctrl[5] = self.gripper

    def step_sim(self, now: float) -> bool:
        """물리 시뮬레이션을 실제 시간에 맞춰 진행하고 성공 판정. 처음 성공한 프레임에서만 True."""
        sim_dt = 0.0
        while self.data.time < now - self.t_sim0:
            mujoco.mj_step(self.model, self.data)
            sim_dt += self.model.opt.timestep
        done = self.placement.update(self.cube(), self.goal_xy, self.gripper_switch.closed, sim_dt)
        first = done and not self.success
        self.success = done
        return first

    # ---- 키 ----
    def toggle_clutch(self) -> None:
        """스페이스: 조종 시작/멈춤. 시작하는 순간의 손 위치 = 로봇의 지금 위치로 맞춘다."""
        ref = self.hand_filter.reference()
        if not self.engaged and ref is not None:
            self.hand_ref = ref
            self.tip_ref = self.ik.tip(self.data.qpos.copy())[0]
            self.q_ik = self.data.qpos.copy()
            self.engaged = True
        else:
            self.engaged = False

    def home(self) -> None:
        self.engaged = False
        self.data.ctrl[:6] = HOME_Q
        self.q_ik[:6] = HOME_Q
        self.target = self.ik.tip(np.concatenate([HOME_Q, self.data.qpos[6:]]))[0]

    # ---- 보조 ----
    def cube(self) -> np.ndarray:
        return cube_pos(self.data, self.ids)

    def cube_quat(self) -> np.ndarray:
        return self.data.qpos[self.ids.cube_qadr + 3:self.ids.cube_qadr + 7]

    def tip(self) -> np.ndarray:
        return self.ik.tip(self.data.qpos.copy())[0]

    def draw_viewer(self, viewer) -> None:
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            if self.engaged:
                add_marker(viewer, self.target, [1.0, 0.3, 0.2, 0.8])  # 빨강: 목표
        viewer.sync()

    def draw_frame(self, frame, obs: HandObservation | None) -> None:
        """웹캠 창에 2D 안내 그림과 손 마디를 그린다."""
        draw_guides(frame, self.tip(), self.gripper_switch.closed, self.cube(), self.cube_quat(), self.goal_xy,
                    WORKSPACE_LO, WORKSPACE_HI)
        if obs is not None:
            pts = obs.pixels.astype(int)
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 200, 0), 2)
            cv2.circle(frame, tuple(obs.palm_center_px.astype(int)), 7, (0, 0, 255), -1)

    def status(self) -> str:
        return (("ENGAGED" if self.engaged else "PAUSED (space)") + ("  CLOSED" if self.gripper_switch.closed else "  open")
                + ("  LOCK" if self.locked and self.engaged else ""))

    def log_row(self, t_s: float, obs: HandObservation | None, detect_ms: float, loop_ms: float, layout) -> list:
        hs = self.hand_smooth if self.hand_smooth is not None else [np.nan] * 3
        r = self.ik_res
        return [f"{t_s:.4f}", int(self.engaged), int(obs is not None), f"{detect_ms:.2f}", f"{self.ik_ms:.2f}",
                f"{loop_ms:.2f}", *[f"{v:.5f}" for v in hs],
                f"{obs.depth_len_m:.5f}" if obs else "", f"{obs.depth_width_m:.5f}" if obs else "",
                f"{obs.pinch:.4f}" if obs else "", int(self.gated), int(self.gripper_switch.closed),
                *[f"{v:.5f}" for v in self.target], *[f"{v:.5f}" for v in self.tip()],
                f"{r.pos_err_m * 1000:.2f}" if r else "", f"{r.dir_err_deg:.2f}" if r else "", f"{self.depth_used:.5f}",
                *[f"{v:.5f}" for v in self.data.qpos[:6]], *[f"{v:.5f}" for v in self.data.ctrl[:6]],
                layout, *[f"{v:.5f}" for v in self.cube()], *[f"{v:.5f}" for v in self.goal_xy], int(self.success),
                *[f"{v:.5f}" for v in self.cube_quat()], obs.handedness if obs else "",
                int(self.locked and self.engaged), *landmark_values(obs)]


def landmark_values(obs) -> list[str]:
    if obs is None:
        return [""] * len(LANDMARK_COLUMNS)
    return ([f"{v:.2f}" for v in obs.pixels.ravel()] + [f"{v:.5f}" for v in obs.rel_z]
            + [f"{v:.5f}" for v in obs.world.ravel()])


def put_lines(frame, lines: list[str], color=(80, 255, 80), y0: int = 28) -> None:
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, y0 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y0 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
