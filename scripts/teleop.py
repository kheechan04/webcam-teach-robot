"""웹캠 손 추적으로 시뮬레이션 SO-101을 조종해 큐브를 목표(초록 사각형)로 옮긴다. 영상은 저장하지 않는다.

실행:
    uv run python scripts/teleop.py
    uv run python scripts/teleop.py --log        # 숫자 기록을 measurements/teleop/에 저장

창이 두 개 뜬다: 웹캠 창(손 추적 상태)과 MuJoCo 창(로봇).
웹캠 창을 클릭해서 선택한 상태에서:
    스페이스  조종 시작/멈춤 (클러치). 시작하는 순간의 손 위치 = 로봇의 지금 위치로 맞춘다
    h         로봇을 처음 자세로
    r         새 배치(큐브·목표 위치를 무작위로 다시)로 처음부터
    q / Esc   끝내기

손 → 로봇 방향 (로봇이 너와 같은 쪽, 카메라 쪽을 보고 서 있다고 생각하면 된다):
    손을 카메라 쪽으로 밀기  → 로봇 집게가 앞으로
    손을 오른쪽으로          → 로봇도 (너 기준) 오른쪽으로
    손을 위로                → 로봇도 위로
    엄지와 검지를 붙이기      → 집게 닫기 (딸깍 스위치: 붙이면 닫히고, 충분히 벌려야 열린다)
    손목 돌리기(roll)는 연결하지 않았다. 과제를 물체가 항상 같은 방향으로 놓이게 짜서 깊이 외 오차 원인을 줄인다.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from webcam_teach_robot.hand_tracking import HAND_CONNECTIONS, HandTracker
from webcam_teach_robot.ik import SO101IK
from webcam_teach_robot.scene import build_model, cube_pos, in_target, place, sample_layout, task_ids
from webcam_teach_robot.teleop_view import draw_guides
from webcam_teach_robot.teleop_mapping import (GripperSwitch, HandFilter, camera_delta_to_robot,
                                               hand_point_camera)

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "measurements" / "teleop"

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


def add_marker(viewer, pos: np.ndarray, rgba) -> None:
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.008, 0, 0]),
                        pos, np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    scn.ngeom += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fov", type=float, default=60.0, help="웹캠 가로 화각 가정값(도)")
    parser.add_argument("--depth", choices=["min", "width", "length"], default="min", help="깊이 추정에 쓸 손바닥 구간")
    # 0.7이었는데 큐브(오른쪽)→목표(왼쪽) 최대 24 cm를 옮기려면 손을 옆으로 약 34 cm 움직여야 해서
    # 45 cm 거리에서도 손이 웹캠 화면 밖으로 나갔다(2026-10-02 사용자 소감). 1.0이면 약 24 cm.
    parser.add_argument("--scale", type=float, default=1.0, help="손 이동량 → 로봇 이동량 배율 (좌우·위아래)")
    parser.add_argument("--scale-depth", type=float, default=0.5,
                        help="앞뒤(카메라 쪽) 배율. 깊이는 범위가 넓고 흔들려서 좌우·위아래보다 작게")
    parser.add_argument("--smooth", type=float, default=0.5,
                        help="손 위치 지수평활 계수(0~1). 1이면 평활 없음, 작을수록 부드럽지만 늦게 따라온다")
    parser.add_argument("--log", action="store_true", help="프레임별 숫자를 CSV로 저장")
    parser.add_argument("--seed", type=int, default=None, help="배치 무작위 시드 (기본: 매번 다름)")
    args = parser.parse_args()

    model = build_model()
    data = mujoco.MjData(model)
    ids = task_ids(model)
    ik = SO101IK(model)
    rng = np.random.default_rng(args.seed)

    def reset_layout():
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = HOME_Q
        data.ctrl[:6] = HOME_Q
        cube_xy, target_xy = sample_layout(rng)
        place(model, data, ids, cube_xy, target_xy)
        return target_xy

    target_xy = reset_layout()
    layout_id = 0
    successes = 0
    success_now = False

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.camera}번을 열 수 없어요.")

    log_file = writer = None
    if args.log:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.csv"
        log_file = open(log_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(log_file)
        writer.writerow(["t_s", "engaged", "hand_found", "detect_ms", "ik_ms", "loop_ms",
                         "hand_x", "hand_y", "hand_z", "depth_len_m", "depth_width_m", "pinch", "gated", "gripper_closed",
                         "target_x", "target_y", "target_z", "tip_x", "tip_y", "tip_z",
                         "ik_pos_err_mm", "ik_dir_err_deg"] + [f"q{i}" for i in range(6)] + [f"ctrl{i}" for i in range(6)]
                        + ["layout", "cube_x", "cube_y", "cube_z", "goal_x", "goal_y", "in_target"])

    engaged = False
    hand_ref = tip_ref = None
    hand_smooth = None
    hand_filter = HandFilter(smooth=args.smooth)
    gripper_switch = GripperSwitch()
    gated = False
    target = ik.tip(data.qpos.copy())[0]
    q_ik = data.qpos.copy()
    gripper = 0.0
    t0 = time.perf_counter()
    t_sim0 = t0  # 시뮬레이션 시각 0에 해당하는 실제 시각. 배치를 새로 하면 다시 맞춘다

    with HandTracker(horizontal_fov_deg=args.fov, depth_segment=args.depth) as tracker, \
            mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [0.2, 0.0, 0.08]
        viewer.cam.distance = 0.75
        viewer.cam.azimuth = 0  # 로봇 뒤에서 앞(+x)을 보는 시점: 너와 같은 방향. (처음에 180으로 해서 좌우가 거울처럼 뒤집혀 보였다)
        viewer.cam.elevation = -25

        while viewer.is_running():
            t_loop = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            t_det = time.perf_counter()
            obs = tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int((t_det - t0) * 1000))
            detect_ms = (time.perf_counter() - t_det) * 1000

            ik_ms = 0.0
            ik_res = None
            gated = False
            if obs is None or not np.isfinite(obs.depth_m):
                hand_filter.lost()
            else:
                hand_filter.found()
                hand = hand_point_camera(obs.palm_center_px, obs.depth_m, w, h, args.fov)
                hand_smooth, gated = hand_filter.update(hand)
                if engaged:
                    scale = np.array([args.scale_depth, args.scale, args.scale])  # 로봇 x(앞뒤), y, z
                    raw = tip_ref + scale * camera_delta_to_robot(hand_smooth - hand_ref)
                    target = np.clip(raw, WORKSPACE_LO, WORKSPACE_HI)
                    # 작업 범위 밖으로 넘친 만큼은 버린다(기준점을 같이 밀어 줌). 안 그러면 손을 되돌려도
                    # 넘친 만큼 돌아올 때까지 로봇이 안 움직인다 — 큐브 옮기기 첫 시도에서 높이 상한(8 cm)에
                    # 막힌 프레임이 27%였고 "내리는 게 인식이 잘 안 된다"는 소감이 나왔다.
                    tip_ref = tip_ref + (target - raw)
                    t_ik = time.perf_counter()
                    ik_res = ik.solve(target, q_ik)
                    ik_ms = (time.perf_counter() - t_ik) * 1000
                    q_ik = ik_res.q
                    gripper = gripper_switch.update(obs.pinch)
                    data.ctrl[:5] = q_ik[:5]
                    data.ctrl[5] = gripper

            # 물리 시뮬레이션을 실제 시간에 맞춰 진행
            sim_target_time = time.perf_counter() - t_sim0
            while data.time < sim_target_time:
                mujoco.mj_step(model, data)
            cube = cube_pos(data, ids)
            done = in_target(cube, target_xy) and not gripper_switch.closed
            if done and not success_now:
                successes += 1
            success_now = done

            tip = ik.tip(data.qpos.copy())[0]
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                if engaged:
                    add_marker(viewer, target, [1.0, 0.3, 0.2, 0.8])  # 빨강: 목표
            viewer.sync()

            # 웹캠 창 표시
            draw_guides(frame, tip, gripper_switch.closed, cube, data.qpos[ids.cube_qadr + 3:ids.cube_qadr + 7],
                        target_xy, WORKSPACE_LO, WORKSPACE_HI)
            if obs is not None:
                pts = obs.pixels.astype(int)
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 200, 0), 2)
                cv2.circle(frame, tuple(obs.palm_center_px.astype(int)), 7, (0, 0, 255), -1)
            status = ("ENGAGED" if engaged else "PAUSED (space)") + ("  CLOSED" if gripper_switch.closed else "  open")
            lines = [f"{status}  loop {(time.perf_counter() - t_loop) * 1000:4.0f} ms",
                     f"layout {layout_id}  {'SUCCESS (r: next)' if success_now else 'move cube to green'}  total {successes}",
                     f"depth {obs.depth_m * 100:5.1f} cm  pinch {obs.pinch:4.2f}" if obs else "hand: not found",
                     f"target ({target[0]*100:4.1f}, {target[1]*100:4.1f}, {target[2]*100:4.1f}) cm"]
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
                cv2.putText(frame, text, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (80, 255, 80) if engaged else (255, 255, 255), 2)
            cv2.imshow("teleop camera (not recorded)", frame)

            loop_ms = (time.perf_counter() - t_loop) * 1000
            if writer:
                hs = hand_smooth if hand_smooth is not None else [np.nan] * 3
                writer.writerow([f"{time.perf_counter() - t0:.4f}", int(engaged), int(obs is not None),
                                 f"{detect_ms:.2f}", f"{ik_ms:.2f}", f"{loop_ms:.2f}",
                                 *[f"{v:.5f}" for v in hs],
                                 f"{obs.depth_len_m:.5f}" if obs else "", f"{obs.depth_width_m:.5f}" if obs else "",
                                 f"{obs.pinch:.4f}" if obs else "", int(gated), int(gripper_switch.closed),
                                 *[f"{v:.5f}" for v in target], *[f"{v:.5f}" for v in tip],
                                 f"{ik_res.pos_err_m * 1000:.2f}" if ik_res else "",
                                 f"{ik_res.dir_err_deg:.2f}" if ik_res else "",
                                 *[f"{v:.5f}" for v in data.qpos[:6]], *[f"{v:.5f}" for v in data.ctrl[:6]],
                                 layout_id, *[f"{v:.5f}" for v in cube], *[f"{v:.5f}" for v in target_xy], int(success_now)])

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                ref = hand_filter.reference()
                if not engaged and ref is not None:
                    hand_ref = ref
                    tip_ref = ik.tip(data.qpos.copy())[0]
                    q_ik = data.qpos.copy()
                    engaged = True
                else:
                    engaged = False
            if key == ord("r"):
                engaged = False
                gripper_switch.closed = False
                target_xy = reset_layout()
                q_ik = data.qpos.copy()
                target = ik.tip(data.qpos.copy())[0]
                t_sim0 = time.perf_counter()
                layout_id += 1
                success_now = False
            if key == ord("h"):
                engaged = False
                data.ctrl[:6] = HOME_Q
                q_ik[:6] = HOME_Q
                target = ik.tip(np.concatenate([HOME_Q, data.qpos[6:]]))[0]

    cap.release()
    cv2.destroyAllWindows()
    if log_file:
        log_file.close()
        print(f"기록 저장: {log_path}")


if __name__ == "__main__":
    main()
