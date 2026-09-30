"""웹캠 손 추적으로 시뮬레이션 SO-101을 조종한다. 영상은 저장하지 않는다.

실행:
    uv run python scripts/teleop.py
    uv run python scripts/teleop.py --log        # 숫자 기록을 measurements/teleop/에 저장

창이 두 개 뜬다: 웹캠 창(손 추적 상태)과 MuJoCo 창(로봇).
웹캠 창을 클릭해서 선택한 상태에서:
    스페이스  조종 시작/멈춤 (클러치). 시작하는 순간의 손 위치 = 로봇의 지금 위치로 맞춘다
    h         로봇을 처음 자세로
    q / Esc   끝내기

손 → 로봇 방향 (로봇이 너와 같은 쪽, 카메라 쪽을 보고 서 있다고 생각하면 된다):
    손을 카메라 쪽으로 밀기  → 로봇 집게가 앞으로
    손을 오른쪽으로          → 로봇도 (너 기준) 오른쪽으로
    손을 위로                → 로봇도 위로
    엄지와 검지를 붙이기      → 집게 닫기
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

from webcam_teach_robot.hand_tracking import HAND_CONNECTIONS, HandTracker, focal_length_px
from webcam_teach_robot.ik import SO101IK

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "third_party" / "robotstudio_so101" / "scene.xml"
LOG_DIR = ROOT / "measurements" / "teleop"

HOME_Q = np.array([0.0, -0.5, 0.8, 1.2, 0.0, 0.0])  # 팔을 굽힌 처음 자세 (집게 끝 약 x=19, z=4 cm)
WORKSPACE_LO = np.array([0.10, -0.20, 0.01])  # 목표 위치를 이 상자 안으로 제한 (m)
WORKSPACE_HI = np.array([0.35, 0.20, 0.25])
GRIPPER_CLOSED, GRIPPER_OPEN = -0.17, 1.0  # 집게 관절 각도(rad)
PINCH_CLOSED, PINCH_OPEN = 0.35, 1.2  # 엄지-검지 거리 / 손바닥 너비. 이 사이를 선형으로 이어 준다


def hand_point_camera(obs, frame_w: int, frame_h: int, fov_deg: float) -> np.ndarray:
    """손바닥 중심을 카메라 좌표(m)로: x 오른쪽, y 아래, z 카메라에서 멀어지는 쪽."""
    f = focal_length_px(frame_w, fov_deg)
    u, v = obs.palm_center_px
    z = obs.depth_m
    return np.array([(u - frame_w / 2) * z / f, (v - frame_h / 2) * z / f, z])


def camera_delta_to_robot(d: np.ndarray) -> np.ndarray:
    """카메라 좌표의 손 이동량 → 로봇 좌표(x 앞, y 왼쪽, z 위)의 이동량. 화면은 좌우 반전된 상태."""
    dx_cam, dy_cam, dz_cam = d
    return np.array([-dz_cam, -dx_cam, -dy_cam])


def pinch_to_gripper(pinch: float) -> float:
    t = np.clip((pinch - PINCH_CLOSED) / (PINCH_OPEN - PINCH_CLOSED), 0.0, 1.0)
    return GRIPPER_CLOSED + t * (GRIPPER_OPEN - GRIPPER_CLOSED)


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
    parser.add_argument("--depth", choices=["width", "length"], default="width", help="깊이 추정에 쓸 손바닥 구간")
    parser.add_argument("--scale", type=float, default=0.7, help="손 이동량 → 로봇 이동량 배율")
    parser.add_argument("--smooth", type=float, default=0.5,
                        help="손 위치 지수평활 계수(0~1). 1이면 평활 없음, 작을수록 부드럽지만 늦게 따라온다")
    parser.add_argument("--log", action="store_true", help="프레임별 숫자를 CSV로 저장")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    ik = SO101IK(model)
    data.qpos[:6] = HOME_Q
    data.ctrl[:6] = HOME_Q
    mujoco.mj_forward(model, data)

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
                         "hand_x", "hand_y", "hand_z", "depth_len_m", "depth_width_m", "pinch",
                         "target_x", "target_y", "target_z", "tip_x", "tip_y", "tip_z",
                         "ik_pos_err_mm", "ik_dir_err_deg"] + [f"q{i}" for i in range(6)] + [f"ctrl{i}" for i in range(6)])

    engaged = False
    hand_ref = tip_ref = None
    hand_smooth = None
    target = ik.tip(data.qpos.copy())[0]
    q_ik = data.qpos.copy()
    gripper = 0.0
    t0 = time.perf_counter()

    with HandTracker(horizontal_fov_deg=args.fov, depth_segment=args.depth) as tracker, \
            mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [0.2, 0.0, 0.08]
        viewer.cam.distance = 0.75
        viewer.cam.azimuth = 180  # 로봇 뒤에서 앞을 보는 시점: 너와 같은 방향
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
            if obs is not None and np.isfinite(obs.depth_m):
                hand = hand_point_camera(obs, w, h, args.fov)
                hand_smooth = hand if hand_smooth is None else args.smooth * hand + (1 - args.smooth) * hand_smooth
                if engaged:
                    target = tip_ref + args.scale * camera_delta_to_robot(hand_smooth - hand_ref)
                    target = np.clip(target, WORKSPACE_LO, WORKSPACE_HI)
                    t_ik = time.perf_counter()
                    ik_res = ik.solve(target, q_ik)
                    ik_ms = (time.perf_counter() - t_ik) * 1000
                    q_ik = ik_res.q
                    gripper = pinch_to_gripper(obs.pinch)
                    data.ctrl[:5] = q_ik[:5]
                    data.ctrl[5] = gripper

            # 물리 시뮬레이션을 실제 시간에 맞춰 진행
            sim_target_time = time.perf_counter() - t0
            while data.time < sim_target_time:
                mujoco.mj_step(model, data)

            tip = ik.tip(data.qpos.copy())[0]
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                if engaged:
                    add_marker(viewer, target, [1.0, 0.3, 0.2, 0.8])  # 빨강: 목표
            viewer.sync()

            # 웹캠 창 표시
            if obs is not None:
                pts = obs.pixels.astype(int)
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 200, 0), 2)
                cv2.circle(frame, tuple(obs.palm_center_px.astype(int)), 7, (0, 0, 255), -1)
            status = "ENGAGED" if engaged else "PAUSED (space)"
            lines = [f"{status}  loop {(time.perf_counter() - t_loop) * 1000:4.0f} ms",
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
                                 f"{obs.pinch:.4f}" if obs else "",
                                 *[f"{v:.5f}" for v in target], *[f"{v:.5f}" for v in tip],
                                 f"{ik_res.pos_err_m * 1000:.2f}" if ik_res else "",
                                 f"{ik_res.dir_err_deg:.2f}" if ik_res else "",
                                 *[f"{v:.5f}" for v in data.qpos[:6]], *[f"{v:.5f}" for v in data.ctrl[:6]]])

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if not engaged and hand_smooth is not None:
                    hand_ref = hand_smooth.copy()
                    tip_ref = ik.tip(data.qpos.copy())[0]
                    q_ik = data.qpos.copy()
                    engaged = True
                else:
                    engaged = False
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
