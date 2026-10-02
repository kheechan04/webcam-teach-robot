"""웹캠 손 추적으로 시뮬레이션 SO-101을 조종해 큐브를 목표(초록 사각형)로 옮긴다. 영상은 저장하지 않는다.

실행:
    uv run python scripts/teleop.py
    uv run python scripts/teleop.py --log        # 숫자 기록을 measurements/teleop/에 저장
    uv run python scripts/teleop.py --depth-correction   # 조건 ③: 손 자세에 따른 깊이 치우침 보정

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
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco.viewer
import numpy as np

from webcam_teach_robot.depth_correction import DepthCorrection
from webcam_teach_robot.hand_tracking import HandTracker
from webcam_teach_robot.scene import sample_layout
from webcam_teach_robot.teleop_rig import (LOG_COLUMNS, RigSettings, TeleopRig, put_lines, run_meta,
                                           setup_viewer_camera)

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "measurements" / "teleop"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fov", type=float, default=60.0, help="웹캠 가로 화각 가정값(도)")
    parser.add_argument("--depth", choices=["min", "width", "length"], default="min", help="깊이 추정에 쓸 손바닥 구간")
    parser.add_argument("--scale", type=float, default=1.0, help="손 이동량 → 로봇 이동량 배율 (좌우·위아래)")
    parser.add_argument("--scale-depth", type=float, default=0.5,
                        help="앞뒤(카메라 쪽) 배율. 깊이는 범위가 넓고 흔들려서 좌우·위아래보다 작게")
    parser.add_argument("--smooth", type=float, default=0.5,
                        help="손 위치 지수평활 계수(0~1). 1이면 평활 없음, 작을수록 부드럽지만 늦게 따라온다")
    parser.add_argument("--log", action="store_true", help="프레임별 숫자를 CSV로 저장")
    parser.add_argument("--seed", type=int, default=None, help="배치 무작위 시드 (기본: 매번 다름)")
    parser.add_argument("--depth-correction", action="store_true",
                        help="조건 ③: 손 자세 특징으로 깊이 치우침 보정 (계수: src/webcam_teach_robot/depth_correction.json)")
    args = parser.parse_args()
    correction = DepthCorrection() if args.depth_correction else None
    if correction and (args.fov != 60.0 or args.depth != "min"):
        raise SystemExit("보정 계수는 화각 60°, --depth min 기준으로 맞춘 것이에요.")

    rig = TeleopRig(RigSettings(fov=args.fov, scale=args.scale, scale_depth=args.scale_depth, smooth=args.smooth),
                    correction)
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    rig.reset(*sample_layout(rng), now=0.0)
    layout_id = 0
    successes = 0

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
        writer.writerow(LOG_COLUMNS)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        log_path.with_suffix(".json").write_text(json.dumps(
            run_meta(rig.s, correction, frame_size, {"args": vars(args)}), ensure_ascii=False, indent=2),
            encoding="utf-8")
    last_flush = t0

    with HandTracker(horizontal_fov_deg=args.fov, depth_segment=args.depth) as tracker,             mujoco.viewer.launch_passive(rig.model, rig.data, show_left_ui=False, show_right_ui=False) as viewer:
        setup_viewer_camera(viewer)
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
            rig.update_hand(obs, w, h, time.perf_counter() - t0, time.perf_counter)
            if rig.step_sim(time.perf_counter() - t0):
                successes += 1
            rig.draw_viewer(viewer)

            rig.draw_frame(frame, obs)
            depth_text = (f"depth {rig.depth_used * 100:5.1f} cm" + (f" (raw {obs.depth_m * 100:4.1f})" if correction else "")
                          + f"  pinch {obs.pinch:4.2f}") if obs else "hand: not found"
            put_lines(frame, [f"{rig.status()}  loop {(time.perf_counter() - t_loop) * 1000:4.0f} ms",
                              f"layout {layout_id}  {'SUCCESS (r: next)' if rig.success else 'move cube to green'}"
                              f"  total {successes}", depth_text,
                              f"target ({rig.target[0]*100:4.1f}, {rig.target[1]*100:4.1f}, {rig.target[2]*100:4.1f}) cm"],
                      (80, 255, 80) if rig.engaged else (255, 255, 255))
            cv2.imshow("teleop camera (not recorded)", frame)

            if writer:
                writer.writerow(rig.log_row(time.perf_counter() - t0, obs, detect_ms,
                                            (time.perf_counter() - t_loop) * 1000, layout_id))
                if time.perf_counter() - last_flush > 1.0:  # 갑자기 꺼져도 잃는 건 최대 1초
                    log_file.flush()
                    last_flush = time.perf_counter()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                rig.toggle_clutch()
            if key == ord("r"):
                if log_file:
                    log_file.flush()
                rig.reset(*sample_layout(rng), now=time.perf_counter() - t0)
                layout_id += 1
            if key == ord("h"):
                rig.home()

    cap.release()
    cv2.destroyAllWindows()
    if log_file:
        log_file.close()
        print(f"기록 저장: {log_path}")


if __name__ == "__main__":
    main()
