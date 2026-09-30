"""웹캠 깊이 추정 오차 측정 (1단계).

손을 카메라에서 정해진 거리(자로 잰 참값)에 두고, 핀홀로 추정한 거리와 비교한다.
화면 안내에 따라 손만 움직이면 된다. 숫자(손 좌표·추정 거리)만 CSV로 저장하고 영상은 저장하지 않는다.

준비:
    줄자를 웹캠 바로 앞에서 책상 위로 똑바로 펼친다. 거리는 화면(웹캠) 면에서 손바닥까지.

실행:
    uv run python scripts/measure_depth.py
    uv run python scripts/measure_depth.py --palm-cm 9.5   # 내 손바닥 실측 길이를 알면 같이 기록

조작:
    스페이스  이 위치에서 측정 시작 (2초 동안 모음)
    r         방금 단계 다시 하기
    q / Esc   끝내기 (끝낸 단계까지는 이미 저장돼 있음)
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from webcam_teach_robot.hand_tracking import HAND_CONNECTIONS, HandTracker, focal_length_px

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "measurements" / "depth"

DISTANCES_CM = [30, 40, 50, 60, 70]
POSES = {
    "flat": "손바닥을 카메라 쪽으로 똑바로 펴세요",
    "tilt": "손바닥을 45도쯤 아래로 기울이세요 (물건 잡으러 갈 때처럼)",
}
FRAMES_PER_STEP = 60  # 30fps면 약 2초

FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def font(size: int):
    if size not in _font_cache:
        _font_cache[size] = (
            ImageFont.truetype(str(FONT_PATH), size) if FONT_PATH.exists() else ImageFont.load_default()
        )
    return _font_cache[size]


def draw_text(frame: np.ndarray, lines: list[str], size: int = 22) -> np.ndarray:
    """OpenCV 기본 글꼴은 한글을 못 그려서 PIL로 그린다."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        d.text((12, 10 + i * (size + 10)), text, font=font(size), fill=(255, 255, 255),
               stroke_width=3, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_hand(frame, obs) -> None:
    pts = obs.pixels.astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 200, 0), 2)
    cv2.line(frame, tuple(pts[0]), tuple(pts[9]), (0, 255, 255), 3)


def csv_header() -> list[str]:
    cols = ["step", "take", "distance_cm", "pose", "frame", "t_s", "infer_ms",
            "palm_px", "palm_m", "depth_m", "handedness"]
    cols += [f"px{i}_{a}" for i in range(21) for a in ("u", "v")]
    cols += [f"relz{i}" for i in range(21)]
    cols += [f"w{i}_{a}" for i in range(21) for a in ("x", "y", "z")]
    return cols


def csv_row(step, take, dist, pose, k, t, infer_ms, obs) -> list:
    row = [step, take, dist, pose, k, f"{t:.4f}", f"{infer_ms:.2f}",
           f"{obs.palm_px:.3f}", f"{obs.palm_m:.5f}", f"{obs.depth_m:.5f}", obs.handedness]
    row += [f"{v:.2f}" for v in obs.pixels.ravel()]
    row += [f"{v:.5f}" for v in obs.rel_z]
    row += [f"{v:.5f}" for v in obs.world.ravel()]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fov", type=float, default=60.0, help="가로 화각 가정값(도). 나중에 데이터로 보정")
    parser.add_argument("--palm-cm", type=float, default=None,
                        help="내 손목 주름~가운뎃손가락 뿌리 마디 실측 길이(cm), 선택")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.camera}번을 열 수 없어요.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"{stamp}.csv"
    meta = {
        "created": stamp,
        "camera_index": args.camera,
        "resolution": [w, h],
        "fov_assumed_deg": args.fov,
        "focal_px_assumed": focal_length_px(w, args.fov),
        "palm_cm_measured": args.palm_cm,
        "distances_cm": DISTANCES_CM,
        "poses": list(POSES),
        "frames_per_step": FRAMES_PER_STEP,
        "mirror": True,
        "note": "distance = webcam(screen) plane to palm, measured with a tape on the desk",
        "completed_steps": [],
    }
    meta_path = csv_path.with_suffix(".json")

    def save_meta():
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    save_meta()
    steps = [(p, d) for p in POSES for d in DISTANCES_CM]

    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(csv_header())
    f.flush()

    t0 = time.perf_counter()
    i = 0
    collecting = False
    buffer: list[list] = []
    takes = [0] * len(steps)  # 단계별 측정 횟수. r로 다시 하면 늘어난다
    quit_ = False

    with HandTracker(horizontal_fov_deg=args.fov) as tracker:
        while i < len(steps) and not quit_:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            t = time.perf_counter() - t0
            t_inf = time.perf_counter()
            obs = tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(t * 1000))
            infer_ms = (time.perf_counter() - t_inf) * 1000

            pose, dist = steps[i]
            if obs is not None:
                draw_hand(frame, obs)

            if collecting:
                if obs is not None:
                    buffer.append(csv_row(i, takes[i], dist, pose, len(buffer), t, infer_ms, obs))
                bar = int(20 * len(buffer) / FRAMES_PER_STEP)
                lines = [f"측정 중... 손을 그대로 두세요  [{'#' * bar}{'.' * (20 - bar)}]"]
                if len(buffer) >= FRAMES_PER_STEP:
                    writer.writerows(buffer)
                    f.flush()
                    meta["completed_steps"].append({"step": i, "take": takes[i], "pose": pose, "distance_cm": dist})
                    save_meta()
                    buffer = []
                    collecting = False
                    i += 1
            else:
                lines = [
                    f"단계 {i + 1}/{len(steps)}:  카메라에서 {dist} cm",
                    POSES[pose],
                    "손 인식됨 - 준비되면 스페이스" if obs is not None else "손이 안 보여요",
                    "r: 이전 단계 다시   q: 끝내기",
                ]
            frame = draw_text(frame, lines)
            cv2.imshow("depth measurement (not recorded)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                quit_ = True
            elif key == ord(" ") and not collecting and obs is not None:
                collecting = True
                buffer = []
            elif key == ord("r") and not collecting and meta["completed_steps"]:
                # 파일에서 지우지 않고 take 번호를 올려서 다시 잰다. 분석은 단계마다 마지막 take만 쓴다.
                meta["completed_steps"].pop()
                save_meta()
                i -= 1
                takes[i] += 1

    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"저장: {csv_path}")
    print(f"완료한 단계: {len(meta['completed_steps'])}/{len(steps)}")


if __name__ == "__main__":
    main()
