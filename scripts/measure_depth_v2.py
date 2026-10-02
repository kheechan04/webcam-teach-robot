"""M5 정식 깊이 측정: 거리 × 손 모양·기울기 × 손 높이, 두 번씩 순서를 섞어서. 화면 가로 위치도 짧게.

예비 측정(measure_depth.py)과 달리 노트북 화면 기울기를 재서 "카메라가 보는 깊이"의 참값을 계산한다.
카메라는 화면에 붙어 있어서, 화면이 뒤로 α만큼 기울면 카메라도 α만큼 위를 본다. 그래서 손을 웹캠보다 낮게 두면
줄자로 잰 거리와 카메라 깊이가 달라진다(예: 화면 기울기 15°, 경첩~웹캠 21 cm, 본체 1.5 cm일 때 줄자 45 cm에서
카메라 깊이 참값은 손이 웹캠 높이면 48.7 cm, 책상에서 9 cm 높이면 45.4 cm).

준비(한 번만 재기):
    --tilt      화면이 수직에서 뒤로 기운 각도(도). 휴대폰 수평계를 화면에 대고 잰다
    --screen    화면 아래 경첩(키보드와 만나는 선)에서 웹캠까지 화면을 따라 잰 길이(cm)
    --base      책상에서 경첩까지 높이(cm), 노트북 본체 두께 정도
    --low       "낮게" 자세에서 손바닥 가운데의 책상 위 높이(cm). 손목을 책상에 댄 채 손바닥을 세운 높이를 자로 잰다
줄자: 0 눈금을 경첩 선에 맞추고 앞쪽(사람 쪽)으로 펼친다. 거리는 경첩 선에서 손바닥 가운데까지.

실행:
    uv run python scripts/measure_depth_v2.py --tilt 15 --screen 21 --base 1.5 --low 9
    uv run python scripts/measure_depth_v2.py --resume measurements/depth_v2/XXX.json   # 끊긴 데서 이어서

조작: 스페이스 = 이 자세 측정(2초) · r = 이전 자세 다시 · q/Esc = 끝내기(끝낸 데까지 저장됨)
영상은 저장하지 않는다. 손 좌표·추정값 숫자만 CSV로.
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from webcam_teach_robot.hand_tracking import HAND_CONNECTIONS, HandTracker

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "measurements" / "depth_v2"
FRAMES_PER_STEP = 60

DISTANCES = [35, 45, 55, 65]
POSES = {
    "open_flat": "편 손, 손바닥을 카메라 쪽으로 똑바로",
    "open_tilt": "편 손, 손바닥을 45도쯤 아래로 기울여서 (물건 잡으러 갈 때처럼)",
    "pinch_flat": "집은 손 (엄지·검지 끝을 붙임), 손바닥은 카메라 쪽으로",
    "pinch_tilt": "집은 손 (엄지·검지 끝을 붙임), 45도쯤 아래로 기울여서",
}
HEIGHTS = {"cam": "웹캠 높이 (손바닥 가운데가 웹캠과 같은 높이)", "low": "낮게 (손목을 책상에 대고 손바닥을 세워서)"}
LATERAL = {"left": "화면 왼쪽 끝 쪽 (손이 화면 안에 다 보이게)", "center": "화면 가운데", "right": "화면 오른쪽 끝 쪽 (손이 화면 안에 다 보이게)"}
LATERAL_DIST = 50

FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")


def make_steps(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    main = [{"block": "main", "distance_cm": d, "pose": p, "height": h, "lateral": "center"}
            for d in DISTANCES for p in POSES for h in HEIGHTS]
    lat = [{"block": "lateral", "distance_cm": LATERAL_DIST, "pose": "open_flat", "height": "cam", "lateral": l}
           for l in LATERAL]
    steps = []
    for rep in range(2):  # 두 번씩, 매번 순서를 섞는다(지치는 효과와 자세 효과가 섞이지 않게)
        order = rng.permutation(len(main))
        steps += [{**main[i], "rep": rep} for i in order]
    for rep in range(2):
        steps += [{**s, "rep": rep} for s in (lat[i] for i in rng.permutation(len(lat)))]
    for i, s in enumerate(steps):
        s["step"] = i
    return steps


def camera_geometry(tilt_deg: float, screen_cm: float, base_cm: float) -> dict:
    """경첩 선 기준 웹캠 위치. 화면이 수직에서 뒤로 tilt만큼 기울었다고 보면 웹캠은 경첩 뒤로 L·sinα, 위로 base + L·cosα."""
    a = math.radians(tilt_deg)
    return {"cam_back_cm": screen_cm * math.sin(a), "cam_height_cm": base_cm + screen_cm * math.cos(a), "tilt_deg": tilt_deg}


def true_camera_depth_cm(distance_cm: float, hand_height_cm: float, geo: dict) -> float:
    """카메라 광축 방향 깊이(핀홀 식의 z) 참값. 광축은 수평에서 위로 tilt만큼 들려 있다고 본다."""
    a = math.radians(geo["tilt_deg"])
    forward = distance_cm + geo["cam_back_cm"]
    up = hand_height_cm - geo["cam_height_cm"]
    return forward * math.cos(a) + up * math.sin(a)


def draw_text(frame: np.ndarray, lines: list[str], size: int = 21) -> np.ndarray:
    font = ImageFont.truetype(str(FONT_PATH), size) if FONT_PATH.exists() else ImageFont.load_default()
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        d.text((12, 10 + i * (size + 9)), text, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def header() -> list[str]:
    cols = ["step", "take", "rep", "block", "distance_cm", "pose", "height", "lateral", "hand_height_cm",
            "true_depth_cm", "frame", "t_s", "infer_ms", "depth_len_m", "depth_width_m", "depth_m", "handedness"]
    cols += [f"px{i}_{a}" for i in range(21) for a in ("u", "v")]
    cols += [f"relz{i}" for i in range(21)]
    cols += [f"w{i}_{a}" for i in range(21) for a in ("x", "y", "z")]
    return cols


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tilt", type=float)
    p.add_argument("--screen", type=float)
    p.add_argument("--base", type=float)
    p.add_argument("--low", type=float)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--resume", type=Path, default=None)
    args = p.parse_args()

    if args.resume:
        meta_path = args.resume
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        csv_path = meta_path.with_suffix(".csv")
    else:
        if None in (args.tilt, args.screen, args.base, args.low):
            raise SystemExit("--tilt --screen --base --low 를 모두 넣어 주세요 (설명은 파일 맨 위).")
        geo = camera_geometry(args.tilt, args.screen, args.base)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUT_DIR / f"{stamp}.csv"
        meta_path = csv_path.with_suffix(".json")
        meta = {"created": stamp, "geometry": geo, "inputs": {"tilt_deg": args.tilt, "screen_cm": args.screen,
                                                               "base_cm": args.base, "low_cm": args.low},
                "hand_height_cm": {"cam": round(geo["cam_height_cm"], 1), "low": args.low},
                "seed": args.seed, "frames_per_step": FRAMES_PER_STEP, "mirror": True, "resolution": [640, 480],
                "steps": make_steps(args.seed), "done": []}
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header())
    geo = meta["geometry"]

    def save_meta():
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    save_meta()
    print(f"웹캠 높이(계산): 책상에서 {geo['cam_height_cm']:.1f} cm, 경첩 뒤로 {geo['cam_back_cm']:.1f} cm")
    print(f"'웹캠 높이' 자세 = 손바닥 가운데를 책상에서 약 {meta['hand_height_cm']['cam']} cm 높이에 두기")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.camera}번을 열 수 없어요.")

    steps = meta["steps"]
    done_ids = {d["step"] for d in meta["done"]}
    queue = [s for s in steps if s["step"] not in done_ids]
    takes = {s["step"]: sum(1 for d in meta["done"] if d["step"] == s["step"]) for s in steps}
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    t0 = time.perf_counter()
    collecting, buffer, i = False, [], 0

    with HandTracker() as tracker:
        while i < len(queue):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            t = time.perf_counter() - t0
            t_inf = time.perf_counter()
            obs = tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(t * 1000))
            infer_ms = (time.perf_counter() - t_inf) * 1000
            s = queue[i]
            hh = meta["hand_height_cm"][s["height"]]
            if obs is not None:
                pts = obs.pixels.astype(int)
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 200, 0), 2)
            if collecting:
                if obs is not None:
                    truth = true_camera_depth_cm(s["distance_cm"], hh, geo)
                    row = [s["step"], takes[s["step"]], s["rep"], s["block"], s["distance_cm"], s["pose"], s["height"],
                           s["lateral"], hh, f"{truth:.2f}", len(buffer), f"{t:.4f}", f"{infer_ms:.2f}",
                           f"{obs.depth_len_m:.5f}", f"{obs.depth_width_m:.5f}", f"{obs.depth_m:.5f}", obs.handedness]
                    row += [f"{v:.2f}" for v in obs.pixels.ravel()] + [f"{v:.5f}" for v in obs.rel_z]
                    row += [f"{v:.5f}" for v in obs.world.ravel()]
                    buffer.append(row)
                bar = int(20 * len(buffer) / FRAMES_PER_STEP)
                lines = [f"측정 중... 그대로  [{'#' * bar}{'.' * (20 - bar)}]"]
                if len(buffer) >= FRAMES_PER_STEP:
                    writer.writerows(buffer)
                    f.flush()
                    meta["done"].append({"step": s["step"], "take": takes[s["step"]]})
                    save_meta()
                    buffer, collecting = [], False
                    i += 1
            else:
                where = LATERAL[s["lateral"]] if s["block"] == "lateral" else ""
                lines = [f"{len(meta['done']) + 1}/{len(steps)}  ·  줄자 {s['distance_cm']} cm  {where}",
                         f"손: {POSES[s['pose']]}",
                         f"높이: {HEIGHTS[s['height']]} — 책상에서 약 {hh} cm",
                         "손 인식됨 · 자세 잡고 스페이스" if obs is not None else "손이 안 보여요",
                         "r: 이전 자세 다시  ·  q: 끝내기(저장됨, --resume로 이어서)"]
            cv2.imshow("depth measurement v2 (not recorded)", draw_text(frame, lines))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and not collecting and obs is not None:
                collecting, buffer = True, []
            if key == ord("r") and not collecting and meta["done"]:
                last = meta["done"].pop()  # 파일 행은 남기고 take 번호를 올려 다시 잰다. 분석은 마지막 take만 쓴다
                takes[last["step"]] += 1
                save_meta()
                prev = next(x for x in steps if x["step"] == last["step"])
                queue.insert(i, prev)
    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"저장: {csv_path}\n완료 {len(meta['done'])}/{len(steps)}" +
          ("" if len(meta["done"]) == len(steps) else f"  → 이어서: uv run python scripts/measure_depth_v2.py --resume {meta_path}"))


if __name__ == "__main__":
    main()
