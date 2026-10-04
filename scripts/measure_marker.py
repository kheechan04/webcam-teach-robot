"""조건 ⑤ 준비: 손목 마커가 거리를 정확히 재는지 확인 (M5 정식 측정과 같은 방식, 더 짧게).

같은 프레임에서 마커 깊이와 MediaPipe 깊이(지금 조종 방식)를 함께 기록해 참값과 비교한다.
거리 4개(35·45·55·65 cm) × 손 3가지(편 손 정면 / 편 손 기울임 / 집고 기울인 손) × 높이 2가지(웹캠 높이 / 낮게) = 24단계, 순서 섞음.

준비:
    - 화면 기울기·경첩~웹캠·본체 두께는 M5와 같다고 보고 그 값을 쓴다(--like). **화면을 M5 때처럼 수직에 가깝게** 세운다.
    - 줄자 0을 경첩 선에 맞추고, 거리는 **경첩 선에서 마커 가운데까지**.
    - 높이는 **마커 가운데**의 책상 위 높이. "웹캠 높이" = 웹캠과 같은 높이(M5: 24.4 cm), "낮게" = 책 위에 손목(15 cm).

실행:
    uv run python scripts/measure_marker.py --like measurements/depth_v2/20261002_232355.json
    uv run python scripts/measure_marker.py --resume measurements/marker/XXX.json
조작: 스페이스 = 측정(2초) · r = 이전 다시 · q = 끝내기(끝낸 데까지 저장). 영상은 저장하지 않는다.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_depth_v2 import draw_text, true_camera_depth_cm  # noqa: E402

from webcam_teach_robot.hand_tracking import HAND_CONNECTIONS, HandTracker  # noqa: E402
from webcam_teach_robot.marker_depth import MarkerTracker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "measurements" / "marker"
FRAMES = 60
DISTANCES = [35, 45, 55, 65]
POSES = {"open_flat": "편 손 · 정면", "open_tilt": "편 손 · 45° 아래로 기울임", "pinch_tilt": "집은 손(엄지·검지 붙임) · 45° 기울임"}
HEIGHTS = {"cam": "웹캠 높이", "low": "낮게(책 위에 손목)"}
COLS = ["step", "take", "distance_cm", "pose", "height", "marker_height_cm", "true_depth_cm", "frame", "t_s",
        "marker_found", "marker_x", "marker_y", "marker_z", "marker_tilt_deg", "marker_side_px",
        "hand_found", "mp_depth_m", "mp_depth_len_m", "mp_depth_width_m", "pinch"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--like", type=Path, help="M5 측정 json (화면·높이 값을 그대로 씀)")
    p.add_argument("--resume", type=Path)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261004)
    args = p.parse_args()

    if args.resume:
        meta_path = args.resume
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        if not args.like:
            raise SystemExit("--like measurements/depth_v2/XXX.json 을 넣어 주세요")
        m5 = json.loads(args.like.read_text(encoding="utf-8"))
        rng = np.random.default_rng(args.seed)
        steps = [{"distance_cm": d, "pose": po, "height": h} for d in DISTANCES for po in POSES for h in HEIGHTS]
        steps = [{**steps[i], "step": k} for k, i in enumerate(rng.permutation(len(steps)))]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_path = OUT_DIR / f"{stamp}.json"
        meta = {"created": stamp, "geometry": m5["geometry"], "inputs": m5["inputs"], "like": args.like.name,
                "marker_height_cm": m5["hand_height_cm"], "marker_size_m": 0.05, "steps": steps, "done": []}
        with open(meta_path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLS)
    csv_path = meta_path.with_suffix(".csv")
    geo = meta["geometry"]
    save = lambda: meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")  # noqa: E731
    save()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없어요.")
    marker = MarkerTracker()
    done = {d["step"] for d in meta["done"]}
    queue = [s for s in meta["steps"] if s["step"] not in done]
    takes = {s["step"]: sum(1 for d in meta["done"] if d["step"] == s["step"]) for s in meta["steps"]}
    f = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    t0 = time.perf_counter()
    collecting, buf, i = False, [], 0
    with HandTracker() as hands:
        while i < len(queue):
            ok, raw = cap.read()
            if not ok:
                break
            m = marker.detect(raw)
            frame = cv2.flip(raw, 1)
            t = time.perf_counter() - t0
            obs = hands.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(t * 1000))
            s = queue[i]
            hh = meta["marker_height_cm"][s["height"]]
            if obs is not None:
                pts = obs.pixels.astype(int)
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 200, 0), 1)
            MarkerTracker.draw(frame, m)
            not_pinched = s["pose"] == "pinch_tilt" and (obs is None or obs.pinch > 0.6)
            if collecting:
                if m is not None:
                    truth = true_camera_depth_cm(s["distance_cm"], hh, geo)
                    buf.append([s["step"], takes[s["step"]], s["distance_cm"], s["pose"], s["height"], hh, f"{truth:.2f}",
                                len(buf), f"{t:.4f}", 1, *[f"{v:.5f}" for v in m.point_cam], f"{m.tilt_deg:.1f}",
                                f"{m.side_px:.2f}", int(obs is not None),
                                f"{obs.depth_m:.5f}" if obs else "", f"{obs.depth_len_m:.5f}" if obs else "",
                                f"{obs.depth_width_m:.5f}" if obs else "", f"{obs.pinch:.3f}" if obs else ""])
                bar = int(20 * len(buf) / FRAMES)
                lines = [f"측정 중... 그대로 [{'#' * bar}{'.' * (20 - bar)}]"]
                if len(buf) >= FRAMES:
                    w.writerows(buf)
                    f.flush()
                    meta["done"].append({"step": s["step"], "take": takes[s["step"]]})
                    save()
                    collecting, buf = False, []
                    i += 1
            else:
                state = ("마커가 안 보여요 — 손목 안쪽을 카메라 쪽으로" if m is None
                         else f"엄지·검지를 붙여 주세요" if not_pinched
                         else f"마커 {m.point_cam[2] * 100:.1f} cm (기울기 {m.tilt_deg:.0f}°) → 스페이스")
                lines = [f"{len(meta['done']) + 1}/{len(meta['steps'])} · 줄자 {s['distance_cm']} cm (경첩 → 마커 가운데)",
                         f"손: {POSES[s['pose']]}", f"높이: {HEIGHTS[s['height']]} = 마커 가운데가 책상에서 {hh} cm",
                         state, "r 이전 다시 · q 끝내기"]
            cv2.imshow("marker depth check (not recorded)", draw_text(frame, lines))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and not collecting and m is not None and not not_pinched:
                collecting, buf = True, []
            if key == ord("r") and not collecting and meta["done"]:
                last = meta["done"].pop()
                takes[last["step"]] += 1
                save()
                queue.insert(i, next(x for x in meta["steps"] if x["step"] == last["step"]))
    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"저장: {csv_path}  완료 {len(meta['done'])}/{len(meta['steps'])}")
    if meta["done"]:
        summarize(csv_path, meta)


def summarize(csv_path: Path, meta: dict) -> None:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    last = {(d["step"]): d["take"] for d in meta["done"]}
    print("\n자세·높이별 평균 오차(cm, 추정 − 참값): 마커 / MediaPipe(지금 조종 방식)")
    for pose in POSES:
        for h in HEIGHTS:
            err_m, err_p, sd_m = [], [], []
            for s in meta["steps"]:
                if s["pose"] != pose or s["height"] != h or s["step"] not in last:
                    continue
                R = [r for r in rows if int(r["step"]) == s["step"] and int(r["take"]) == last[s["step"]]]
                truth = float(R[0]["true_depth_cm"])
                mz = np.array([float(r["marker_z"]) for r in R]) * 100
                err_m.append(np.median(mz) - truth)
                sd_m.append(np.std(mz))
                mp = [float(r["mp_depth_m"]) * 100 for r in R if r["mp_depth_m"]]
                if mp:
                    err_p.append(np.median(mp) - truth)
            if err_m:
                print(f"  {POSES[pose]:24s} {HEIGHTS[h]:12s} 마커 {np.mean(err_m):+5.1f} (흔들림 {np.mean(sd_m):.2f}) / "
                      f"MediaPipe {np.mean(err_p) if err_p else float('nan'):+5.1f}")


if __name__ == "__main__":
    main()
