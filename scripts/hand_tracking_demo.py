"""웹캠으로 손을 추적해서 화면에 보여 준다. 영상은 저장하지 않는다.

실행:
    uv run python scripts/hand_tracking_demo.py
    uv run python scripts/hand_tracking_demo.py --camera 1   # 다른 카메라

화면에 나오는 값:
    wrist px   손목의 사진 위 위치 (픽셀)
    palm       손목~가운뎃손가락 뿌리 길이: 사진 속 픽셀 / 실제 cm
    depth      핀홀로 추정한 카메라~손 거리 (화각 60° 가정, 아직 보정 전)
    infer/fps  손 추적 한 번에 걸린 시간, 초당 프레임 수

q 또는 Esc로 끝낸다.
"""

import argparse
import time

import cv2

from webcam_teach_robot.hand_tracking import HAND_CONNECTIONS, HandTracker


def draw(frame, obs) -> None:
    pts = obs.pixels.astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 200, 0), 2)
    for p in pts:
        cv2.circle(frame, tuple(p), 4, (0, 0, 255), -1)
    # 깊이 추정에 쓰는 구간(손목~가운뎃손가락 뿌리)을 노란색으로 강조
    cv2.line(frame, tuple(pts[0]), tuple(pts[9]), (0, 255, 255), 3)


def put_lines(frame, lines) -> None:
    for i, text in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fov", type=float, default=60.0, help="웹캠 가로 화각(도), 보정 전 가정값")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.camera}번을 열 수 없어요.")

    t0 = time.perf_counter()
    last = t0
    fps = 0.0
    with HandTracker(horizontal_fov_deg=args.fov) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울처럼 보이게

            t_infer = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            obs = tracker.detect(rgb, int((t_infer - t0) * 1000))
            infer_ms = (time.perf_counter() - t_infer) * 1000

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1 / max(now - last, 1e-6))
            last = now

            h, w = frame.shape[:2]
            lines = [f"{w}x{h}  infer {infer_ms:4.1f} ms  fps {fps:4.1f}"]
            if obs is None:
                lines.append("hand: not found")
            else:
                draw(frame, obs)
                u, v = obs.pixels[0]
                lines += [
                    f"hand: {obs.handedness}",
                    f"wrist px ({u:4.0f}, {v:4.0f})",
                    f"palm {obs.palm_px:5.1f} px / {obs.palm_m * 100:4.1f} cm",
                    f"depth ~{obs.depth_m * 100:5.1f} cm (fov {args.fov:.0f} assumed)",
                ]
            put_lines(frame, lines)

            cv2.imshow("hand tracking (not recorded)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
