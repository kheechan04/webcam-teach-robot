"""조종 기록(관절 각도)만으로 시뮬레이션 로봇 움직임을 다시 그려 애니메이션(webp)으로 저장한다.
웹캠 영상은 쓰지 않으므로 얼굴·방이 나오지 않는다.

실행:
    uv run python scripts/render_replay.py measurements/teleop/XXX.csv --start 25 --end 40
"""

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "third_party" / "robotstudio_so101" / "scene.xml"
FONT = Path("C:/Windows/Fonts/malgun.ttf")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--start", type=float, default=0.0, help="시작 시각(초, 기록 기준)")
    p.add_argument("--end", type=float, default=1e9)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--width", type=int, default=560)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows = [r for r in csv.DictReader(open(args.log, encoding="utf-8"))
            if args.start <= float(r["t_s"]) <= args.end]
    if not rows:
        raise SystemExit("그 구간에 기록이 없어요.")
    t = np.array([float(r["t_s"]) for r in rows])
    q = np.array([[float(r[f"q{i}"]) for i in range(6)] for r in rows])
    engaged = np.array([r["engaged"] == "1" for r in rows])
    closed = np.array([r.get("gripper_closed") == "1" for r in rows])

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, 0.0, 0.06]
    cam.distance = 0.62
    cam.azimuth = 0  # 조종할 때와 같은 시점(로봇 뒤)
    cam.elevation = -22
    font = ImageFont.truetype(str(FONT), 16) if FONT.exists() else ImageFont.load_default()

    frames = []
    times = np.arange(t[0], t[-1], 1 / args.fps)
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for ts in times:
            i = min(np.searchsorted(t, ts), len(t) - 1)
            data.qpos[:6] = q[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(img)
            label = f"{ts - t[0]:4.1f}s  " + ("조종 중" if engaged[i] else "멈춤(클러치)") + ("  · 집게 닫힘" if closed[i] else "")
            d.text((10, 8), label, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            frames.append(img)

    out = args.out or ROOT / "docs" / "figures" / f"replay_{args.log.stem}_{int(args.start)}-{int(min(args.end, t[-1]))}.webp"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, quality=70)
    print(f"저장: {out}  ({len(frames)}프레임, {out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
