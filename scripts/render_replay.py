"""조종 기록(관절 각도, 큐브 위치·자세, 목표 위치)만으로 시뮬레이션 장면을 다시 그려 애니메이션(webp)으로 저장한다.
웹캠 영상은 쓰지 않으므로 얼굴·방이 나오지 않는다. 큐브 자세가 기록되지 않은 옛 기록은 처음 방향으로 그린다.

실행:
    uv run python scripts/render_replay.py measurements/teleop/XXX.csv --start 25 --end 40
"""

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from webcam_teach_robot.dataset import default_cube_quat
from webcam_teach_robot.scene import build_model, task_ids, yaw_facing_robot

ROOT = Path(__file__).resolve().parent.parent
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
    p.add_argument("--azimuth", type=float, default=0.0, help="0이면 조종할 때처럼 로봇 뒤에서. 큐브가 가리면 120~150")
    p.add_argument("--elevation", type=float, default=-22.0)
    args = p.parse_args()

    rows = [r for r in csv.DictReader(open(args.log, encoding="utf-8"))
            if args.start <= float(r["t_s"]) <= args.end]
    if not rows:
        raise SystemExit("그 구간에 기록이 없어요.")
    t = np.array([float(r["t_s"]) for r in rows])
    q = np.array([[float(r[f"q{i}"]) for i in range(6)] for r in rows])
    engaged = np.array([r["engaged"] == "1" for r in rows])
    closed = np.array([r.get("gripper_closed") == "1" for r in rows])
    has_task = "cube_x" in rows[0]
    if has_task:
        cube = np.array([[float(r[f"cube_{a}"]) for a in "xyz"] for r in rows])
        goal = np.array([[float(r["goal_x"]), float(r["goal_y"])] for r in rows])
        quat = (np.array([[float(r[f"cube_q{a}"]) for a in "wxyz"] for r in rows]) if "cube_qw" in rows[0]
                else np.tile(default_cube_quat(cube[0, :2]), (len(rows), 1)))
        success = np.array([r.get("placed_success", r.get("in_target")) == "1" for r in rows])

    model = build_model()
    ids = task_ids(model)
    data = mujoco.MjData(model)
    if not has_task:  # 과제 장면 이전 기록: 큐브·목표를 화면 밖으로
        data.qpos[ids.cube_qadr:ids.cube_qadr + 3] = [0, 0, -1]
        data.mocap_pos[ids.target_mocap] = [0, 0, -1]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, 0.0, 0.06]
    cam.distance = 0.62
    cam.azimuth = args.azimuth  # 0: 조종할 때와 같은 시점(로봇 뒤)
    cam.elevation = args.elevation
    font = ImageFont.truetype(str(FONT), 16) if FONT.exists() else ImageFont.load_default()

    frames = []
    times = np.arange(t[0], t[-1], 1 / args.fps)
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for ts in times:
            i = min(np.searchsorted(t, ts), len(t) - 1)
            data.qpos[:6] = q[i]
            if has_task:
                data.qpos[ids.cube_qadr:ids.cube_qadr + 3] = cube[i]
                data.qpos[ids.cube_qadr + 3:ids.cube_qadr + 7] = quat[i]
                yaw = yaw_facing_robot(goal[i])
                data.mocap_pos[ids.target_mocap] = [goal[i, 0], goal[i, 1], 0.0005]
                data.mocap_quat[ids.target_mocap] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(img)
            label = (f"{ts - t[0]:4.1f}s  " + ("조종 중" if engaged[i] else "멈춤(클러치)") + ("  · 집게 닫힘" if closed[i] else "")
                     + ("  · 성공" if has_task and success[i] else ""))
            d.text((10, 8), label, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            frames.append(img)

    out = args.out or ROOT / "docs" / "figures" / f"replay_{args.log.stem}_{int(args.start)}-{int(min(args.end, t[-1]))}.webp"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, quality=70)
    print(f"저장: {out}  ({len(frames)}프레임, {out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
