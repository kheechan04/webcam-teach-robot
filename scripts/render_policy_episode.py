"""학습한 정책이 평가 배치 하나를 혼자 하는 장면을 그린다(포트폴리오·확인용).

왼쪽: 옆에서 본 장면 / 오른쪽: 정책이 실제로 보는 카메라 화면 2개(앞쪽, 손목, 128×128).
정책 입력은 eval_policy.py와 똑같이 만든다.

실행:
    uv run python scripts/render_policy_episode.py kheechan04/webcam-teach-robot-act-cond1-pilot --layout 0
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from webcam_teach_robot.dataset import FPS, SceneRenderer
from webcam_teach_robot.scene import PlacementTracker, build_model, cube_pos, load_layouts, place, task_ids

ROOT = Path(__file__).resolve().parent.parent
FONT = Path("C:/Windows/Fonts/malgun.ttf")
HOME_Q = np.array([0.0, -0.21, 0.346, 1.434, 0.0, 1.0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--layout", type=int, default=0)
    p.add_argument("--split", default="eval")
    p.add_argument("--max-s", type=float, default=8.0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "eval_policy.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    policy, pre, post = ev.load_policy(args.policy, "cpu")

    model = build_model()
    data = mujoco.MjData(model)
    ids = task_ids(model)
    cams = SceneRenderer(model, ids)
    side = mujoco.Renderer(model, 256, 384)
    view = mujoco.MjvCamera()
    view.lookat[:] = [0.21, 0.0, 0.04]
    view.distance = 0.55
    view.azimuth = 135
    view.elevation = -28
    font = ImageFont.truetype(str(FONT), 15) if FONT.exists() else ImageFont.load_default()

    cube_xy, goal_xy = load_layouts(args.split)[args.layout]
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME_Q
    data.ctrl[:6] = HOME_Q
    place(model, data, ids, cube_xy, goal_xy)
    policy.reset()
    tracker = PlacementTracker()
    steps = int(round(1 / (FPS * model.opt.timestep)))
    frames, done_at = [], None
    for k in range(int(args.max_s * FPS)):
        imgs = cams.render_data(data)
        obs = {"observation.state": torch.from_numpy(data.qpos[:6].astype(np.float32)), "task": "x"}
        for key, img in imgs.items():
            obs[f"observation.images.{key}"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            a = post(policy.select_action(pre(obs))).squeeze(0).numpy()
        data.ctrl[:6] = a
        for _ in range(steps):
            mujoco.mj_step(model, data)
        if tracker.update(cube_pos(data, ids), goal_xy, a[5] < 0.5, 1 / FPS) and done_at is None:
            done_at = k / FPS

        side.update_scene(data, camera=view)
        canvas = Image.new("RGB", (384 + 128, 256), (20, 20, 25))
        canvas.paste(Image.fromarray(side.render()), (0, 0))
        canvas.paste(Image.fromarray(imgs["front"]), (384, 0))
        canvas.paste(Image.fromarray(imgs["wrist"]), (384, 128))
        d = ImageDraw.Draw(canvas)
        d.text((8, 6), f"{k / FPS:4.1f}s  학습한 정책 혼자" + ("  · 성공" if done_at is not None else ""),
               font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        d.text((390, 4), "정책이 보는 화면", font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        frames.append(canvas)
        if done_at is not None and k / FPS > done_at + 0.8:
            break
    cams.close()
    side.close()
    out = args.out or ROOT / "outputs" / f"policy_{Path(args.policy).name}_{args.split}{args.layout}.webp"
    frames[0].save(out, save_all=True, append_images=frames[1::2], duration=67, loop=0, quality=70)
    print(f"저장: {out} ({len(frames)}프레임, {out.stat().st_size / 1e6:.2f} MB), 성공 {done_at}")


if __name__ == "__main__":
    main()
