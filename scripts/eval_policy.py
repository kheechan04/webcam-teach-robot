"""학습한 정책을 시뮬레이션에서 혼자 돌려 성공률을 잰다.

평가 배치는 시범에 쓰지 않은 고정 목록(experiments/layouts_v1.json의 eval)이다. 성공 판정은 시범·조종과 같은
PlacementTracker. 카메라 화면은 데이터셋을 만들 때와 같은 SceneRenderer로 그린다.

실행:
    uv run python scripts/eval_policy.py kheechan04/webcam-teach-robot-act-cond1-pilot --n 100
    uv run python scripts/eval_policy.py outputs/train/xxx/checkpoints/last/pretrained_model --n 5 --render 2
결과: experiments/eval/<이름>.json (에피소드별 성공·걸린 시간), --render면 outputs/에 재생 애니메이션.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from webcam_teach_robot.dataset import CAMERAS, FPS, SceneRenderer
from webcam_teach_robot.scene import PlacementTracker, build_model, cube_pos, load_layouts, place, task_ids

ROOT = Path(__file__).resolve().parent.parent
HOME_Q = np.array([0.0, -0.21, 0.346, 1.434, 0.0, 1.0])  # 시범과 같은 처음 자세
MAX_S = float(__import__("os").environ.get("EVAL_MAX_S", "20.0"))  # 에피소드 시간 제한. 스크립트 시범은 약 6초


def load_policy(path: str, device: str):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(path)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def run_episode(policy, pre, post, model, data, ids, renderer, cube_xy, goal_xy, device, frames=None):
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME_Q
    data.ctrl[:6] = HOME_Q
    place(model, data, ids, cube_xy, goal_xy)
    policy.reset()
    tracker = PlacementTracker()
    steps = int(round(1 / (FPS * model.opt.timestep)))
    for k in range(int(MAX_S * FPS)):
        imgs = renderer.render_data(data)
        obs = {"observation.state": torch.from_numpy(data.qpos[:6].astype(np.float32))}
        for key, img in imgs.items():
            obs[f"observation.images.{key}"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        obs["task"] = "Pick up the red cube and place it on the green square."
        with torch.inference_mode():
            action = post(policy.select_action(pre(obs)))
        a = action.squeeze(0).cpu().numpy()
        data.ctrl[:6] = a
        for _ in range(steps):
            mujoco.mj_step(model, data)
        if frames is not None:
            frames.append(Image.fromarray(np.concatenate([imgs[k] for k in CAMERAS], 1)))
        if tracker.update(cube_pos(data, ids), goal_xy, a[5] < 0.5, 1 / FPS):
            return True, (k + 1) / FPS, tracker.was_lifted
    return False, MAX_S, tracker.was_lifted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("policy", help="Hub 저장소 이름 또는 로컬 pretrained_model 폴더")
    p.add_argument("--split", default="eval")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--render", type=int, default=0, help="앞에서 몇 개를 애니메이션으로 저장")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    policy, pre, post = load_policy(args.policy, args.device)
    model = build_model()
    data = mujoco.MjData(model)
    ids = task_ids(model)
    renderer = SceneRenderer(model, ids)
    layouts = load_layouts(args.split)[:args.n]

    results = []
    t0 = time.perf_counter()
    for i, (cube_xy, goal_xy) in enumerate(layouts):
        frames = [] if i < args.render else None
        ok, secs, lifted = run_episode(policy, pre, post, model, data, ids, renderer, cube_xy, goal_xy,
                                       args.device, frames)
        results.append({"layout": i, "success": ok, "seconds": secs, "lifted": lifted,
                        "final_cube": cube_pos(data, ids).round(4).tolist()})
        rate = np.mean([r["success"] for r in results])
        print(f"{i:3d} {'성공' if ok else '실패'} ({secs:4.1f}s, 들었나 {lifted}) | 누적 {rate * 100:5.1f}% ({len(results)}개)")
        if frames:
            out = ROOT / "outputs" / f"eval_{args.name or Path(args.policy).name}_{i}.webp"
            frames[0].save(out, save_all=True, append_images=frames[2::2], duration=67, loop=0, quality=60)
    renderer.close()

    succ = np.array([r["success"] for r in results])
    # 95% 신뢰구간(윌슨). 평가 배치가 유한하니 성공률과 함께 남긴다.
    n, ph, z = len(succ), succ.mean(), 1.96
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    summary = {"policy": args.policy, "split": args.split, "n": n, "success_rate": float(ph),
               "wilson95": [float(centre - half), float(centre + half)],
               "lifted_rate": float(np.mean([r["lifted"] for r in results])),
               "mean_success_seconds": float(np.mean([r["seconds"] for r in results if r["success"]])) if succ.any() else None,
               "eval_wall_s": round(time.perf_counter() - t0, 1), "date": datetime.now().isoformat(timespec="seconds"),
               "device": args.device, "episodes": results}
    out_dir = ROOT / "experiments" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.name or Path(args.policy).name}_{args.split}{n}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n성공률 {ph * 100:.1f}% ({succ.sum()}/{n}), 95% 구간 {summary['wilson95'][0] * 100:.0f}~{summary['wilson95'][1] * 100:.0f}% → {out}")


if __name__ == "__main__":
    main()
