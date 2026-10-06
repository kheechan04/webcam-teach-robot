"""학습한 정책이 "어디서 집게를 닫는가"를 잰다 (M9 해석: 왜 ②가 ③보다, ④가 ④0보다 못한가).

eval_policy.py와 같은 평가(같은 배치·같은 렌더·같은 판정)를 돌리면서, 집게를 처음 닫는 순간 집게 끝과 큐브의 위치 차이를
로봇 좌표(x 앞뒤 = 웹캠 깊이 방향, y 좌우, z 높이)로 기록한다. 시범에서는 집는 위치가 ②·③ 비슷했으므로,
정책이 닫는 위치가 앞뒤로 더 흩어지거나 치우치는지 본다.

    uv run python scripts/probe_policy_grasp.py --n 30 --workers 8
    uv run python scripts/probe_policy_grasp.py --set 100k --n 30 --workers 4   # 10만 스텝 ②·③, 시드 5개
결과: experiments/policy_grasp_probe.json (10만 스텝은 policy_grasp_probe_100k.json)
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_policy import HOME_Q, load_policy  # noqa: E402

from webcam_teach_robot.dataset import FPS, SceneRenderer  # noqa: E402
from webcam_teach_robot.ik import SO101IK  # noqa: E402
from webcam_teach_robot.scene import TASK_TEXT, PlacementTracker, base_pos, build_model, cube_pos, load_layouts, place, task_ids  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = {c: [f"kheechan04/webcam-teach-robot-act-{c}-s{s}" for s in (1000, 2000, 3000)]
          for c in ("cond2-webcam", "cond3-webcam-corrected", "cond4-depth-error", "cond4z-no-error", "cond5-webcam-marker")}
MODELS_100K = {c: [f"kheechan04/webcam-teach-robot-act-{c}-s{s}-100k" for s in (1000, 2000, 3000, 4000, 5000)]
               for c in ("cond2-webcam", "cond3-webcam-corrected")}
MAX_S = 60.0
_W = {}


def init(path):
    torch.set_num_threads(1)
    policy, pre, post = load_policy(path, "cpu")
    model = build_model()
    _W.update(policy=policy, pre=pre, post=post, model=model, data=mujoco.MjData(model), ids=task_ids(model),
              r=SceneRenderer(model, task_ids(model)), ik=SO101IK(model))


def run(job):
    i, cube_xy, goal_xy = job
    w = _W
    m, d, ids = w["model"], w["data"], w["ids"]
    mujoco.mj_resetData(m, d)
    d.qpos[:6] = HOME_Q
    d.ctrl[:6] = HOME_Q
    place(m, d, ids, cube_xy, goal_xy)
    w["policy"].reset()
    tr = PlacementTracker()
    steps = int(round(1 / (FPS * m.opt.timestep)))
    closes = []
    was_closed = False
    for k in range(int(MAX_S * FPS)):
        imgs = w["r"].render_data(d)
        obs = {"observation.state": torch.from_numpy(d.qpos[:6].astype(np.float32))}
        for key, img in imgs.items():
            obs[f"observation.images.{key}"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        obs["task"] = TASK_TEXT
        with torch.inference_mode():
            a = w["post"](w["policy"].select_action(w["pre"](obs))).squeeze(0).cpu().numpy()
        closed = a[5] < 0.5
        if closed and not was_closed:
            tip = w["ik"].tip(d.qpos.copy())[0]
            c = cube_pos(d, ids)
            closes.append({"t": k / FPS, "offset_mm": ((tip - c) * 1000).round(1).tolist(),
                           "cube_moved_mm": float(np.linalg.norm(c[:2] - cube_xy) * 1000)})
        was_closed = closed
        d.ctrl[:6] = a
        for _ in range(steps):
            mujoco.mj_step(m, d)
        if tr.update(cube_pos(d, ids), goal_xy, closed, 1 / FPS, base_pos(d, ids)):
            return {"layout": i, "success": True, "closes": closes, "lifted": tr.was_lifted}
    return {"layout": i, "success": False, "closes": closes, "lifted": tr.was_lifted}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--set", choices=["20k", "100k"], default="20k")
    args = p.parse_args()
    models = MODELS_100K if args.set == "100k" else MODELS
    layouts = load_layouts("eval")[:args.n]
    jobs = [(i, c, g) for i, (c, g) in enumerate(layouts)]
    out_path = ROOT / "experiments" / ("policy_grasp_probe_100k.json" if args.set == "100k" else "policy_grasp_probe.json")
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for cond, paths in models.items():
        for path in paths:
            if path in out and len(out[path]) >= args.n:
                continue
            with mp.get_context("spawn").Pool(args.workers, init, (path,)) as pool:
                res = sorted(pool.map(run, jobs), key=lambda r: r["layout"])
            out[path] = res
            out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
            first = np.array([r["closes"][0]["offset_mm"] for r in res if r["closes"]])
            print(f"{path.split('act-')[1]:28s} 성공 {sum(r['success'] for r in res)}/{len(res)}  첫 닫힘 위치 "
                  f"x {first[:, 0].mean():+5.1f}±{first[:, 0].std():4.1f}  y {first[:, 1].mean():+5.1f}±{first[:, 1].std():4.1f}"
                  f"  z {first[:, 2].mean():+5.1f}  닫기 횟수 {np.mean([len(r['closes']) for r in res]):.1f}", flush=True)


if __name__ == "__main__":
    main()
