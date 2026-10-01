"""스크립트 시범으로 큐브 옮기기가 되는지 확인한다 (물리·접촉 점검).

실행:
    uv run python scripts/test_scripted_grasp.py              # 무작위 배치 20개, 성공률
    uv run python scripts/test_scripted_grasp.py --render 3   # 처음 3개를 애니메이션으로 저장
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from webcam_teach_robot.ik import SO101IK
from webcam_teach_robot.scene import TARGET_HALF, build_model, cube_pos, place, task_ids
from webcam_teach_robot.scripted import CONTROL_HZ, cube_on_table, plan, trajectory

ROOT = Path(__file__).resolve().parent.parent
HOME_Q = np.array([0.0, -0.21, 0.346, 1.434, 0.0, 1.0])


def sample_layout(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """큐브는 오른쪽(y<0), 목표는 왼쪽(y>0) 영역에서 무작위. 둘 다 작업 범위(x 16~28, y ±15 cm) 안."""
    cube = np.array([rng.uniform(0.18, 0.26), rng.uniform(-0.12, -0.04)])
    target = np.array([rng.uniform(0.18, 0.26), rng.uniform(0.04, 0.12)])
    return cube, target


def in_target(cube_xyz: np.ndarray, target_xy: np.ndarray) -> bool:
    yaw = np.arctan2(target_xy[1], target_xy[0])
    d = cube_xyz[:2] - target_xy
    local = np.array([np.cos(yaw) * d[0] + np.sin(yaw) * d[1], -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]])
    return bool(np.all(np.abs(local) <= TARGET_HALF) and cube_on_table(cube_xyz[2]))


def run(model, data, ids, ik, cube_xy, target_xy, renderer=None, frames=None):
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME_Q
    data.ctrl[:6] = HOME_Q
    place(model, data, ids, cube_xy, target_xy)
    start = ik.tip(data.qpos.copy())[0]
    q_ik = data.qpos.copy()
    steps_per_ctrl = int(round(1 / (CONTROL_HZ * model.opt.timestep)))
    max_lift = 0.0
    for pos, grip in trajectory(plan(cube_xy, target_xy, start)):
        q_ik = ik.solve(pos, q_ik).q
        data.ctrl[:5] = q_ik[:5]
        data.ctrl[5] = grip
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(model, data)
        max_lift = max(max_lift, cube_pos(data, ids)[2])
        if renderer is not None:
            renderer.update_scene(data, camera="front")
            frames.append(Image.fromarray(renderer.render()))
    for _ in range(int(0.5 / model.opt.timestep)):  # 0.5초 더 두고 큐브가 자리 잡게
        mujoco.mj_step(model, data)
    final = cube_pos(data, ids)
    return in_target(final, target_xy), final, max_lift


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", type=int, default=0)
    args = p.parse_args()

    model = build_model()
    data = mujoco.MjData(model)
    ids = task_ids(model)
    ik = SO101IK(model)
    rng = np.random.default_rng(args.seed)

    ok = 0
    renderer = mujoco.Renderer(model, 240, 320) if args.render else None
    for i in range(args.n):
        cube_xy, target_xy = sample_layout(rng)
        frames = [] if i < args.render else None
        success, final, lift = run(model, data, ids, ik, cube_xy, target_xy,
                                   renderer if frames is not None else None, frames)
        ok += success
        err = np.linalg.norm(final[:2] - target_xy) * 100
        print(f"{i:2d} 큐브 ({cube_xy[0]*100:4.1f},{cube_xy[1]*100:5.1f}) → 목표 ({target_xy[0]*100:4.1f},{target_xy[1]*100:5.1f}) "
              f"| {'성공' if success else '실패'}  최고 높이 {lift*100:4.1f} cm  최종 ({final[0]*100:4.1f},{final[1]*100:5.1f},{final[2]*100:4.1f})  목표까지 {err:4.1f} cm")
        if frames:
            out = ROOT / "outputs" / f"scripted_{i}.webp"
            frames[0].save(out, save_all=True, append_images=frames[1:][::2], duration=67, loop=0, quality=60)
    print(f"\n성공 {ok}/{args.n} ({ok / args.n * 100:.0f}%)")


if __name__ == "__main__":
    main()
