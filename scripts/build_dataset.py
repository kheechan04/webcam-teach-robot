"""시범을 LeRobot 데이터셋으로 만든다. 데이터셋은 data/lerobot/<이름>/ 에 생긴다(git에는 안 올림).

조건 ① 스크립트 시범 (고정 배치 experiments/layouts_v1.json의 train 50개):
    uv run python scripts/build_dataset.py scripted --name cond1_scripted
조건 ② 웹캠 시범 (조종 기록 CSV들에서 성공한 배치만):
    uv run python scripts/build_dataset.py webcam --name cond2_webcam measurements/teleop/2026100*.csv
M8 녹화(record_demos.py)에서 조건별로 (progress.json에서 성공으로 끝난 시도만):
    uv run python scripts/build_dataset.py webcam --name cond2_webcam --condition 2 measurements/demos_m8/u*.csv
    uv run python scripts/build_dataset.py webcam --name cond3_webcam_corrected --condition 3 measurements/demos_m8/u*.csv

웹캠 기록 처리:
    - 배치(layout)마다 한 에피소드. 처음 조종을 건 순간부터, 성공 판정 1초 뒤(또는 배치 끝)까지
    - 클러치로 멈춘 구간은 뺀다(그동안 로봇은 정지라 빼도 움직임이 이어진다)
    - 기록 시각이 고르지 않아서(루프 중앙값 약 30 ms) 1/30초 간격으로 다시 뽑는다
    - 관측과 행동 짝: 기록 한 줄은 "명령을 보내고 물리를 진행한 뒤"의 상태라서, 행동 t는 한 줄 앞의 상태와 짝짓는다
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

from webcam_teach_robot.dataset import (FPS, Episode, SceneRenderer, default_cube_quat, features, resample,
                                        write_episode)
from webcam_teach_robot.ik import SO101IK
from webcam_teach_robot.scene import PlacementTracker, build_model, cube_pos, load_layouts, place, task_ids
from webcam_teach_robot.scripted import CONTROL_HZ, plan, trajectory

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "lerobot"
HOME_Q = np.array([0.0, -0.21, 0.346, 1.434, 0.0, 1.0])
TAIL_S = 1.0  # 성공 뒤 남길 시간
assert CONTROL_HZ == FPS


def scripted_episodes(model, split: str, n: int | None):
    data = mujoco.MjData(model)
    ids = task_ids(model)
    ik = SO101IK(model)
    steps = int(round(1 / (FPS * model.opt.timestep)))
    layouts = load_layouts(split)[:n]
    for i, (cube_xy, goal_xy) in enumerate(layouts):
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = HOME_Q
        data.ctrl[:6] = HOME_Q
        place(model, data, ids, cube_xy, goal_xy)
        traj = trajectory(plan(cube_xy, goal_xy, ik.tip(data.qpos.copy())[0]))
        traj += [traj[-1]] * int(TAIL_S * FPS)
        q_ik = data.qpos.copy()
        tracker = PlacementTracker()
        rec = {k: [] for k in ("qpos", "action", "cube_pos", "cube_quat")}
        for pos, grip in traj:
            rec["qpos"].append(data.qpos[:6].copy())  # 행동을 보내기 전 상태
            rec["cube_pos"].append(cube_pos(data, ids))
            rec["cube_quat"].append(data.qpos[ids.cube_qadr + 3:ids.cube_qadr + 7].copy())
            q_ik = ik.solve(pos, q_ik).q
            action = np.r_[q_ik[:5], grip]
            rec["action"].append(action)
            data.ctrl[:6] = action
            for _ in range(steps):
                mujoco.mj_step(model, data)
            tracker.update(cube_pos(data, ids), goal_xy, grip < 0.5, 1 / FPS)
        ok = tracker.succeeded
        yield Episode(**{k: np.array(v) for k, v in rec.items()}, goal_xy=goal_xy, success=ok,
                      info={"source": "scripted", "split": split, "layout": i,
                            "cube_xy": cube_xy.tolist(), "goal_xy": goal_xy.tolist()})


def webcam_episodes(paths: list[Path], condition: int | None = None):
    progress_cache = {}
    for path in paths:
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if condition is not None and meta.get("condition", 2) != condition:
            continue
        if "plan_unit" in meta:  # record_demos.py 녹화: 녹화 화면이 성공으로 끝낸 시도만
            prog_path = path.parent / "progress.json"
            if prog_path not in progress_cache:
                progress_cache[prog_path] = json.loads(prog_path.read_text(encoding="utf-8"))
            tries = progress_cache[prog_path]["units"][str(meta["plan_unit"]["unit"])]["tries"]
            if not any(t["file"] == path.name and t["result"] == "success" for t in tries):
                continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        if not rows or "layout" not in rows[0]:
            print(f"건너뜀(과제 장면 이전 기록): {path.name}")
            continue
        col = lambda k, R: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in R])
        for layout in sorted({int(r["layout"]) for r in rows}):
            R = [r for r in rows if int(r["layout"]) == layout]
            eng = col("engaged", R) == 1
            t_all = col("t_s", R)
            # 성공 판정은 조종 화면과 같은 PlacementTracker로 다시 계산한다(기록의 in_target 열은 예전 느슨한 판정일 수 있다)
            tracker, succ = PlacementTracker(), np.zeros(len(R), bool)
            for k, r in enumerate(R):
                dt = 0.0 if k == 0 else t_all[k] - t_all[k - 1]
                succ[k] = tracker.update(np.array([float(r[f"cube_{a}"]) for a in "xyz"]),
                                         np.array([float(r["goal_x"]), float(r["goal_y"])]),
                                         r["gripper_closed"] == "1", dt)
            if not eng.any() or not succ.any():
                continue
            start = int(np.argmax(eng))
            first_succ = int(np.argmax(succ))
            end = int(np.searchsorted(t_all, t_all[first_succ] + TAIL_S))
            keep = np.arange(start, min(end, len(R)))
            keep = keep[eng[keep] | (keep >= first_succ)]  # 클러치로 멈춘 줄 제외(성공 뒤 꼬리는 남김)
            Rk = [R[i] for i in keep]
            t = col("t_s", Rk)
            dt = np.diff(t, prepend=t[0])
            dt[dt > 0.1] = 1.0 / FPS  # 멈춘 구간을 뺀 자리의 시간 공백을 한 칸으로
            t = np.cumsum(dt)
            qpos = np.stack([col(f"q{i}", Rk) for i in range(6)], 1)
            ctrl = np.stack([col(f"ctrl{i}", Rk) for i in range(6)], 1)
            cube = np.stack([col(f"cube_{a}", Rk) for a in "xyz"], 1)
            goal = np.array([float(Rk[0]["goal_x"]), float(Rk[0]["goal_y"])])
            if "cube_qw" in Rk[0]:
                quat = np.stack([col(f"cube_q{a}", Rk) for a in "wxyz"], 1)
                quat_note = "logged"
            else:
                quat = np.tile(default_cube_quat(cube[0, :2]), (len(Rk), 1))
                quat_note = "approx (not logged; only valid while the cube is untouched)"
            # 행동 t ↔ 한 줄 앞의 상태
            v = resample(t[1:], {"qpos": qpos[:-1], "action": ctrl[1:], "cube_pos": cube[:-1], "cube_quat": quat[:-1]})
            yield Episode(**v, goal_xy=goal, success=True,
                          info={"source": "webcam", "log": path.name, "layout": layout,
                                "condition": meta.get("condition", 2), "plan_unit": meta.get("plan_unit"),
                                "cube_quat": quat_note, "raw_rows": len(Rk)})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", choices=["scripted", "webcam"])
    p.add_argument("logs", nargs="*", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--split", default="train", help="고정 배치 목록 (train/eval)")
    p.add_argument("--n", type=int, default=None, help="배치 앞에서 n개만 (시험용)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--condition", type=int, choices=[2, 3], default=None, help="웹캠 기록 중 이 조건만 (기록 JSON 기준)")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = OUT_DIR / args.name
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} 가 이미 있어요. --overwrite로 덮어쓰기")
        shutil.rmtree(root)
    model = build_model()
    ds = LeRobotDataset.create(repo_id=f"local/{args.name}", fps=FPS, features=features(), root=root,
                               robot_type="so101_sim", use_videos=True)
    renderer = SceneRenderer(model, task_ids(model))
    episodes = (scripted_episodes(model, args.split, args.n) if args.source == "scripted"
                else webcam_episodes(sorted(args.logs), args.condition))
    infos, skipped = [], 0
    for ep in episodes:
        if not ep.success:
            skipped += 1
            print(f"  실패 시범 제외: {ep.info}")
            continue
        write_episode(ds, renderer, ep)
        infos.append({**ep.info, "frames": len(ep.qpos), "seconds": round(len(ep.qpos) / FPS, 2)})
        print(f"  에피소드 {len(infos) - 1}: {len(ep.qpos)}프레임 ({len(ep.qpos) / FPS:.1f}초) {ep.info}")
    renderer.close()
    ds.finalize()
    (OUT_DIR / f"{args.name}_episodes.json").write_text(
        json.dumps({"episodes": infos, "skipped_failures": skipped}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {root}  에피소드 {len(infos)}개, 실패 제외 {skipped}개")


if __name__ == "__main__":
    main()
