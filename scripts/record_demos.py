"""M8 웹캠 시범 녹화: 조건 ②(보정 없음)와 ③(깊이 보정)을 번갈아, 고정 학습 배치 50개씩.

실행:
    uv run python scripts/record_demos.py                 # 이어서 녹화 (처음이면 녹화 계획을 만든다)
    uv run python scripts/record_demos.py --status        # 진행 상황만 보기

규칙 (docs/09-recording.md):
    - 학습 배치 50개(experiments/layouts_v1.json의 train)를 5개씩 10블록으로 나눈다. 블록마다 같은 5개를
      ②로 한 번, ③으로 한 번 한다. 어느 쪽을 먼저 할지는 블록마다 ABBA 순서로 바꾼다(연습 효과 상쇄).
    - 화면에는 지금이 ②인지 ③인지 보여 주지 않는다. 기록 파일(JSON)에만 남는다.
    - 한 시도는 조종을 처음 건 때부터 90초. 같은 배치·조건은 3번까지 다시 한다. 3번 다 실패하면 실패로 두고 넘어간다.
    - 시도마다 CSV 하나 + JSON 하나. 진행 상황은 시도가 끝날 때마다 progress.json에 저장한다.
      중간에 꺼도(q, 창 닫기, 노트북 꺼짐) 다음 실행 때 끝나지 않은 시도부터 다시 한다. 끝나지 않은 시도는 횟수에 넣지 않는다.

키 (웹캠 창을 클릭해서 선택한 상태에서):
    스페이스  조종 시작/멈춤 (클러치)          h  로봇을 처음 자세로
    x         이번 시도 포기 (실패 1회로 셈)      n  다음으로 (시도가 끝난 뒤)
    q / Esc   끄기 (나중에 이어서)
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco.viewer
import numpy as np

from webcam_teach_robot.depth_correction import DepthCorrection
from webcam_teach_robot.hand_tracking import HandTracker
from webcam_teach_robot.scene import load_layouts
from webcam_teach_robot.teleop_rig import (LOG_COLUMNS, RigSettings, TeleopRig, put_lines, run_meta,
                                           setup_viewer_camera, warn_palm)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "measurements" / "demos_m8"
PLAN_SEED = 20261003
BLOCK = 5
TIME_LIMIT_S = 90.0
MAX_TRIES = 3
TAIL_S = 1.5  # 성공 뒤 더 기록하는 시간 (데이터셋은 성공 1초 뒤까지 쓴다)


def make_plan(n_layouts: int) -> dict:
    """블록마다 같은 배치 5개를 [먼저 할 조건 5개, 다른 조건 5개]로. 먼저 할 조건은 ABBA 반복:
    블록 0 A, 1 B, 2 B, 3 A, 4 A, 5 B, ... (A가 ②인지 ③인지는 시드로 정한다). 블록 안 배치 순서도 조건마다 섞는다."""
    rng = np.random.default_rng(PLAN_SEED)
    order = rng.permutation(n_layouts)
    a, b = (2, 3) if rng.random() < 0.5 else (3, 2)
    units = []
    for blk in range(n_layouts // BLOCK):
        first, second = (a, b) if blk % 4 in (0, 3) else (b, a)
        layouts = [int(x) for x in order[blk * BLOCK:(blk + 1) * BLOCK]]
        for cond in (first, second):
            for lay in rng.permutation(layouts):
                units.append({"unit": len(units), "block": blk, "condition": cond, "layout": int(lay)})
    return {"seed": PLAN_SEED, "block_size": BLOCK, "time_limit_s": TIME_LIMIT_S, "max_tries": MAX_TRIES,
            "split": "train", "units": units}


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)  # 쓰는 중에 꺼져도 예전 파일은 남는다


def load_state(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    plan_path, prog_path = out / "plan.json", out / "progress.json"
    if not plan_path.exists():
        write_json(plan_path, make_plan(len(load_layouts("train"))))
        print(f"녹화 계획을 만들었어요: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prog = json.loads(prog_path.read_text(encoding="utf-8")) if prog_path.exists() else {"units": {}}
    return plan, prog, prog_path


def unit_state(prog, k: int) -> dict:
    return prog["units"].setdefault(str(k), {"status": "todo", "tries": []})


def next_unit(plan, prog):
    for u in plan["units"]:
        if unit_state(prog, u["unit"])["status"] == "todo":
            return u
    return None


def print_status(plan, prog) -> None:
    by = {2: [0, 0, 0], 3: [0, 0, 0]}  # 성공, 실패, 남음
    for u in plan["units"]:
        st = unit_state(prog, u["unit"])["status"]
        by[u["condition"]][{"success": 0, "failed": 1}.get(st, 2)] += 1
    done = sum(v[0] + v[1] for v in by.values())
    print(f"진행 {done}/{len(plan['units'])}  (조건별 성공/실패/남음은 녹화가 다 끝난 뒤에 보는 게 좋아요)")
    return by


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--status", action="store_true", help="진행 상황만 보고 끝내기")
    args = p.parse_args()

    plan, prog, prog_path = load_state(args.out)
    if args.status:
        by = print_status(plan, prog)
        if next_unit(plan, prog) is None:
            print("조건 ②", by[2], "/ 조건 ③", by[3], "(성공, 실패, 남음)")
        return
    if next_unit(plan, prog) is None:
        print("다 끝났어요.")
        return
    layouts = load_layouts(plan["split"])
    settings = RigSettings()
    correction = DepthCorrection()
    rig = TeleopRig(settings, None)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.camera}번을 열 수 없어요.")
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if frame_size != (640, 480):
        raise SystemExit(f"웹캠 해상도가 {frame_size}예요. 보정 계수는 640×480 기준이에요.")

    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0  # noqa: E731

    # ---- 시도 하나의 상태 ----
    st = {}

    def start_attempt():
        u = next_unit(plan, prog)
        if u is None:
            return False
        us = unit_state(prog, u["unit"])
        tries_done = len([t for t in us["tries"] if t["result"] != "interrupted"])
        rig.correction = correction if u["condition"] == 3 else None
        cube_xy, goal_xy = layouts[u["layout"]]
        rig.reset(cube_xy, goal_xy, now())
        stem = f"u{u['unit']:03d}_try{tries_done + 1}_{datetime.now():%Y%m%d_%H%M%S}"
        path = args.out / f"{stem}.csv"
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(LOG_COLUMNS)
        write_json(path.with_suffix(".json"), run_meta(settings, rig.correction, frame_size, {
            "plan_unit": u, "try": tries_done + 1, "started": datetime.now().isoformat(timespec="seconds"),
            "time_limit_s": plan["time_limit_s"], "layout_split": plan["split"]}))
        st.update(unit=u, us=us, try_no=tries_done + 1, file=f, writer=w, path=path, t_engage=None,
                  t_success=None, result=None, t_start=now(), last_flush=now())
        return True

    def finish_attempt(result: str):
        st["file"].close()
        engaged_s = (now() - st["t_engage"]) if st["t_engage"] is not None else 0.0
        if st["t_success"] is not None:
            engaged_s = st["t_success"] - st["t_engage"]
        st["us"]["tries"].append({"file": st["path"].name, "result": result, "seconds_from_engage": round(engaged_s, 2),
                                  "seconds_total": round(now() - st["t_start"], 2)})
        if result == "success":
            st["us"]["status"] = "success"
        elif result != "interrupted" and st["try_no"] >= plan["max_tries"]:
            st["us"]["status"] = "failed"
        write_json(prog_path, prog)
        st["result"] = result

    start_attempt()
    print_status(plan, prog)

    with HandTracker(horizontal_fov_deg=settings.fov, depth_segment="min") as tracker, \
            mujoco.viewer.launch_passive(rig.model, rig.data, show_left_ui=False, show_right_ui=False) as viewer:
        setup_viewer_camera(viewer)
        while viewer.is_running():
            t_loop = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            t_det = time.perf_counter()
            obs = tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int((t_det - t0) * 1000))
            detect_ms = (time.perf_counter() - t_det) * 1000

            recording = st["result"] is None
            if recording:
                rig.update_hand(obs, w, h, now(), time.perf_counter)
                if rig.engaged and st["t_engage"] is None:
                    st["t_engage"] = now()  # 90초는 조종을 처음 건 때부터
            first_success = rig.step_sim(now())
            rig.draw_viewer(viewer)

            if recording:
                st["writer"].writerow(rig.log_row(now(), obs, detect_ms, (time.perf_counter() - t_loop) * 1000,
                                                  st["unit"]["layout"]))
                if now() - st["last_flush"] > 1.0:  # 갑자기 꺼져도 잃는 건 최대 1초
                    st["file"].flush()
                    st["last_flush"] = now()
                if first_success:
                    st["t_success"] = now()
                if st["t_success"] is not None and now() - st["t_success"] > TAIL_S:
                    finish_attempt("success")
                elif (st["t_success"] is None and st["t_engage"] is not None
                      and now() - st["t_engage"] > plan["time_limit_s"]):
                    finish_attempt("timeout")

            # ---- 화면 ----
            rig.draw_frame(frame, obs)
            warn_palm(frame, obs)
            u = st["unit"]
            n_units = len(plan["units"])
            head = f"block {u['block'] + 1}/{n_units // (2 * BLOCK)}  demo {u['unit'] + 1}/{n_units}  try {st['try_no']}/{plan['max_tries']}"
            if st["result"] is None:
                left = plan["time_limit_s"] - (now() - st["t_engage"]) if st["t_engage"] is not None else plan["time_limit_s"]
                lines = [head, f"{rig.status()}  time left {left:4.0f} s",
                         "SUCCESS!" if st["t_success"] is not None else ("move cube to green" if obs else "hand: not found")]
                put_lines(frame, lines, (80, 255, 80) if rig.engaged else (255, 255, 255))
            else:
                msg = {"success": "SUCCESS - saved", "timeout": "time over", "abandoned": "abandoned"}[st["result"]]
                nxt = next_unit(plan, prog)
                lines = [head, msg]
                if nxt is None:
                    lines += ["ALL DONE - press q"]
                else:
                    if nxt["unit"] == u["unit"]:
                        lines += ["same layout again: press n"]
                    elif nxt["block"] != u["block"]:
                        lines += [f"block {u['block'] + 1} done. rest if you like", "n: next   q: quit (resume later)"]
                    else:
                        lines += ["n: next demo"]
                put_lines(frame, lines, (80, 220, 255))
            cv2.imshow("recording camera (not recorded)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if st["result"] is None:
                if key == ord(" "):
                    rig.toggle_clutch()
                elif key == ord("h"):
                    rig.home()
                elif key == ord("x") and st["t_success"] is None:
                    finish_attempt("abandoned")
            elif key == ord("n") and next_unit(plan, prog) is not None:
                start_attempt()

    if st["result"] is None:  # 시도 도중에 껐다: 횟수에 넣지 않고 다음에 다시(성공한 뒤 꼬리 기록 중이었으면 성공)
        finish_attempt("success" if st["t_success"] is not None else "interrupted")
    cap.release()
    cv2.destroyAllWindows()
    print_status(plan, prog)


if __name__ == "__main__":
    main()
