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

import os
import sys

# 쌓기 과제: --task stack 이면 장면 모듈을 불러오기 전에 과제를 정한다(scene.TASK는 불러올 때 한 번 정해진다)
if "--task" in sys.argv and sys.argv[sys.argv.index("--task") + 1] == "stack":
    os.environ["WTR_TASK"] = "stack"

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_depth_v2 import draw_text  # noqa: E402

from webcam_teach_robot.depth_correction import DepthCorrection
from webcam_teach_robot.hand_tracking import HandTracker
from webcam_teach_robot.marker_depth import MarkerTracker
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


# 쌓기 과제(--task stack --session stack): M8과 같은 규칙(②·③ ABBA, 50개씩, 조건 가림, 3번까지)에 시드만 다르다.
# 시간 제한은 사용자 시험 조종(docs/13-stacking-plan.md) 뒤 녹화 전에 확정한다.
STACK_PLAN_SEED = PLAN_SEED + 200
STACK_TIME_LIMIT_S = 120.0


def make_plan(n_layouts: int, seed: int = PLAN_SEED, time_limit_s: float = TIME_LIMIT_S, session: str = "m8") -> dict:
    """블록마다 같은 배치 5개를 [먼저 할 조건 5개, 다른 조건 5개]로. 먼저 할 조건은 ABBA 반복:
    블록 0 A, 1 B, 2 B, 3 A, 4 A, 5 B, ... (A가 ②인지 ③인지는 시드로 정한다). 블록 안 배치 순서도 조건마다 섞는다."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_layouts)
    a, b = (2, 3) if rng.random() < 0.5 else (3, 2)
    units = []
    for blk in range(n_layouts // BLOCK):
        first, second = (a, b) if blk % 4 in (0, 3) else (b, a)
        layouts = [int(x) for x in order[blk * BLOCK:(blk + 1) * BLOCK]]
        for cond in (first, second):
            for lay in rng.permutation(layouts):
                units.append({"unit": len(units), "block": blk, "condition": cond, "layout": int(lay)})
    return {"seed": seed, "session": session, "block_size": BLOCK, "time_limit_s": time_limit_s,
            "max_tries": MAX_TRIES, "split": "train", "units": units}


def make_plan_m8b(n_layouts: int) -> dict:
    """세션 m8b (2026-10-04): 조건 ⑤(손목 마커 깊이) 50개 + 조건 ② 20개(며칠 사이 조종 실력 변화 확인용).
    ⑤ 5개 묶음 10개 사이에 ② 5개 묶음 4개를 고르게 끼운다. ②의 배치는 ⑤에서 이미 한 배치 중에서 고른다.
    조작자는 내내 마커를 차고 있고, 화면에는 조건을 표시하지 않는다(가림 유지)."""
    rng = np.random.default_rng(PLAN_SEED + 1)
    order = [int(x) for x in rng.permutation(n_layouts)]
    five = [order[i * BLOCK:(i + 1) * BLOCK] for i in range(n_layouts // BLOCK)]
    two_after = {1: 0, 3: 2, 6: 5, 8: 7}  # ⑤ 묶음 k 뒤에, ⑤ 묶음 j의 배치로 ② 묶음
    units, blk = [], 0
    for k, lays in enumerate(five):
        for lay in lays:
            units.append({"unit": len(units), "block": blk, "condition": 5, "layout": lay})
        blk += 1
        if k in two_after:
            for lay in rng.permutation(five[two_after[k]]):
                units.append({"unit": len(units), "block": blk, "condition": 2, "layout": int(lay)})
            blk += 1
    return {"seed": PLAN_SEED + 1, "session": "m8b", "block_size": BLOCK, "time_limit_s": TIME_LIMIT_S,
            "max_tries": MAX_TRIES, "split": "train", "units": units}


PARTICIPANT_LAYOUTS = 20  # 다른 시범자: 학습 배치 중 20개 × ②·③
PRACTICE_MIN_SUCCESS = 3  # 연습 5개 중 이만큼 성공해야 본 녹화로 (2026-10-03 미리 정한 기준)


def make_plan_participant(n_layouts: int, participant: int) -> dict:
    """다른 시범자 세션: 연습 5개(연습용 배치, 학습·평가에 안 씀, ②·③ 번갈아) → 기준 통과 시 본 녹화
    ② 20개 + ③ 20개(모든 시범자에게 같은 학습 배치 20개). 5개 묶음, 먼저 할 조건은 ABBA(시범자마다 시작 조건을 바꿈)."""
    rng = np.random.default_rng(PLAN_SEED + 100 + participant)
    common = [int(x) for x in np.random.default_rng(PLAN_SEED + 99).permutation(n_layouts)[:PARTICIPANT_LAYOUTS]]
    units = [{"unit": k, "block": -1, "condition": (2, 3)[k % 2], "layout": k, "split": "practice", "practice": True}
             for k in range(5)]
    a, b = (2, 3) if participant % 2 == 0 else (3, 2)
    for blk in range(PARTICIPANT_LAYOUTS // BLOCK):
        first, second = (a, b) if blk % 4 in (0, 3) else (b, a)
        lays = common[blk * BLOCK:(blk + 1) * BLOCK]
        for cond in (first, second):
            for lay in rng.permutation(lays):
                units.append({"unit": len(units), "block": blk, "condition": cond, "layout": int(lay), "split": "train"})
    return {"seed": PLAN_SEED + 100 + participant, "session": f"p{participant}", "participant": participant,
            "block_size": BLOCK, "time_limit_s": TIME_LIMIT_S, "max_tries": MAX_TRIES, "split": "train",
            "practice_min_success": PRACTICE_MIN_SUCCESS, "units": units}


INTRO = [
    "안내 (모든 시범자에게 같은 문장)",
    "· 웹캠 영상은 저장하지 않아요. 손 마디 좌표 숫자만 남고, 결과에는 이름 대신 '시범자 번호'로 적어요.",
    "· 오른손만 써요. 손바닥이 카메라를 보게 하고, 카메라에서 약 50 cm에 손을 두세요.",
    "· 스페이스 = 조종 시작/멈춤. 손이 화면 끝에 닿으면 멈추고 손을 가운데로 옮겨 다시 시작해요.",
    "· 손목은 돌리지 말고, 창문 닦듯이 팔 전체로 옮겨요. 엄지·검지를 붙이면 집게가 닫혀요.",
    "· 빨간 큐브를 집어 초록 사각형 위에 내려놓고 손가락을 펴면 성공. 한 번에 90초.",
    "· 먼저 연습 5개를 하고, 5개 중 3개 이상 성공하면 본 녹화로 넘어가요.",
    "준비되면 스페이스",
]


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)  # 쓰는 중에 꺼져도 예전 파일은 남는다


def load_state(out: Path, session: str = "m8", participant: int | None = None):
    out.mkdir(parents=True, exist_ok=True)
    plan_path, prog_path = out / "plan.json", out / "progress.json"
    if not plan_path.exists():
        n = len(load_layouts("train"))
        write_json(plan_path, make_plan_participant(n, participant) if session == "person"
                   else make_plan_m8b(n) if session == "m8b"
                   else make_plan(n, STACK_PLAN_SEED, STACK_TIME_LIMIT_S, "stack") if session == "stack" else make_plan(n))
        print(f"녹화 계획을 만들었어요: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prog = json.loads(prog_path.read_text(encoding="utf-8")) if prog_path.exists() else {"units": {}}
    return plan, prog, prog_path


def unit_state(prog, k: int) -> dict:
    return prog["units"].setdefault(str(k), {"status": "todo", "tries": []})


def practice_passed(plan, prog) -> bool | None:
    """연습 단계가 있는 세션: 연습이 다 끝났으면 기준 통과 여부, 아직이면 None, 연습이 없는 세션이면 True."""
    pr = [u for u in plan["units"] if u.get("practice")]
    if not pr or prog.get("practice_skipped"):
        return True
    st = [unit_state(prog, u["unit"])["status"] for u in pr]
    if "todo" in st:
        return None
    return sum(x == "success" for x in st) >= plan.get("practice_min_success", PRACTICE_MIN_SUCCESS)


def next_unit(plan, prog):
    passed = practice_passed(plan, prog)
    for u in plan["units"]:
        if u.get("practice") and prog.get("practice_skipped"):
            continue
        if unit_state(prog, u["unit"])["status"] == "todo":
            if not u.get("practice") and passed is False:
                return None  # 연습 기준 미달: 본 녹화는 하지 않는다(연습 기록은 남는다)
            return u
    return None


def print_status(plan, prog) -> None:
    by = {c: [0, 0, 0] for c in sorted({u["condition"] for u in plan["units"]})}  # 성공, 실패, 남음
    for u in plan["units"]:
        st = unit_state(prog, u["unit"])["status"]
        by[u["condition"]][{"success": 0, "failed": 1}.get(st, 2)] += 1
    done = sum(v[0] + v[1] for v in by.values())
    print(f"진행 {done}/{len(plan['units'])}  (조건별 성공/실패/남음은 녹화가 다 끝난 뒤에 보는 게 좋아요)")
    return by


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["place", "stack"], default="place", help="과제 (stack = 큐브 쌓기)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--status", action="store_true", help="진행 상황만 보고 끝내기")
    p.add_argument("--session", choices=["m8", "m8b", "person", "stack"], default="m8",
                   help="m8b = 조건 ⑤(손목 마커) 50개 + ② 20개 / person = 다른 시범자(연습 5 + ②·③ 20개씩)")
    p.add_argument("--participant", type=int, default=None, help="--session person일 때 시범자 번호(2, 3, ...)")
    p.add_argument("--skip-practice", action="store_true",
                   help="연습 5개를 건너뛰고 바로 본 녹화(자유 연습으로 대신했을 때). 기록에 '건너뜀'으로 남는다")
    args = p.parse_args()

    if (args.session == "stack") != (args.task == "stack"):
        raise SystemExit("쌓기 녹화는 --task stack --session stack 을 같이 써 주세요")
    if args.session == "stack" and args.out == OUT_DIR:
        args.out = ROOT / "measurements" / "demos_stack"
    if args.session == "m8b" and args.out == OUT_DIR:
        args.out = ROOT / "measurements" / "demos_m8b"
    if args.session == "person":
        if not args.participant or args.participant < 2:
            raise SystemExit("--participant 2 처럼 시범자 번호를 넣어 주세요 (1은 프로젝트 진행자)")
        if args.out == OUT_DIR:
            args.out = ROOT / "measurements" / f"demos_p{args.participant}"
    plan, prog, prog_path = load_state(args.out, args.session, args.participant)
    if args.skip_practice and any(u.get("practice") for u in plan["units"]) and not prog.get("practice_skipped"):
        prog["practice_skipped"] = {"when": datetime.now().isoformat(timespec="seconds"),
                                    "why": "자유 연습(teleop.py)으로 대신하고 연습 5개·참가 기준을 건너뜀"}
        write_json(prog_path, prog)
    if args.status:
        by = print_status(plan, prog)
        if next_unit(plan, prog) is None:
            print("조건 ②", by[2], "/ 조건 ③", by[3], "(성공, 실패, 남음)")
        return
    if next_unit(plan, prog) is None:
        print("다 끝났어요.")
        return
    layouts = {sp: load_layouts(sp) for sp in {u.get("split", plan["split"]) for u in plan["units"]}}
    settings = RigSettings()
    correction = DepthCorrection()
    rig = TeleopRig(settings, None)
    marker = MarkerTracker() if plan.get("session") == "m8b" else None  # m8b는 ② 단계에서도 마커를 기록만 한다

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
        rig.marker_mode = u["condition"] == 5
        cube_xy, goal_xy = layouts[u.get("split", plan["split"])][u["layout"]]
        rig.reset(cube_xy, goal_xy, now())
        stem = f"u{u['unit']:03d}_try{tries_done + 1}_{datetime.now():%Y%m%d_%H%M%S}"
        path = args.out / f"{stem}.csv"
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(LOG_COLUMNS)
        meta = run_meta(settings, rig.correction, frame_size, {
            "plan_unit": u, "try": tries_done + 1, "started": datetime.now().isoformat(timespec="seconds"),
            "time_limit_s": plan["time_limit_s"], "layout_split": plan["split"], "session": plan.get("session", "m8")})
        if u["condition"] == 5:
            meta.update(condition=5, hand_source="wrist_marker", marker={"dict": "4x4_50", "id": 0, "size_m": 0.05})
        write_json(path.with_suffix(".json"), meta)
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
        elif result != "interrupted" and (st["try_no"] >= plan["max_tries"] or st["unit"].get("practice")):
            st["us"]["status"] = "failed"  # 연습은 한 번씩만
        write_json(prog_path, prog)
        st["result"] = result

    start_attempt()
    # 다른 시범자 세션: 처음(아무 시도도 안 했을 때) 같은 안내문을 띄운다
    show_intro = plan.get("session", "").startswith("p") and not any(v["tries"] for v in prog["units"].values() if v["tries"])
    print_status(plan, prog)

    with HandTracker(horizontal_fov_deg=settings.fov, depth_segment="min") as tracker, \
            mujoco.viewer.launch_passive(rig.model, rig.data, show_left_ui=False, show_right_ui=False) as viewer:
        setup_viewer_camera(viewer)
        while viewer.is_running():
            t_loop = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            marker_obs = marker.detect(frame) if marker else None  # 거울상이 되기 전 원본에서
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            t_det = time.perf_counter()
            obs = tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int((t_det - t0) * 1000))
            detect_ms = (time.perf_counter() - t_det) * 1000

            recording = st["result"] is None
            if recording:
                rig.update_hand(obs, w, h, now(), time.perf_counter, marker_obs)
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
            if marker:
                MarkerTracker.draw(frame, marker_obs)
                if marker_obs is None:
                    cv2.putText(frame, "MARKER NOT SEEN", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
                    cv2.putText(frame, "MARKER NOT SEEN", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
            u = st["unit"]
            n_units = len(plan["units"])
            n_blocks = len({x["block"] for x in plan["units"] if not x.get("practice")})
            head = (f"PRACTICE {u['unit'] + 1}/5" if u.get("practice") else
                    f"block {u['block'] + 1}/{n_blocks}  demo {u['unit'] + 1}/{n_units}  try {st['try_no']}/{plan['max_tries']}")
            if st["result"] is None:
                left = plan["time_limit_s"] - (now() - st["t_engage"]) if st["t_engage"] is not None else plan["time_limit_s"]
                lines = [head, f"{rig.status()}  time left {left:4.0f} s",
                         "SUCCESS!" if st["t_success"] is not None else ("move cube to green" if obs else "hand: not found")]
                put_lines(frame, lines, (80, 255, 80) if rig.engaged else (255, 255, 255))
            else:
                msg = {"success": "SUCCESS - saved", "timeout": "time over", "abandoned": "abandoned"}[st["result"]]
                nxt = next_unit(plan, prog)
                lines = [head, msg]
                if nxt is None and practice_passed(plan, prog) is False:
                    ok = sum(unit_state(prog, x["unit"])["status"] == "success" for x in plan["units"] if x.get("practice"))
                    lines += [f"practice {ok}/5 - below 3, recording ends here. thank you!", "press q"]
                elif nxt is None:
                    lines += ["ALL DONE - press q"]
                else:
                    if nxt["unit"] == u["unit"]:
                        lines += ["same layout again: press n"]
                    elif u.get("practice") and not nxt.get("practice"):
                        lines += ["practice done. main recording next", "n: start   q: quit (resume later)"]
                    elif nxt["block"] != u["block"]:
                        lines += [f"block {u['block'] + 1} done. rest if you like", "n: next   q: quit (resume later)"]
                    else:
                        lines += ["n: next demo"]
                put_lines(frame, lines, (80, 220, 255))
            if show_intro:
                frame = draw_text(frame, INTRO, 17)
            cv2.imshow("recording camera (not recorded)", frame)

            key = cv2.waitKey(1) & 0xFF
            if show_intro:
                if key == ord(" "):
                    show_intro = False
                continue
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
