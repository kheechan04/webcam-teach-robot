"""조건 ④·④0 시범 만들기 (가상 조작자, src/webcam_teach_robot/virtual_operator.py).

    uv run python scripts/make_cond4.py
→ measurements/cond4_virtual/{err,clean}/u*.csv (+ .json), 웹캠 녹화와 같은 형식이라 build_dataset.py로 그대로 만든다:
    uv run python scripts/build_dataset.py webcam --name cond4_depth_error --condition 4 measurements/cond4_virtual/err/*.csv
    uv run python scripts/build_dataset.py webcam --name cond4z_no_error --condition 40 measurements/cond4_virtual/clean/*.csv

학습 배치 50개마다: 오차 재료로 쓸 ② 시범은 **같은 배치의 ② 시범**(사람이 그 배치에서 실제로 보인 손 자세).
실패하면 사람 녹화처럼 3번까지 다시(흔들림 시작 위치를 바꿔서). ④0은 같은 배치를 오차 없이 한 번.
"""

import csv
import json
from pathlib import Path

import numpy as np

from webcam_teach_robot.depth_correction import DepthCorrection
from webcam_teach_robot.scene import load_layouts
from webcam_teach_robot.teleop_rig import LOG_COLUMNS, RigSettings, TeleopRig, run_meta
from webcam_teach_robot.virtual_operator import load_error_source, run_episode

ROOT = Path(__file__).resolve().parent.parent
DEMOS = ROOT / "measurements" / "demos_m8"
OUT = ROOT / "measurements" / "cond4_virtual"
MAX_TRIES = 3


def cond2_demo_for_layout() -> dict[int, Path]:
    plan = json.loads((DEMOS / "plan.json").read_text(encoding="utf-8"))
    prog = json.loads((DEMOS / "progress.json").read_text(encoding="utf-8"))
    out = {}
    for u in plan["units"]:
        if u["condition"] == 2:
            ok = [t["file"] for t in prog["units"][str(u["unit"])]["tries"] if t["result"] == "success"]
            out[u["layout"]] = DEMOS / ok[0]
    return out


def main() -> None:
    layouts = load_layouts("train")
    correction = DepthCorrection()
    demos = cond2_demo_for_layout()
    rig = TeleopRig(RigSettings(), None)
    summary = {"err": [], "clean": []}
    for variant in ("clean", "err"):
        d = OUT / variant
        d.mkdir(parents=True, exist_ok=True)
        for lay, (cube_xy, goal_xy) in enumerate(layouts):
            src = load_error_source(demos[lay], correction) if variant == "err" else None
            tries = []
            for k in range(1 if variant == "clean" else MAX_TRIES):
                if src is not None and k > 0:
                    src.jitter = np.roll(src.jitter, 97 * k)  # 다시 할 때는 흔들림을 다른 데서부터
                path = d / f"layout{lay:02d}_try{k + 1}.csv"
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(LOG_COLUMNS)
                    r = run_episode(rig, cube_xy, goal_xy, src, w, lay)
                meta = run_meta(rig.s, None, (640, 480), {
                    "virtual_operator": True, "depth_error": variant == "err",
                    "error_source": src.name if src else None, "try": k + 1, "layout": lay, "layout_split": "train"})
                meta["condition"] = 4 if variant == "err" else 40
                meta["result"] = "success" if r["success"] else "failed"
                path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
                tries.append({"file": path.name, **r})
                if r["success"]:
                    break
            summary[variant].append({"layout": lay, "tries": tries})
            print(variant, lay, [(t["success"], t["seconds"]) for t in tries])
    for v, s in summary.items():
        ok = sum(any(t["success"] for t in x["tries"]) for x in s)
        n = sum(len(x["tries"]) for x in s)
        first = sum(x["tries"][0]["success"] for x in s)
        print(f"{v}: 배치 성공 {ok}/50, 시도 {n}, 첫 시도 성공 {first}")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
