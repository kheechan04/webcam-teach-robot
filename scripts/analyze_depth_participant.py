"""다른 시범자의 짧은 깊이 측정(measure_depth_v2.py --short) 분석: 시범자 1 손으로 맞춘 보정(조건 ③)이 이 사람 손에도 통하나.

    uv run python scripts/analyze_depth_participant.py measurements/depth_v2/XXX.csv
비교 (모두 같은 16단계, 거리 45·55 cm, 웹캠 높이):
    지금 방식(②)            길이·너비 중 작은 값, 화각 60° 가정
    시범자 1 보정(③)         depth_correction.json 그대로
    이 사람 1회차로 맞춘 보정  같은 특징으로 이 사람 1회차만 써서 다시 맞추고 2회차로 평가 (사람마다 맞추면 얼마나 나아지나)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_depth_v2 import POSE_KO, arrays, load  # noqa: E402
from fit_depth_correction import build, fit_ridge, pinhole_min, predict  # noqa: E402

from webcam_teach_robot.depth_correction import COEF_FILE  # noqa: E402

import json  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1])
    meta, data = load(path)
    rows = build(data)
    coef1 = json.loads(COEF_FILE.read_text(encoding="utf-8"))
    tr = [r for r in rows if r["step"]["rep"] == 0]
    te = [r for r in rows if r["step"]["rep"] == 1]
    coef_own = fit_ridge(tr, coef1["alpha"])
    step_t = [float(np.median(np.log(r["truth"] / r["d"]))) for r in tr]
    coef_own["log_factor_range"] = [min(step_t), max(step_t)]

    def table(rs, est):
        out = {}
        for pose in POSE_KO:
            v = [float(np.median(est(r)) - r["truth"]) * 100 for r in rs if r["step"]["pose"] == pose]
            out[pose] = float(np.mean(v)) if v else float("nan")
        return out

    methods = {"지금 방식(②)": lambda r: r["d"], "시범자 1 보정(③)": lambda r: predict(coef1, r["X"], r["d"]),
               "이 사람으로 맞춘 보정": lambda r: predict(coef_own, r["X"], r["d"])}
    print(f"{path.name}: 시범자 {meta.get('participant')}, 단계 {len(rows)} (맞춤 1회차 {len(tr)}, 평가 2회차 {len(te)})")
    print("\n2회차 평가, 자세별 평균 오차(cm, 추정 − 참값)와 자세 간 범위·평균 절대 오차")
    summary = {}
    for name, est in methods.items():
        t = table(te, est)
        rng_ = max(t.values()) - min(t.values())
        mae = float(np.mean([abs(np.median(est(r)) - r["truth"]) * 100 for r in te]))
        summary[name] = {"by_pose": t, "pose_range": rng_, "mae": mae}
        print(f"  {name:14s} " + "  ".join(f"{POSE_KO[p]} {v:+5.1f}" for p, v in t.items())
              + f"  | 범위 {rng_:4.1f}  절대 {mae:4.1f}")
    out = Path("experiments") / f"depth_participant_{path.stem}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
