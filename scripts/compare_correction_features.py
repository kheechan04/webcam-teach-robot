"""탐색(사후) 분석: 쌓기 ③ 녹화의 손 자세 특징이 보정을 맞춘 측정(M5 1회차) 범위 안이었나.

docs/13-stacking-plan.md 결과 뒤 "손이 화면 위쪽에 있었다"를 숫자로 확인한다. 계획에 없던 분석이다.
    uv run python scripts/compare_correction_features.py
특징 6개(log_d, pinch, foreshorten, normal_z, u, v)마다 M5 맞춤 데이터의 1~99% 범위와, 옮기기 ③·쌓기 ③ 녹화 프레임 중
그 범위 밖인 비율을 낸다. 녹화 쪽 핀홀 거리는 조종에 쓴 것(depth_len·width 중 작은 값)과 같은 방식으로 다시 계산한다.
"""

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_depth_v2 import load  # noqa: E402
from fit_depth_correction import build  # noqa: E402

from webcam_teach_robot.depth_correction import FEATURE_NAMES, posture_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
W, H = 640, 480


def recording_features(folder: Path) -> np.ndarray:
    out = []
    for f in sorted(folder.glob("u*.csv")):
        meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("condition") != 3:
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["engaged"] != "1" or not r.get("px0_u") or not r["depth_len_m"]:
                continue
            P = np.array([[float(r[f"px{i}_{a}"]) for a in "uv"] for i in range(21)])
            Wd = np.array([[float(r[f"w{i}_{a}"]) for a in "xyz"] for i in range(21)])
            d = min(float(r["depth_len_m"]), float(r["depth_width_m"]))
            out.append(posture_features(P, Wd, d, W, H))
    return np.array(out)


def main() -> None:
    _, data = load(ROOT / "measurements" / "depth_v2" / "20261002_232355.csv")
    rows = [r for r in build(data) if r["step"]["rep"] == 0]  # 보정을 맞춘 1회차
    fit = np.concatenate([r["X"] for r in rows])
    lo, hi = np.percentile(fit, 1, axis=0), np.percentile(fit, 99, axis=0)
    rec = {"옮기기 ③": recording_features(ROOT / "measurements" / "demos_m8"),
           "쌓기 ③": recording_features(ROOT / "measurements" / "demos_stack")}
    summary = {"fit_p1_p99": {n: [float(a), float(b)] for n, a, b in zip(FEATURE_NAMES, lo, hi)}}
    print(f"{'특징':12s} {'M5 맞춤 1~99%':>18s} | " + " | ".join(f"{k} 중앙 · 범위 밖" for k in rec))
    for j, n in enumerate(FEATURE_NAMES):
        cells = []
        for k, X in rec.items():
            outside = float(np.mean((X[:, j] < lo[j]) | (X[:, j] > hi[j])))
            cells.append(f"{np.median(X[:, j]):+7.3f} · {outside * 100:4.1f}%")
            summary.setdefault(k, {})[n] = {"median": float(np.median(X[:, j])), "outside_fit_range": outside}
        print(f"{n:12s} {lo[j]:+8.3f} ~ {hi[j]:+8.3f} | " + " | ".join(cells))
    for k, X in rec.items():
        any_out = float(np.mean(np.any((X < lo) | (X > hi), axis=1)))
        summary[k]["frames"] = len(X)
        summary[k]["any_feature_outside"] = any_out
        print(f"{k}: 프레임 {len(X)}, 특징 하나라도 범위 밖 {any_out * 100:.1f}%")
    (ROOT / "experiments" / "correction_feature_range.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
