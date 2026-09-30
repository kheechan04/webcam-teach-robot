"""measure_depth.py로 모은 깊이 측정을 분석한다.

실행:
    uv run python scripts/analyze_depth.py                          # 가장 최근 측정
    uv run python scripts/analyze_depth.py measurements/depth/XXX.csv

추정 방법 두 가지를 비교한다 (둘 다 깊이 = 초점거리 × 실제 길이 / 사진 속 길이).
    palm length  손목(0) ~ 가운뎃손가락 뿌리(9). 지금 hand_tracking.py가 쓰는 방법
    palm width   검지 뿌리(5) ~ 새끼 뿌리(17). 손바닥을 앞뒤로 기울여도 덜 짧아질 거라는 가설
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEPTH_DIR = ROOT / "measurements" / "depth"
FIG_DIR = ROOT / "docs" / "figures"

ESTIMATORS = {"palm length (0-9)": (0, 9), "palm width (5-17)": (5, 17)}
TITLES = {"palm length (0-9)": "손바닥 길이로 추정 (손목~가운뎃손가락 뿌리)",
          "palm width (5-17)": "손바닥 너비로 추정 (검지 뿌리~새끼 뿌리)"}
POSE_STYLE = {"flat": ("#2a78d6", "o", "펴기"), "tilt": ("#eb6834", "s", "기울이기")}
LABEL_NUDGE = {"flat": -10, "tilt": 10}  # 끝점이 겹칠 때 라벨이 안 겹치게 위아래로 비킨다

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def latest_csv() -> Path:
    files = [p for p in sorted(DEPTH_DIR.glob("*.csv")) if sum(1 for _ in open(p, encoding="utf-8")) > 1]
    if not files:
        raise SystemExit("측정 파일이 없어요.")
    return files[-1]


def load(path: Path) -> dict[tuple[str, int], list[dict]]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    last_take: dict[int, int] = {}
    for r in rows:
        last_take[int(r["step"])] = max(last_take.get(int(r["step"]), 0), int(r["take"]))
    groups = defaultdict(list)
    for r in rows:
        if int(r["take"]) == last_take[int(r["step"])]:
            groups[(r["pose"], int(r["distance_cm"]))].append(r)
    return groups


def depth_cm(rows: list[dict], a: int, b: int, focal: float) -> np.ndarray:
    px = np.array([[float(r[f"px{i}_{c}"]) for c in "uv"] for r in rows for i in (a, b)]).reshape(-1, 2, 2)
    w = np.array([[float(r[f"w{i}_{c}"]) for c in "xyz"] for r in rows for i in (a, b)]).reshape(-1, 2, 3)
    l_px = np.linalg.norm(px[:, 0] - px[:, 1], axis=1)
    l_m = np.linalg.norm(w[:, 0] - w[:, 1], axis=1)
    return focal * l_m / l_px * 100


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_csv()
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    focal = meta["focal_px_assumed"]
    groups = load(path)
    poses = [p for p in POSE_STYLE if any(k[0] == p for k in groups)]
    dists = sorted({k[1] for k in groups})

    print(f"파일: {path.name}  (화각 {meta['fov_assumed_deg']}° 가정, 손바닥 실측 {meta['palm_cm_measured']} cm)")
    rows_all = [r for v in groups.values() for r in v]
    infer = np.array([float(r["infer_ms"]) for r in rows_all])
    print(f"손 추적 시간: 중앙값 {np.median(infer):.1f} ms, 95% {np.percentile(infer, 95):.1f} ms (프레임 {len(infer)}개)")
    for pose in poses:
        pl = [np.mean([np.linalg.norm([float(r[f'w0_{c}']) - float(r[f'w9_{c}']) for c in 'xyz'])
                       for r in groups[(pose, d)]]) * 100 for d in dists if (pose, d) in groups]
        print(f"MediaPipe 월드 손바닥 길이 [{pose}]: {np.round(pl, 1)} cm")

    fig, axes = plt.subplots(1, len(ESTIMATORS), figsize=(11, 5), sharey=True)
    summary = {}
    for ax, (name, (a, b)) in zip(axes, ESTIMATORS.items()):
        print(f"\n[{name}]  거리(cm) | 추정 평균 | 오차 | 떨림(표준편차)")
        lim = max(dists) + 25
        ax.plot([0, lim], [0, lim], color="#9a9893", lw=1, ls="--", zorder=1)
        ax.text(lim - 2, lim - 5, "추정 = 실제", color="#52514e", ha="right", fontsize=9)
        for pose in poses:
            color, marker, label = POSE_STYLE[pose]
            ds = [d for d in dists if (pose, d) in groups]
            est = [depth_cm(groups[(pose, d)], a, b, focal) for d in ds]
            mean = np.array([e.mean() for e in est])
            std = np.array([e.std() for e in est])
            ax.errorbar(ds, mean, yerr=std, color=color, marker=marker, ms=8, lw=2, capsize=4,
                        mec="#fcfcfb", mew=2, label=label, zorder=3)
            ax.annotate(label, (ds[-1], mean[-1]), xytext=(8, LABEL_NUDGE[pose]), textcoords="offset points",
                        color="#0b0b0b", va="center", fontsize=10)
            err = mean - np.array(ds)
            summary[(name, pose)] = {"err_mean": float(err.mean()), "jitter_mean": float(std.mean())}
            for d, m, e, s in zip(ds, mean, err, std):
                print(f"  {pose:4s} {d:3d} | {m:6.1f} | {e:+6.1f} | {s:5.2f}")
            print(f"  {pose:4s} 평균 오차 {err.mean():+.1f} cm, 평균 떨림 {std.mean():.2f} cm")
        ax.set_title(TITLES[name], fontsize=12, color="#0b0b0b")
        ax.set_xlabel("실제 거리 (cm, 줄자)", color="#52514e")
        ax.set_xlim(20, lim)
        ax.set_ylim(20, lim)
        ax.grid(color="#e9e8e4", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("웹캠 추정 거리 (cm)", color="#52514e")
    fig.suptitle(f"웹캠 깊이 추정: 실제 vs 추정 (막대 = 2초간 떨림 ±1σ, 화각 {meta['fov_assumed_deg']:.0f}° 가정)",
                 fontsize=13)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"depth_{path.stem}.png"
    fig.savefig(out, dpi=130, facecolor="#fcfcfb")
    print(f"\n그림 저장: {out}")


if __name__ == "__main__":
    main()
