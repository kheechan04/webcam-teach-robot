"""M9 결과 정리: 조건별 성공률(시드 평균·범위), 같은 시드끼리 짝 비교, 그림.

    uv run python scripts/analyze_m9.py
입력: experiments/eval/m9/act-<조건>-s<시드>.json (eval_policy.py 결과, Hub m9-results에서 받음)
출력: experiments/m9_summary.json, docs/figures/m9_success.png

통계: 평가 배치 100개는 모든 조건·시드에 같다. 조건 차이의 불확실성은 (1) 시드 간 흔들림과 (2) 평가 배치 표본
두 가지라서, 같은 시드끼리 차이를 내고 + 평가 배치를 다시 뽑는 부트스트랩(시드별로 같은 배치 묶음)으로 95% 구간을 낸다.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "experiments" / "eval" / "m9"
CONDS = [("cond1-scripted", "① 스크립트"), ("cond4z-no-error", "④0 가상 조작자"), ("cond4-depth-error", "④ 가상 + 오차 주입"),
         ("cond3-webcam-corrected", "③ 웹캠 + 보정"), ("cond2-webcam", "② 웹캠")]
SEEDS = ["1000", "2000", "3000", "4000", "5000"]
COMPARISONS = [  # (뒤, 앞, 이름): 앞 − 뒤
    ("cond1-scripted", "cond2-webcam", "웹캠 시범의 손해 (② − ①)"),
    ("cond4z-no-error", "cond4-depth-error", "깊이 오차만의 몫 (④ − ④0)"),
    ("cond2-webcam", "cond3-webcam-corrected", "보정으로 회복 (③ − ②)"),
    ("cond1-scripted", "cond4z-no-error", "조종 코드를 거친 효과 (④0 − ①)"),
    ("cond4-depth-error", "cond2-webcam", "사람 시범의 추가 손해 (② − ④)"),
    ("cond1-scripted", "cond3-webcam-corrected", "보정해도 남는 손해 (③ − ①)"),
]
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load():
    out = {}
    for c, _ in CONDS:
        for s in SEEDS:
            p = EVAL / f"act-{c}-s{s}.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                out[(c, s)] = {"succ": np.array([e["success"] for e in d["episodes"]], float),
                               "lift": np.array([e["lifted"] for e in d["episodes"]], float),
                               "secs": np.array([e["seconds"] for e in d["episodes"] if e["success"]]),
                               "device": d.get("device")}
    return out


def main() -> None:
    R = load()
    rng = np.random.default_rng(0)
    summary = {"conditions": {}, "comparisons": {}}
    print("조건별 성공 (100개 중), 시드 1000/2000/3000 → 평균")
    for c, name in CONDS:
        v = [int(R[(c, s)]["succ"].sum()) if (c, s) in R else None for s in SEEDS]
        got = [x for x in v if x is not None]
        lift = [R[(c, s)]["lift"].mean() * 100 for s in SEEDS if (c, s) in R]
        secs = np.concatenate([R[(c, s)]["secs"] for s in SEEDS if (c, s) in R]) if got else np.array([])
        summary["conditions"][c] = {"name": name, "per_seed": v, "mean": float(np.mean(got)) if got else None,
                                    "lift_mean": float(np.mean(lift)) if lift else None,
                                    "success_seconds_median": float(np.median(secs)) if len(secs) else None}
        print(f"  {name:14s} {v} → {np.mean(got):5.1f}  (들어 올리기 {np.mean(lift):4.1f}%, 성공까지 중앙 {np.median(secs):4.1f}초)")

    print("\n비교 (같은 시드끼리 차이, 평균 [95% 구간: 평가 배치 부트스트랩 + 시드 평균])")
    for a, b, name in COMPARISONS:
        seeds = [s for s in SEEDS if (a, s) in R and (b, s) in R]
        if not seeds:
            continue
        diffs = [int(R[(b, s)]["succ"].sum() - R[(a, s)]["succ"].sum()) for s in seeds]
        boots = []
        for _ in range(5000):
            idx = rng.integers(0, 100, 100)
            sd = rng.choice(seeds, len(seeds))  # 시드도 다시 뽑는다
            boots.append(np.mean([R[(b, s)]["succ"][idx].sum() - R[(a, s)]["succ"][idx].sum() for s in sd]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        summary["comparisons"][name] = {"per_seed": diffs, "mean": float(np.mean(diffs)), "ci95": [float(lo), float(hi)],
                                        "n_seeds": len(seeds)}
        print(f"  {name:28s} 시드별 {diffs} → {np.mean(diffs):+6.1f}  [{lo:+.1f}, {hi:+.1f}]  (시드 {len(seeds)}개)")

    (ROOT / "experiments" / "m9_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(CONDS))
    colors = ["#9a9893", "#1baf7a", "#2a78d6", "#7b5fd6", "#eb6834"]
    for i, (c, name) in enumerate(CONDS):
        v = [R[(c, s)]["succ"].sum() for s in SEEDS if (c, s) in R]
        ax.bar(i, np.mean(v), 0.6, color=colors[i], alpha=0.85)
        ax.scatter([i] * len(v), v, color="#17171d", s=18, zorder=3)
        ax.text(i, 4, f"{np.mean(v):.1f}", ha="center", fontsize=11, color="white", fontweight="bold")
    ax.set_xticks(x, [n for _, n in CONDS], fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("처음 보는 배치 100개 중 성공")
    ax.set_title("조건별 학습한 로봇의 성공 (막대: 시드 평균, 점: 시드별)", fontsize=11.5)
    ax.grid(axis="y", color="#e9e8e4")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "figures" / "m9_success.png", dpi=130, facecolor="#fcfcfb")
    print("\n→ experiments/m9_summary.json, docs/figures/m9_success.png")


if __name__ == "__main__":
    main()
