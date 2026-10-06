"""쌓기 과제 결과 분석 — docs/13-stacking-plan.md의 "분석"에 미리 적은 그대로.

    uv run python scripts/analyze_stack.py
입력: experiments/eval/stack/act-stack*.json (RunPod 평가, Hub m9-results에서 받음)
출력: experiments/stack_summary.json

1. 주 가설 H1(③ > ②, 한쪽): 같은 시드끼리 차이 5개, 시드 부호 뒤집기(한쪽·양쪽), 평가 배치와 시드를 함께 다시 뽑는
   부트스트랩 95% 구간(5,000번, 시드 0 — analyze_m9.py와 같은 방법). 판정: 5/5 재현됨, 4/5 대체로 재현, 3 이하 재현 안 됨.
2. 두 과제 합친 한쪽 부호 뒤집기: 옮기기 10만 스텝 ③ − ② 시드 5개 + 쌓기 시드 5개.
3. 보조: 같은 배치에서 한쪽만 성공한 횟수, ①과의 차이.
"""

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "experiments" / "eval" / "stack"
SEEDS = [1000, 2000, 3000, 4000, 5000]
CUBE_DIFFS = [1, 2, 24, 8, 4]  # 옮기기 10만 스텝 ③ − ② (docs/11-m9-results.md)


def succ(name: str) -> np.ndarray:
    d = json.loads((EVAL / f"act-{name}.json").read_text(encoding="utf-8"))
    return np.array([e["success"] for e in d["episodes"]], float)


def sign_flip(d, one_sided: bool) -> float:
    d = np.asarray(d, float)
    obs = d.mean()
    vals = [(d * np.array(s)).mean() for s in itertools.product([1, -1], repeat=len(d))]
    if one_sided:
        return float(np.mean([v >= obs - 1e-9 for v in vals]))
    return float(np.mean([abs(v) >= abs(obs) - 1e-9 for v in vals]))


def main() -> None:
    c2 = np.array([succ(f"stack2-webcam-s{s}-60k") for s in SEEDS])
    c3 = np.array([succ(f"stack3-webcam-corrected-s{s}-60k") for s in SEEDS])
    c1 = np.array([succ(f"stack1-scripted-s{s}-20k") for s in SEEDS[:3]])
    diffs = (c3.sum(1) - c2.sum(1)).astype(int).tolist()
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(5000):
        sd = rng.choice(len(SEEDS), len(SEEDS))
        ly = rng.choice(c2.shape[1], c2.shape[1])
        boot.append((c3[sd][:, ly].sum(1) - c2[sd][:, ly].sum(1)).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    n_pos = sum(d > 0 for d in diffs)
    verdict = "재현됨" if n_pos == 5 else "대체로 재현" if n_pos == 4 else "재현 안 됨"
    both = CUBE_DIFFS + diffs
    out = {
        "per_seed": {"cond1": c1.sum(1).astype(int).tolist(), "cond2": c2.sum(1).astype(int).tolist(),
                     "cond3": c3.sum(1).astype(int).tolist()},
        "mean": {"cond1": float(c1.sum(1).mean()), "cond2": float(c2.sum(1).mean()), "cond3": float(c3.sum(1).mean())},
        "h1": {"diffs_3_minus_2": diffs, "mean": float(np.mean(diffs)), "seeds_positive": n_pos,
               "p_one_sided": sign_flip(diffs, True), "p_two_sided": sign_flip(diffs, False),
               "ci95": [float(lo), float(hi)], "verdict": verdict},
        "combined_two_tasks": {"diffs": both, "mean": float(np.mean(both)), "seeds_positive": int(sum(d > 0 for d in both)),
                               "p_one_sided": sign_flip(both, True), "p_two_sided": sign_flip(both, False)},
        "discordant": {"cond3_only": int(((c3 == 1) & (c2 == 0)).sum()), "cond2_only": int(((c2 == 1) & (c3 == 0)).sum())},
        "vs_cond1": {"cond2_minus_1_mean_seeds_1_3": float((c2[:3].sum(1) - c1.sum(1)).mean()),
                     "cond3_minus_1_mean_seeds_1_3": float((c3[:3].sum(1) - c1.sum(1)).mean())},
    }
    (ROOT / "experiments" / "stack_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
