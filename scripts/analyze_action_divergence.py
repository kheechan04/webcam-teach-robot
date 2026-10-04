"""시범의 "행동 들쭉날쭉함" (Belkhale et al. NeurIPS 2023의 action divergence를 단순화) — M9 해석용.

    uv run python scripts/analyze_action_divergence.py
집기 직전 구간(집게 열림, 큐브가 바닥, 집게 끝이 큐브에서 6 cm 안)만 본다 — 학습한 로봇이 실패하는 곳.
상태 = 집게 끝 − 큐브 (로봇 좌표, 3차원). 행동 = 다음 0.3초(9프레임) 동안 집게 끝 목표가 움직인 방향(단위벡터)과 거리.
시범의 한 프레임마다 **다른 시범**의 프레임 중 상태가 가장 가까운 5개를 찾아, 행동 방향이 얼마나 다른지(1 − 코사인)를 잰다.
같은 상황에서 시범마다 다른 쪽으로 움직였다면 값이 크다.
"""

import csv
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_demos import demo_files  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H = 9


def frames(path: Path):
    R = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["engaged"] == "1"]
    g = lambda k: np.array([float(r[k]) for r in R])  # noqa: E731
    T = np.stack([g("target_x"), g("target_y"), g("target_z")], 1)
    tip = np.stack([g("tip_x"), g("tip_y"), g("tip_z")], 1)
    C = np.stack([g("cube_x"), g("cube_y"), g("cube_z")], 1)
    cl = g("gripper_closed")
    S, A = [], []
    for k in range(len(R) - H):
        rel = tip[k] - C[k]
        if cl[k] == 0 and C[k, 2] < 0.02 and np.linalg.norm(rel) < 0.06:
            d = T[k + H] - T[k]
            n = np.linalg.norm(d)
            if n > 0.002:  # 움직이는 프레임만 (멈춤·잠금 제외)
                S.append(rel)
                A.append(d / n)
    return np.array(S).reshape(-1, 3), np.array(A).reshape(-1, 3)


def divergence(demos, rng, k=5, max_frames=3000):
    S = [frames(p) for p in demos]
    S = [(s, a) for s, a in S if len(s)]
    allS = np.concatenate([s for s, _ in S]); allA = np.concatenate([a for _, a in S])
    owner = np.concatenate([np.full(len(s), i) for i, (s, _) in enumerate(S)])
    pick = rng.choice(len(allS), min(max_frames, len(allS)), replace=False)
    vals, xs = [], []
    for i in pick:
        m = owner != owner[i]
        dist = np.linalg.norm(allS[m] - allS[i], axis=1)
        nn = np.argsort(dist)[:k]
        if dist[nn].max() > 0.01:  # 1 cm 안에 이웃이 없으면 비교하지 않음
            continue
        a = allA[m][nn]
        vals.append(np.mean(1 - a @ allA[i]))
        xs.append(np.mean(np.abs(a[:, 0] - allA[i][0])))  # 앞뒤(깊이 방향) 성분 차이
    return {"frames": int(len(allS)), "compared": len(vals), "divergence_mean": float(np.mean(vals)),
            "divergence_median": float(np.median(vals)), "x_component_diff": float(np.mean(xs))}


def main() -> None:
    rng = np.random.default_rng(0)
    F = demo_files()
    names = {2: "② 웹캠", 3: "③ 웹캠 + 보정", 4: "④ 가상 + 오차 주입", 40: "④0 가상, 오차 없음"}
    out = {}
    for c in (2, 3, 4, 40):
        out[c] = divergence(F[c], rng)
        r = out[c]
        print(f"{names[c]:14s} 집기 직전 프레임 {r['frames']:5d}, 비교 {r['compared']:4d} | 행동 방향 차이 평균 {r['divergence_mean']:.3f} "
              f"중앙 {r['divergence_median']:.3f} | 앞뒤 성분 차이 {r['x_component_diff']:.3f}")
    (ROOT / "experiments" / "action_divergence.json").write_text(json.dumps({str(k): v for k, v in out.items()}, indent=1),
                                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
