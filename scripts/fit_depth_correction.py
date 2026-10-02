"""조건 ③ 깊이 보정 맞추기·평가 (M7).

RootNet(ICCV 2019)식 구조: 핀홀로 구한 거리 × 손 자세로 예측한 보정 계수.
    d_보정 = d_핀홀 × exp(w · 특징)
d_핀홀은 지금 조종 방식(손바닥 길이·너비 중 작은 값, 가정 초점거리). 특징은 조종 중에 매 프레임 얻을 수 있는 값만 쓴다
(src/webcam_teach_robot/depth_correction.py의 posture_features).

M5 정식 측정(measurements/depth_v2/)의 1회차로만 맞추고 2회차로 평가한다. 정규화 세기는 1회차 안에서
"자세×높이 조건 하나씩 빼고 맞추기"로 고른다(2회차는 고르는 데도 쓰지 않음).

실행:
    uv run python scripts/fit_depth_correction.py                  # 가장 최근 측정
결과: src/webcam_teach_robot/depth_correction.json (조종에서 읽는 계수), experiments/depth_correction_eval.json, 그림
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_depth_v2 import F_ASSUMED, POSE_KO, HEIGHT_KO, arrays, load  # noqa: E402

from webcam_teach_robot.depth_correction import COEF_FILE, FEATURE_NAMES, posture_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES_W, RES_H = 640, 480
ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]


def pinhole_min(P, W, f=F_ASSUMED):
    lp = np.linalg.norm(P[:, 0] - P[:, 9], axis=1)
    lw = np.linalg.norm(P[:, 5] - P[:, 17], axis=1)
    mp_ = np.linalg.norm(W[:, 0] - W[:, 9], axis=1)
    mw = np.linalg.norm(W[:, 5] - W[:, 17], axis=1)
    return np.minimum(f * mp_ / lp, f * mw / lw)


def build(data):
    """프레임 단위 표. 단계 정보, 핀홀 거리(m), 특징, 참값(m)."""
    rows = []
    for s, rs in data:
        P, W = arrays(rs)
        d = pinhole_min(P, W)
        X = np.array([posture_features(P[k], W[k], d[k], RES_W, RES_H) for k in range(len(rs))])
        rows.append({"step": s, "truth": float(rs[0]["true_depth_cm"]) / 100, "d": d, "X": X})
    return rows


def fit_ridge(rows, alpha):
    X = np.concatenate([r["X"] for r in rows])
    y = np.concatenate([np.log(r["truth"] / r["d"]) for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z1 = np.hstack([Z, np.ones((len(Z), 1))])
    reg = alpha * np.eye(Z1.shape[1])
    reg[-1, -1] = 0  # 절편은 벌주지 않는다
    w = np.linalg.solve(Z1.T @ Z1 + reg, Z1.T @ y)
    return {"mean": mu.tolist(), "std": sd.tolist(), "w": w[:-1].tolist(), "b": float(w[-1])}


def predict(coef, X, d):
    """조종에서 쓰는 것과 같은 계산(범위 자르기 포함)."""
    Z = (X - np.array(coef["mean"])) / np.array(coef["std"])
    lo, hi = coef.get("log_factor_range", (-np.inf, np.inf))
    return d * np.exp(np.clip(Z @ np.array(coef["w"]) + coef["b"], lo, hi))


def cond_key(s):
    return (s["pose"], s["height"]) if s["block"] == "main" else ("lateral", s["lateral"])


def evaluate(rows, est):
    """est(row) → 프레임별 추정 거리(m). 단계 중앙값 오차(cm)와 단계 안 흔들림."""
    out = []
    for r in rows:
        e = est(r)
        out.append({**r["step"], "err": float(np.median(e) - r["truth"]) * 100, "sd": float(np.std(e)) * 100,
                    "truth_cm": r["truth"] * 100})
    return out


def summarize(res):
    main = [r for r in res if r["block"] == "main"]
    by = {}
    for pose in POSE_KO:
        for h in HEIGHT_KO:
            v = [r["err"] for r in main if r["pose"] == pose and r["height"] == h]
            by[f"{pose}|{h}"] = float(np.mean(v))
    # 같은 거리·높이에서 손 모양만 바꿨을 때 추정 거리가 얼마나 달라지나 (조종 중 손 모양을 바꾸면 로봇이 이만큼 앞뒤로 튄다)
    jumps = []
    for h in HEIGHT_KO:
        for dist in sorted({r["distance_cm"] for r in main}):
            e = {r["pose"]: r["err"] for r in main if r["height"] == h and r["distance_cm"] == dist}
            if len(e) == 4:
                jumps.append(max(e.values()) - min(e.values()))
    # 같은 손 모양에서 거리를 바꿨을 때 추정 이동량 / 실제 이동량 (조종의 앞뒤 배율이 틀어지는 정도)
    gains = []
    for pose in POSE_KO:
        for h in HEIGHT_KO:
            v = [r for r in main if r["pose"] == pose and r["height"] == h]
            t = np.array([r["truth_cm"] for r in v])
            est = t + np.array([r["err"] for r in v])
            gains.append(float(np.polyfit(t, est, 1)[0]))
    lat = {r["lateral"]: r["err"] for r in res if r["block"] == "lateral"}
    return {"mae": float(np.mean([abs(r["err"]) for r in main])), "by_condition": by,
            "cond_range": max(by.values()) - min(by.values()),
            "pose_jump_mean": float(np.mean(jumps)), "gain_mean": float(np.mean(gains)),
            "gain_range": [min(gains), max(gains)], "jitter": float(np.mean([r["sd"] for r in main])),
            "lateral": lat}


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "measurements" / "depth_v2").glob("*.csv"))[-1]
    meta, data = load(path)
    rows = build(data)
    train = [r for r in rows if r["step"]["rep"] == 0]
    test = [r for r in rows if r["step"]["rep"] == 1]
    print(f"{path.name}: 1회차(맞춤) {len(train)}단계, 2회차(평가) {len(test)}단계, 특징 {FEATURE_NAMES}")

    # 정규화 세기: 1회차 안에서 조건 하나씩 빼고 맞춘 뒤 뺀 조건의 평균 절대 오차
    conds = sorted({cond_key(r["step"]) for r in train})
    cv = {}
    for a in ALPHAS:
        errs = []
        for c in conds:
            fit = fit_ridge([r for r in train if cond_key(r["step"]) != c], a)
            for r in train:
                if cond_key(r["step"]) == c:
                    errs.append(abs(np.median(predict(fit, r["X"], r["d"])) - r["truth"]) * 100)
        cv[a] = float(np.mean(errs))
    alpha = min(cv, key=cv.get)
    print("조건 빼고 맞추기 평균 절대 오차(cm):", {a: round(v, 2) for a, v in cv.items()}, "→ 고른 값", alpha)

    coef = fit_ridge(train, alpha)
    step_targets = [float(np.median(np.log(r["truth"] / r["d"]))) for r in train]
    coef.update({"log_factor_range": [min(step_targets), max(step_targets)], "features": FEATURE_NAMES, "alpha": alpha, "fit_on": f"{path.name} rep 0",
                 "f_assumed_px": F_ASSUMED, "resolution": [RES_W, RES_H]})
    # 비교용: 상수 배율만(= 초점거리 다시 맞추기와 같음)
    scale = float(np.exp(np.mean(np.concatenate([np.log(r["truth"] / r["d"]) for r in train]))))

    methods = {
        "baseline": lambda r: r["d"],
        "scale_only": lambda r: r["d"] * scale,
        "correction": lambda r: predict(coef, r["X"], r["d"]),
    }
    report = {}
    for name, est in methods.items():
        report[name] = {"train": summarize(evaluate(train, est)), "test": summarize(evaluate(test, est))}
    report["test_steps"] = {name: evaluate(test, est) for name, est in methods.items()}

    print("\n2회차(평가) 결과, cm")
    print(f"{'':12s} {'절대오차':>8s} {'조건간범위':>10s} {'손모양 튐':>9s} {'앞뒤 배율':>9s} {'흔들림':>7s}")
    for name in methods:
        t = report[name]["test"]
        print(f"{name:12s} {t['mae']:8.1f} {t['cond_range']:10.1f} {t['pose_jump_mean']:9.1f} "
              f"{t['gain_mean']:9.2f} {t['jitter']:7.2f}   가로 {({k: round(v, 1) for k, v in t['lateral'].items()})}")
    print("\n자세×높이별 평균 오차 (2회차): 지금 → 보정")
    for k in report["baseline"]["test"]["by_condition"]:
        p, h = k.split("|")
        print(f"  {POSE_KO[p]:10s} {HEIGHT_KO[h]:6s} {report['baseline']['test']['by_condition'][k]:+6.1f} → "
              f"{report['correction']['test']['by_condition'][k]:+6.1f}")
    print("\n계수(표준화 특징):", {n: round(w, 3) for n, w in zip(FEATURE_NAMES, coef["w"])}, "절편", round(coef["b"], 3))

    COEF_FILE.write_text(json.dumps(coef, indent=1), encoding="utf-8")
    out = ROOT / "experiments" / "depth_correction_eval.json"
    out.write_text(json.dumps({"alpha_cv": cv, "scale_only": scale, "coef": coef, **report}, ensure_ascii=False,
                              indent=1), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    keys = list(report["baseline"]["test"]["by_condition"])
    x = np.arange(len(keys))
    for k, (name, label, color) in enumerate([("baseline", "지금 방식 (조건 ②)", "#eb6834"),
                                              ("scale_only", "배율만 다시 맞춤", "#9a9893"),
                                              ("correction", "자세 보정 (조건 ③)", "#2a78d6")]):
        ax.bar(x + (k - 1) * 0.26, [report[name]["test"]["by_condition"][c] for c in keys], 0.24, color=color, label=label)
    ax.axhline(0, color="#52514e", lw=1)
    ax.set_xticks(x, [f"{POSE_KO[c.split('|')[0]]}\n{HEIGHT_KO[c.split('|')[1]]}" for c in keys], fontsize=9)
    ax.set_ylabel("평균 깊이 오차 (cm, 추정 - 참값)")
    ax.set_title("깊이 보정 평가 — 1회차로 맞추고 2회차로 평가", fontsize=12)
    ax.grid(axis="y", color="#e9e8e4")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig_path = ROOT / "docs" / "figures" / "depth_correction_eval.png"
    fig.savefig(fig_path, dpi=130, facecolor="#fcfcfb")
    print(f"\n계수 {COEF_FILE}\n평가 {out}\n그림 {fig_path}")


if __name__ == "__main__":
    main()
