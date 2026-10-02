"""M5 정식 깊이 측정 분석 (measure_depth_v2.py 결과).

실행:
    uv run python scripts/analyze_depth_v2.py                  # 가장 최근 측정
    uv run python scripts/analyze_depth_v2.py measurements/depth_v2/XXX.csv

참값: 측정 때 계산해 둔 카메라 광축 방향 깊이(true_depth_cm).
추정 방식(모두 깊이 = 초점거리 × 실제 길이 / 사진 속 길이):
    len3d      손목(0)–가운뎃손가락 뿌리(9), MediaPipe 3D 길이 전체        — 예비 측정의 "길이"
    width3d    검지 뿌리(5)–새끼 뿌리(17), 3D 전체                         — 예비 측정의 "너비"
    min3d      위 둘 중 작은 값                                              — 지금 조종(조건 ②)이 쓰는 방식
    len_xy     0–9, 3D 구간의 화면과 나란한 성분(x, y)만                    — 보정 후보 1
    width_xy   5–17, 화면과 나란한 성분
    pinky_xy   손목(0)–새끼 뿌리(17), 화면과 나란한 성분                    — 보정 후보 1+2
    ring_xy    손목(0)–약지 뿌리(13), 화면과 나란한 성분
초점거리: 가정값(화각 60° → 554 px)과, 1회차 "편 손·정면·웹캠 높이·가운데" 단계로 맞춘 값 두 가지로 본다.
맞춘 값은 그 단계들로만 정하고 나머지(2회차 전부 포함)에 적용한다.
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
F_ASSUMED = 554.256
ESTIMATORS = {
    "len3d": ((0, 9), False), "width3d": ((5, 17), False), "len_xy": ((0, 9), True),
    "width_xy": ((5, 17), True), "pinky_xy": ((0, 17), True), "ring_xy": ((0, 13), True),
}
POSE_KO = {"open_flat": "편 손·정면", "open_tilt": "편 손·기울임", "pinch_flat": "집은 손·정면", "pinch_tilt": "집은 손·기울임"}
HEIGHT_KO = {"cam": "웹캠 높이", "low": "낮게"}

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load(path: Path):
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        by[(int(r["step"]), int(r["take"]))].append(r)
    steps = {s["step"]: s for s in meta["steps"]}
    out = []
    for d in meta["done"]:
        rs = by[(d["step"], d["take"])][-60:]  # 같은 번호로 두 번 저장된 단계는 마지막 60프레임
        out.append((steps[d["step"]], rs))
    return meta, out


def arrays(rs):
    P = np.array([[[float(r[f"px{i}_{a}"]) for a in "uv"] for i in range(21)] for r in rs])
    W = np.array([[[float(r[f"w{i}_{a}"]) for a in "xyz"] for i in range(21)] for r in rs])
    return P, W


def depth_cm(P, W, est, f):
    (a, b), xy = ESTIMATORS[est]
    lp = np.linalg.norm(P[:, a] - P[:, b], axis=1)
    wv = W[:, a] - W[:, b]
    lw = np.linalg.norm(wv[:, :2] if xy else wv, axis=1)
    return f * lw / lp * 100


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "measurements" / "depth_v2").glob("*.csv"))[-1]
    meta, data = load(path)
    print(f"파일 {path.name}: 단계 {len(data)}, 기하 {meta['geometry']}, 입력 {meta['inputs']}")

    # 초점거리 맞추기: 1회차 편 손·정면·웹캠 높이·가운데 단계만
    fit = defaultdict(list)
    for s, rs in data:
        if s["block"] == "main" and s["pose"] == "open_flat" and s["height"] == "cam" and s["rep"] == 0:
            P, W = arrays(rs)
            for est in ESTIMATORS:
                fit[est].append(np.median(float(rs[0]["true_depth_cm"]) / (depth_cm(P, W, est, 1.0))))
    f_fit = {est: float(np.median(v)) for est, v in fit.items()}
    print("맞춘 초점거리(px):", {k: round(v) for k, v in f_fit.items()}, f"/ 가정값 {F_ASSUMED:.0f}")

    # 단계별 결과
    recs = []
    for s, rs in data:
        P, W = arrays(rs)
        truth = float(rs[0]["true_depth_cm"])
        pinch = np.array([float(np.linalg.norm(W[k, 4] - W[k, 8]) / np.linalg.norm(W[k, 5] - W[k, 17])) for k in range(len(rs))])
        r = {**s, "truth": truth, "pinch_med": float(np.median(pinch)), "pinch_gt08": float(np.mean(pinch > 0.8)),
             "pinch_gt10": float(np.mean(pinch > 1.0))}
        for est in ESTIMATORS:
            for tag, f in (("assumed", F_ASSUMED), ("fit", f_fit[est])):
                d = depth_cm(P, W, est, f)
                r[f"{est}_{tag}_err"] = float(np.median(d) - truth)
                r[f"{est}_{tag}_sd"] = float(np.std(d))
        d = np.minimum(depth_cm(P, W, "len3d", F_ASSUMED), depth_cm(P, W, "width3d", F_ASSUMED))
        r["min3d_assumed_err"], r["min3d_assumed_sd"] = float(np.median(d) - truth), float(np.std(d))
        recs.append(r)

    main_recs = [r for r in recs if r["block"] == "main"]
    eval_recs = [r for r in main_recs if not (r["pose"] == "open_flat" and r["height"] == "cam" and r["rep"] == 0)]

    def table(rs, key):
        out = {}
        for pose in POSE_KO:
            for h in HEIGHT_KO:
                v = [r[key] for r in rs if r["pose"] == pose and r["height"] == h]
                out[(pose, h)] = (float(np.mean(v)), float(np.mean(np.abs(v))), len(v)) if v else None
        return out

    print("\n[1] 지금 조종 방식(min3d, 가정 초점거리): 자세 × 높이별 평균 오차 / 평균 절대 오차 (cm, 거리 4개 × 2회 평균)")
    t = table(main_recs, "min3d_assumed_err")
    for (pose, h), v in t.items():
        print(f"  {POSE_KO[pose]:10s} {HEIGHT_KO[h]:6s} 평균 {v[0]:+6.1f}  절대 {v[1]:5.1f}  (n={v[2]})")

    print("\n[2] 추정 방식 비교 (맞춘 초점거리, 초점 맞춤에 쓴 단계 제외): 평균 절대 오차 / 자세·높이 조건 간 평균오차 범위 / 단계 안 흔들림")
    summary = {}
    for est in ESTIMATORS:
        errs = [r[f"{est}_fit_err"] for r in eval_recs]
        cond = table(eval_recs, f"{est}_fit_err")
        means = [v[0] for v in cond.values() if v]
        sds = [r[f"{est}_fit_sd"] for r in eval_recs]
        summary[est] = {"mae": float(np.mean(np.abs(errs))), "cond_range": float(max(means) - min(means)),
                        "jitter": float(np.mean(sds)), "by_condition": {f"{p}|{h}": v[0] for (p, h), v in cond.items() if v}}
        print(f"  {est:9s} 절대오차 {summary[est]['mae']:5.1f}  조건간 범위 {summary[est]['cond_range']:5.1f}  흔들림 {summary[est]['jitter']:4.2f}")

    print("\n[3] 거리별 (min3d, 가정 초점거리, 평균 오차 cm)")
    for dist in sorted({r["distance_cm"] for r in main_recs}):
        v = [r["min3d_assumed_err"] for r in main_recs if r["distance_cm"] == dist]
        print(f"  줄자 {dist} cm: {np.mean(v):+6.1f}")

    print("\n[4] 가로 위치 (편 손·정면·웹캠 높이, 50 cm): min3d 가정 / len_xy 맞춤 평균 오차")
    for lat in ("left", "center", "right"):
        v = [r for r in recs if r["block"] == "lateral" and r["lateral"] == lat]
        print(f"  {lat:6s} {np.mean([r['min3d_assumed_err'] for r in v]):+6.1f} / {np.mean([r['len_xy_fit_err'] for r in v]):+6.1f}")

    print("\n[5] 집은 손을 '붙인 상태'로 유지했나: 엄지-검지 비율 중앙값, 비율 >0.8 프레임(조종에선 '덜 붙음'), >1.0 프레임(조종에선 집게가 열림)")
    for pose in ("pinch_flat", "pinch_tilt"):
        for h in HEIGHT_KO:
            for dist in sorted({r["distance_cm"] for r in main_recs}):
                v = [r for r in main_recs if r["pose"] == pose and r["height"] == h and r["distance_cm"] == dist]
                if v:
                    print(f"  {POSE_KO[pose]:10s} {HEIGHT_KO[h]:6s} {dist} cm: 중앙 {np.mean([x['pinch_med'] for x in v]):.2f}  "
                          f">0.8 {np.mean([x['pinch_gt08'] for x in v]) * 100:4.0f}%  >1.0 {np.mean([x['pinch_gt10'] for x in v]) * 100:4.0f}%")
    for pose in ("open_flat", "open_tilt"):
        v = [r for r in main_recs if r["pose"] == pose]
        print(f"  (비교) {POSE_KO[pose]:8s}: 중앙 {np.mean([x['pinch_med'] for x in v]):.2f}  <0.8 {np.mean([1 - x['pinch_gt08'] for x in v]) * 100:4.0f}% (편 손인데 '집음'으로 읽힐 수 있는 프레임)")

    # 그림: 자세×높이별 평균 오차, 지금 방식 vs 보정 후보
    fig, ax = plt.subplots(figsize=(10, 4.6))
    conds = [(p, h) for p in POSE_KO for h in HEIGHT_KO]
    x = np.arange(len(conds))
    show = [("min3d_assumed_err", "지금 방식 (길이·너비 중 작은 값, 가정 초점)", "#eb6834"),
            ("len_xy_fit_err", "후보: 손바닥 길이, 화면과 나란한 성분 (맞춘 초점)", "#2a78d6"),
            ("pinky_xy_fit_err", "후보: 손목–새끼 뿌리, 화면과 나란한 성분 (맞춘 초점)", "#1baf7a")]
    w = 0.26
    for k, (key, label, color) in enumerate(show):
        tt = table(main_recs, key)
        ax.bar(x + (k - 1) * w, [tt[c][0] for c in conds], w - 0.02, color=color, label=label)
    ax.axhline(0, color="#52514e", lw=1)
    ax.set_xticks(x, [f"{POSE_KO[p]}\n{HEIGHT_KO[h]}" for p, h in conds], fontsize=9)
    ax.set_ylabel("평균 깊이 오차 (cm, 추정 - 참값)")
    ax.set_title("자세·손 높이별 깊이 오차 (거리 35~65 cm, 2회 평균)", fontsize=12)
    ax.grid(axis="y", color="#e9e8e4")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / f"depth_v2_{path.stem}.png"
    fig.savefig(out, dpi=130, facecolor="#fcfcfb")
    res = ROOT / "experiments" / f"depth_v2_{path.stem}_summary.json"
    res.write_text(json.dumps({"f_fit": f_fit, "estimators": summary, "steps": recs}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n그림 {out}\n요약 {res}")


if __name__ == "__main__":
    main()
