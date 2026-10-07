"""논문용 숫자·표·그림을 결과 파일에서 바로 만든다 (손으로 옮기지 않는다).

    uv run python paper/make_assets.py
→ paper/numbers.tex (본문에서 쓰는 숫자 매크로), paper/tables/*.tex, paper/figures/*.pdf

부트스트랩 95% 구간은 analyze_m9.py와 같은 방법(같은 시드끼리 짝, 시드와 평가 배치 100개를 함께 다시 뽑기 5,000번)이고,
비교마다 난수 시드 0으로 새로 시작해 다시 실행해도 같은 값이 나온다.
"""

import itertools
import json
from pathlib import Path

import matplotlib
import matplotlib.patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "paper"
EV = ROOT / "experiments" / "eval"
SEEDS = [1000, 2000, 3000, 4000, 5000]
NUM: dict[str, str] = {}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.titlesize": 9})
C = {"1": "#7f7f7f", "40": "#2f8a5b", "4": "#8fbf9f", "5": "#b77a00", "3": "#2a6fc4", "2": "#c2502a"}


def succ(path: Path) -> np.ndarray:
    d = json.loads(path.read_text(encoding="utf-8"))
    return np.array([e["success"] for e in d["episodes"]], float)


def m9(cond: str, seed: int, suffix: str = "") -> np.ndarray | None:
    p = EV / "m9" / f"act-{cond}-s{seed}{suffix}.json"
    return succ(p) if p.exists() else None


def stack(cond: str, seed: int, suffix: str) -> np.ndarray:
    return succ(EV / "stack" / f"act-{cond}-s{seed}{suffix}.json")


def paired(a: dict, b: dict) -> dict:
    """a, b: 시드 → 100개 성공 배열. b − a."""
    seeds = sorted(set(a) & set(b))
    A = np.array([a[s] for s in seeds])
    B = np.array([b[s] for s in seeds])
    diffs = (B.sum(1) - A.sum(1)).astype(int)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(5000):
        sd = rng.integers(0, len(seeds), len(seeds))
        ly = rng.integers(0, A.shape[1], A.shape[1])
        boots.append((B[sd][:, ly].sum(1) - A[sd][:, ly].sum(1)).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = list(itertools.product([1, -1], repeat=len(diffs)))
    m = diffs.mean()
    p2 = np.mean([abs((diffs * np.array(s)).mean()) >= abs(m) - 1e-9 for s in signs])
    p1 = np.mean([(diffs * np.array(s)).mean() >= m - 1e-9 for s in signs])
    return {"diffs": diffs.tolist(), "mean": float(m), "lo": float(lo), "hi": float(hi), "p2": float(p2), "p1": float(p1),
            "n": len(seeds), "pos": int((diffs > 0).sum())}


DIGIT = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def num(name: str, value) -> None:
    # LaTeX 명령 이름에는 숫자를 못 쓴다
    NUM["".join(DIGIT.get(ch, ch) for ch in name)] = str(value)


def f1(x: float) -> str:
    return f"{x:.1f}".replace("-", "\\textminus")


def signed(x: float) -> str:
    body = f"{abs(x):.0f}" if float(x).is_integer() else f"{abs(x):.1f}"
    return ("+" if x > 0 else "\\textminus" if x < 0 else "") + body


def signed1(x: float) -> str:
    return ("+" if x > 0 else "\\textminus" if x < 0 else "") + f"{abs(x):.1f}"


def main() -> None:
    (P / "tables").mkdir(parents=True, exist_ok=True)
    (P / "figures").mkdir(parents=True, exist_ok=True)
    conds = [("40", "cond4z-no-error", r"\condVz{} virtual, no depth error"), ("1", "cond1-scripted", r"\condI{} scripted"),
             ("4", "cond4-depth-error", r"\condIV{} virtual + depth error"), ("5", "cond5-webcam-marker", r"\condV{} webcam + marker"),
             ("3", "cond3-webcam-corrected", r"\condIII{} webcam + correction"), ("2", "cond2-webcam", r"\condII{} webcam")]
    short = {k: {s: m9(c, s) for s in SEEDS if m9(c, s) is not None} for k, c, _ in conds}
    long_ = {k: {s: m9(c, s, "-100k") for s in SEEDS if m9(c, s, "-100k") is not None} for k, c, _ in conds}

    # ---- 표: 조건별 성공 (2만·10만) ----
    rows = []
    for k, c, name in conds:
        s20 = [int(short[k][s].sum()) for s in SEEDS]
        s100 = [int(long_[k][s].sum()) for s in sorted(long_[k])]
        rows.append(f"{name} & {', '.join(map(str, s20))} & {np.mean(s20):.1f} & {', '.join(map(str, s100))} & {np.mean(s100):.1f} \\\\")
        num(f"sr{k}short", f"{np.mean(s20):.1f}")
        num(f"sr{k}long", f"{np.mean(s100):.1f}")
    (P / "tables" / "main_results.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n & \\multicolumn{2}{c}{20k steps} & \\multicolumn{2}{c}{100k steps} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nCondition & per seed & mean & per seed & mean \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # ---- 표: 짝 비교 (10만 스텝) ----
    comps = [("1", "2", "Webcam loss", r"\condII{} $-$ \condI{}", "loss"),
             ("40", "4", "Depth error alone (no human)", r"\condIV{} $-$ \condVz{}", "depthvirt"),
             ("2", "5", "Depth share in human demos", r"\condV{} $-$ \condII{}", "depthhuman"),
             ("1", "5", "Residual human-demo loss", r"\condV{} $-$ \condI{}", "residual"),
             ("1", "40", "Effect of the teleop pipeline", r"\condVz{} $-$ \condI{}", "pipeline"),
             ("2", "3", "Correction (placing task)", r"\condIII{} $-$ \condII{}", "corr"),
             ("3", "5", "Marker vs.\\ correction", r"\condV{} $-$ \condIII{}", "markervscorr")]
    rows = []
    res = {}
    for a, b, label, expr, key in comps:
        r = paired(long_[a], long_[b])
        res[key] = r
        rows.append(f"{label} & {expr} & {', '.join(signed(d) for d in r['diffs'])} & {signed1(r['mean'])} & "
                    f"[{f1(r['lo'])}, {f1(r['hi'])}] \\\\")
        num(f"{key}Mean", signed1(r["mean"]))
        num(f"{key}Lo", f1(r["lo"]))
        num(f"{key}Hi", f1(r["hi"]))
        num(f"{key}N", r["n"])
    (P / "tables" / "comparisons.tex").write_text(
        "\\begin{tabular}{llllr}\n\\toprule\nComparison & & per seed & mean & 95\\% CI \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    num("corrPtwo", f"{res['corr']['p2']:.4f}".rstrip("0"))
    num("corrPone", f"{res['corr']['p1']:.3f}")
    r20 = paired(short["1"], short["2"])
    num("lossShortMean", signed1(r20["mean"]))
    num("lossShortLo", f1(r20["lo"]))
    num("lossShortHi", f1(r20["hi"]))

    # ---- 용량-반응 ----
    dose = [("0", "cond4z-no-error"), ("0.5", "cond4h-error-half"), ("1", "cond4-depth-error"), ("2", "cond4d-error-double")]
    dv = [np.mean([m9(c, s).sum() for s in (1000, 2000, 3000)]) for _, c in dose]
    for (lab, _), v in zip(dose, dv):
        num("dose" + lab.replace(".", "p"), f"{v:.1f}")

    # ---- 시범자 2 (5만 스텝) ----
    p2 = {c: {s: m9(f"{c}-p2", s, "-50k") for s in SEEDS} for c in ("cond2", "cond3")}
    num("ptwoTwo", f"{np.mean([v.sum() for v in p2['cond2'].values()]):.1f}")
    num("ptwoThree", f"{np.mean([v.sum() for v in p2['cond3'].values()]):.1f}")

    # ---- 쌓기 ----
    st2 = {s: stack("stack2-webcam", s, "-60k") for s in SEEDS}
    st3 = {s: stack("stack3-webcam-corrected", s, "-60k") for s in SEEDS}
    st1 = {s: stack("stack1-scripted", s, "-20k") for s in (1000, 2000, 3000)}
    rs = paired(st2, st3)
    both = res["corr"]["diffs"] + rs["diffs"]
    signs = list(itertools.product([1, -1], repeat=len(both)))
    mb = float(np.mean(both))
    pb1 = float(np.mean([(np.array(both) * np.array(s)).mean() >= mb - 1e-9 for s in signs]))
    num("stackOne", f"{np.mean([v.sum() for v in st1.values()]):.1f}")
    num("stackTwo", f"{np.mean([v.sum() for v in st2.values()]):.1f}")
    num("stackThree", f"{np.mean([v.sum() for v in st3.values()]):.1f}")
    num("stackCorrMean", signed1(rs["mean"]))
    num("stackCorrLo", f1(rs["lo"]))
    num("stackCorrHi", f1(rs["hi"]))
    num("stackPone", f"{rs['p1']:.2f}")
    num("stackPtwo", f"{rs['p2']:.3f}")
    num("stackPos", rs["pos"])
    num("bothMean", signed1(mb))
    num("bothPone", f"{pb1:.2f}")
    num("bothPos", int(sum(d > 0 for d in both)))
    disc3 = int(sum(((st3[s] == 1) & (st2[s] == 0)).sum() for s in SEEDS))
    disc2 = int(sum(((st2[s] == 1) & (st3[s] == 0)).sum() for s in SEEDS))
    num("stackDiscThree", disc3)
    num("stackDiscTwo", disc2)
    rows = [f"\\condI{{}} scripted (20k) & {', '.join(str(int(st1[s].sum())) for s in (1000, 2000, 3000))} & {np.mean([v.sum() for v in st1.values()]):.1f} \\\\",
            f"\\condII{{}} webcam (60k) & {', '.join(str(int(st2[s].sum())) for s in SEEDS)} & {np.mean([v.sum() for v in st2.values()]):.1f} \\\\",
            f"\\condIII{{}} webcam + correction (60k) & {', '.join(str(int(st3[s].sum())) for s in SEEDS)} & {np.mean([v.sum() for v in st3.values()]):.1f} \\\\",
            f"\\condIII{{}} $-$ \\condII{{}} & {', '.join(signed(d) for d in rs['diffs'])} & {signed1(rs['mean'])} \\\\"]
    (P / "tables" / "stacking.tex").write_text(
        "\\begin{tabular}{llr}\n\\toprule\nCondition & per seed & mean \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    # 쌓기 실패 단계 (사후)
    stages = {}
    for key, runs in (("two", st2), ("three", st3)):
        cnt = {"succ": 0, "floor": 0, "onbase": 0, "nolift": 0}
        for s in SEEDS:
            name = "stack2-webcam" if key == "two" else "stack3-webcam-corrected"
            for e in json.loads((EV / "stack" / f"act-{name}-s{s}-60k.json").read_text(encoding="utf-8"))["episodes"]:
                z = e["final_cube"][2]
                k = ("succ" if e["success"] else "nolift" if not e["lifted"] else "onbase" if z > 0.045
                     else "floor" if z < 0.025 else None)
                if k:
                    cnt[k] += 1
        stages[key] = cnt
        for k, v in cnt.items():
            num(f"stage{key.capitalize()}{k.capitalize()}", v)
    fr = json.loads((ROOT / "experiments" / "correction_feature_range.json").read_text(encoding="utf-8"))
    num("outPlace", f"{fr['옮기기 ③']['v']['outside_fit_range'] * 100:.1f}")
    num("outStack", f"{fr['쌓기 ③']['v']['outside_fit_range'] * 100:.1f}")
    num("outAnyPlace", f"{fr['옮기기 ③']['any_feature_outside'] * 100:.1f}")
    num("outAnyStack", f"{fr['쌓기 ③']['any_feature_outside'] * 100:.1f}")

    # 집는 위치 (정책 탐침)
    def grasp(path, conds_):
        d = json.loads(path.read_text(encoding="utf-8"))
        out = {}
        for c in conds_:
            f = np.array([r["closes"][0]["offset_mm"] for k, v in d.items() if c in k for r in v if r["closes"]])
            out[c] = float(np.median(np.abs(f[:, 0])))
        return out
    g100 = grasp(ROOT / "experiments" / "policy_grasp_probe_100k.json", ("cond2-webcam", "cond3-webcam-corrected"))
    gst = grasp(ROOT / "experiments" / "policy_grasp_probe_stack.json", ("stack2-webcam", "stack3-webcam-corrected"))
    num("graspPlaceTwo", f"{g100['cond2-webcam']:.1f}")
    num("graspPlaceThree", f"{g100['cond3-webcam-corrected']:.1f}")
    num("graspStackTwo", f"{gst['stack2-webcam']:.1f}")
    num("graspStackThree", f"{gst['stack3-webcam-corrected']:.1f}")

    # 깊이 보정 평가 (측정 2회차)
    ce = json.loads((ROOT / "experiments" / "depth_correction_eval.json").read_text(encoding="utf-8"))
    num("jumpBase", f"{ce['baseline']['test']['pose_jump_mean']:.1f}")
    num("jumpCorr", f"{ce['correction']['test']['pose_jump_mean']:.1f}")
    num("rangeBase", f"{ce['baseline']['test']['cond_range']:.1f}")
    num("rangeCorr", f"{ce['correction']['test']['cond_range']:.1f}")

    # ---- 그림 1: 학습량별 성공 ----
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    order = ["1", "40", "4", "5", "3", "2"]
    labels = {"1": "C1", "40": "C4$_0$", "4": "C4", "5": "C5", "3": "C3", "2": "C2"}
    x = np.arange(len(order))
    s_ = [np.mean([v.sum() for v in short[k].values()]) for k in order]
    l_ = [np.mean([v.sum() for v in long_[k].values()]) for k in order]
    ax.bar(x - 0.2, s_, 0.38, color="#cfcfcf", label="20k steps")
    ax.bar(x + 0.2, l_, 0.38, color=[C[k] for k in order], label="100k steps")
    ax.set_xticks(x, [labels[k] for k in order])
    ax.set_ylabel("successes / 100")
    ax.set_ylim(0, 100)
    ax.legend(handles=[matplotlib.patches.Patch(color="#cfcfcf", label="20k steps"),
                       matplotlib.patches.Patch(color="#555555", label="100k steps (colored)")],
              frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(P / "figures" / "training_length.pdf")
    plt.close(fig)

    # ---- 그림 2: 용량-반응 ----
    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    ax.plot([0, 0.5, 1, 2], dv, "-o", color="#2a6fc4")
    for xv, yv in zip([0, 0.5, 1, 2], dv):
        ax.annotate(f"{yv:.1f}", (xv, yv), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xlabel("injected depth-error scale")
    ax.set_ylabel("successes / 100")
    ax.set_ylim(40, 100)
    fig.tight_layout()
    fig.savefig(P / "figures" / "dose_response.pdf")
    plt.close(fig)

    # ---- 그림 3: 재현 (시드별 차이) ----
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    for i, d in enumerate((res["corr"]["diffs"], rs["diffs"])):
        xs = i + np.linspace(-0.15, 0.15, len(d))
        ax.scatter(xs, d, s=18, color=["#2a6fc4" if v > 0 else "#c2502a" for v in d], zorder=3)
        ax.hlines(np.mean(d), i - 0.25, i + 0.25, color="black", lw=1.5)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks([0, 1], ["placing\n(exploratory)", "stacking\n(pre-registered)"])
    ax.set_ylabel("3 $-$ 2 (successes / 100)")
    fig.tight_layout()
    fig.savefig(P / "figures" / "replication.pdf")
    plt.close(fig)

    # ---- 측정 ----
    m5 = json.loads((ROOT / "experiments" / "depth_v2_20261002_232355_summary.json").read_text(encoding="utf-8"))
    st = [r for r in m5["steps"] if r["block"] == "main"]
    for pose in ("open_flat", "open_tilt", "pinch_flat", "pinch_tilt"):
        for h in ("cam", "low"):
            v = np.mean([r["min3d_assumed_err"] for r in st if r["pose"] == pose and r["height"] == h])
            num("depth" + pose.replace("_", "").capitalize() + h.capitalize(), signed1(float(v)))

    # ---- 녹화 ----
    def rec(folder, conds_=(2, 3)):
        plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
        prog = json.loads((folder / "progress.json").read_text(encoding="utf-8"))["units"]
        out = {}
        for c in conds_:
            us = [u for u in plan["units"] if u["condition"] == c]
            tr = [t for u in us for t in prog[str(u["unit"])]["tries"] if t["result"] != "interrupted"]
            ok = [t for u in us for t in prog[str(u["unit"])]["tries"] if t["result"] == "success"]
            out[c] = {"n": len(us), "tries": len(tr), "succ": len(ok),
                      "med": float(np.median([t["seconds_from_engage"] for t in ok]))}
        return out
    r8 = rec(ROOT / "measurements" / "demos_m8")
    rst = rec(ROOT / "measurements" / "demos_stack")
    for tag, r in (("Place", r8), ("Stack", rst)):
        for c in (2, 3):
            num(f"rec{tag}{'Two' if c == 2 else 'Three'}Tries", r[c]["tries"])
            num(f"rec{tag}{'Two' if c == 2 else 'Three'}Med", f"{r[c]['med']:.1f}")

    lines = ["% 자동 생성: paper/make_assets.py — 직접 고치지 말 것"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(NUM.items())]
    (P / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"numbers.tex {len(NUM)}개, 표 4개, 그림 3개")


if __name__ == "__main__":
    main()
