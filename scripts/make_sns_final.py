"""SNS 마무리 게시물용 그림 (중간 정리 이후의 내용: 쌓기 재현, 사후 분석, 최종 결론).

    uv run python scripts/make_sns_final.py
→ outputs/sns/ (git에 안 올림). 숫자는 결과 파일에서 읽는다.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sns_media import OUT, ROOT, S, font  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
INK, DIM, BLUE, ORANGE, GREY = "#17171d", "#50505e", "#2a78d6", "#eb6834", "#9a9893"


def sq(title, sub):
    fig = plt.figure(figsize=(7.2, 7.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    fig.text(0.07, 0.93, title, fontsize=20, fontweight="bold", color=INK)
    fig.text(0.07, 0.885, sub, fontsize=11.5, color=DIM)
    ax = fig.add_axes([0.1, 0.14, 0.85, 0.66])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e9e8e4")
    ax.set_axisbelow(True)
    return fig, ax


def card(path, title, sub, rows, foot):
    im = Image.new("RGB", (S, S), (252, 252, 251))
    d = ImageDraw.Draw(im)
    d.text((70, 70), title, font=font(52), fill=(23, 23, 29))
    d.text((70, 148), sub, font=font(26, False), fill=(80, 80, 94))
    y = 250
    for k, v in rows:
        d.text((70, y), k, font=font(32), fill=(42, 120, 214))
        for i, line in enumerate(v.split("\n")):
            d.text((70, y + 48 + i * 40), line, font=font(28, False), fill=(23, 23, 29))
        y += 70 + 40 * len(v.split("\n")) + 24
    d.text((70, S - 90), foot, font=font(24, False), fill=(125, 125, 140))
    im.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / "measurements" / "demos_stack" / "plan.json").read_text(encoding="utf-8"))
    card(OUT / "03_녹화_전에_정한_것.png", "결과를 보기 전에 정한 것", "쌓기 과제 · 계획 문서를 녹화 전에 저장소에 올림",
         [("시범", f"보정 끈 시범 50개, 켠 시범 50개 · 순서 교대 · 조건 가림 · {plan['time_limit_s']:.0f}초"),
          ("학습", "같은 조건으로 5번씩(시드 5개) · 약 18에폭"),
          ("판정", "5번 모두 보정 쪽이 높으면 \"재현됨\"\n4번이면 \"대체로\", 3번 이하면 \"재현 안 됨\""),
          ("약속", "결과가 어떻든 시드를 더 늘리지 않고 그대로 보고")],
         "결과를 본 뒤 기준을 바꾸면, 원하는 결과가 나올 때까지 고친 셈이 되니까")

    ev = Path(ROOT / "experiments" / "eval" / "stack")
    cats = {"성공": [], "바닥에 떨어뜨림·내려놓음": [], "받침 위에 있지만 실패": [], "안 들림": []}
    for cond in ("stack2-webcam", "stack3-webcam-corrected"):
        c = dict.fromkeys(cats, 0)
        for f in sorted(ev.glob(f"act-{cond}-s*-60k.json")):
            for e in json.loads(f.read_text(encoding="utf-8"))["episodes"]:
                z = e["final_cube"][2]
                k = ("성공" if e["success"] else "안 들림" if not e["lifted"]
                     else "받침 위에 있지만 실패" if z > 0.045 else "바닥에 떨어뜨림·내려놓음" if z < 0.025 else None)
                if k:
                    c[k] += 1
        for k in cats:
            cats[k].append(c[k])
    fig, ax = sq("보정한 로봇은 '놓을 때' 실패했다", "쌓기 평가 500판(시드 5개 × 100판) 중 실패를 단계별로 나눔")
    labels = ["보정 안 한\n시범으로 학습", "보정한\n시범으로 학습"]
    fail = [k for k in cats if k != "성공"]
    x = range(len(fail))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [cats[k][0] for k in fail], w, color=ORANGE, label="보정 안 한 시범")
    ax.bar([i + w / 2 for i in x], [cats[k][1] for k in fail], w, color=BLUE, label="보정한 시범")
    for i, k in enumerate(fail):
        ax.text(i - w / 2, cats[k][0] + 1.5, str(cats[k][0]), ha="center", fontsize=13, fontweight="bold")
        ax.text(i + w / 2, cats[k][1] + 1.5, str(cats[k][1]), ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(list(x), [k.replace("·", "·\n") if len(k) > 8 else k for k in fail], fontsize=11)
    ax.set_ylabel("실패 횟수 (500판 중)", fontsize=11)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    fig.text(0.07, 0.03, f"성공은 {cats['성공'][0]} vs {cats['성공'][1]}판 · 집는 위치는 보정한 쪽이 오히려 더 정확했다 (2.6 vs 3.2 mm)",
             fontsize=10.5, color=BLUE)
    fig.savefig(OUT / "05_어디서_실패했나.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    fr = json.loads((ROOT / "experiments" / "correction_feature_range.json").read_text(encoding="utf-8"))
    v = [fr["옮기기 ③"]["v"]["outside_fit_range"] * 100, fr["쌓기 ③"]["v"]["outside_fit_range"] * 100]
    fig, ax = sq("보정이 배운 적 없는 곳에서 쓰였다", "손이 화면 세로로 보정을 만들 때 재 본 범위 밖에 있던 비율 (보정 켠 시범)")
    ax.bar([0, 1], v, 0.5, color=[GREY, ORANGE])
    for i, val in enumerate(v):
        ax.text(i, val + 1.5, f"{val:.0f}%", ha="center", fontsize=16, fontweight="bold")
    ax.set_xticks([0, 1], ["큐브 옮기기", "큐브 쌓기"], fontsize=12)
    ax.set_ylim(0, 70)
    ax.set_ylabel("측정 범위 밖 프레임 (%)", fontsize=11)
    fig.text(0.07, 0.03, "쌓기는 큐브를 받침 위로 들고 가느라 손이 화면 위쪽에 있었다 · 결과를 본 뒤 찾은 설명이라 확정은 아님",
             fontsize=10, color=ORANGE)
    fig.savefig(OUT / "06_측정_범위_밖.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    card(OUT / "07_최종_결론.png", "그래서, 결론", "웹캠 시범의 깊이 오차는 로봇 학습에 얼마나 손해일까",
         [("1. 웹캠 시범은 얼마나 손해인가", "충분히 학습하면 약 17%p (짧게 학습하면 33%p로 부풀어 보임)"),
          ("2. 깊이 오차가 원인인가", "그렇다. 오차만 넣고 빼도 매번 손해, 키울수록 88 → 59"),
          ("3. 가볍게 보정하면 되나", "과제마다 달랐다 (+7.8 → -8.2%p), 일반적인 결론 아님"),
          ("공개", "코드 · 데이터셋 14개 · 학습한 모델 35개")],
         "github.com/kheechan04/webcam-teach-robot")
    print("→", OUT)


if __name__ == "__main__":
    main()
