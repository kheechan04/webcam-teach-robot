"""시범 데이터 자체의 품질 비교: 왜 학습 결과가 달라졌나 (M9 해석용).

    uv run python scripts/analyze_demos.py
조건 ②·③(사람 웹캠 녹화, measurements/demos_m8)과 ④·④0(가상 조작자, measurements/cond4_virtual)의 성공 시범을
같은 지표로 비교한다. 조건 ①은 조종 기록이 없어 빼지만, 스크립트라 집게 여닫기 1번씩·되돌림 없음이다.

지표 (조종을 건 구간, 집게 끝 목표 위치 기준):
    grip_toggles   집게 여닫기 횟수 (스크립트 = 2: 닫기 1, 열기 1)
    regrasps       닫았다가 큐브를 못 들고 다시 연 횟수 (헛집기)
    path_ratio     집게 끝이 움직인 거리 / 꼭 필요한 거리(시작→큐브→목표)
    approach_x_jit 큐브를 들어 올린 집기 직전 2초 동안 앞뒤(로봇 x = 웹캠 깊이 방향) 목표의 떨림
                   (5프레임 이동평균과의 차이의 표준편차, mm)
"""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def demo_files():
    plan = json.loads((ROOT / "measurements/demos_m8/plan.json").read_text(encoding="utf-8"))
    prog = json.loads((ROOT / "measurements/demos_m8/progress.json").read_text(encoding="utf-8"))
    out = {2: [], 3: []}
    for u in plan["units"]:
        for t in prog["units"][str(u["unit"])]["tries"]:
            if t["result"] == "success":
                out[u["condition"]].append(ROOT / "measurements/demos_m8" / t["file"])
    for v, c in (("err", 4), ("clean", 40)):
        for j in sorted((ROOT / "measurements/cond4_virtual" / v).glob("*.json")):
            if json.loads(j.read_text(encoding="utf-8"))["result"] == "success":
                out.setdefault(c, []).append(j.with_suffix(".csv"))
    return out


def metrics(path: Path) -> dict:
    R = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["engaged"] == "1"]
    g = lambda k: np.array([float(r[k]) for r in R])  # noqa: E731
    T = np.stack([g("target_x"), g("target_y"), g("target_z")], 1)
    cl = g("gripper_closed")
    cz = g("cube_z")
    tog = np.where(np.diff(cl) != 0)[0] + 1
    closes = [k for k in tog if cl[k] == 1]
    regr = 0
    for k in closes:
        nxt = [j for j in tog if j > k]
        end = nxt[0] if nxt else len(R)
        if cz[k:end].max() < 0.025:
            regr += 1
    path = np.linalg.norm(np.diff(T, axis=0), axis=1).sum()
    cube0 = np.array([float(R[0]["cube_x"]), float(R[0]["cube_y"])])
    goal = np.array([float(R[0]["goal_x"]), float(R[0]["goal_y"])])
    need = np.linalg.norm(T[0, :2] - cube0) + np.linalg.norm(cube0 - goal) + 2 * 0.06
    x = T[:, 0]
    # 큐브를 실제로 들어 올린 첫 집기 직전 2초 (시작하자마자 손이 오므려져 있던 짧은 닫힘 등은 건너뜀)
    jit = np.nan
    lifting = [k for k in closes if cz[k:k + 90].max() > 0.025 and k > 10]
    if lifting:
        k = lifting[0]
        seg = x[max(0, k - 60):k]
        ma = np.convolve(seg, np.ones(5) / 5, mode="same")
        jit = float(np.std((seg - ma)[2:-2]) * 1000)
    return {"grip_toggles": len(tog), "regrasps": regr, "path_ratio": path / need, "approach_x_jit_mm": jit, "seconds": len(R) / 30}


def main() -> None:
    files = demo_files()
    names = {2: "② 웹캠", 3: "③ 웹캠 + 보정", 4: "④ 가상 + 깊이 오차", 40: "④0 가상, 오차 없음"}
    out = {}
    keys = ["grip_toggles", "regrasps", "path_ratio", "approach_x_jit_mm", "seconds"]
    print(f"{'':16s}" + "".join(f"{k:>18s}" for k in keys))
    for c in (2, 3, 4, 40):
        M = [metrics(p) for p in files[c]]
        out[c] = {k: [m[k] for m in M] for k in keys}
        print(f"{names[c]:14s} n={len(M):2d}" + "".join(f"{np.nanmedian(out[c][k]):12.2f} ({np.nanmean(out[c][k]):5.2f})" for k in keys))
    # ②·③ 같은 배치끼리 짝 비교
    (ROOT / "experiments" / "demo_quality.json").write_text(json.dumps({str(k): v for k, v in out.items()}, indent=1),
                                                            encoding="utf-8")
    print("(중앙값 (평균)) → experiments/demo_quality.json")


if __name__ == "__main__":
    main()
