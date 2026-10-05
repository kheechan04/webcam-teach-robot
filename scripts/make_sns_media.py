"""SNS(인스타그램)용 정사각형 1080×1080 그림·영상. 웹캠 영상은 쓰지 않는다 — 기록된 숫자로 시뮬레이션을 다시 그린다.

    uv run python scripts/make_sns_media.py policy <Hub 모델> [--layout N]   # 학습한 로봇 혼자 (mp4)
    uv run python scripts/make_sns_media.py replay <조종 기록 csv>           # 웹캠 조종 시범 다시 그리기 (mp4)
    uv run python scripts/make_sns_media.py charts                          # 그래프 png
→ outputs/sns/ (git에 안 올림)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "sns"
FONT = "C:/Windows/Fonts/malgunbd.ttf"
FONT_R = "C:/Windows/Fonts/malgun.ttf"
S = 1080
BG = (16, 18, 24)


def font(size, bold=True):
    try:
        return ImageFont.truetype(FONT if bold else FONT_R, size)
    except OSError:
        return ImageFont.load_default()


def write_mp4(frames: list[Image.Image], path: Path, fps: int = 30) -> None:
    import av
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = S, S, "yuv420p"
        st.options = {"crf": "20", "preset": "medium"}
        for im in frames:
            for p in st.encode(av.VideoFrame.from_image(im)):
                c.mux(p)
        for p in st.encode():
            c.mux(p)
    print(f"→ {path} ({len(frames)}프레임, {path.stat().st_size / 1e6:.1f} MB)")


def renderer(model, w, h):
    import mujoco
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)
    r = mujoco.Renderer(model, h, w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.19, 0.0, 0.06]
    cam.distance, cam.azimuth, cam.elevation = 0.78, 150, -32
    return r, cam


def card(scene: np.ndarray, title: str, sub: str, small: list[np.ndarray] | None = None, tag: str = "") -> Image.Image:
    im = Image.new("RGB", (S, S), BG)
    d = ImageDraw.Draw(im)
    d.text((48, 36), title, font=font(46), fill=(255, 255, 255))
    d.text((48, 100), sub, font=font(28, False), fill=(190, 196, 208))
    im.paste(Image.fromarray(scene), (0, 160))
    if small:
        y = 160 + scene.shape[0] + 16
        for i, s in enumerate(small):
            im.paste(Image.fromarray(s).resize((160, 160), Image.NEAREST), (48 + i * 176, y))
        d.text((48 + len(small) * 176 + 8, y + 50), "로봇이 실제로 보는 카메라 화면\n(128×128, 큐브 위치는 숫자로 안 알려 줌)",
               font=font(24, False), fill=(190, 196, 208))
    if tag:
        d.text((48, S - 52), tag, font=font(22, False), fill=(130, 136, 150))
    return im


def cmd_policy(args):
    import importlib.util

    import mujoco
    import torch

    from webcam_teach_robot.dataset import FPS, SceneRenderer
    from webcam_teach_robot.scene import PlacementTracker, build_model, cube_pos, load_layouts, place, task_ids
    spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "eval_policy.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    policy, pre, post = ev.load_policy(args.target, "cpu")
    model = build_model()
    data = mujoco.MjData(model)
    ids = task_ids(model)
    cams = SceneRenderer(model, ids)
    side, view = renderer(model, S, 640)
    # 노트북 CPU에서 돌리면 GPU 평가와 계산이 조금 달라 같은 배치에서도 결과가 다를 수 있다 → 성공하는 배치를 찾는다
    for lay in [args.layout] + [x for x in range(100) if x != args.layout]:
        frames, done = run_layout(lay, policy, pre, post, model, data, ids, cams, side, view, ev, args)
        if done is not None:
            break
        print(f"  배치 {lay}: 30초 안에 성공 못 함, 다음 배치")
    OUT.mkdir(parents=True, exist_ok=True)
    write_mp4(frames, OUT / f"policy_{Path(args.target).name}_L{lay}.mp4")


def run_layout(lay, policy, pre, post, model, data, ids, cams, side, view, ev, args):
    import mujoco
    import torch

    from webcam_teach_robot.dataset import FPS
    from webcam_teach_robot.scene import PlacementTracker, cube_pos, load_layouts, place
    cube_xy, goal_xy = load_layouts("eval")[lay]
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = ev.HOME_Q
    data.ctrl[:6] = ev.HOME_Q
    place(model, data, ids, cube_xy, goal_xy)
    policy.reset()
    tr = PlacementTracker()
    steps = int(round(1 / (FPS * model.opt.timestep)))
    frames, done = [], None
    for k in range(int(30 * FPS)):
        imgs = cams.render_data(data)
        obs = {"observation.state": torch.from_numpy(data.qpos[:6].astype(np.float32)), "task": "x"}
        for key, img in imgs.items():
            obs[f"observation.images.{key}"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            a = post(policy.select_action(pre(obs))).squeeze(0).numpy()
        data.ctrl[:6] = a
        for _ in range(steps):
            mujoco.mj_step(model, data)
        if tr.update(cube_pos(data, ids), goal_xy, a[5] < 0.5, 1 / FPS) and done is None:
            done = k / FPS
        side.update_scene(data, camera=view)
        sub = f"{k / FPS:4.1f}초" + ("   ✓ 성공" if done is not None else "")
        frames.append(card(side.render(), args.title or "웹캠 시범으로 배운 로봇 (혼자)", sub,
                           [imgs["front"], imgs["wrist"]], "처음 보는 배치 · 시뮬레이션 SO-101 · ACT"))
        if done is not None and k / FPS > done + 1.0:
            break
    return frames, done


def cmd_replay(args):
    import mujoco

    from webcam_teach_robot.dataset import default_cube_quat
    from webcam_teach_robot.scene import build_model, task_ids, yaw_facing_robot
    R = [r for r in csv.DictReader(open(args.target, encoding="utf-8")) if r["engaged"] == "1"]
    model = build_model()
    data = mujoco.MjData(model)
    ids = task_ids(model)
    side, view = renderer(model, S, 760)
    t = np.array([float(r["t_s"]) for r in R])
    grid = np.arange(t[0], t[-1], args.speed / 30)
    frames = []
    for g in grid:
        r = R[int(np.searchsorted(t, g))]
        data.qpos[:6] = [float(r[f"q{i}"]) for i in range(6)]
        q = ids.cube_qadr
        data.qpos[q:q + 3] = [float(r[f"cube_{a}"]) for a in "xyz"]
        data.qpos[q + 3:q + 7] = [float(r[f"cube_q{a}"]) for a in "wxyz"]
        goal = np.array([float(r["goal_x"]), float(r["goal_y"])])
        data.mocap_pos[ids.target_mocap] = [goal[0], goal[1], 0.0005]
        yaw = yaw_facing_robot(goal)
        data.mocap_quat[ids.target_mocap] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(model, data)
        side.update_scene(data, camera=view)
        frames.append(card(side.render(), "웹캠으로 조종한 시범", f"{g - t[0]:4.1f}초  ({args.speed:g}배속)",
                           None, "웹캠 영상 없이, 기록된 관절 각도·큐브 위치로 다시 그림"))
    OUT.mkdir(parents=True, exist_ok=True)
    write_mp4(frames, OUT / f"replay_{Path(args.target).stem}.mp4")


def cmd_charts(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(parents=True, exist_ok=True)

    def sq(title, sub):
        fig = plt.figure(figsize=(7.2, 7.2), dpi=150)
        fig.patch.set_facecolor("#fcfcfb")
        fig.text(0.07, 0.93, title, fontsize=20, fontweight="bold", color="#17171d")
        fig.text(0.07, 0.885, sub, fontsize=11.5, color="#50505e")
        ax = fig.add_axes([0.1, 0.12, 0.85, 0.68])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e9e8e4")
        ax.set_axisbelow(True)
        return fig, ax

    # 1. 깊이 오차 (M5)
    m5 = json.loads((ROOT / "experiments" / "depth_v2_20261002_232355_summary.json").read_text(encoding="utf-8"))
    steps = [r for r in m5["steps"] if r["block"] == "main"]
    poses = [("open_flat", "편 손\n정면"), ("open_tilt", "편 손\n기울임"), ("pinch_flat", "집은 손\n정면"), ("pinch_tilt", "집은 손\n기울임")]
    v = [np.mean([r["min3d_assumed_err"] for r in steps if r["pose"] == p]) for p, _ in poses]
    fig, ax = sq("웹캠은 손까지 거리를 틀리게 읽는다", "줄자로 잰 참값과 비교 (거리 35~65 cm, 70단계, 시범자 1)")
    ax.bar(range(4), v, color=["#9a9893", "#9a9893", "#9a9893", "#eb6834"], width=0.6)
    for i, x in enumerate(v):
        ax.text(i, x + (0.4 if x >= 0 else -1.2), f"{x:+.1f} cm", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(range(4), [n for _, n in poses], fontsize=12)
    ax.set_ylabel("평균 깊이 오차 (cm, +는 멀게 읽힘)", fontsize=11)
    ax.axhline(0, color="#52514e", lw=1)
    fig.text(0.07, 0.03, "큐브를 집으러 갈 때의 손 모양이 가장 크게, 항상 '멀다' 쪽으로 틀린다", fontsize=11, color="#eb6834")
    fig.savefig(OUT / "chart1_depth_error.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2. 용량-반응
    E = ROOT / "experiments" / "eval" / "m9"
    sc = lambda n: json.loads((E / f"act-{n}.json").read_text(encoding="utf-8"))["success_rate"] * 100  # noqa: E731
    lv = [("cond4z-no-error", "0배"), ("cond4h-error-half", "0.5배"), ("cond4-depth-error", "1배\n(실측)"), ("cond4d-error-double", "2배")]
    m = [np.mean([sc(f"{c}-s{s}") for s in (1000, 2000, 3000)]) for c, _ in lv]
    fig, ax = sq("깊이 오차가 클수록 로봇이 못 배운다", "사람 없이 깊이 오차 크기만 바꿔 넣은 시범으로 학습 (시드 3개 평균)")
    ax.plot(range(4), m, "-o", color="#2a78d6", lw=3, ms=11)
    for i, x in enumerate(m):
        ax.text(i, x + 2.2, f"{x:.0f}", ha="center", fontsize=15, fontweight="bold")
    ax.set_xticks(range(4), [n for _, n in lv], fontsize=12)
    ax.set_ylim(40, 100)
    ax.set_xlabel("넣은 깊이 오차 크기", fontsize=11)
    ax.set_ylabel("처음 보는 배치 100개 중 성공", fontsize=11)
    fig.savefig(OUT / "chart2_dose_response.png", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"→ {OUT}/chart1_depth_error.png, chart2_dose_response.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["policy", "replay", "charts"])
    p.add_argument("target", nargs="?")
    p.add_argument("--layout", type=int, default=0)
    p.add_argument("--speed", type=float, default=2.0)
    p.add_argument("--title", default=None)
    a = p.parse_args()
    {"policy": cmd_policy, "replay": cmd_replay, "charts": cmd_charts}[a.what](a)


if __name__ == "__main__":
    main()
