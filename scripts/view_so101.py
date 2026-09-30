"""SO-101 모델을 MuJoCo로 불러와서 관절 정보를 출력하고 화면에 띄운다.

실행:
    uv run python scripts/view_so101.py            # 창을 띄워 직접 돌려 보기
    uv run python scripts/view_so101.py --snapshot  # 창 없이 사진 한 장만 저장
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "third_party" / "robotstudio_so101" / "scene.xml"


def print_joints(model: mujoco.MjModel) -> None:
    print(f"관절 {model.njnt}개, 모터(actuator) {model.nu}개")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = np.degrees(model.jnt_range[j])
        print(f"  {name:14s} 범위 {lo:7.1f}° ~ {hi:6.1f}°")


def save_snapshot(model: mujoco.MjModel, data: mujoco.MjData, path: Path) -> None:
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.15, 0.0, 0.15]
        cam.distance = 0.8
        cam.azimuth = 135
        cam.elevation = -20
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()

    Image.fromarray(pixels).save(path)
    print(f"사진 저장: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", action="store_true", help="창 없이 사진만 저장")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    print_joints(model)

    if args.snapshot:
        out = ROOT / "outputs" / "so101_snapshot.png"
        out.parent.mkdir(exist_ok=True)
        save_snapshot(model, data, out)
    else:
        # 창 오른쪽 Control 패널의 슬라이더로 관절을 하나씩 움직여 볼 수 있다.
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
