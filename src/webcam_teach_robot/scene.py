"""과제 장면: 큐브 옮기기(place, 기본) 또는 쌓기(stack, 환경 변수 WTR_TASK=stack).

Menagerie SO-101 장면(third_party, 수정하지 않음)을 불러와서 코드로 덧붙인다.
    - cube   : 3 cm 정육면체, 자유롭게 움직임(freejoint)
    - target : 목표 표시. 바닥에 그린 초록 사각형(충돌 없음). 에피소드마다 옮기려고 mocap 몸체로 둔다
    - front  : 정책이 볼 고정 카메라 (로봇 뒤 위쪽에서 작업 공간을 내려다봄)
    - wrist_cam : 원래 모델에 있던 손목 카메라

집게는 로봇 앞뒤 방향(집게 기준 x축)으로 닫힌다. 손목 돌리기를 고정했으므로,
큐브는 면이 로봇 쪽을 정면으로 보게(yaw = atan2(y, x)) 놓는다.

쌓기(stack, 2026-10-06 추가): 초록 목표 사각형 대신 초록 받침 블록(4 cm, 자유롭게 움직임)을 목표 자리에 놓고,
빨간 큐브를 그 위에 올려놓는다. "목표 위치(goal_xy)"는 받침 블록의 처음 위치다. 받침을 밀어 옮기면 실패.
과제는 프로세스 시작 때 WTR_TASK로 한 번 정하고, 기록·데이터셋 정보에 함께 남긴다.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE_SCENE = ROOT / "third_party" / "robotstudio_so101" / "scene.xml"

CUBE_HALF = 0.015  # 3 cm 정육면체
CUBE_MASS = 0.03  # kg
TARGET_HALF = 0.03  # 목표 표시 6 cm × 6 cm

TASK = os.environ.get("WTR_TASK", "place")
assert TASK in ("place", "stack"), f"WTR_TASK는 place 또는 stack: {TASK}"
BASE_HALF = 0.02  # 쌓기 받침 블록 4 cm 정육면체 (위 큐브 3 cm보다 커서 놓을 자리가 사방 0.5 cm 넓다)
BASE_MASS = 0.1  # kg
STACK_Z = 2 * BASE_HALF + CUBE_HALF  # 받침 위에 놓인 큐브 중심 높이
TASK_TEXT = ("Pick up the red cube and stack it on the green block." if TASK == "stack"
             else "Pick up the red cube and place it on the green square.")


# 정책이 볼 앞쪽 카메라. 128×128로 줄였을 때 큐브가 몇 픽셀밖에 안 돼서(처음: 위치 (0.50, −0.30, 0.38), 화각 55°)
# 후보 4개를 같은 배치로 그려 비교한 뒤, 큐브·목표·집게가 크게 보이고 배치 끝도 화면에 들어오는 것으로 정했다.
FRONT_CAM_POS = np.array([0.46, -0.20, 0.30])
FRONT_CAM_LOOKAT = np.array([0.21, 0.0, 0.04])
FRONT_CAM_FOVY = 45


def lookat_xyaxes(pos: np.ndarray, target: np.ndarray) -> list[float]:
    """카메라가 target을 보게 하는 xyaxes. MuJoCo 카메라는 자기 -z 방향을 본다."""
    f = target - pos
    f = f / np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x)
    y = np.cross(x, f)
    return [*x, *y]


@dataclass
class TaskIds:
    cube_body: int
    cube_qadr: int  # freejoint qpos 시작 위치 (x, y, z, qw, qx, qy, qz)
    cube_dadr: int
    target_mocap: int
    tip_site: int
    base_qadr: int = -1  # 쌓기 받침 블록 freejoint (옮기기 과제에선 없음)
    base_dadr: int = -1


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(BASE_SCENE))
    world = spec.worldbody

    cube = world.add_body(name="cube", pos=[0.22, -0.06, CUBE_HALF])
    cube.add_freejoint(name="cube_free")
    cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                  mass=CUBE_MASS, rgba=[0.85, 0.25, 0.2, 1], condim=4,
                  friction=[1.0, 0.03, 0.003], solref=[0.01, 1])

    target = world.add_body(name="target", mocap=True, pos=[0.22, 0.08, 0.0])
    # 쌓기에선 목표 사각형을 보이지 않게 둔다(TaskIds·렌더 코드를 두 과제가 같이 쓰려고 몸체는 남김)
    target.add_geom(name="target", type=mujoco.mjtGeom.mjGEOM_BOX, size=[TARGET_HALF, TARGET_HALF, 0.0005],
                    rgba=[0.2, 0.75, 0.35, 0.0 if TASK == "stack" else 0.6], contype=0, conaffinity=0, group=1)
    if TASK == "stack":
        base = world.add_body(name="stack_base", pos=[0.22, 0.08, BASE_HALF])
        base.add_freejoint(name="stack_base_free")
        base.add_geom(name="stack_base", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BASE_HALF] * 3,
                      mass=BASE_MASS, rgba=[0.2, 0.7, 0.3, 1], condim=4,
                      friction=[1.0, 0.03, 0.003], solref=[0.01, 1])

    # 작업 공간 앞쪽 비스듬한 위에서 로봇 쪽을 보는 카메라. (로봇 뒤 위에서 보면 팔이 큐브를 가렸다)
    world.add_camera(name="front", pos=FRONT_CAM_POS, xyaxes=lookat_xyaxes(FRONT_CAM_POS, FRONT_CAM_LOOKAT), fovy=FRONT_CAM_FOVY)
    return spec.compile()


def task_ids(model: mujoco.MjModel) -> TaskIds:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "stack_base_free")
    return TaskIds(
        cube_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube"),
        cube_qadr=model.jnt_qposadr[jid],
        cube_dadr=model.jnt_dofadr[jid],
        target_mocap=model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")],
        tip_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"),
        base_qadr=int(model.jnt_qposadr[bid]) if bid >= 0 else -1,
        base_dadr=int(model.jnt_dofadr[bid]) if bid >= 0 else -1,
    )


def yaw_facing_robot(xy: np.ndarray) -> float:
    return float(np.arctan2(xy[1], xy[0]))


def place(model: mujoco.MjModel, data: mujoco.MjData, ids: TaskIds,
          cube_xy: np.ndarray, target_xy: np.ndarray) -> None:
    """큐브와 목표를 놓는다. 큐브는 면이 로봇을 향하게."""
    yaw = yaw_facing_robot(cube_xy)
    q = ids.cube_qadr
    data.qpos[q:q + 3] = [cube_xy[0], cube_xy[1], CUBE_HALF]
    data.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    data.qvel[ids.cube_dadr:ids.cube_dadr + 6] = 0
    tyaw = yaw_facing_robot(target_xy)
    data.mocap_pos[ids.target_mocap] = [target_xy[0], target_xy[1], 0.0005]
    data.mocap_quat[ids.target_mocap] = [np.cos(tyaw / 2), 0, 0, np.sin(tyaw / 2)]
    if ids.base_qadr >= 0:
        b = ids.base_qadr
        data.qpos[b:b + 3] = [target_xy[0], target_xy[1], BASE_HALF]
        data.qpos[b + 3:b + 7] = [np.cos(tyaw / 2), 0, 0, np.sin(tyaw / 2)]
        data.qvel[ids.base_dadr:ids.base_dadr + 6] = 0
    mujoco.mj_forward(model, data)


def cube_pos(data: mujoco.MjData, ids: TaskIds) -> np.ndarray:
    return data.qpos[ids.cube_qadr:ids.cube_qadr + 3].copy()


def base_pos(data: mujoco.MjData, ids: TaskIds) -> np.ndarray | None:
    """쌓기 받침 블록 위치. 옮기기 과제에선 None."""
    return data.qpos[ids.base_qadr:ids.base_qadr + 3].copy() if ids.base_qadr >= 0 else None


def base_quat(data: mujoco.MjData, ids: TaskIds) -> np.ndarray | None:
    return data.qpos[ids.base_qadr + 3:ids.base_qadr + 7].copy() if ids.base_qadr >= 0 else None


def sample_layout(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """큐브는 오른쪽(y<0), 목표는 왼쪽(y>0) 영역에서 무작위. 둘 다 조종 작업 범위(x 16~28, y ±15 cm) 안."""
    cube = np.array([rng.uniform(0.18, 0.26), rng.uniform(-0.12, -0.04)])
    target = np.array([rng.uniform(0.18, 0.26), rng.uniform(0.04, 0.12)])
    return cube, target


def cube_on_table(cube_z: float) -> bool:
    return abs(cube_z - CUBE_HALF) < 0.005


def in_target(cube_xyz: np.ndarray, target_xy: np.ndarray) -> bool:
    """성공 판정: 큐브 중심이 목표 사각형(목표 방향 기준) 안에 있고 바닥에 놓여 있다."""
    yaw = yaw_facing_robot(target_xy)
    d = cube_xyz[:2] - target_xy
    local = np.array([np.cos(yaw) * d[0] + np.sin(yaw) * d[1], -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]])
    return bool(np.all(np.abs(local) <= TARGET_HALF) and cube_on_table(cube_xyz[2]))


def on_base(cube_xyz: np.ndarray, base_xyz: np.ndarray) -> bool:
    """쌓기 판정: 큐브가 받침 윗면에 얹혀 있다(높이 ±5 mm, 큐브 중심이 받침 윗면 안, 받침 방향 기준)."""
    yaw = yaw_facing_robot(base_xyz[:2])
    d = cube_xyz[:2] - base_xyz[:2]
    local = np.array([np.cos(yaw) * d[0] + np.sin(yaw) * d[1], -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]])
    return bool(np.all(np.abs(local) <= BASE_HALF)
                and abs(cube_xyz[2] - (base_xyz[2] + BASE_HALF + CUBE_HALF)) < 0.005)


LIFT_Z = CUBE_HALF + 0.01  # 큐브 중심이 이보다 높으면 "들렸다" (바닥에서 1 cm 넘게)
PUSH_TOL = 0.01  # 내려앉은 뒤 이만큼 넘게 움직이면 밀린 것
STABLE_S = 0.5  # 성공 상태가 이만큼 이어져야 성공


class PlacementTracker:
    """성공 판정: 큐브를 "들어서 옮겨 놓았다".

    in_target만 쓰면 바닥에서 밀어 넣어도 성공으로 쳤다(2026-10-02 조종 기록 4개의 "성공"이 전부 그랬다).
    그래서 아래를 모두 만족해야 성공이다.
        1. 큐브가 바닥에서 1 cm 넘게 들렸다가
        2. 목표 사각형 안에 내려앉았고 (in_target)
        3. 내려앉은 뒤 1 cm 넘게 밀리지 않았고
        4. 집게가 열린 채로 0.5초 이어졌다
    스크립트 시범·웹캠 조종 화면·데이터셋 만들기·학습한 정책 평가가 모두 이 판정을 쓴다.

    쌓기(TASK == "stack")에서는 2를 "받침 블록 윗면에 얹혔다(on_base)"로 바꾸고, 하나를 더한다:
        5. 받침 블록이 처음 자리(goal_xy)에서 1 cm 넘게 밀리지 않았고 바닥에 서 있다
    그래서 update에 받침 위치(base_xyz)를 꼭 넘겨야 한다.
    """

    def __init__(self):
        self.was_lifted = False
        self.landing_xy: np.ndarray | None = None
        self.ok_time = 0.0
        self.succeeded = False

    def update(self, cube_xyz: np.ndarray, goal_xy: np.ndarray, gripper_closed: bool, dt: float,
               base_xyz: np.ndarray | None = None) -> bool:
        if TASK == "stack":
            if base_xyz is None:
                raise ValueError("쌓기 과제는 PlacementTracker.update에 base_xyz(받침 블록 위치)가 필요하다")
            landed = on_base(cube_xyz, base_xyz)
            placed = landed and (np.linalg.norm(base_xyz[:2] - goal_xy) <= PUSH_TOL
                                 and abs(base_xyz[2] - BASE_HALF) < 0.005)
            lifted = cube_xyz[2] > LIFT_Z and not landed
        else:
            landed = cube_on_table(cube_xyz[2])
            placed = in_target(cube_xyz, goal_xy)
            lifted = cube_xyz[2] > LIFT_Z
        if lifted:
            self.was_lifted = True
            self.landing_xy = None
        elif self.was_lifted and self.landing_xy is None and landed:
            self.landing_xy = cube_xyz[:2].copy()
        ok = (self.landing_xy is not None and not gripper_closed and placed
              and np.linalg.norm(cube_xyz[:2] - self.landing_xy) <= PUSH_TOL)
        self.ok_time = self.ok_time + dt if ok else 0.0
        if self.ok_time >= STABLE_S:
            self.succeeded = True
        return self.succeeded


# 쌓기는 배치 목록을 따로 둔다(같은 영역, 다른 시드). 받침 블록 자리 = goal_xy.
LAYOUTS_FILE = ROOT / "experiments" / ("layouts_stack_v1.json" if TASK == "stack" else "layouts_v1.json")


def make_layouts(seed: int, n: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        cube, goal = sample_layout(rng)
        out.append({"id": i, "cube_xy": [round(float(v), 4) for v in cube],
                    "goal_xy": [round(float(v), 4) for v in goal]})
    return out


def load_layouts(split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """고정 배치 목록. split = "train"(시범용) 또는 "eval"(학습한 정책 평가용, 시범에 안 씀)."""
    import json
    data = json.loads(LAYOUTS_FILE.read_text(encoding="utf-8"))
    return [(np.array(d["cube_xy"]), np.array(d["goal_xy"])) for d in data[split]]
