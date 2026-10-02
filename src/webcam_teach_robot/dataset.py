"""시범 → LeRobot 데이터셋.

조종할 때는 숫자(관절 각도·명령, 큐브 위치·자세, 목표 위치)만 저장하고, 데이터셋을 만들 때 그 숫자로 장면을
다시 세워 시뮬레이션 카메라 화면을 그린다. 조종 중에 그리면 느리고(카메라당 약 75 ms), 웹캠 영상이 데이터에
들어갈 일도 없다.

모든 조건(①~④)이 같은 함수로 데이터셋을 만든다: 같은 관측·행동 정의, 같은 카메라, 같은 렌더 설정.
    observation.state         관절 각도 6개 (rad)
    action                    관절 명령 6개 (rad) — 위치 제어 모터에 보낸 목표 각도
    observation.images.front  앞쪽 고정 카메라
    observation.images.wrist  손목 카메라
큐브·목표 위치는 정책 입력에 넣지 않는다(카메라로 봐야 한다). 대신 에피소드별 정보 파일에 따로 남긴다.
"""

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from webcam_teach_robot.scene import TaskIds, yaw_facing_robot

FPS = 30
IMAGE_HW = (128, 128)
CAMERAS = {"front": "front", "wrist": "wrist_cam"}
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TASK_TEXT = "Pick up the red cube and place it on the green square."


def features() -> dict:
    h, w = IMAGE_HW
    f = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"{n}.pos" for n in JOINT_NAMES]},
        "action": {"dtype": "float32", "shape": (6,), "names": [f"{n}.pos" for n in JOINT_NAMES]},
    }
    for key in CAMERAS:
        f[f"observation.images.{key}"] = {"dtype": "video", "shape": (h, w, 3),
                                          "names": ["height", "width", "channels"]}
    return f


@dataclass
class Episode:
    """제어 주기(1/FPS초)마다 한 줄. 렌더에 필요한 장면 상태를 모두 담는다."""
    qpos: np.ndarray  # (T, 6) 로봇 관절
    action: np.ndarray  # (T, 6) 관절 명령
    cube_pos: np.ndarray  # (T, 3)
    cube_quat: np.ndarray  # (T, 4) wxyz
    goal_xy: np.ndarray  # (2,)
    success: bool
    info: dict = field(default_factory=dict)  # 출처, 배치, 조건 등


class SceneRenderer:
    """기록된 숫자로 장면을 세우고 카메라 화면을 그린다. 그림자·바닥 반사는 끈다(카메라당 75 → 15 ms)."""

    def __init__(self, model: mujoco.MjModel, ids: TaskIds):
        self.model, self.ids = model, ids
        self.data = mujoco.MjData(model)
        self.r = mujoco.Renderer(model, *IMAGE_HW)

    def frames(self, ep: Episode, t: int) -> dict[str, np.ndarray]:
        d, q = self.data, self.ids.cube_qadr
        d.qpos[:6] = ep.qpos[t]
        d.qpos[q:q + 3] = ep.cube_pos[t]
        d.qpos[q + 3:q + 7] = ep.cube_quat[t]
        d.mocap_pos[self.ids.target_mocap] = [ep.goal_xy[0], ep.goal_xy[1], 0.0005]
        yaw = yaw_facing_robot(ep.goal_xy)
        d.mocap_quat[self.ids.target_mocap] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        # 카메라 위치는 mj_kinematics가 아니라 mj_camlight에서 계산된다. 빼먹으면 카메라가 원점(로봇 받침대 안)에
        # 있는 것으로 그려진다(첫 시험 데이터셋에서 실제로 그랬다). mj_forward는 둘 다 한다.
        mujoco.mj_forward(self.model, d)
        return self.render_data(d)

    def render_data(self, d: mujoco.MjData) -> dict[str, np.ndarray]:
        """지금 장면 상태 그대로 카메라 화면을 그린다. 데이터셋 만들 때와 정책 평가 때 같은 함수를 쓴다."""
        out = {}
        for key, cam in CAMERAS.items():
            self.r.update_scene(d, camera=cam)
            self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            out[key] = self.r.render().copy()
        return out

    def close(self) -> None:
        self.r.close()


def write_episode(ds, renderer: SceneRenderer, ep: Episode) -> None:
    for t in range(len(ep.qpos)):
        imgs = renderer.frames(ep, t)
        frame = {"observation.state": ep.qpos[t].astype(np.float32),
                 "action": ep.action[t].astype(np.float32), "task": TASK_TEXT}
        for key, img in imgs.items():
            frame[f"observation.images.{key}"] = img
        ds.add_frame(frame)
    ds.save_episode()


def resample(t_s: np.ndarray, values: dict[str, np.ndarray], fps: int = FPS) -> dict[str, np.ndarray]:
    """시각이 고르지 않은 기록을 1/fps 간격으로 다시 뽑는다(각 시각 직전 기록값 사용)."""
    grid = np.arange(t_s[0], t_s[-1] + 1e-9, 1.0 / fps)
    idx = np.clip(np.searchsorted(t_s, grid, side="right") - 1, 0, len(t_s) - 1)
    return {k: v[idx] for k, v in values.items()}


def default_cube_quat(cube_xy: np.ndarray) -> np.ndarray:
    yaw = yaw_facing_robot(cube_xy)
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

