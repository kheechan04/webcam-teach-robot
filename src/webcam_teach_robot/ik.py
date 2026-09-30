"""SO-101 역기구학(IK): "집게 끝을 여기에" → 관절 각도.

방법: 감쇠 최소제곱(damped least squares, DLS)을 여러 번 반복한다.

1. 지금 관절 각도에서 집게 끝이 어디 있는지 계산한다 (순기구학, FK).
2. 목표와의 차이(오차)를 구한다.
3. 야코비안 J를 구한다. J는 "각 관절을 아주 조금 돌리면 집게 끝이 어느 쪽으로 얼마나 움직이나"를 모은 표다.
   (view_so101로 관절마다 +20°씩 돌려 본 표를 아주 작은 각도로 만든 것과 같다.)
4. 오차를 줄이는 방향으로 관절을 조금 돌린다: dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e  (dls_pinv)
   λ(감쇠)는 팔이 쭉 펴져서 어떤 방향으로는 못 움직이는 자세(특이점) 근처에서 관절이 튀지 않게 막는다.
5. 오차가 충분히 작아질 때까지 반복한다.

SO-101은 팔 관절이 5개라서 집게 끝 위치(3) + 방향(3)을 전부 맞출 수 없다.
그래서 우선순위를 둔다(task-priority IK).
    1순위: 집게 끝 위치. 관절 4개로 위치 3개를 맞추면 자유도가 1개 남는다.
    2순위: 집게가 향하는 방향(예: 아래). 남은 자유도 1개로, 위치를 건드리지 않는 범위에서만 맞춘다.
"위치를 건드리지 않는 관절 움직임"을 영공간(null space)이라고 부른다. 팔꿈치를 굽히면서 손목을 반대로 펴면
집게 끝은 제자리에 있고 방향만 바뀌는 것 같은 움직임이다.
(처음엔 위치·방향을 가중치로 섞었는데, 높은 곳에서 둘이 서로 양보해서 위치가 최대 8.7 cm 틀어졌다.)
wrist_roll(집게 돌리기)은 위치에도 향하는 방향에도 영향이 없어서 IK에서 빼고 따로 정한다.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
TIP_SITE = "gripperframe"  # 이 site의 x축이 집게가 향하는 방향


@dataclass
class IKResult:
    q: np.ndarray  # 모델 전체 qpos (IK 관절만 바뀜)
    pos_err_m: float
    dir_err_deg: float
    iters: int


def dls_pinv(J: np.ndarray, damping: float) -> np.ndarray:
    """감쇠 의사역행렬: Jᵀ (J Jᵀ + λ² I)⁻¹."""
    return J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(J.shape[0]))


class SO101IK:
    def __init__(self, model: mujoco.MjModel, damping: float = 0.05,
                 max_iters: int = 30, tol_m: float = 1e-3, max_step_rad: float = 0.2):
        self.model = model
        self.data = mujoco.MjData(model)  # 계산용 복사본. 시뮬레이션 상태는 건드리지 않는다
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dofadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0]
        self.hi = model.jnt_range[jids, 1]
        self.damping = damping
        self.max_iters = max_iters
        self.tol_m = tol_m
        self.max_step_rad = max_step_rad
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def tip(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """순기구학: 관절 각도 → (집게 끝 위치, 집게가 향하는 방향 단위벡터)."""
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3)[:, 0].copy()

    def _priority_step(self, Jp, Jr, e_pos, e_dir, use_dir: bool) -> np.ndarray:
        # 1순위: 위치
        Jp_pinv = dls_pinv(Jp, self.damping)
        dq = Jp_pinv @ e_pos
        # 2순위: 방향. 위치를 안 바꾸는 움직임(영공간 N)만 쓴다
        if use_dir:
            # 영공간은 감쇠 없는 정확한 의사역행렬로 만든다. 감쇠 넣은 걸 쓰면 방향 보정이 위치로 새어 들어가서
            # 위치가 1순위인데도 위치가 틀어진다(실제로 격자 테스트에서 최대 12 cm 틀어짐).
            N = np.eye(Jp.shape[1]) - np.linalg.pinv(Jp, rcond=1e-4) @ Jp
            dq = dq + N @ dls_pinv(Jr @ N, self.damping) @ (e_dir - Jr @ dq)
        return dq

    def solve(self, target_pos: np.ndarray, q_init: np.ndarray,
              target_dir: np.ndarray | None = np.array([0.0, 0.0, -1.0])) -> IKResult:
        q = q_init.copy()
        for it in range(1, self.max_iters + 1):
            pos, direction = self.tip(q)
            e_pos = target_pos - pos
            # 방향 오차: 지금 방향을 목표 방향으로 돌리는 회전축×각도 (외적으로 근사)
            e_dir = np.cross(direction, target_dir) if target_dir is not None else np.zeros(3)
            if np.linalg.norm(e_pos) < self.tol_m and np.linalg.norm(e_dir) < 1e-2:
                break

            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
            Jp = self._jp[:, self.dofadr]  # (3, 4) 위치 야코비안
            Jr = self._jr[:, self.dofadr]  # (3, 4) 회전 야코비안

            # 관절 한계에 걸리는 관절은 이번 걸음에서 빼고(열을 0으로) 다시 푼다.
            # 그냥 잘라 내면 2순위(방향)가 관절을 한계로 밀 때 1순위(위치)가 깨진다.
            qa = q[self.qadr]
            free = np.ones(len(self.qadr), bool)
            for _ in range(len(self.qadr)):
                dq = self._priority_step(Jp * free, Jr * free, e_pos, e_dir, target_dir is not None)
                over = free & ((qa + dq < self.lo) | (qa + dq > self.hi))
                if not over.any():
                    break
                free &= ~over
            dq = np.clip(dq, -self.max_step_rad, self.max_step_rad)
            q[self.qadr] = np.clip(qa + dq, self.lo, self.hi)
            if np.linalg.norm(e_pos) < self.tol_m and np.linalg.norm(dq) < 1e-4:
                break  # 위치는 맞췄고 방향은 더 나아지지 않음

        pos, direction = self.tip(q)
        dir_err = 0.0 if target_dir is None else np.degrees(np.arccos(np.clip(direction @ target_dir, -1, 1)))
        return IKResult(q=q, pos_err_m=float(np.linalg.norm(target_pos - pos)), dir_err_deg=float(dir_err), iters=it)
