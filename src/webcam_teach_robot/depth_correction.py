"""조건 ③ 깊이 보정: 핀홀 거리 × 손 자세로 예측한 보정 계수 (RootNet식, docs/08-depth-correction.md).

M5 측정에서 남는 가장 큰 오차는 손 자세에 따른 일정한 치우침이었다(집은 손·기울임 +10~12 cm, 항상 멀다 쪽).
조종 중에 매 프레임 얻을 수 있는 손 자세 특징으로 log(참값 / 핀홀 거리)를 예측해서 곱한다.
계수는 scripts/fit_depth_correction.py가 M5 1회차로 맞춰 depth_correction.json에 저장한다.
조건 ②에서는 쓰지 않는다(모든 웹캠 조건 공통 처리는 teleop_mapping.py).
"""

import json
from pathlib import Path

import numpy as np

COEF_FILE = Path(__file__).with_name("depth_correction.json")
FEATURE_NAMES = ["log_d", "pinch", "foreshorten", "normal_z", "u", "v"]


def posture_features(pixels: np.ndarray, world: np.ndarray, d_pinhole: float, frame_w: int, frame_h: int) -> list[float]:
    """pixels (21, 2) 픽셀, world (21, 3) MediaPipe 월드 좌표(m), d_pinhole 핀홀 거리(m).
        log_d       핀홀 거리(로그). 거리가 멀수록 더 멀게 읽히는 경향(M5: 65 cm에서 +9.6 cm)
        pinch       엄지-검지 거리 / 손바닥 너비. 집은 손일수록 작다
        foreshorten 사진 속 (길이/너비) ÷ 3D (길이/너비) 의 로그. 손바닥을 앞뒤로 기울이면 사진 속 길이가 더 짧아진다
        normal_z    손바닥 법선의 카메라 방향 성분 크기(월드 좌표). 손바닥이 카메라를 정면으로 볼수록 1
        u, v        손바닥 중심의 화면 위치(가운데 0, 가장자리 ±0.5). 가장자리일수록 렌즈 왜곡·비스듬한 시선
    """
    len_px = np.linalg.norm(pixels[0] - pixels[9])
    wid_px = np.linalg.norm(pixels[5] - pixels[17])
    len_m = np.linalg.norm(world[0] - world[9])
    wid_m = np.linalg.norm(world[5] - world[17])
    pinch = np.linalg.norm(world[4] - world[8]) / max(wid_m, 1e-6)
    n = np.cross(world[5] - world[0], world[17] - world[0])
    n_z = abs(n[2]) / max(np.linalg.norm(n), 1e-9)
    c = pixels[[0, 5, 9, 13, 17]].mean(axis=0)
    return [float(np.log(d_pinhole)), float(min(pinch, 1.6)),
            float(np.log((len_px / max(wid_px, 1e-6)) / max(len_m / max(wid_m, 1e-6), 1e-6))),
            float(n_z), float(c[0] / frame_w - 0.5), float(c[1] / frame_h - 0.5)]


class DepthCorrection:
    def __init__(self, path: Path = COEF_FILE):
        coef = json.loads(Path(path).read_text(encoding="utf-8"))
        assert coef["features"] == FEATURE_NAMES, "특징 목록이 바뀌었다 — 다시 맞춰야 한다"
        self.mean, self.std = np.array(coef["mean"]), np.array(coef["std"])
        self.w, self.b = np.array(coef["w"]), coef["b"]
        self.resolution = tuple(coef["resolution"])
        # 맞출 때 본 보정 범위 밖으로는 나가지 않는다(측정하지 않은 자세·거리에서 엉뚱하게 크게 고치지 않게)
        self.log_lo, self.log_hi = coef["log_factor_range"]

    @staticmethod
    def coef_text(path: Path = COEF_FILE) -> str:
        return Path(path).read_text(encoding="utf-8")

    def factor(self, pixels: np.ndarray, world: np.ndarray, d_pinhole: float, frame_w: int, frame_h: int) -> float:
        x = np.array(posture_features(pixels, world, d_pinhole, frame_w, frame_h))
        return float(np.exp(np.clip(((x - self.mean) / self.std) @ self.w + self.b, self.log_lo, self.log_hi)))

    def __call__(self, pixels, world, d_pinhole, frame_w, frame_h) -> float:
        """보정한 거리(m)."""
        return d_pinhole * self.factor(pixels, world, d_pinhole, frame_w, frame_h)
