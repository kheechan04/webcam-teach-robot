"""웹캠 프레임에서 손 위치를 읽는다 (MediaPipe Hand Landmarker).

좌표 두 종류를 같이 쓴다.
- 화면 좌표 (landmarks): 사진 위 위치. x, y는 0~1로 정규화, z는 손목 기준 상대 깊이일 뿐이라 카메라까지 거리가 아니다.
- 월드 좌표 (world_landmarks): 미터 단위지만 원점이 손 한가운데라서, 손 모양(마디 사이 길이)은 알려 주지만 카메라까지 거리는 모른다.

그래서 카메라까지 거리(깊이)는 따로 추정해야 한다. 여기서는 핀홀 카메라 관계를 쓴다.
    깊이 = 초점거리(픽셀) × 실제 길이(m) / 사진 속 길이(픽셀)
멀리 있는 물체가 작게 보이는 원리다. 실제 길이는 월드 좌표에서, 사진 속 길이는 화면 좌표에서 가져온다.

어느 구간 길이를 쓰느냐에 따라 오차가 다르다 (docs/01-depth-measurement.md).
    "length": 손목~가운뎃손가락 뿌리. 손바닥을 앞뒤로 기울이면 크게 틀린다(예비 측정 평균 +12.7 cm)
    "width" : 검지 뿌리~새끼 뿌리. 앞뒤 기울기에 덜 민감하다(같은 측정 +4.9 cm).
              대신 손을 옆으로 돌리면 너비가 짧아 보여서 크게 틀린다(첫 조종 기록에서 2.4% 프레임이 100 cm 넘음)
    "min"   : 둘 중 작은 값. 기울이거나 돌리면 사진 속 길이가 짧아져서 항상 "멀다" 쪽으로 틀리므로,
              덜 짧아진 구간(= 더 가깝게 나온 값)을 고른다. 기본값
"""

from dataclasses import dataclass
from pathlib import Path
import math
import urllib.request

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5  # 검지 뿌리 마디
INDEX_TIP = 8
MIDDLE_MCP = 9  # 가운뎃손가락 뿌리 마디
PINKY_MCP = 17  # 새끼 뿌리 마디
PALM_POINTS = [0, 5, 9, 13, 17]  # 손바닥 중심을 구할 때 쓰는 점들
DEPTH_SEGMENTS = {"length": (WRIST, MIDDLE_MCP), "width": (INDEX_MCP, PINKY_MCP)}

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(exist_ok=True)
        print(f"손 추적 모델 내려받는 중: {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def focal_length_px(image_width: int, horizontal_fov_deg: float) -> float:
    """화각(FOV)으로 초점거리를 픽셀 단위로 구한다. 보정(calibration) 전 임시값."""
    return (image_width / 2) / math.tan(math.radians(horizontal_fov_deg) / 2)


@dataclass
class HandObservation:
    pixels: np.ndarray  # (21, 2) 사진 위 위치, 픽셀
    rel_z: np.ndarray  # (21,) MediaPipe 화면 z (손목 기준 상대값, 단위 없음)
    world: np.ndarray  # (21, 3) 손 한가운데 기준, 미터
    handedness: str  # "Left" / "Right" (MediaPipe 기준)
    palm_px: float  # 손목~가운뎃손가락 뿌리, 사진 속 길이(픽셀)
    palm_m: float  # 같은 구간의 실제 길이(미터, 월드 좌표에서)
    depth_len_m: float  # 손바닥 길이로 추정한 카메라~손 거리(미터)
    depth_width_m: float  # 손바닥 너비로 추정한 거리(미터)
    depth_m: float  # 둘 중 선택한 방법의 값
    palm_center_px: np.ndarray  # (2,) 손바닥 중심, 사진 위 픽셀
    pinch: float  # 엄지 끝~검지 끝 거리 / 손바닥 너비 (월드 좌표, 거리와 무관). 작을수록 집은 상태


class HandTracker:
    def __init__(self, horizontal_fov_deg: float = 60.0, depth_segment: str = "min"):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.horizontal_fov_deg = horizontal_fov_deg
        self.depth_segment = depth_segment

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> HandObservation | None:
        h, w = frame_rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None

        lms = result.hand_landmarks[0]
        pixels = np.array([[p.x * w, p.y * h] for p in lms])
        rel_z = np.array([p.z for p in lms])
        world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]])

        f = focal_length_px(w, self.horizontal_fov_deg)

        def seg_depth(a: int, b: int) -> tuple[float, float, float]:
            l_px = float(np.linalg.norm(pixels[a] - pixels[b]))
            l_m = float(np.linalg.norm(world[a] - world[b]))
            return l_px, l_m, (f * l_m / l_px if l_px > 1 else float("nan"))

        palm_px, palm_m, depth_len = seg_depth(*DEPTH_SEGMENTS["length"])
        width_px, width_m, depth_width = seg_depth(*DEPTH_SEGMENTS["width"])
        depth_m = {"width": depth_width, "length": depth_len}.get(self.depth_segment, min(depth_len, depth_width))
        pinch = float(np.linalg.norm(world[THUMB_TIP] - world[INDEX_TIP]) / max(width_m, 1e-6))

        return HandObservation(
            pixels=pixels,
            rel_z=rel_z,
            world=world,
            handedness=result.handedness[0][0].category_name,
            palm_px=palm_px,
            palm_m=palm_m,
            depth_len_m=depth_len,
            depth_width_m=depth_width,
            depth_m=depth_m,
            palm_center_px=pixels[PALM_POINTS].mean(axis=0),
            pinch=pinch,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
