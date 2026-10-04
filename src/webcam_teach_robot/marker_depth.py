"""조건 ⑤: 손목에 붙인 인쇄 마커(ArUco)로 손 위치를 읽는다 — 같은 웹캠, 깊이 오차만 거의 없앤 조건.

MediaPipe 손바닥 크기로 거리를 추정하면 손 모양(집기·기울임)에 따라 치우친다(M5: 집고 기울인 손 +10~12 cm).
크기를 정확히 아는 단단한 평면 마커는 네 꼭짓점으로 자세까지 풀어서(solvePnP) 기울어도 거리를 맞힌다.
조건 ⑤에서는 손 위치(카메라 좌표 x, y, z)를 전부 마커에서 가져오고, 집기(엄지-검지)만 MediaPipe로 읽는다.

마커: docs/marker/wrist_marker_A4.pdf (ArUco 4x4_50, id 0, 한 변 50 mm). 손목 안쪽, 카메라를 향하게.
주의: ArUco는 거울상을 다른 무늬로 본다. 그래서 좌우를 뒤집기 **전** 원본 프레임에서 찾고, 결과를 뒤집은 화면 좌표로 바꾼다.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from webcam_teach_robot.hand_tracking import focal_length_px

MARKER_SIZE_M = 0.050
MARKER_ID = 0


@dataclass
class MarkerObservation:
    point_cam: np.ndarray  # (3,) 마커 중심, 카메라 좌표(m), 화면을 좌우로 뒤집은 기준(x 오른쪽, y 아래, z 멀어지는 쪽)
    center_px: np.ndarray  # (2,) 뒤집은 화면에서의 픽셀 위치
    corners_px: np.ndarray  # (4, 2) 뒤집은 화면 기준
    tilt_deg: float  # 마커 면이 카메라 정면에서 기운 각도
    side_px: float  # 사진 속 한 변 평균 길이


class MarkerTracker:
    def __init__(self, frame_w: int = 640, frame_h: int = 480, fov_deg: float = 60.0, size_m: float = MARKER_SIZE_M):
        f = focal_length_px(frame_w, fov_deg)  # MediaPipe 깊이와 같은 가정값(화각 60°)
        self.K = np.array([[f, 0, frame_w / 2], [0, f, frame_h / 2], [0, 0, 1]], dtype=np.float64)
        self.w = frame_w
        s = size_m / 2
        # IPPE_SQUARE가 요구하는 순서: 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래 (ArUco 꼭짓점 순서와 같음)
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)

    def detect(self, frame_bgr_unflipped: np.ndarray) -> MarkerObservation | None:
        gray = cv2.cvtColor(frame_bgr_unflipped, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None or MARKER_ID not in ids.ravel():
            return None
        c = corners[list(ids.ravel()).index(MARKER_ID)].reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(self.obj, c, self.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        tilt = float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0, 1))))
        t = tvec.ravel()
        flipped = c.copy()
        flipped[:, 0] = self.w - 1 - flipped[:, 0]
        side = float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))
        return MarkerObservation(point_cam=np.array([-t[0], t[1], t[2]]), center_px=flipped.mean(axis=0),
                                 corners_px=flipped, tilt_deg=tilt, side_px=side)

    @staticmethod
    def draw(frame_flipped: np.ndarray, m: MarkerObservation | None) -> None:
        if m is None:
            return
        pts = m.corners_px.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame_flipped, [pts], True, (255, 160, 0), 2)
        cv2.circle(frame_flipped, tuple(m.center_px.astype(int)), 5, (255, 160, 0), -1)
