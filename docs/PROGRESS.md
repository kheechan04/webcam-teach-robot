# 진행 상황

> 노트북을 껐다 켜도 여기서부터 이어 간다. 작업 단위가 끝날 때마다 갱신하고 커밋한다.

## 지금 상태 (2026-10-01)

- 단계: **1단계 진행 중**
- 완료
  - 0단계 조사, 방향 결정 (`docs/00-research.md`)
  - 개발 환경: uv + Python 3.12 (`.python-version`), mujoco 3.14.0, mediapipe 1.0.1 설치 확인
  - GitHub 공개 저장소: https://github.com/kheechan04/webcam-teach-robot
  - SO-101 모델(MuJoCo Menagerie) 가져와서 `third_party/`에 두고 렌더링 확인 (`scripts/view_so101.py`)
- 진행 중인 것: 없음 (깨끗한 상태)

## 다음 할 일

1. ~~SO-101 창 띄워 관절 직접 움직여 보기~~ 완료: 집게 끝을 내리는 관절 = shoulder_lift, elbow_flex, wrist_flex (셋 다 +방향). 각 관절 +20° 때 z 변화 -11.8 / -10.2 / -5.5 cm (FK 계산값)
2. 웹캠 + MediaPipe로 손 위치 읽기 (영상은 저장하지 않음) — 코드 완료 `scripts/hand_tracking_demo.py`, 빈 이미지로 모델 로드 확인(추론 ~11 ms, 손 없을 때). 데모는 선택 사항
3. 깊이 오차 측정: `scripts/measure_depth.py` 완성. 거리 30~70cm(10cm 간격) × 자세 2개(flat, tilt), 단계마다 60프레임. 결과는 `measurements/depth/*.csv, *.json` (숫자만, 커밋함). **예비 측정 완료** → 결과 `docs/01-depth-measurement.md`. 펴기 평균 오차 −2.7 cm, 기울이기 +12.7 cm(떨림 8배). 원인: MediaPipe 월드 좌표가 기울이면 손바닥을 6.5~7 cm로 눌러 봄. 손바닥 너비로 추정하면 기울이기 오차 +4.9 cm
4. 역기구학: 손 위치 → SO-101 관절 각도

## 결정 기록

| 날짜 | 결정 | 이유 |
|---|---|---|
| 2026-10-01 | 차별점 A(깊이 문제), SO-101, RunPod | `docs/00-research.md` 4절 |
| 2026-10-01 | Python 3.12 + uv | LeRobot 0.6.1이 Python ≥3.12 요구. 기존 3.11은 그대로 둠 |
| 2026-10-01 | SO-101 모델을 저장소에 그대로 복사(vendor) | 커밋 해시를 고정해 재현성 확보. Apache-2.0이라 라이선스 파일 포함하면 재배포 가능 |

## 포트폴리오 반영 기록

| 날짜 | 반영한 내용 |
|---|---|
| (아직 없음) | 1단계 마일스톤(웹캠으로 시뮬 SO-101 조종 + 깊이 오차 측정값)이 나오면 첫 반영 |
