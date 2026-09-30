# webcam-teach-robot

**웹캠으로 보여 준 시범만으로 시뮬레이션 로봇팔(SO-101)을 가르치고, 웹캠의 깊이 오차가 모방 학습 성공률을 얼마나 깎아 먹는지 재는 프로젝트.**

*Teaching a simulated SO-101 arm from webcam hand demonstrations, and measuring how monocular depth error affects imitation-learning success.*

- 쉬운 설명: [프로젝트_소개.md](프로젝트_소개.md)
- 관련 연구 조사: [docs/00-research.md](docs/00-research.md)
- 진행 상황: [docs/PROGRESS.md](docs/PROGRESS.md)

## 상태

진행 중 (1단계: 웹캠 → 시뮬레이션 SO-101 조종).

## 실행

[uv](https://docs.astral.sh/uv/) 설치 후:

```
uv run python scripts/view_so101.py
```

## 라이선스 · 출처

- SO-101 모델: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `robotstudio_so101` (Apache-2.0). 자세한 내용은 [third_party/README.md](third_party/README.md).
- 사용 라이브러리: MuJoCo, MediaPipe (Apache-2.0).
- 손 추적 모델: MediaPipe Hand Landmarker `hand_landmarker.task` (Apache-2.0, 모델 카드에 명시). 실행 시 자동으로 내려받고 저장소에는 넣지 않는다.
- 이 프로젝트는 Claude Code와 함께 만들고 있다.
