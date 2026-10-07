# webcam-teach-robot

**웹캠 한 대로 보여 준 시범으로 시뮬레이션 로봇팔(SO-101)을 가르치고, 웹캠의 깊이 오차가 모방 학습 결과에 무엇을 하는지 잰 프로젝트.**

*Teaching a simulated SO-101 arm from webcam hand demonstrations, and measuring what monocular depth error does to imitation learning.*

- 쉬운 설명: [프로젝트_소개.md](프로젝트_소개.md) · 진행 기록: [docs/PROGRESS.md](docs/PROGRESS.md)
- 결과 정리: [docs/11-m9-results.md](docs/11-m9-results.md)

## 질문

보통 SO-100/101 로봇팔은 조종용 팔을 하나 더 두고 시범을 보여 가르친다. 노트북 웹캠만으로 손을 추적해 시범을 모으면 장비는 싸지만, 한 대의 카메라는 손까지의 거리(깊이)를 손 크기로 짐작해야 해서 틀린다. **이 깊이 오차가 학습한 로봇의 성공률을 얼마나 깎는지, 보정하면 얼마나 돌아오는지**를 잰다.

웹캠 조종 → 시뮬레이션 → ACT 학습 파이프라인 자체는 이미 있었다(예: handrobot). 찾은 범위에서(2026-10-03, 관련 논문 3편을 인용한 487편 포함) 단안 카메라 조종이 **조종 성능**에서 손해라는 비교(AnyTeleop, TeleOpBench)는 있었지만, 그 시범으로 **학습한 로봇의 성공률**과 깊이 오차의 몫을 잰 연구는 찾지 못했다 — [docs/00-research.md](docs/00-research.md), [docs/07-related-work-v2.md](docs/07-related-work-v2.md).

## 실험

과제: 3 cm 큐브를 집어 표시된 목표 위에 내려놓기(MuJoCo, SO-101). 시범용 배치 50개와 평가용 배치 100개는 고정. 정책은 LeRobot ACT 기본값.

| 조건 | 시범 | 깊이 |
|---|---|---|
| ① | 스크립트(정답 위치를 아는 프로그램) | — |
| ② | 사람, 웹캠 | MediaPipe 손 크기로 추정 (오차 그대로) |
| ③ | 사람, 웹캠 | ② + 손 자세 특징으로 치우침 보정 (CPU 실시간) |
| ④ / ④0 | 가상 조작자(②·③과 같은 조종 코드) | 실측 깊이 오차 주입 / 오차 없음 |
| ⑤ | 사람, 웹캠 + 손목 인쇄 마커 | 마커로 측정 (자세에 따른 오차 거의 없음) |

**재현 과제(쌓기):** 빨간 큐브를 초록 받침 블록(4 cm) 위에 올리기. ①·②·③만, ②·③ 시범 50개씩. 녹화 전에 시드 수·검정·판정 기준을 고정했다([docs/13-stacking-plan.md](docs/13-stacking-plan.md)).

시범 녹화: 시범자 1이 270개(옮기기 ② 70·③ 50·⑤ 50, 쌓기 ②·③ 100), 시범자 2가 옮기기 ②·③ 20번씩, 녹화 중 조건은 화면에 표시하지 않음, 연습 효과를 상쇄하려고 블록 단위로 번갈아 녹화. 웹캠 영상은 저장하지 않고, 데이터셋 이미지는 기록된 숫자로 다시 그린 시뮬레이션 카메라 화면뿐이다.

## 결과 (요약, 자세한 표와 한계는 [docs/11-m9-results.md](docs/11-m9-results.md))

**연구 질문에 대한 답**

1. **웹캠 시범은 학습 성공률을 얼마나 깎나:** 충분히 학습하면 스크립트 시범보다 약 17%p 낮다(옮기기). 학습이 짧으면 33%p로 부풀어 보인다.
2. **그중 깊이 오차의 몫은:** 깊이 오차는 원인이다. 사람 없이 같은 조종 코드에 오차만 넣고 빼면 모든 시드에서 손해(약 7%p, 구간 −12 \~ −3)였고, 오차를 키울수록 성공률이 떨어졌다. 사람 시범에서 손목 마커로 깊이를 정확히 쟀을 때는 약 9%p 높았다(시드 3개 모두 같은 방향, 구간 −1 \~ +20).
3. **보정하면 회복되나:** 노트북 CPU로 도는 손 자세 보정은 **과제마다 결과가 달랐다**. 옮기기에서는 +7.8%p였지만, 미리 계획을 고정한 쌓기에서는 −8.2%p였다. 쌓기에서는 프레임의 63%에서 보정이 맞춰 본 손 자세 범위 밖에서 쓰였다(사후 분석).


처음 보는 평가 배치 100개 중 성공, 10만 스텝 학습(②·③ 시드 5개, 나머지 3개 평균):

| ④0 가상, 오차 없음 | ① 스크립트 | ④ 가상 + 오차 | ⑤ 웹캠 + 마커 | ③ 웹캠 + 보정 | ② 웹캠 |
|---|---|---|---|---|---|
| 96.7 | 92.7 | 89.3 | 85.3 | 84.4 | 76.6 |

- **깊이 측정:** 큐브를 집으러 갈 때의 손(엄지·검지를 붙이고 기울인 손)이 평균 10\~12 cm 멀게 읽혔다(참값과 비교한 70단계, [docs/06-depth-formal.md](docs/06-depth-formal.md)). 손목 마커는 같은 자세에서 −0.9\~+1.2 cm([docs/12-marker.md](docs/12-marker.md)).
- **학습량을 맞추자 결론이 바뀌었다:** 웹캠 시범은 길어서 같은 스텝이면 덜 배운다. 2만 스텝에서 본 웹캠 손해(② − ① −33%p) 대부분은 학습량 부족이었고, 충분히 학습하면 **약 17%p**(구간 −30 \~ −5)다.
- **그중 깊이 오차의 몫은 약 9%p**(⑤ − ②, 시드 3개 모두 같은 방향이지만 구간 −1 \~ +20으로 0을 살짝 걸침), 나머지 약 7%p는 깊이가 정확해도 남는 사람 시범의 특성(⑤ − ①)이다.
- **깊이 오차는 원인이다:** 사람 없이 깊이 오차만 넣고 뺀 가상 조작자 비교에서 세 시드 모두 손해(④ − ④0 −7.3, 구간 −12 \~ −3), 오차 크기 0 / 0.5 / 1 / 2배에 성공률 88 / 87 / 70 / 59(2만 스텝).
- **조종 성능으로는 보이지 않는다:** 녹화할 때 ②와 ③은 성공률·시간 차이가 없었지만(50/50 vs 50/50), 학습한 로봇에서는 차이가 났다.
- **왜:** 깊이 오차가 있으면 집기 직전의 앞뒤(웹캠 깊이 방향) 움직임이 큐브 위치와 거의 상관없어지고, 학습한 로봇은 앞뒤로 어긋난 곳에서 집게를 닫아 큐브를 민다.
- **보정은 재현되지 않았다:** 노트북 CPU로 도는 손 자세 보정(③)은 옮기기에서 시드 5개 모두 ②보다 높았다(평균 +7.8%p, 양쪽 p = 0.06). 그런데 녹화 전에 계획을 고정하고 돌린 **쌓기 과제에서는 다섯 시드 중 넷에서 오히려 낮았다**(③ − ② = −8.2%p, 양쪽 p = 0.125, [docs/13-stacking-plan.md](docs/13-stacking-plan.md)). 두 과제를 합치면 효과가 없다(평균 −0.2, 한쪽 p = 0.53). 그래서 "보정이 학습 성공률을 올린다"는 일반적인 결론으로 쓰지 않는다. 시범자 2에게는 시범자 1의 보정이 효과가 없었다(54.0 vs 54.4).

한계: 시범자 2명(오른손), 과제 2개(옮기기, 쌓기는 보정 효과 재현용), 시뮬레이션만.

## 실행

[uv](https://docs.astral.sh/uv/) 설치 후 (Python 3.12, `.python-version`):

```
uv run python scripts/teleop.py                       # 웹캠으로 조종해 보기 (영상 저장 안 함)
uv run python scripts/teleop.py --depth-correction    # 조건 ③ 깊이 보정 켜기
uv run python scripts/record_demos.py                 # 시범 녹화 (조건 ②·③ 번갈아, 이어 하기 지원)
uv run python scripts/build_dataset.py scripted --name cond1_scripted   # 조건 ① 데이터셋
uv run python scripts/eval_policy.py <Hub 모델 또는 로컬 폴더> --n 100   # 정책 평가
```

학습은 GPU 대여 서버에서 `runpod/run_m9.sh`(학습 + 평가 + 결과 업로드, 할 일 목록 `runpod/jobs_*.txt`). 분석: `scripts/analyze_m9.py`, `scripts/analyze_action_divergence.py`, `scripts/probe_policy_grasp.py`.

| 폴더 | 내용 |
|---|---|
| `src/webcam_teach_robot/` | 손 추적, 역기구학, 조종 처리, 깊이 보정, 손목 마커, 가상 조작자, 과제 장면, 데이터셋 |
| `scripts/` | 측정·조종·녹화·데이터셋·평가·분석 |
| `measurements/` | 깊이 측정·조종·시범 녹화 기록 (숫자만) |
| `experiments/` | 배치 목록, 평가 결과, 분석 요약 |
| `docs/` | 단계별 기록 |

## 라이선스

개발 중이라 아직 라이선스를 정하지 않았다(모든 권리 보유: 코드를 볼 수는 있지만 복사·수정·재배포는 허락하지 않는다). 공개할 때 코드는 Apache-2.0, 데이터셋은 CC BY 4.0으로 바꿀 예정이다. `third_party/` 안의 SO-101 모델과 내려받아 쓰는 모델은 각자의 라이선스(Apache-2.0)를 따른다.

## 출처

- SO-101 모델: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `robotstudio_so101` (Apache-2.0). 받은 커밋과 라이선스 파일은 [third_party/README.md](third_party/README.md).
- 손 추적 모델: MediaPipe Hand Landmarker `hand_landmarker.task` (Apache-2.0, 모델 카드에 명시). 실행 시 자동으로 내려받고 저장소에는 넣지 않는다.
- 정책·학습: [LeRobot](https://github.com/huggingface/lerobot) 0.6.1 ACT (Apache-2.0); Zhao et al., Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (RSS 2023).
- 사용 라이브러리: [MuJoCo](https://github.com/google-deepmind/mujoco)·[MediaPipe](https://github.com/google-ai-edge/mediapipe)·OpenCV (ArUco 포함, Apache-2.0), NumPy (BSD-3-Clause 등), Pillow (MIT-CMU), matplotlib (PSF 계열).
- 깊이 보정 구조의 근거: Moon et al., RootNet (ICCV 2019).
- 코드는 Claude Code(AI 코딩 도구)를 써서 작성했다.
