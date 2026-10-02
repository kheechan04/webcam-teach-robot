# 0단계 조사 기록

조사일: 2026-10-01. 수치는 각 저장소 README/공식 문서에 적힌 값을 그대로 옮긴 것이고, 직접 재현한 값이 아니다.

## 1. 기존 프로젝트: 웹캠·손 포즈로 로봇팔 조종

| 프로젝트 | 무엇을 하나 | 시뮬/실물 | 학습까지? | 라이선스 | 비고 |
|---|---|---|---|---|---|
| [guptabhishekumar/handrobot](https://github.com/guptabhishekumar/handrobot) | 웹캠 손 → MuJoCo 로봇팔 조종 → 시범 녹화 → ACT 학습 → 평가 | 시뮬만 (Panda 기본, SO-101 옵션) | O. README 기준 단일 과제 95% (38/40), 30Hz | MIT | **우리 계획 1~3단계와 거의 같다.** 스스로 적은 한계: "Monocular depth is the weak link", 실물 미배포 |
| [steven-tired/mediapipe-so101](https://github.com/steven-tired/mediapipe-so101) | 웹캠 손목 포즈 → IK(placo) → 실물 SO-101, LeRobot v3.0 데이터셋 10Hz | 실물만 | O. ACT/Diffusion/SmolVLA 시도, "not approved for autonomous deployment" | Apache-2.0 | 깊이: 단안 스케일 가정 또는 OAK-D 스테레오. 조작자 손 영상은 녹화 안 함 |
| [Joeclinton1/hand-teleop](https://github.com/Joeclinton1/hand-teleop) | 웹캠 → 관절 각도, LeRobot용 | 실물 | X | Apache-2.0 | 별 50. 주 백엔드 WiLoR(GPU 필요), MediaPipe 백엔드는 "almost working". LeRobot 최신 버전과 호환 안 됨, 본체에 합쳐지지 않음 |
| [MattiArlo/so101-hand-teleop](https://github.com/MattiArlo/so101-hand-teleop) | 양손 추적으로 SO-101 조종 | 둘 다 | X | Apache-2.0 | 깊이: 손 크기 정규화 + 밀고 당기는 제스처 |
| [edangelux/so-arm100-teleop](https://github.com/edangelux/so-arm100-teleop) | ROS 2 + Gazebo + MoveIt 2 + MediaPipe | 시뮬 | X | 미확인 | |
| AnyTeleop ([arXiv 2307.04577](https://arxiv.org/abs/2307.04577)) | 비전 기반 팔+손 원격 조종 일반 시스템 (논문) | 둘 다 | | | 관련 연구로 읽을 것 |

추가 조사에서 찾은 것:

| 프로젝트 | 내용 | 비고 |
|---|---|---|
| [ml-research-lab/video-to-robotics-motion](https://github.com/ml-research-lab/video-to-robotics-motion) | 웹캠 → MediaPipe → 차분 IK(mink) → MuJoCo SO-101 | MIT. handrobot을 단순화해 다시 만든 것이라고 밝힘. 녹화·학습은 "다음 단계"로 남겨 둠. 측정값 없음 |
| [niicoofdezz/hand-teleop-imitation](https://github.com/niicoofdezz/hand-teleop-imitation) | 웹캠 손 → MuJoCo + 모방 학습 | 개발 초기, 커밋 3개, 라이선스 없음 |
| [aviadarn/so101-lerobot](https://github.com/aviadarn/so101-lerobot), [RajatDandekar/sim-engine](https://github.com/RajatDandekar/sim-engine), [Younus-Elazzouzi/mujoco-so101-act](https://github.com/Younus-Elazzouzi/mujoco-so101-act) | MuJoCo SO-101 + ACT (스크립트 시범 등) | 웹캠 조종 아님. 시뮬 환경 참고용 |

**handrobot README 정밀 확인 결과 (중요):** 보고된 ACT 성공률(Panda 95% 38/40, SO-101 95% 19/20)은 **스크립트 시범 150개로 학습한 기준선**이다. README 원문: "A baseline policy trained only on scripted demonstrations, so that every number below is reproducible without a camera." 즉 **웹캠으로 사람이 보여 준 시범으로 학습했을 때의 성공률은 보고되어 있지 않다.** 깊이는 손의 한 마디 길이 + 핀홀 관계로 추정하고, 두 번째 웹캠 스테레오 옵션이 있다.

**결론: "웹캠으로 시범 → 시뮬 SO-10x → ACT 학습" 파이프라인 자체는 이미 있다.** 그러나 "웹캠 시범이 학습 결과를 얼마나 깎아 먹는가"를 잰 곳은 찾지 못했다.

## 1-2. 관련 논문 (데이터 품질 · 시범 방식)

| 논문 | 핵심 | 우리와의 관계 |
|---|---|---|
| Mandlekar et al., "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation", CoRL 2021 ([robomimic](https://robomimic.github.io/study/)) | 숙련도가 다른 조작자 6명의 시범(MH 데이터셋)으로 학습 비교 | 조작자 품질 → 성능. 입력 장치 오차는 다루지 않음 |
| Belkhale, Cui, Sadigh, "Data Quality in Imitation Learning", NeurIPS 2023 ([arXiv 2306.02437](https://arxiv.org/abs/2306.02437)) | 데이터 품질을 action divergence·transition diversity로 정식화. 같은 상태에서 행동이 일관되지 않으면 성능 하락 | 웹캠 오차가 "행동 비일관성"을 늘리는지 재 볼 수 있음 |
| Li, Cui, Sadigh, "How to Train Your Robots? The Impact of Demonstration Modality on Imitation Learning", ICRA 2025 ([arXiv 2503.07017](https://arxiv.org/abs/2503.07017)) | 키네스테틱 / VR / 스페이스마우스 비교. 키네스테틱이 가장 깨끗 | **웹캠 손 추적은 비교 대상에 없음** → 우리가 채울 수 있는 칸 |
| Kulkarni, Dhar, Cui, "Learning from the Best: Smoothness-Driven Metrics…", 2026 ([arXiv 2604.23000](https://arxiv.org/abs/2604.23000)) | 궤적 매끄러움으로 시범을 골라내면 적은 데이터로 성공률 상승 | 웹캠 떨림이 있는 시범을 걸러 내는 방법으로 응용 가능 |
| AnyTeleop (2023), Robotic Telekinesis ([arXiv 2202.10448](https://arxiv.org/abs/2202.10448)) | 단일 RGB 카메라로 조종하는 시스템 | 조종 시스템 쪽 관련 연구 |

**찾지 못한 것:** 단안 웹캠 조종의 깊이 오차를 측정하고, 그 오차가 모방 학습 성공률에 주는 영향을 통제된 실험으로 분리한 연구. (못 찾았다는 것이지 없다는 보장은 아니다.)

## 2. SO-100/101 시뮬레이션 환경

| 이름 | 내용 | 라이선스 |
|---|---|---|
| MuJoCo Menagerie `trs_so_arm100` | SO-100 MJCF 모델, 5DOF | Apache-2.0 |
| MuJoCo Menagerie `robotstudio_so101` | SO-101 MJCF 모델, MuJoCo 3.1.3 이상 | Apache-2.0 |
| [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | 원본 하드웨어 저장소, `Simulation` 폴더 있음 | Apache-2.0 |
| [isaka1022/so101-sim2real](https://github.com/isaka1022/so101-sim2real) | SO-101 pick-cube 환경, LeRobot v0.6.0+ 플러그인 | Apache-2.0 (Windows 설명 없음) |
| [ilonajulczuk/gym-so100-c](https://github.com/ilonajulczuk/gym-so100-c) | gym-aloha 방식 SO-100 환경 | 미확인 |
| LeRobot `gym_hil` | 공식 시뮬이지만 **Franka Panda** 중심, NVIDIA GPU 요구 | |

## 3. GPU 비용 (3단계용)

- LeRobot 하드웨어 가이드: ACT는 배치 8에서 VRAM ~2–6GB, L4/A10G로 50 에피소드 5 에폭에 ~1–2시간 (문서 스스로 ±50% 오차라고 함). CPU로는 학습하지 말라고 함.
- Hugging Face Jobs 가격(분 단위 과금): T4 small $0.40/h, L4 $0.80/h, A10G small $1.00/h.
- 예상(추정): L4로 20시간이면 $16. 실패한 실행을 넣어도 3만 원 안쪽일 가능성이 높다. 3단계에서 실제로 재고 갱신한다.

## 4. 결정 (2026-10-01)

- 방향: **A. 깊이 문제를 파고들기.** 연구 질문: "단안 웹캠으로 모은 시범은 모방 학습 성공률을 얼마나 깎아 먹고, 그중 깊이 오차의 몫은 얼마이며, 보정하면 얼마나 회복되는가?"
- 로봇: **SO-101** (MuJoCo Menagerie `robotstudio_so101`, Apache-2.0)
- 학습 GPU: RunPod 대여 (사용자가 써 본 적 있음)
- 기각: B(실물 로봇)는 A 이후 여건이 되면 → 2026-10-03 범위를 시뮬레이션으로 한정. C(LeRobot 기여)는 여유 있을 때 덧붙임.

## 5. 재조사 (2026-10-03)

M5 결과와 피드백을 받아 비전 학회(손 절대 위치)·로봇 학회(RGB 조종) 쪽을 다시 찾았다. 자세한 표는 `docs/07-related-work-v2.md`. 요약:
- RootNet(ICCV 2019)이 "핀홀 거리 × 학습한 보정 계수"로 사람 몸의 절대 깊이를 풀었다 — 우리 깊이 문제와 구조가 같아 조건 ③ 설계의 근거로 쓴다.
- AnyTeleop(RSS 2023)이 RGB만 vs RGB-D 조종 성능을 비교했다(피아노 과제 오류 28.1% vs 21.8%). 정책 학습 결과 비교는 없다.
- 같은 SO-101로 손 추적 + IK 조종을 한 2026년 연구(IEEE Access)는 RGB-D 카메라를 쓰고 손 시범으로 정책을 학습하지 않았다.
- 차별점("단안 웹캠 시범 → 정책 성공률 손해와 깊이 몫")은 이번에도 유지된다.
- 인용 따라가기(가까운 논문 3편을 인용한 487편 → 키워드로 44편): TeleOpBench(2025)가 시뮬레이션에서 단안 카메라 손 추적 조종을 다른 장비와 비교(조종 성능만, 정책 학습·깊이 분석 없음). 노트북 카메라로 5관절 팔을 조종한 2026 연구도 조종만. 차별점 유지.
