# M6 관련 연구 다시 조사 (2026-10-03)

동기 피드백(비전 학회도 보라)과 M5 결과(남는 오차 = 손 자세에 따른 일정한 치우침)를 받아, 조건 ③ 보정을 만들기 전에 다시 찾았다. 찾은 범위에서의 정리이며 없다는 보장은 아니다. 논문 내용은 초록·공개 페이지 기준으로 확인한 것만 적었다.

## A. 비전: 한 대의 RGB 카메라로 손(사람)의 실제 거리 추정

| 연구 | 한 것 | 우리와의 관계 |
|---|---|---|
| Moon et al., **RootNet** — "Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image", ICCV 2019 ([논문](https://openaccess.thecvf.com/content_ICCV_2019/html/Moon_Camera_Distance-Aware_Top-Down_Approach_for_3D_Multi-Person_Pose_Estimation_From_ICCV_2019_paper.html), [arXiv](https://arxiv.org/abs/1907.11346)) | 핀홀 관계로 거리 k = √(αx·αy·A_real / A_img) (사람 실제 면적을 고정값으로 가정)를 구하고, **이미지에서 보정 계수 γ를 학습해 곱한다.** 논문이 든 실패 이유: 자세가 바뀌면 같은 거리에서도 A_img가 달라지고(웅크림), 아이와 어른은 A_img가 비슷해도 거리가 다르다 | **우리 문제와 구조가 같다.** 우리는 손바닥 길이·너비로 핀홀 거리를 구하고, 손 자세(집기·기울임)에 따라 일정하게 틀린다. RootNet은 사람 몸에 대해 "기하 추정 × 학습한 보정"으로 풀었다 → 조건 ③을 이 구조의 가벼운 손 버전(손 자세 특징 → 보정)으로 설계할 근거 |
| I2L-MeshNet (+ RootNet) | 루트 기준 손·몸 모양 따로, 절대 깊이는 별도 네트워크 | 두 단계 방식의 대표 |
| Huang et al., **Neural Voting Field**, CVPR 2023 ([논문](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Neural_Voting_Field_for_Camera-Space_3D_Hand_Pose_Estimation_CVPR_2023_paper.pdf)) | 카메라 공간(절대 위치 포함) 3D 손 자세를 한 번에 추정하는 암시적 표현 | 학습 기반 절대 손 위치 |
| Valassakis & Garcia-Hernando, **HandDGP**, ECCV 2024 ([arXiv 2407.15844](https://arxiv.org/abs/2407.15844)) | 손 메시를 카메라 공간에서 바로 예측, 미분 가능한 전역 위치 모듈, 입력을 같은 카메라로 찍은 것처럼 정규화해 크기-깊이 모호성을 다룸 | 같은 문제("크기-깊이 모호성")를 학습으로 정면 해결 |
| Pavlakos et al., **HaMeR**, CVPR 2024 ([페이지](https://geopavlakos.github.io/hamer/)) | ViT 기반 손 메시 복원, MANO와 카메라 파라미터 출력 | 강한 손 복원 모델. 큰 ViT라 노트북 CPU 실시간은 어려울 것(확인 안 함) |
| Potamias et al., **WiLoR**, CVPR 2025 ([arXiv 2409.12259](https://arxiv.org/abs/2409.12259)) | 실시간 손 검출(RTX 4090에서 138~175 FPS) + 트랜스포머 복원 | 0단계에서 본 hand-teleop이 이걸 썼고 GPU가 필요했다 |
| 단안 metric depth: Depth Anything V2, UniDepth(CVPR 2024), Metric3D v2 | 한 장의 사진에서 장면 전체의 실제 거리(미터) 추정 | 손 위치의 깊이를 읽는 다른 길. 노트북 CPU로 30 fps는 어려울 것으로 보임(확인 안 함) |

## B. 로봇: RGB 카메라 손 추적으로 조종

| 연구 | 한 것 | 우리와의 관계 |
|---|---|---|
| Qin et al., **AnyTeleop**, RSS 2023 ([arXiv 2307.04577](https://arxiv.org/abs/2307.04577)) | RGB만 있을 때 손목 위치는 FrankMocap처럼 "약원근 크기(weak perspective scale)를 예측하는 신경망"으로 구함. 저자: "깊이 카메라보다 오차가 크지만 많은 조종 과제엔 충분". 부록의 피아노 과제: **단일 RGB 109초·오류 28.1%, 단일 RGB-D 87초·21.8%, RGB-D 두 대 74초·12.5%** | RGB만 쓴 조종이 실제로 손해라는 걸 **조종 성능**으로 보였다. 그 시범으로 학습한 **정책의 성공률**은 비교하지 않았다 → 우리가 채우는 칸 |
| Chiche et al., "Vision-Based Hand Shadowing for Robotic Manipulation via Inverse Kinematics", IEEE Access 2026 ([arXiv 2603.11383](https://arxiv.org/abs/2603.11383)) | **SO-ARM101**을 안경에 단 **RGB-D** 카메라 손 추적 + IK로 조종. IK 위치 오차 평균 36.4 mm, 구조화된 과제 86.7%, 비구조 환경 9.3%(손 가림). 손 시범으로 정책을 학습하진 않고, 리더-팔로워 데이터로 학습한 ACT·SmolVLA·π0.5·GR00T와 비교 | 같은 로봇, 비슷한 IK 조종. 깊이 카메라를 쓰고, 손 시범 → 모방 학습은 안 했다 |
| Qin, Su, Wang, "From One Hand to Multiple Hands", 2022 ([arXiv 2204.12490](https://arxiv.org/abs/2204.12490)) | 단일 카메라 조종으로 손 시범을 모아 모방 학습 | 초록만으로는 카메라가 RGB인지 RGB-D인지 확인 못 함 |

## 결론 — 무엇이 바뀌었나

1. **조건 ③ 설계:** RootNet의 "핀홀 기하 추정 × 학습한 보정 계수" 구조를 손 조종에 맞게 가볍게 쓴다. 입력은 이미 있는 MediaPipe 값(엄지-검지 비율, 손바닥 기울기, 길이·너비 비, 화면 위치, 기하 추정 거리), 출력은 거리 보정(곱 또는 빼기). M5 1회차로 맞추고 2회차로 평가. CPU에서 실시간으로 돈다.
2. **더 무거운 대안(비교용, 선택):** HaMeR·WiLoR·HandDGP 같은 학습 기반 절대 손 위치, 또는 단안 metric depth. 정확할 수 있지만 GPU가 필요하고, "비싼 장비 없이 웹캠만으로"라는 취지와 조종 지연을 생각하면 주 보정으로는 맞지 않는다. 시간이 되면 오프라인으로 M5 데이터에 돌려 "가벼운 보정이 무거운 모델과 비교해 어느 정도인지"를 볼 수 있다.
3. **차별점은 그대로:** RGB만 쓴 조종이 손해라는 건 AnyTeleop이 조종 성능으로 보였지만, **단안 웹캠 시범으로 학습한 정책의 성공률이 얼마나 깎이는지, 그중 자세에 따른 깊이 치우침의 몫이 얼마인지**를 잰 연구는 이번에도 찾지 못했다(2026-10-03 기준).
