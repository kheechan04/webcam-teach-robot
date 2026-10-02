# 3단계: 학습과 평가

코드: `runpod/train_act.sh`(RunPod 학습), `scripts/eval_policy.py`(평가), `scripts/build_dataset.py`(데이터셋)

## 공통 설정

- 정책: LeRobot 0.6.1 ACT 기본값(ResNet18 백본, chunk 100 = 약 3.3초, lr 1e-5 고정·스케줄러 없음, VAE 사용). 바꾸는 것은 스텝·배치·시드뿐.
- 관측: 관절 6개 + 카메라 2대(앞쪽, 손목) 128×128. 행동: 관절 명령 6개. 큐브·목표 위치는 입력에 없음(카메라로 봐야 함).
- 평가: `experiments/layouts_v1.json`의 eval 배치(시범에 안 쓴 100개), 에피소드 20초 제한, `PlacementTracker` 판정, 윌슨 95% 구간. 카메라는 데이터셋과 같은 `SceneRenderer`로 그린다.

## 시험 학습: 조건 ① (2026-10-02)

- 데이터: `kheechan04/webcam-teach-robot-cond1-scripted` (스크립트 시범 50개, 9078프레임, 비공개)
- 모델: `kheechan04/webcam-teach-robot-act-cond1-pilot` (비공개)
- RunPod RTX 3090(커뮤니티), 20000스텝, 배치 8(약 17.6 에폭), 시드 1000. **학습 982초**, 스크립트 전체 약 17분(설치 포함). 약 22 step/s, GPU 메모리 0.97 GB.
- loss: 200스텝 6.30 → 2천 1.08 → 6천 0.20 → 1만 0.116 → 2만 **0.065** (l1 0.052).
- 발견: LeRobot이 모델 카드에 `license: apache-2.0`을 자동으로 적었다(우리는 지정 안 함) → 지우고, 학습 스크립트에서 자동으로 지우게 함. 데이터셋 업로드 때도 같은 기본값이 있어서 `license=None`으로 올렸다.
- 평가 결과: (진행 중)
