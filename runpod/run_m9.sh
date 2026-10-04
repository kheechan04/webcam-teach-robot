#!/usr/bin/env bash
# M9: 다섯 조건 × 시드 3개 ACT 학습 + 평가를 RunPod 한 대에서 차례로. 끝나면 서버를 스스로 끈다.
#
# 사용 (RunPod 웹 터미널에서):
#   export HF_TOKEN=hf_...            # 이번 학습용으로 새로 만든 쓰기 토큰 (끝나면 지운다)
#   curl -LsSf https://raw.githubusercontent.com/kheechan04/webcam-teach-robot/main/runpod/run_m9.sh -o run_m9.sh
#   nohup bash run_m9.sh > /dev/null 2>&1 &           # M9
#   JOBS=jobs_m9b.txt nohup bash run_m9.sh > /dev/null 2>&1 &   # 보강 실험 목록
#   tail -f /workspace/logs/m9_*.log  # 진행 보기 (창을 닫아도 계속 돈다)
#
# 이어 하기: 같은 명령을 다시 실행하면, Hub에 평가 결과가 이미 있는 (조건, 시드)는 건너뛴다.
# 결과: 모델 kheechan04/webcam-teach-robot-act-<조건>-s<시드> (비공개),
#       평가 JSON은 모델 저장소의 eval/ 폴더와 kheechan04/webcam-teach-robot-m9-results (비공개 데이터셋 저장소).
set -uo pipefail

: "${HF_TOKEN:?HF_TOKEN 환경 변수를 먼저 넣어 주세요 (export HF_TOKEN=hf_...)}"
USER_HF=kheechan04
CONDS="cond1-scripted cond2-webcam cond3-webcam-corrected cond4-depth-error cond4z-no-error"
SEEDS="${SEEDS:-1000 2000 3000}"
STEPS=20000
BATCH=8
WORKERS="${WORKERS:-$(( $(nproc) > 16 ? 16 : $(nproc) ))}"
RESULTS="$USER_HF/webcam-teach-robot-m9-results"

mkdir -p /workspace/logs
# 두 번 겹쳐 실행되지 않게 (2026-10-03 첫 시도에서 토큰 없이 한 번, 넣고 한 번 실행돼 둘이 같이 돌았다)
exec 9>/workspace/run_m9.lock
flock -n 9 || { echo "이미 실행 중이에요. 진행은: tail -f /workspace/logs/m9_*.log"; exit 1; }
LOG="/workspace/logs/m9_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

finish() {
  echo "=== 끝 $(date) ==="
  (cd /workspace/webcam-teach-robot 2>/dev/null && uv run hf upload "$RESULTS" "$LOG" "logs/$(basename "$LOG")" --repo-type dataset --private) || true
  # runpodctl은 설정 파일이 없으면 못 끈다(첫 시도: "config file not found"). Pod에 들어 있는 API 키로 먼저 설정.
  # 끝까지 돌았을 때만 서버를 끈다. 사람이 pkill로 멈춘 경우에도 끄면 바로 다시 실행할 수 없다(2026-10-03 실제로 그랬다).
  if [ "${ALL_DONE:-0}" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    [ -n "${RUNPOD_API_KEY:-}" ] && runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1
    runpodctl stop pod "$RUNPOD_POD_ID" || echo "!!! 서버를 스스로 끄지 못함 — RunPod 화면에서 직접 Stop/Terminate 해 주세요"
  fi
}
trap finish EXIT

echo "=== 시작 $(date) | 조건 $CONDS | 시드 $SEEDS | 평가 프로세스 $WORKERS ==="
nvidia-smi
apt-get update -qq && apt-get install -y -qq ffmpeg git libgl1 libglib2.0-0 libegl1 libosmesa6 > /dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
cd /workspace
[ -d webcam-teach-robot ] || git clone -q https://github.com/kheechan04/webcam-teach-robot.git
cd webcam-teach-robot && git pull -q
uv sync -q
uv cache clean -q || true  # 설치 캐시 정리 (디스크 절약)
# 화면 없는 서버에서 MuJoCo 카메라 그리기: GPU(EGL)가 되면 그걸, 안 되면 CPU(OSMesa)로
export MUJOCO_GL=egl
if ! uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco/>'); r=mujoco.Renderer(m,64,64); r.render()" 2>/dev/null; then
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
fi
echo "MUJOCO_GL=$MUJOCO_GL"
uv run python -c "import torch, mujoco; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mujoco', mujoco.__version__)"
uv run hf auth login --token "$HF_TOKEN"
# 학습은 M4 시험 학습과 같은 방식의 별도 환경(LeRobot 0.6.1을 pip로 설치)에서 한다. 첫 시도에서 프로젝트 환경(uv sync)으로
# 학습하니 RTX 4000 Ada에서 초당 5~7스텝이었다(3090 시험 학습은 22스텝). 평가는 프로젝트 환경에서 한다.
[ -d /workspace/.venv-train ] || uv venv -q -p 3.12 /workspace/.venv-train
# --no-config: 프로젝트의 [tool.uv] 설정(opencv-python-headless 빼기)이 학습 환경에 적용되면 cv2가 없어 학습이 안 뜬다
# (2026-10-03 두 번째 시도에서 실제로 그랬다).
VIRTUAL_ENV=/workspace/.venv-train uv pip install -q --no-config "lerobot[dataset,training]==0.6.1" opencv-python-headless
TRAIN=/workspace/.venv-train/bin
$TRAIN/python -c "import torch, cv2, lerobot.scripts.lerobot_train; print('train env torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cv2', cv2.__version__)"   || { echo "!!! 학습 환경이 안 뜸 — 멈춤"; exit 1; }
uv run hf repo create "$RESULTS" --repo-type dataset --private --exist-ok || true

# 웹 터미널이 끊겨도 노트북에서 진행을 볼 수 있게 로그를 10분마다 Hub에 올린다
( while sleep 600; do
    tr '\r' '\n' < "$LOG" | grep -v '^Training:' | tail -n 400 > /workspace/logs/latest.txt
    date -u >> /workspace/logs/latest.txt  # 내용이 같아도 매번 새로 올라가게
    uv run hf upload "$RESULTS" /workspace/logs/latest.txt "logs/latest${JOBS:+_${JOBS%.txt}}.txt" --repo-type dataset --private >/dev/null 2>&1
  done ) &

# 할 일 목록: 기본은 M9(5조건 × 시드 1000~3000). JOBS=파일 이름을 주면 그 목록(줄마다 "조건 시드 스텝 [이름꼬리]")을 따른다.
if [ -n "${JOBS:-}" ]; then
  JOBLIST=$(grep -v '^#' "runpod/$JOBS" | grep -v '^\s*$')
else
  JOBLIST=$(for SEED in $SEEDS; do for COND in $CONDS; do echo "$COND $SEED $STEPS"; done; done)
fi
echo "할 일 $(echo "$JOBLIST" | wc -l)개"

while read -r COND SEED STEPS SUFFIX; do
  {
    NAME="act-$COND-s$SEED${SUFFIX:-}"
    POLICY="$USER_HF/webcam-teach-robot-$NAME"
    if uv run python - <<PY
import sys
from huggingface_hub import HfApi
files = HfApi().list_repo_files("$RESULTS", repo_type="dataset")
sys.exit(0 if "eval/$NAME.json" in files else 1)
PY
    then echo "--- $NAME: 이미 끝남, 건너뜀"; continue; fi

    # GPU가 사라지면(2026-10-03 첫 서버: 몇 시간 뒤 "Failed to initialize NVML", 학습이 CPU로 넘어가 한 스텝 5.5초) 멈춘다
    $TRAIN/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"       || { echo "!!! GPU가 안 보임 — 멈춤. Pod를 새로 만들어 같은 명령으로 이어 하세요"; exit 1; }
    echo "=== $NAME 학습 시작 $(date) ==="
    T0=$(date +%s)
    rm -rf "/workspace/outputs/$NAME"
    $TRAIN/lerobot-train \
      --dataset.repo_id="$USER_HF/webcam-teach-robot-$COND" \
      --policy.type=act --policy.device=cuda \
      --policy.repo_id="$POLICY" --policy.push_to_hub=true --policy.private=true \
      --output_dir="/workspace/outputs/$NAME" --job_name="$NAME" \
      --batch_size=$BATCH --steps=$STEPS --save_freq=$STEPS --log_freq=500 \
      --seed=$SEED --wandb.enable=false &
    TPID=$!
    sleep 240; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/GPU 사용률·메모리 (학습 4분째): /'
    wait $TPID || { echo "!!! $NAME 학습 실패 — 나머지도 같은 이유로 실패할 수 있어 멈춤"; exit 1; }
    echo "학습 시간(초): $(( $(date +%s) - T0 ))"
    uv run python - <<PY || true
from huggingface_hub import ModelCard
card = ModelCard.load("$POLICY"); card.data.license = None
card.push_to_hub("$POLICY", commit_message="Remove auto-added license (not decided yet)")
PY

    echo "=== $NAME 평가 시작 $(date) ==="
    T1=$(date +%s)
    uv run python scripts/eval_policy.py "/workspace/outputs/$NAME/checkpoints/last/pretrained_model" \
      --n 100 --workers "$WORKERS" --name "$NAME" --out-dir /workspace/eval || { echo "!!! $NAME 평가 실패"; continue; }
    echo "평가 시간(초): $(( $(date +%s) - T1 ))"
    J="/workspace/eval/${NAME}_eval100.json"
    uv run hf upload "$POLICY" "$J" "eval/$NAME.json" --private
    uv run hf upload "$RESULTS" "$J" "eval/$NAME.json" --repo-type dataset --private
    rm -rf "/workspace/outputs/$NAME"  # 모델은 Hub에 있다. 체크포인트(약 0.6 GB)가 쌓이면 컨테이너 디스크 30 GB가 찬다
  } < /dev/null
done <<< "$JOBLIST"
ALL_DONE=1
