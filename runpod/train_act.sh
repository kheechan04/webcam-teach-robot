#!/usr/bin/env bash
# RunPod GPU 서버에서 ACT를 학습한다. 끝나면(성공이든 실패든) 학습 기록을 Hub에 올리고 서버를 스스로 끈다.
#
# 사용 (RunPod 웹 터미널에서):
#   export HF_TOKEN=hf_...            # Hugging Face 쓰기 권한 토큰
#   curl -LsSf https://raw.githubusercontent.com/kheechan04/webcam-teach-robot/main/runpod/train_act.sh -o train_act.sh
#   nohup bash train_act.sh kheechan04/webcam-teach-robot-cond1-scripted kheechan04/webcam-teach-robot-act-cond1-pilot 20000 &
#
# 인자: [데이터셋 저장소] [모델 저장소] [학습 스텝]
# 모델은 Hub에 비공개로 올린다(라이선스는 공개할 때 정한다).
set -euo pipefail

DATASET="${1:-kheechan04/webcam-teach-robot-cond1-scripted}"
POLICY="${2:-kheechan04/webcam-teach-robot-act-cond1-pilot}"
STEPS="${3:-20000}"
BATCH=8
SEED=1000
: "${HF_TOKEN:?HF_TOKEN 환경 변수를 먼저 넣어 주세요 (export HF_TOKEN=hf_...)}"

NAME="$(basename "$POLICY")"
mkdir -p /workspace/logs
LOG="/workspace/logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

finish() {
  code=$?
  echo "=== 끝 (종료 코드 $code) $(date) ==="
  # 학습 기록을 모델 저장소에 남긴다(실패해도). 그다음 서버를 끈다 — 요금이 계속 나가지 않게.
  if command -v hf >/dev/null 2>&1; then
    hf upload "$POLICY" "$LOG" "train_logs/$(basename "$LOG")" --repo-type model --private || true
  fi
  if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
trap finish EXIT

echo "=== 시작 $(date) | 데이터셋 $DATASET | 모델 $POLICY | 스텝 $STEPS | 배치 $BATCH | 시드 $SEED ==="
nvidia-smi

# LeRobot 0.6.1은 Python 3.12 이상이 필요해서 uv로 따로 환경을 만든다. 영상 읽기용 ffmpeg도 깐다.
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
cd /workspace
uv venv -p 3.12 .venv-lerobot
source .venv-lerobot/bin/activate
uv pip install "lerobot[dataset,training]==0.6.1"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

hf auth login --token "$HF_TOKEN"

START=$(date +%s)
# ACT 기본값 그대로(chunk 100, lr 1e-5 고정, 스케줄러 없음). 바꾸는 건 스텝·배치·시드뿐.
lerobot-train \
  --dataset.repo_id="$DATASET" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id="$POLICY" \
  --policy.push_to_hub=true \
  --policy.private=true \
  --output_dir="/workspace/outputs/$NAME" \
  --job_name="$NAME" \
  --batch_size=$BATCH \
  --steps=$STEPS \
  --save_freq=5000 \
  --log_freq=200 \
  --seed=$SEED \
  --wandb.enable=false
echo "학습 시간(초): $(( $(date +%s) - START ))"

# LeRobot이 모델 카드에 license: apache-2.0을 자동으로 적는다. 라이선스는 공개할 때 정하므로 지운다.
python - <<PY
from huggingface_hub import ModelCard
card = ModelCard.load("$POLICY")
card.data.license = None
card.push_to_hub("$POLICY", commit_message="Remove auto-added license (not decided yet)")
print("모델 카드 라이선스 지움")
PY
