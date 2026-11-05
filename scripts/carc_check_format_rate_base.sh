#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=qasper-format-check-base
#SBATCH --output=format-check-base_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

# Same real-rollout check as carc_check_format_rate.sh, but for an
# UNTRAINED base model directly -- no LoRA adapter to load, since there is
# none yet. Needed as the "before RL training" half of a before/after
# comparison (e.g. for R7's 7B checkpoint, whose "after" half is already
# in eval/rollout_scoring-r7-7b-validation-seed599/summary.md).
#
# Usage: sbatch scripts/carc_check_format_rate_base.sh <model-name> <run-name> [split] [seed] [quotas]
# e.g.:  sbatch scripts/carc_check_format_rate_base.sh Qwen/Qwen2.5-7B-Instruct r7-7b-untrained validation 599 extractive=150,free_form=40,yes_no=70,unanswerable=60

MODEL="${1:?Usage: carc_check_format_rate_base.sh <model-name> <run-name> [split] [seed] [quotas]}"
RUN_NAME="${2:?Usage: carc_check_format_rate_base.sh <model-name> <run-name> [split] [seed] [quotas]}"
SPLIT="${3:-train}"
SEED="${4:-}"
QUOTAS="${5:-}"

module load ver/2607 gcc/15.3.0 cuda/13.0.3
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
command -v nvcc > /dev/null || { echo 'FATAL: nvcc still not on PATH after module load' >&2; exit 1; }
export HF_HOME="${HF_HOME:-/scratch1/${USER}/hf-cache}"

PYTHON="${GRPO_PYTHON:-python3}"
if [[ ! -x .venv-grpo/bin/python ]]; then
    "$PYTHON" -m venv .venv-grpo
fi
source .venv-grpo/bin/activate
python -m pip install -r requirements-training.txt

OUT="eval/rollout_scoring-${RUN_NAME}"
[[ "$SPLIT" == train ]] || OUT="${OUT}-${SPLIT}"
[[ -z "$SEED" ]] || OUT="${OUT}-seed${SEED}"
PID=''
trap 'if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi' EXIT

PORT=$((20000 + SLURM_JOB_ID % 20000))
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.80 \
    > "grpo-format-check-base-vllm_${SLURM_JOB_ID}.log" 2>&1 &
PID=$!
READY=0
for ((i=0; i<120; i++)); do
    kill -0 "$PID" 2>/dev/null || { tail -80 "grpo-format-check-base-vllm_${SLURM_JOB_ID}.log"; exit 1; }
    if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null; then READY=1; break; fi
    sleep 5
done
[[ "$READY" == 1 ]] || { tail -80 "grpo-format-check-base-vllm_${SLURM_JOB_ID}.log"; exit 1; }

SEED_ARGS=()
[[ -z "$SEED" ]] || SEED_ARGS=(--seed "$SEED")
QUOTAS_ARGS=()
[[ -z "$QUOTAS" ]] || QUOTAS_ARGS=(--quotas "$QUOTAS")
python -m scripts.score_real_rollouts --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --split "$SPLIT" --out-dir "$OUT" "${SEED_ARGS[@]}" "${QUOTAS_ARGS[@]}"

echo "Done. Compare $OUT/summary.md against the trained checkpoint's own eval/rollout_scoring-*/summary.md."
