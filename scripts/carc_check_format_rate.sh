#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=qasper-format-check
#SBATCH --output=format-check_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

# Compares a trained checkpoint's format-error rate against the untrained
# baseline already committed at eval/rollout_scoring/summary.md (33/100
# repeated_format_error), by pointing A's score_real_rollouts.py at the
# SAME base model served with the checkpoint loaded as a vLLM LoRA adapter
# -- no changes to her script, no new eval tooling, same metric.
#
# Usage: sbatch scripts/carc_check_format_rate.sh <checkpoint-dir> <adapter-name> [split] [seed]
# e.g.:  sbatch scripts/carc_check_format_rate.sh checkpoints/ref ref
#        sbatch scripts/carc_check_format_rate.sh checkpoints/ref ref validation
#        sbatch scripts/carc_check_format_rate.sh checkpoints/ref ref train 601
#
# [split] defaults to "train" (same split GRPO training draws from -- the
# original behavior, kept for comparability with earlier runs). Pass
# "validation" or "test" for a truly held-out check: these QASPER splits are
# NOT used anywhere in training (see corpus/qasper/papers.jsonl's own split
# field), so there is no "studying the exam" overlap risk there.
# [seed] defaults to score_real_rollouts.py's own default (599); since that
# script already samples at temperature=1.0 (real stochastic rollouts, not
# greedy), a different seed genuinely produces different rollouts -- unlike
# the temperature=0 inference ablations in carc_eval_agent_ablation.sh.

CHECKPOINT="${1:?Usage: carc_check_format_rate.sh <checkpoint-dir> <adapter-name> [split] [seed]}"
ADAPTER_NAME="${2:?Usage: carc_check_format_rate.sh <checkpoint-dir> <adapter-name> [split] [seed]}"
SPLIT="${3:-train}"
SEED="${4:-}"
[[ -f "$CHECKPOINT/adapter_config.json" ]] || { echo "FATAL: $CHECKPOINT/adapter_config.json not found" >&2; exit 1; }

module load ver/2607 gcc/15.3.0 cuda/13.0.3
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
command -v nvcc > /dev/null || { echo 'FATAL: nvcc still not on PATH after module load' >&2; exit 1; }

PYTHON="${GRPO_PYTHON:-python3}"
if [[ ! -x .venv-grpo/bin/python ]]; then
    "$PYTHON" -m venv .venv-grpo
fi
source .venv-grpo/bin/activate
python -m pip install -r requirements-training.txt

OUT="eval/rollout_scoring-${ADAPTER_NAME}"
[[ "$SPLIT" == train ]] || OUT="${OUT}-${SPLIT}"
[[ -z "$SEED" ]] || OUT="${OUT}-seed${SEED}"
PID=''
trap 'if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi' EXIT

vllm serve Qwen/Qwen2.5-3B-Instruct --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.80 \
    --enable-lora --max-lora-rank 16 --lora-modules "${ADAPTER_NAME}=${CHECKPOINT}" \
    > "grpo-format-check-vllm_${SLURM_JOB_ID}.log" 2>&1 &
PID=$!
READY=0
for ((i=0; i<120; i++)); do
    kill -0 "$PID" 2>/dev/null || { tail -80 "grpo-format-check-vllm_${SLURM_JOB_ID}.log"; exit 1; }
    if curl --fail --silent --max-time 2 "http://127.0.0.1:8000/v1/models" > /dev/null; then READY=1; break; fi
    sleep 5
done
[[ "$READY" == 1 ]] || { tail -80 "grpo-format-check-vllm_${SLURM_JOB_ID}.log"; exit 1; }

# Fail fast if vLLM didn't actually pick up the adapter under this name.
curl -s "http://127.0.0.1:8000/v1/models" | grep -q "\"${ADAPTER_NAME}\"" \
    || { echo "FATAL: adapter '${ADAPTER_NAME}' not listed by the server" >&2; exit 1; }

SEED_ARGS=()
[[ -z "$SEED" ]] || SEED_ARGS=(--seed "$SEED")
python -m scripts.score_real_rollouts --base-url http://127.0.0.1:8000 \
    --model "${ADAPTER_NAME}" --split "$SPLIT" --out-dir "$OUT" "${SEED_ARGS[@]}"

echo "Done. Compare $OUT/summary.md's 'How rollouts ended' table against the untrained baseline in eval/rollout_scoring/summary.md."
