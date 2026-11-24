#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=rashidin_1753
#SBATCH --job-name=qasper-demo-loadtest
#SBATCH --output=demo-loadtest_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

# Runs the full live-demo stack (vLLM + LoRA checkpoint, then the FastAPI
# backend, then the real load test against it) in one job, entirely on
# CARC's local disk -- no Drive, no scp, no FUSE/mmap surprises. Gave up on
# doing this in Colab after repeated "SafetensorError: header too large"
# crashes that moved around between the LoRA file and the base model
# depending on where HF_HOME/the LoRA dir were pointed, none of which
# reproduced here; CARC already serves this same checkpoint successfully via
# carc_check_format_rate.sh, so this reuses that exact vLLM invocation
# instead of re-debugging Colab's environment.
#
# Usage: sbatch scripts/carc_run_demo_loadtest.sh <checkpoint-dir> <adapter-name> [concurrency...] [repeats] [base-model]
# e.g.:  sbatch scripts/carc_run_demo_loadtest.sh checkpoints/p1-data1000-step1000 p1-data1000-step1000
#        sbatch scripts/carc_run_demo_loadtest.sh checkpoints/p1-data1000-step1000 p1-data1000-step1000 "2 3" 5
#
# [concurrency] is a single quoted, space-separated list (default "2 3", the
# P3 stability check -- NOT the P0 "find where it breaks" sweep, which
# already has its answer recorded in docs/CONCURRENCY_REPAIR_PLAN.md).
# [repeats] defaults to 5.

CHECKPOINT="${1:?Usage: carc_run_demo_loadtest.sh <checkpoint-dir> <adapter-name> [concurrency] [repeats] [base-model]}"
ADAPTER_NAME="${2:?Usage: carc_run_demo_loadtest.sh <checkpoint-dir> <adapter-name> [concurrency] [repeats] [base-model]}"
CONCURRENCY="${3:-2 3}"
REPEATS="${4:-5}"
BASE_MODEL="${5:-Qwen/Qwen2.5-3B-Instruct}"
[[ -f "$CHECKPOINT/adapter_config.json" ]] || { echo "FATAL: $CHECKPOINT/adapter_config.json not found" >&2; exit 1; }

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
# Training deps cover vLLM; the app itself (backend/, scripts/load_test_api.py)
# needs its own small set that requirements-training.txt has no reason to include.
python -m pip install 'fastapi>=0.115,<1' 'uvicorn>=0.30,<1' 'httpx>=0.27,<1' \
    'pydantic-settings>=2.6,<3' 'pymupdf>=1.25,<2' 'rank-bm25>=0.2.2,<0.3'

VLLM_PID=''
APP_PID=''
cleanup() {
    for pid in "$APP_PID" "$VLLM_PID"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
    for pid in "$APP_PID" "$VLLM_PID"; do
        [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# Same per-job port randomization as carc_check_format_rate.sh/
# carc_eval_agent_ablation.sh -- a fixed port collides when several of these
# jobs land on the same shared GPU node.
VLLM_PORT=$((20000 + SLURM_JOB_ID % 20000))
APP_PORT=$((40000 + SLURM_JOB_ID % 20000))

vllm serve "$BASE_MODEL" --host 127.0.0.1 --port "$VLLM_PORT" \
    --dtype bfloat16 --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.80 \
    --enable-lora --max-lora-rank 16 --lora-modules "${ADAPTER_NAME}=${CHECKPOINT}" \
    > "vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
READY=0
for ((i=0; i<120; i++)); do
    kill -0 "$VLLM_PID" 2>/dev/null || { tail -80 "vllm_${SLURM_JOB_ID}.log"; exit 1; }
    if curl --fail --silent --max-time 2 "http://127.0.0.1:$VLLM_PORT/v1/models" > /dev/null; then READY=1; break; fi
    sleep 5
done
[[ "$READY" == 1 ]] || { echo "FATAL: vLLM did not become ready" >&2; tail -80 "vllm_${SLURM_JOB_ID}.log"; exit 1; }
curl -s "http://127.0.0.1:$VLLM_PORT/v1/models" | grep -q "\"${ADAPTER_NAME}\"" \
    || { echo "FATAL: adapter '${ADAPTER_NAME}' not listed by the server" >&2; exit 1; }

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}"
export VLLM_MODEL="$ADAPTER_NAME"
export MAX_CONCURRENT_CHATS="${MAX_CONCURRENT_CHATS:-2}"
export MAX_QUEUED_CHATS="${MAX_QUEUED_CHATS:-5}"
export CHAT_QUEUE_TIMEOUT_SECONDS="${CHAT_QUEUE_TIMEOUT_SECONDS:-60}"
APP_LOG="app_${SLURM_JOB_ID}.log"
python -m uvicorn backend.main:app --host 127.0.0.1 --port "$APP_PORT" > "$APP_LOG" 2>&1 &
APP_PID=$!
READY=0
for ((i=0; i<30; i++)); do
    kill -0 "$APP_PID" 2>/dev/null || { tail -80 "$APP_LOG"; exit 1; }
    if curl --fail --silent --max-time 2 "http://127.0.0.1:$APP_PORT/api/health" > /dev/null; then READY=1; break; fi
    sleep 2
done
[[ "$READY" == 1 ]] || { echo "FATAL: backend did not become ready" >&2; tail -80 "$APP_LOG"; exit 1; }
curl -s "http://127.0.0.1:$APP_PORT/api/health" | grep -q "\"${ADAPTER_NAME}\"" \
    || { echo "FATAL: backend is not reporting model=${ADAPTER_NAME} -- check VLLM_MODEL" >&2; exit 1; }

OUT="eval/load_test/carc-${ADAPTER_NAME}-${SLURM_JOB_ID}"
python -m scripts.load_test_api --url "http://127.0.0.1:${APP_PORT}" --output "$OUT" \
    --concurrency $CONCURRENCY --repeats "$REPEATS" --server-log "$APP_LOG"

echo "Done. Results: $OUT/summary.csv"
cat "$OUT/summary.csv"
