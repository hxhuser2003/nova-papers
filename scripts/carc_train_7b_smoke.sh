#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --account=xiangren_1987
#SBATCH --job-name=qasper-grpo-7b
#SBATCH --output=grpo_7b_%j.log
set -euo pipefail

# Submit from the checkout containing this change. Never pull over a running
# experiment or install into the user's shared ~/.local Python environment.
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"

# A compute node's default environment has no nvcc on PATH. vLLM's FlashInfer
# JIT only warns about this at server startup ("FlashInfer kernels are
# disabled") but later fails hard when something actually needs to compile
# ("RuntimeError: Could not find nvcc and default cuda_home=... doesn't
# exist") -- confirmed from a real CARC job's vllm.log, not a dtype/precision
# issue as earlier suspected. Load the module stack verified to provide nvcc
# explicitly, rather than depending on whatever the submitting login shell
# happened to have loaded (explicit setup also makes the job reproducible with --export=ALL).
module load ver/2607 gcc/15.3.0 cuda/13.0.3
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
command -v nvcc > /dev/null || { echo 'FATAL: nvcc still not on PATH after module load' >&2; exit 1; }

# Independent 7B capacity/optimizer probe; no dependency on the 3B job.
# 0.45 is a starting vLLM budget for A40 48GB, not a proven fit.
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=$((20000 + SLURM_JOB_ID % 30000))
export HF_HOME="${HF_HOME:-/scratch1/${USER}/hf-cache}"
export PYTHONUNBUFFERED=1
MODE="${1:-all}"
if [[ "$MODE" != probe && "$MODE" != train && "$MODE" != all ]]; then
    echo 'Usage: sbatch scripts/carc_train_7b_smoke.sh [probe|train|all]' >&2
    exit 2
fi
PYTHON="${GRPO_PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3,11), "Load Python 3.11+ before submitting"'
if [[ ! -x .venv-grpo-7b/bin/python ]]; then
    "$PYTHON" -m venv .venv-grpo-7b
fi
source .venv-grpo-7b/bin/activate
python -m pip install -r requirements-training.txt
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'No CUDA GPU visible'
assert torch.cuda.get_device_capability()[0] >= 8, 'Use A40/A100/L40S, not P100/V100'
print(torch.cuda.get_device_name(), torch.__version__)
x = torch.ones(1, device='cuda')
print('CUDA kernel probe:', (x + 1).item())
PY
OUT="eval/training-7b/${SLURM_JOB_ID}"
mkdir -p "$OUT"
python -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/base-commit.txt"
git diff > "$OUT/local-changes.patch"
VLLM_PID=''
GPU_MONITOR_PID=''
cleanup() {
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    if [[ -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
if [[ "$MODE" != train ]]; then
    vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
        --dtype bfloat16 --max-model-len 8192 --enforce-eager \
        --gpu-memory-utilization 0.80 --return-tokens-as-token-ids \
        > "$OUT/vllm.log" 2>&1 &
    VLLM_PID=$!
    READY=0
    for ((i=0; i<120; i++)); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            tail -80 "$OUT/vllm.log"
            echo 'vLLM exited before readiness' >&2
            exit 1
        fi
        if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null; then
            READY=1
            break
        fi
        sleep 5
    done
    [[ "$READY" == 1 ]] || { tail -80 "$OUT/vllm.log"; exit 1; }
    python -m scripts.check_vllm_rollout --model "$MODEL" --base-url "http://127.0.0.1:$PORT" --output "$OUT/live-token-probe.json"
    cleanup
    VLLM_PID=''
fi
if [[ "$MODE" != probe ]]; then
    : > "$OUT/device-memory.csv"
    if GPU_UUID=$(python -m scripts.gpu_monitor 2> "$OUT/device-monitor-errors.log") &&
       nvidia-smi --id="$GPU_UUID" --query-gpu=timestamp,memory.used --format=csv,noheader,nounits > "$OUT/device-memory-preflight.csv" 2>> "$OUT/device-monitor-errors.log" &&
       python -c 'import csv,sys; from datetime import datetime; rows=list(csv.reader(open(sys.argv[1]))); assert len(rows)==1 and len(rows[0])==2; datetime.strptime(rows[0][0].strip(), "%Y/%m/%d %H:%M:%S.%f"); assert float(rows[0][1])>=0' "$OUT/device-memory-preflight.csv"; then
        printf '%s\n' "$GPU_UUID" > "$OUT/monitored-gpu-uuid.txt"
        nvidia-smi --id="$GPU_UUID" --query-gpu=timestamp,memory.used --format=csv,noheader,nounits \
            --loop-ms=200 > "$OUT/device-memory.csv" 2>> "$OUT/device-monitor-errors.log" &
        GPU_MONITOR_PID=$!
    else
        echo 'WARNING: whole-device monitoring unavailable; training continues, device peaks will be null' >&2
    fi
    python -m scripts.train_grpo --model "$MODEL" --gpu-memory-utilization 0.45 --max-steps 2 --num-generations 2 --limit 8 \
        --output-dir "checkpoints/grpo-7b-smoke-${SLURM_JOB_ID}"
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    GPU_MONITOR_PID=''
    python -m scripts.summarize_training_metrics \
        "checkpoints/grpo-7b-smoke-${SLURM_JOB_ID}/phase_metrics.jsonl" \
        --device-samples "$OUT/device-memory.csv" --output "$OUT/phase-summary.json"
fi
