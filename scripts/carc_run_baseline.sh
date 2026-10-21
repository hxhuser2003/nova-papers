#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --account=rashidin_1753
#SBATCH --job-name=qasper-baseline
#SBATCH --output=baseline_%j.log

set -e

REPO_DIR="$HOME/nova-papers"

if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/hxhuser2003/nova-papers.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull

# A compute node's default environment has no nvcc on PATH. vLLM's FlashInfer
# JIT only warns about this at server startup ("FlashInfer kernels are
# disabled") but fails hard later when something actually needs to compile:
# "RuntimeError: Could not find nvcc and default cuda_home=... doesn't
# exist" -- this is the confirmed cause of this script's earlier crash
# (job 13264048), not a dtype/precision issue. Load the module stack
# verified to provide nvcc explicitly; sbatch does not inherit whatever the
# submitting login shell happened to have loaded.
module load ver/2607 gcc/15.3.0 cuda/13.0.3
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
command -v nvcc > /dev/null || { echo "FATAL: nvcc still not on PATH after module load" >&2; exit 1; }

pip install --quiet vllm openai rank_bm25

# pip installs console scripts to ~/.local/bin, which a batch job's shell
# does not reliably have on PATH even though an interactive login shell
# does (it never sources ~/.bashrc the same way) -- this is what made the
# first run fail with "vllm: command not found" despite pip install itself
# succeeding silently.
export PATH="$HOME/.local/bin:$PATH"

if ! command -v vllm > /dev/null; then
    echo "FATAL: vllm is still not on PATH after pip install. Aborting before wasting the GPU allocation." >&2
    exit 1
fi

# Start vLLM in the background, log its own output separately.
vllm serve Qwen/Qwen2.5-3B-Instruct --dtype half --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --port 8000 > vllm_server.log 2>&1 &
VLLM_PID=$!

# Poll until the server actually answers, instead of guessing a fixed sleep.
# Also check the process is still alive: without this, a server that crashes
# seconds in still burns the full 10-minute timeout before this script
# notices, as happened in job 13264048.
echo "Waiting for vLLM to come up..."
VLLM_READY=0
for i in $(seq 1 60); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "FATAL: vLLM process exited before becoming ready. See vllm_server.log." >&2
        tail -80 vllm_server.log >&2
        exit 1
    fi
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "vLLM is up after ${i}0s."
        VLLM_READY=1
        break
    fi
    sleep 10
done

if [ "$VLLM_READY" -eq 0 ]; then
    echo "FATAL: vLLM never responded after 10 minutes. See vllm_server.log. Aborting instead of running 40 questions against a dead server." >&2
    kill "$VLLM_PID" 2>/dev/null || true
    exit 1
fi

python scripts/run_baseline.py --modes agent

kill "$VLLM_PID"
