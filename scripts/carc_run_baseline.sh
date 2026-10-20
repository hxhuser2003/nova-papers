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
echo "Waiting for vLLM to come up..."
VLLM_READY=0
for i in $(seq 1 60); do
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
