#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=v100:1
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

# Start vLLM in the background, log its own output separately.
vllm serve Qwen/Qwen2.5-3B-Instruct --dtype half --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --port 8000 > vllm_server.log 2>&1 &
VLLM_PID=$!

# Poll until the server actually answers, instead of guessing a fixed sleep.
echo "Waiting for vLLM to come up..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "vLLM is up after ${i}0s."
        break
    fi
    sleep 10
done

python scripts/run_baseline.py --modes agent

kill "$VLLM_PID"
