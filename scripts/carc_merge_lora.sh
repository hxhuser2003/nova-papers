#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=rashidin_1753
#SBATCH --job-name=qasper-merge-lora
#SBATCH --output=merge-lora_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

# Runs scripts/merge_lora_and_push.py on an actual compute node instead of
# the login node -- running it directly on the login node segfaulted deep
# inside scipy's compiled BLAS extension (login nodes aren't meant for
# loading a full model into memory; this is the same reason every other
# heavy step in this repo goes through sbatch, not a bare shell command).
#
# Usage: sbatch scripts/carc_merge_lora.sh <checkpoint-dir> <output-dir> [base-model]
# e.g.:  sbatch scripts/carc_merge_lora.sh checkpoints/p1-data1000-step1000 ~/merged-p1

CHECKPOINT="${1:?Usage: carc_merge_lora.sh <checkpoint-dir> <output-dir> [base-model]}"
OUTPUT_DIR="${2:?Usage: carc_merge_lora.sh <checkpoint-dir> <output-dir> [base-model]}"
BASE_MODEL="${3:-Qwen/Qwen2.5-3B-Instruct}"
[[ -f "$CHECKPOINT/adapter_config.json" ]] || { echo "FATAL: $CHECKPOINT/adapter_config.json not found" >&2; exit 1; }

module load ver/2607 gcc/15.3.0 cuda/13.0.3
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export HF_HOME="${HF_HOME:-/scratch1/${USER}/hf-cache}"

PYTHON="${GRPO_PYTHON:-python3}"
if [[ ! -x .venv-grpo/bin/python ]]; then
    "$PYTHON" -m venv .venv-grpo
fi
source .venv-grpo/bin/activate
python -m pip install -r requirements-training.txt

python -m scripts.merge_lora_and_push "$CHECKPOINT" --output-dir "$OUTPUT_DIR" --base-model "$BASE_MODEL"

TARBALL="${OUTPUT_DIR%/}.tar.gz"
tar -czf "$TARBALL" -C "$OUTPUT_DIR" .
echo "Merged model saved to $OUTPUT_DIR, tarball at $TARBALL"
ls -la "$TARBALL"
