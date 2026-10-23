#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=qasper-lora-reload
#SBATCH --output=lora-reload_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load ver/2607 gcc/15.3.0 cuda/13.0.3
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
# Reuse the exact installed training environment read-only; no pip upgrades.
TRAIN_DIR="$HOME/nova-papers-grpo-fixed"
PYTHON="$TRAIN_DIR/.venv-grpo/bin/python"
test -x "$PYTHON"
OUT="eval/lora-reload/$SLURM_JOB_ID"
mkdir -p "$OUT"
"$PYTHON" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/commit.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
"$PYTHON" -m scripts.verify_lora_reload --checkpoint "$TRAIN_DIR/checkpoints/grpo-smoke-13310264" --output "$OUT"
