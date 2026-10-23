#!/bin/bash
# Run on CARC. Creates a unique backup directory, never overwrites a checkpoint.
set -euo pipefail
REPO="${1:-$HOME/nova-papers-grpo-fixed}"
JOB="${2:-13310264}"
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
cd "$REPO"
test -s "checkpoints/grpo-smoke-$JOB/adapter_model.safetensors"
test -s "checkpoints/grpo-smoke-$JOB/adapter_config.json"
test -s "checkpoints/grpo-smoke-$JOB/training_verification.json"
BACKUP=$(mktemp -d "$HOME/nova-grpo-backup-$JOB-XXXXXX")
ARCHIVE="$BACKUP/grpo-$JOB.tar.gz"
tar -czf "$ARCHIVE" "checkpoints/grpo-smoke-$JOB" "eval/training/$JOB" "grpo_$JOB.log" scripts/train_grpo.py scripts/carc_train_smoke.sh backend/token_rollout.py requirements-training.txt
(cd "$BACKUP" && sha256sum "grpo-$JOB.tar.gz" > SHA256SUMS && sha256sum -c SHA256SUMS)
echo "Backup directory: $BACKUP"
echo 'Download this entire directory to your Mac; keep SHA256SUMS with the archive.'
