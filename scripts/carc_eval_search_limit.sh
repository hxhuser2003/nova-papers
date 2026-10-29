#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=qasper-search-limit
#SBATCH --output=search-limit_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

# R5 (inference-side half): how many passages the <search> tool returns per
# call (QasperCorpus.SEARCH_LIMIT, currently a fixed 3) affects Answer F1 /
# Evidence F1 without needing to retrain anything -- this script serves an
# already-trained checkpoint once via vLLM (same --enable-lora pattern as
# carc_check_format_rate.sh) and reruns scripts/run_baseline.py's agent mode
# against it at search-limit in {1, 3, 5}, using run_baseline.py's own
# Answer F1 / Evidence F1 scoring (now threaded through QasperCorpus's new
# `search_limit` constructor arg -- see backend/qasper_corpus.py). This is
# NOT the training-side version of R5 from the original plan (varying
# retrieval during GRPO rollout collection itself, which needs a real
# retraining run per k); it only tells us whether a trained checkpoint's
# behavior is sensitive to retrieval breadth at inference time.
#
# Usage: sbatch scripts/carc_eval_search_limit.sh <checkpoint-dir> <adapter-name> [samples-file] [workers]
# e.g.:  sbatch scripts/carc_eval_search_limit.sh checkpoints/ref ref
#        sbatch scripts/carc_eval_search_limit.sh checkpoints/ref ref eval/sample_questions_test500.jsonl 8
CHECKPOINT="${1:?Usage: carc_eval_search_limit.sh <checkpoint-dir> <adapter-name> [samples-file] [workers]}"
ADAPTER_NAME="${2:?Usage: carc_eval_search_limit.sh <checkpoint-dir> <adapter-name> [samples-file] [workers]}"
SAMPLES_FILE="${3:-eval/sample_questions.jsonl}"
WORKERS="${4:-16}"
[[ -f "$CHECKPOINT/adapter_config.json" ]] || { echo "FATAL: $CHECKPOINT/adapter_config.json not found" >&2; exit 1; }
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'Tracked source files modified'; exit 1; }

module load ver/2607 gcc/15.3.0 cuda/13.0.3
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
command -v nvcc > /dev/null || { echo 'FATAL: nvcc still not on PATH after module load' >&2; exit 1; }

PYTHON="${GRPO_PYTHON:-python3}"
if [[ ! -x .venv-grpo/bin/python ]]; then
    "$PYTHON" -m venv .venv-grpo
fi
source .venv-grpo/bin/activate
python -m pip install -r requirements-training.txt openai

OUT="eval/search-limit-${ADAPTER_NAME}"
mkdir -p "$OUT"
PID=''
trap 'if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi' EXIT

vllm serve Qwen/Qwen2.5-3B-Instruct --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.80 \
    --enable-lora --max-lora-rank 16 --lora-modules "${ADAPTER_NAME}=${CHECKPOINT}" \
    > "grpo-search-limit-vllm_${SLURM_JOB_ID}.log" 2>&1 &
PID=$!
READY=0
for ((i=0; i<120; i++)); do
    kill -0 "$PID" 2>/dev/null || { tail -80 "grpo-search-limit-vllm_${SLURM_JOB_ID}.log"; exit 1; }
    if curl --fail --silent --max-time 2 "http://127.0.0.1:8000/v1/models" > /dev/null; then READY=1; break; fi
    sleep 5
done
[[ "$READY" == 1 ]] || { tail -80 "grpo-search-limit-vllm_${SLURM_JOB_ID}.log"; exit 1; }

# Fail fast if vLLM didn't actually pick up the adapter under this name.
curl -s "http://127.0.0.1:8000/v1/models" | grep -q "\"${ADAPTER_NAME}\"" \
    || { echo "FATAL: adapter '${ADAPTER_NAME}' not listed by the server" >&2; exit 1; }

for K in 1 3 5; do
    python -m scripts.run_baseline --modes agent --search-limit "$K" \
        --samples "$SAMPLES_FILE" --base-url http://127.0.0.1:8000/v1 --model "${ADAPTER_NAME}" \
        --workers "$WORKERS" --out-dir "$OUT/k${K}"
done

python - "$OUT" <<'REPORT'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
report = {}
for k in (1, 3, 5):
    rows = [json.loads(s) for s in (root / f'k{k}' / 'agent.jsonl').read_text().splitlines() if s.strip()]
    report[f'k{k}'] = {
        'answer_f1': sum(r['answer_f1'] for r in rows) / len(rows),
        'evidence_f1': sum(r['evidence_f1'] for r in rows) / len(rows),
        'latency_s': sum(r['latency_s'] for r in rows) / len(rows),
        'answered': sum(r.get('terminated_reason') == 'answered' for r in rows),
        'errors': sum('error' in r for r in rows),
        'n': len(rows),
    }
(root / 'comparison.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
REPORT
echo "Done. Comparison written to $OUT/comparison.json."
