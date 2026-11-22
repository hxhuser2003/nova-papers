#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=rashidin_1753
#SBATCH --job-name=qasper-rag-vs-agent
#SBATCH --output=rag-vs-agent_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

# Fair RAG-vs-Agent comparison on the SAME trained checkpoint (not the
# untrained-base-model direct/rag/agent split from job 13318062, which never
# loaded any GRPO checkpoint). Serves the checkpoint once via vLLM as a LoRA
# adapter (same --enable-lora pattern as carc_check_format_rate.sh /
# carc_eval_search_limit.sh), then runs scripts/run_baseline.py's "rag" mode
# (one BM25 retrieval + single-shot answer) and "agent" mode (the trained
# multi-round search/read/answer loop) against the identical adapter, so the
# only variable is the inference strategy -- not the weights. Scored with
# run_baseline.py's own Answer F1 / Evidence F1 (official QASPER metric),
# significance via paired_ttest_agent.py (per-question, question_id-aligned,
# matches this project's existing |t|>=1.96 decision rule).
#
# Held-out by default: eval/sample_questions_test500.jsonl is QASPER's test
# split (480 questions), which no checkpoint in this project has ever been
# trained on -- generated here if missing (fixed seed=2026, deterministic).
#
# Usage: sbatch scripts/carc_eval_rag_vs_agent.sh <checkpoint-dir> <adapter-name> [base-model] [samples-file] [workers]
# e.g.:  sbatch scripts/carc_eval_rag_vs_agent.sh checkpoints/p1-7b-data1000-step1000 p1-7b Qwen/Qwen2.5-7B-Instruct
CHECKPOINT="${1:?Usage: carc_eval_rag_vs_agent.sh <checkpoint-dir> <adapter-name> [base-model] [samples-file] [workers]}"
ADAPTER_NAME="${2:?Usage: carc_eval_rag_vs_agent.sh <checkpoint-dir> <adapter-name> [base-model] [samples-file] [workers]}"
BASE_MODEL="${3:-Qwen/Qwen2.5-7B-Instruct}"
SAMPLES_FILE="${4:-eval/sample_questions_test500.jsonl}"
WORKERS="${5:-16}"
[[ -f "$CHECKPOINT/adapter_config.json" ]] || { echo "FATAL: $CHECKPOINT/adapter_config.json not found" >&2; exit 1; }
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'Tracked source files modified' >&2; exit 1; }

module load ver/2607 gcc/15.3.0 cuda/13.0.3
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
command -v nvcc > /dev/null || { echo 'FATAL: nvcc still not on PATH after module load' >&2; exit 1; }
export HF_HOME="${HF_HOME:-/scratch1/${USER}/hf-cache}"

PYTHON="${GRPO_PYTHON:-python3}"
if [[ ! -x .venv-grpo/bin/python ]]; then
    "$PYTHON" -m venv .venv-grpo
fi
source .venv-grpo/bin/activate
python -m pip install -r requirements-training.txt openai

if [[ ! -f "$SAMPLES_FILE" ]]; then
    echo "Samples file $SAMPLES_FILE not found, generating (test split, 480 questions, fixed seed)..."
    python -m scripts.make_held_out_sample --splits test --target-total 480 --out "$SAMPLES_FILE"
fi

OUT="eval/rag-vs-agent-${ADAPTER_NAME}"
mkdir -p "$OUT"
PID=''
trap 'if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi' EXIT

PORT=$((20000 + SLURM_JOB_ID % 20000))
vllm serve "$BASE_MODEL" --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.80 \
    --enable-lora --max-lora-rank 16 --lora-modules "${ADAPTER_NAME}=${CHECKPOINT}" \
    > "grpo-rag-vs-agent-vllm_${SLURM_JOB_ID}.log" 2>&1 &
PID=$!
READY=0
for ((i=0; i<120; i++)); do
    kill -0 "$PID" 2>/dev/null || { tail -80 "grpo-rag-vs-agent-vllm_${SLURM_JOB_ID}.log"; exit 1; }
    if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null; then READY=1; break; fi
    sleep 5
done
[[ "$READY" == 1 ]] || { tail -80 "grpo-rag-vs-agent-vllm_${SLURM_JOB_ID}.log"; exit 1; }

curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q "\"${ADAPTER_NAME}\"" \
    || { echo "FATAL: adapter '${ADAPTER_NAME}' not listed by the server" >&2; exit 1; }

python -m scripts.run_baseline --modes rag agent \
    --samples "$SAMPLES_FILE" --base-url "http://127.0.0.1:$PORT/v1" --model "${ADAPTER_NAME}" \
    --workers "$WORKERS" --out-dir "$OUT"

echo "=== Answer F1 (official QASPER metric), paired by question_id ==="
python -m scripts.paired_ttest_agent "$OUT/rag.jsonl" "$OUT/agent.jsonl" answer_f1
echo "=== Evidence F1 ==="
python -m scripts.paired_ttest_agent "$OUT/rag.jsonl" "$OUT/agent.jsonl" evidence_f1

echo "Done. Per-question records in $OUT/{rag,agent}.jsonl, summary in $OUT/summary.md."
