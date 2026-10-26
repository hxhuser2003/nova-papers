#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=qasper-hybrid-retrieval
#SBATCH --output=hybrid-retrieval_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'Tracked source files modified'; exit 1; }
module load ver/2607 gcc/15.3.0 cuda/13.0.3
command -v nvcc
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
OUT="$PWD/eval/hybrid-retrieval/${SLURM_JOB_ID}"
mkdir -p "$OUT"
python3 -m venv "$OUT/venv"
source "$OUT/venv/bin/activate"
python -m pip install -r requirements-training.txt openai sentence-transformers
python -m pip check
python -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/commit.txt"
nvcc --version > "$OUT/nvcc.txt"
cp eval/sample_questions.jsonl eval/system_prompt.txt "$OUT/"
sha256sum corpus/qasper/*.jsonl "$OUT/sample_questions.jsonl" "$OUT/system_prompt.txt" > "$OUT/input-checksums.txt"
PORT=$((20000 + SLURM_JOB_ID % 20000))
PID=''
trap 'if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi' EXIT
vllm serve Qwen/Qwen2.5-3B-Instruct --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.80 --seed 42 > "$OUT/vllm.log" 2>&1 &
PID=$!
READY=0
for ((i=0; i<240; i++)); do
    kill -0 "$PID" 2>/dev/null || { cat "$OUT/vllm.log"; exit 1; }
    if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json"; then READY=1; break; fi
    sleep 5
done
[[ "$READY" == 1 ]] || { cat "$OUT/vllm.log"; exit 1; }
# control: production BM25-only QasperCorpus, unmodified. treatment: same
# model/prompt/agent_loop, only the retriever swapped (HybridQasperCorpus:
# BM25 + bge-m3 dense, fused with RRF -- see backend/hybrid_retrieval.py
# and TRAINING_NOTES.md's R10 entry).
python -m scripts.run_baseline --modes agent --samples "$OUT/sample_questions.jsonl" \
    --prompt "$OUT/system_prompt.txt" --base-url "http://127.0.0.1:$PORT/v1" --out-dir "$OUT/control"
python -m scripts.run_hybrid_retrieval --modes agent --samples "$OUT/sample_questions.jsonl" \
    --prompt "$OUT/system_prompt.txt" --base-url "http://127.0.0.1:$PORT/v1" --out-dir "$OUT/treatment"
for ARM in control treatment; do
    python - "$OUT/$ARM/agent.jsonl" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(s) for s in Path(sys.argv[1]).read_text().splitlines() if s.strip()]
assert len(rows) == 40, 'Incomplete evaluation'
errors = [r for r in rows if 'error' in r or r.get('terminated_reason') == 'exception']
assert not errors, f'{len(errors)} evaluation errors; inspect JSONL, do not treat as valid comparison'
PY
done
echo "Both 40-question evaluations finished without recorded exceptions. Results: $OUT"
