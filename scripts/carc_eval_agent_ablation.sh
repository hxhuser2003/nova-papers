#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=qasper-ablation
#SBATCH --output=ablation_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# Usage: sbatch scripts/carc_eval_agent_ablation.sh <experiment> [temperature] [seed] [samples-file] [workers]
# temperature defaults to 0 (greedy, deterministic -- the original behavior).
# At temperature=0 on this fixed 40-question set, rerunning with a different
# seed produces byte-identical output: there is no sampling to vary. To get
# a genuine second/third data point for seed-robustness, pass a nonzero
# temperature (e.g. 1.0, matching GRPOConfig's training-time sampling) plus
# a seed, e.g.: sbatch scripts/carc_eval_agent_ablation.sh combined 1.0 601
#
# samples-file defaults to eval/sample_questions.jsonl (the original 40
# questions). For the power-analysis follow-up in TRAINING_NOTES.md
# (required_n ~= 436 at the observed effect size), generate a larger
# held-out file first with scripts.make_held_out_sample, then pass it here,
# e.g.: sbatch scripts/carc_eval_agent_ablation.sh combined 0 42 eval/sample_questions_test500.jsonl 8
# workers defaults to 1 (the original behavior); raise it for larger sample
# files so a few hundred questions don't run fully serially.
EXPERIMENT="${1:?Use first-search, final-answer, or combined}"
TEMPERATURE="${2:-0}"
SEED="${3:-42}"
SAMPLES_FILE="${4:-eval/sample_questions.jsonl}"
WORKERS="${5:-1}"
export SAMPLES_FILE
[[ "$EXPERIMENT" == first-search || "$EXPERIMENT" == final-answer || "$EXPERIMENT" == combined ]] || exit 2
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo 'Tracked source files modified'; exit 1; }
module load ver/2607 gcc/15.3.0 cuda/13.0.3
command -v nvcc
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
OUT="$PWD/eval/$EXPERIMENT/${SLURM_JOB_ID}"
mkdir -p "$OUT"
export OUT
python3 -m venv "$OUT/venv"
source "$OUT/venv/bin/activate"
python -m pip install -r requirements-training.txt openai
python -m pip check
python -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/commit.txt"
nvcc --version > "$OUT/nvcc.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
python - <<'PY'
import os, json
from pathlib import Path
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability()[0] >= 8
print('GPU:', torch.cuda.get_device_name(), 'Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('CUDA test:', (torch.ones(1, device='cuda') + 1).item())
out = Path(os.environ['OUT'])
prompt = Path('eval/system_prompt.txt').read_text()
lines = prompt.splitlines(keepends=True)
matches = [s for s in lines if 'never repeat the exact same search query twice' in s]
assert len(matches) == 1, 'Expected exactly one no-repeat instruction'
(out/'treatment.txt').write_text(''.join(s for s in lines if s != matches[0]))
(out/'control.txt').write_text(''.join(s for s in lines if s != matches[0]))
samples_path = Path(os.environ['SAMPLES_FILE'])
samples = [json.loads(s) for s in samples_path.read_text().splitlines() if s.strip()]
assert samples, f'{samples_path} is empty'
(out/'samples.jsonl').write_text(samples_path.read_text())
print(f'Using {len(samples)} questions from {samples_path}')
PY
sha256sum corpus/qasper/*.jsonl "$OUT/"*.txt "$OUT/samples.jsonl" > "$OUT/input-checksums.txt"
PORT=$((20000 + SLURM_JOB_ID % 20000))
PID=''
trap 'if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi' EXIT
vllm serve Qwen/Qwen2.5-3B-Instruct --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.80 --seed "$SEED" > "$OUT/vllm.log" 2>&1 &
PID=$!
READY=0
for ((i=0; i<240; i++)); do
    kill -0 "$PID" 2>/dev/null || { cat "$OUT/vllm.log"; exit 1; }
    if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json"; then READY=1; break; fi
    sleep 5
done
[[ "$READY" == 1 ]] || { cat "$OUT/vllm.log"; exit 1; }
for ARM in control treatment; do
    python -m scripts.run_agent_ablation --experiment "$EXPERIMENT" --arm "$ARM" --modes agent --workers "$WORKERS" --temperature "$TEMPERATURE" --max-tokens 400 --samples "$OUT/samples.jsonl" --prompt "$OUT/$ARM.txt" --base-url "http://127.0.0.1:$PORT/v1" --out-dir "$OUT/$ARM"
    python - "$OUT/$ARM/agent.jsonl" "$OUT/samples.jsonl" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(s) for s in Path(sys.argv[1]).read_text().splitlines() if s.strip()]
expected = len([s for s in Path(sys.argv[2]).read_text().splitlines() if s.strip()])
assert len(rows) == expected, f'Incomplete evaluation: {len(rows)}/{expected} rows'
errors = [r for r in rows if 'error' in r or r.get('terminated_reason') == 'exception']
assert not errors, f'{len(errors)} evaluation errors; inspect JSONL, do not treat as valid comparison'
PY
done
echo "Both evaluations ($(python -c "import json;print(len(open('$OUT/samples.jsonl').readlines()))") questions each) finished without recorded exceptions. Results: $OUT"
python - "$OUT" <<'REPORT'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
report={}
for arm in ('control','treatment'):
    rows=[json.loads(s) for s in (root/arm/'agent.jsonl').read_text().splitlines()]
    report[arm]={k:sum(r[k] for r in rows)/len(rows) for k in ('answer_f1','evidence_f1','latency_s','prompt_tokens','completion_tokens')}
    report[arm]['extra_answer_attempts']=sum(r['extra_answer_attempted'] for r in rows)
    report[arm]['extra_answer_seconds']=sum(r['extra_answer_seconds'] for r in rows)
    report[arm]['extra_completion_tokens']=sum((r['extra_answer_tokens'] or {}).get('completion_tokens',0) for r in rows)
(root/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
REPORT
