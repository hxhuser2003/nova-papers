"""Paired t-test between two run_baseline.py/run_agent_ablation.py agent.jsonl
files (one flat row per question, not grouped like score_real_rollouts.py),
aligned by question_id.

Usage: python3 scripts/paired_ttest_agent.py <control.jsonl> <treatment.jsonl> [metric]
  metric defaults to answer_f1 (also accepts evidence_f1)
"""
import json
import statistics
import sys


def per_question(path: str, metric: str) -> dict[str, float]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["question_id"]] = r[metric]
    return out


def main() -> None:
    control_path, treatment_path = sys.argv[1], sys.argv[2]
    metric = sys.argv[3] if len(sys.argv) > 3 else "answer_f1"
    control = per_question(control_path, metric)
    treatment = per_question(treatment_path, metric)
    shared = sorted(set(control) & set(treatment))
    if len(shared) < len(control) or len(shared) < len(treatment):
        print(f"WARNING: only {len(shared)} question_ids shared "
              f"(control has {len(control)}, treatment has {len(treatment)})", file=sys.stderr)
    diffs = [treatment[q] - control[q] for q in shared]
    n = len(diffs)
    mean_diff = statistics.fmean(diffs)
    sample_std = statistics.stdev(diffs) if n > 1 else float("nan")
    t_stat = mean_diff / (sample_std / n ** 0.5) if sample_std else float("inf")
    print(f"metric={metric} n={n} mean_diff={mean_diff:+.4f} sample_std={sample_std:.4f} t_stat={t_stat:.3f}")
    print("SIGNIFICANT at |t|>=1.96" if abs(t_stat) >= 1.96 else "NOT significant at |t|>=1.96")


if __name__ == "__main__":
    main()
