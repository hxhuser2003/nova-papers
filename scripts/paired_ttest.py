"""Paired t-test between two score_real_rollouts.py rollouts.jsonl files,
aligned by question_id, averaging each question's G rollouts first.

Usage: python3 scripts/paired_ttest.py <baseline.jsonl> <treatment.jsonl> [metric]
  metric defaults to answer_f1 (also accepts evidence_f1, reward)
"""
import json
import statistics
import sys


def per_question_means(path: str, metric: str) -> dict[str, float]:
    out = {}
    with open(path) as f:
        for line in f:
            g = json.loads(line)
            out[g["question_id"]] = statistics.fmean(r[metric] for r in g["rollouts"]) if metric != "reward" \
                else statistics.fmean(g["rewards"])
    return out


def main() -> None:
    baseline_path, treatment_path = sys.argv[1], sys.argv[2]
    metric = sys.argv[3] if len(sys.argv) > 3 else "answer_f1"
    baseline = per_question_means(baseline_path, metric)
    treatment = per_question_means(treatment_path, metric)
    shared = sorted(set(baseline) & set(treatment))
    if len(shared) < len(baseline) or len(shared) < len(treatment):
        print(f"WARNING: only {len(shared)} question_ids shared "
              f"(baseline has {len(baseline)}, treatment has {len(treatment)})", file=sys.stderr)
    diffs = [treatment[q] - baseline[q] for q in shared]
    n = len(diffs)
    mean_diff = statistics.fmean(diffs)
    sample_std = statistics.stdev(diffs) if n > 1 else float("nan")
    t_stat = mean_diff / (sample_std / n ** 0.5) if sample_std else float("inf")
    print(f"metric={metric} n={n} mean_diff={mean_diff:+.4f} sample_std={sample_std:.4f} t_stat={t_stat:.3f}")
    print("SIGNIFICANT at |t|>=1.96" if abs(t_stat) >= 1.96 else "NOT significant at |t|>=1.96")


if __name__ == "__main__":
    main()
