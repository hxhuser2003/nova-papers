"""Build a larger held-out sample-questions file from QASPER, for the
combined-experiment power analysis in TRAINING_NOTES.md. Reuses
score_real_rollouts.usable()'s filter (annotators agree on answer type, no
non-text evidence, answerable questions have evidence) so these questions
are screened the same way the rest of the project's real evaluations are.

The combined experiment (carc_eval_agent_ablation.sh) serves the raw,
untrained Qwen2.5-3B-Instruct base model directly -- no GRPO checkpoint is
loaded -- so unlike the R1-R4 reward ablations, there is no train/validation/
test leakage concern here: nothing has ever been fine-tuned on any QASPER
split. --splits can therefore pool train+validation+test to reach a larger
target (2026-11-02: the first 480-question test-split-only run measured a
smaller, noisier effect than hoped, needing ~1839 questions for the actual
observed effect size/std -- more than any single split's usable pool holds).

Usage: python -m scripts.make_held_out_sample --out eval/sample_questions_test500.jsonl
       python -m scripts.make_held_out_sample --splits train,validation,test --target-total 1839 --out eval/sample_questions_pooled1839.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from backend.qasper_corpus import QasperCorpus
from scripts.score_real_rollouts import usable


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="corpus/qasper")
    p.add_argument("--splits", default="test", help="comma-separated QASPER splits to pool from")
    p.add_argument("--target-total", type=int, default=480)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    corpus = QasperCorpus(Path(args.corpus))
    pools: dict[str, list[dict]] = defaultdict(list)
    for split in (s.strip() for s in args.splits.split(",")):
        for q in corpus.questions_for_split(split):
            kind = usable(q)
            if kind:
                pools[kind].append(dict(q, answer_type=kind))

    # Take every usable question from the three scarce categories, then fill
    # the rest from the abundant "extractive" pool up to this target total.
    rng = random.Random(args.seed)
    target_total = args.target_total
    picked: list[dict] = []
    for kind in ("free_form", "unanswerable", "yes_no"):
        pool = sorted(pools[kind], key=lambda q: q["question_id"])
        picked += pool
    remaining = target_total - len(picked)
    extractive_pool = sorted(pools["extractive"], key=lambda q: q["question_id"])
    picked += rng.sample(extractive_pool, min(remaining, len(extractive_pool)))

    counts = defaultdict(int)
    for q in picked:
        counts[q["answer_type"]] += 1
    print(f"picked {len(picked)} questions from splits={args.splits}: {dict(counts)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in picked:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
