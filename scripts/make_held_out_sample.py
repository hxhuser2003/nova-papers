"""Build a larger held-out sample-questions file from QASPER's test split, for
the combined-experiment power analysis in TRAINING_NOTES.md (2026-10-27): the
original 40-question eval/sample_questions.jsonl is too small to tell a real
+0.03 Answer F1 effect from noise (required_n ~= 436 at the observed
std_diff). Reuses score_real_rollouts.usable()'s filter (annotators agree on
answer type, no non-text evidence, answerable questions have evidence) so
these questions are screened the same way the rest of the project's real
evaluations are, and the test split is never touched by GRPO training or by
checkpoint/parameter selection (that's what validation is for).

Usage: python -m scripts.make_held_out_sample --out eval/sample_questions_test500.jsonl
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
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    corpus = QasperCorpus(Path(args.corpus))
    pools: dict[str, list[dict]] = defaultdict(list)
    for q in corpus.questions_for_split(args.split):
        kind = usable(q)
        if kind:
            pools[kind].append(dict(q, answer_type=kind))

    # Take every usable question from the three scarce categories, then fill
    # the rest from the abundant "extractive" pool up to this target total.
    rng = random.Random(args.seed)
    target_total = 480
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
    print(f"picked {len(picked)} questions from split={args.split}: {dict(counts)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in picked:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
