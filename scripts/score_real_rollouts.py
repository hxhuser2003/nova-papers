"""Score real Qwen rollouts with the exact reward used in GRPO training.

Part "4 (data/validation)": before spending CARC time on training, run the
*real* model on a stratified set of train-split questions, G rollouts per
question (same sampling as training: temperature 1.0, <=500 tokens/round,
<=5 rounds), score every rollout with scripts.train_grpo._reward_func (the
function GRPOTrainer actually calls, not a re-implementation), and compute
group advantages. Answers two questions:

  1. Does the reward behave on real (not scripted) model output?
  2. Do rollouts in a group get *different* rewards? A zero-variance group
     has all-zero advantages, so GRPO learns nothing from it.

Needs a running vLLM server (OpenAI-compatible). In Colab:
    python -m scripts.score_real_rollouts --base-url http://localhost:8000

Outputs (in --out-dir, default eval/rollout_scoring/):
    rollouts.jsonl  one line per question: G rollouts + rewards + advantages
    summary.md      aggregate tables
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from backend.agent_loop import AgentTrajectory, run_group
from backend.qasper_corpus import QasperCorpus
from backend.qasper_metrics import answer_f1, evidence_f1
from backend.reward import group_advantages
from backend.vllm_client import VLLMModelClient
from scripts.train_grpo import _reward_func

# Same passage-header pattern backend/token_rollout.py uses for training.
PASSAGE = re.compile(r"^\[(p_\d+)\] \([^\n]*?\) (.*)$", re.MULTILINE)
TYPES = ("extractive", "free_form", "yes_no", "unanswerable")


def usable(q: dict) -> str | None:
    """Same spirit as the 40-question sample-set rules: annotators agree on the
    answer type, no table/figure evidence, answerable questions have evidence."""
    types = {a["answer_type"] for a in q["answers"]}
    if len(types) != 1:
        return None
    kind = types.pop()
    if any(a.get("has_non_text_evidence") for a in q["answers"]):
        return None
    if kind != "unanswerable" and not all(a.get("evidence_para_ids") for a in q["answers"]):
        return None
    return kind


def pick_questions(corpus: QasperCorpus, split: str, quotas: dict[str, int], seed: int) -> list[dict]:
    pools = defaultdict(list)
    for q in corpus.questions_for_split(split):
        kind = usable(q)
        if kind:
            pools[kind].append(q)
    rng = random.Random(seed)
    picked = []
    for kind, n in quotas.items():
        pool = sorted(pools[kind], key=lambda q: q["question_id"])
        picked += [dict(q, answer_type=kind) for q in rng.sample(pool, min(n, len(pool)))]
    return picked


def describe(traj: AgentTrajectory, answers: list[dict]) -> dict:
    ids, passages = [], []
    for step in traj.steps:
        if step.action.kind in ("search", "read") and step.tool_result:
            for pid, text in PASSAGE.findall(step.tool_result):
                if pid not in ids:
                    ids.append(pid)
                passages.append(text)
    return {
        "final_answer": traj.final_answer,
        "terminated_reason": traj.terminated_reason,
        "rounds": len(traj.steps),
        "actions": [s.action.kind or "format_error" for s in traj.steps],
        "model_outputs": [s.model_output for s in traj.steps],
        "evidence_para_ids": ids,
        "evidence_passages": passages,
        "answer_f1": answer_f1(traj.final_answer, answers),  # eval/BASELINE.md metric
        "evidence_f1": evidence_f1(set(ids), answers),
    }


def score_group(q: dict, rollouts: list[dict]) -> list[float]:
    n = len(rollouts)
    return _reward_func(
        completions=["".join(r["model_outputs"]) for r in rollouts],
        answers=[q["answers"]] * n,
        question=[q["question"]] * n,
        evidence_passages=[r["evidence_passages"] for r in rollouts],
        evidence_para_ids=[r["evidence_para_ids"] for r in rollouts],
        model_outputs=[r["model_outputs"] or [""] for r in rollouts],
        final_answer=[r["final_answer"] for r in rollouts],
    )


def run_question(q: dict, corpus: QasperCorpus, model: VLLMModelClient, group_size: int) -> dict:
    start = time.monotonic()
    trajs = run_group(q["question_id"], q["question"], q["paper_id"], corpus, model, group_size=group_size)
    rollouts = [describe(t, q["answers"]) for t in trajs]
    rewards = score_group(q, rollouts)
    for r, reward in zip(rollouts, rewards):
        r["reward"] = round(reward, 4)
        del r["evidence_passages"]  # large; ids are enough for inspection
    return {
        "question_id": q["question_id"], "paper_id": q["paper_id"], "question": q["question"],
        "answer_type": q["answer_type"],
        "gold_answers": [a.get("answer_text") or "Unanswerable" for a in q["answers"]],
        "rollouts": rollouts,
        "rewards": [round(x, 4) for x in rewards],
        "advantages": [round(x, 4) for x in group_advantages(rewards)],
        "reward_std": round(statistics.pstdev(rewards), 4),
        "seconds": round(time.monotonic() - start, 1),
    }


def summarize(groups: list[dict], args) -> str:
    rows = [r for g in groups for r in g["rollouts"]]
    zero = [g for g in groups if g["reward_std"] < 1e-6]
    lines = [
        "# Real-rollout reward check", "",
        f"Model `{args.model}`, {len(groups)} train-split questions x G={args.group_size}, "
        f"temperature {args.temperature}, max {args.max_tokens} tokens/round, seed {args.seed}.", "",
        "## Overall", "",
        "| metric | value |", "| --- | --- |",
        f"| rollouts | {len(rows)} |",
        f"| mean reward | {statistics.fmean(r['reward'] for r in rows):.3f} |",
        f"| mean Answer F1 | {statistics.fmean(r['answer_f1'] for r in rows):.3f} |",
        f"| mean Evidence F1 | {statistics.fmean(r['evidence_f1'] for r in rows):.3f} |",
        f"| mean rounds | {statistics.fmean(r['rounds'] for r in rows):.2f} |",
        f"| groups with zero reward variance (no GRPO signal) | {len(zero)}/{len(groups)} |",
        f"| mean seconds per question (G rollouts) | {statistics.fmean(g['seconds'] for g in groups):.1f} |",
        "", "## How rollouts ended", "", "| terminated_reason | count | mean reward |", "| --- | --- | --- |",
    ]
    by_end = defaultdict(list)
    for r in rows:
        by_end[r["terminated_reason"]].append(r["reward"])
    for end, vals in sorted(by_end.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {end} | {len(vals)} | {statistics.fmean(vals):.3f} |")
    lines += ["", "## By question type", "",
              "| type | questions | mean reward | mean Answer F1 | zero-variance groups |",
              "| --- | --- | --- | --- | --- |"]
    for kind in TYPES:
        gs = [g for g in groups if g["answer_type"] == kind]
        if not gs:
            continue
        rs = [r for g in gs for r in g["rollouts"]]
        lines.append(f"| {kind} | {len(gs)} | {statistics.fmean(r['reward'] for r in rs):.3f} | "
                     f"{statistics.fmean(r['answer_f1'] for r in rs):.3f} | "
                     f"{sum(g['reward_std'] < 1e-6 for g in gs)}/{len(gs)} |")
    # Sanity check: correct answers should outscore wrong/unanswered ones.
    right = [r["reward"] for r in rows if r["answer_f1"] >= 0.5]
    wrong = [r["reward"] for r in rows if r["final_answer"] is not None and r["answer_f1"] < 0.5]
    none = [r["reward"] for r in rows if r["final_answer"] is None]
    lines += ["", "## Sanity check: reward vs. answer quality", "",
              "| group | rollouts | mean reward |", "| --- | --- | --- |"]
    for name, vals in (("answered, Answer F1 >= 0.5", right), ("answered, Answer F1 < 0.5", wrong),
                       ("no answer", none)):
        lines.append(f"| {name} | {len(vals)} | {statistics.fmean(vals):.3f} |" if vals else f"| {name} | 0 | - |")
    actions = Counter(a for r in rows for a in r["actions"])
    lines += ["", f"Actions used: {dict(actions)}", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--corpus", default="corpus/qasper")
    p.add_argument("--split", default="train")
    p.add_argument("--quotas", default="extractive=7,free_form=5,yes_no=5,unanswerable=3")
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)  # matches GRPOConfig in train_grpo.py
    p.add_argument("--max-tokens", type=int, default=500)  # matches --max-round-tokens
    p.add_argument("--parallel-questions", type=int, default=4)
    p.add_argument("--seed", type=int, default=599)
    p.add_argument("--out-dir", default="eval/rollout_scoring")
    args = p.parse_args()

    httpx.get(f"{args.base_url.rstrip('/')}/v1/models", timeout=10).raise_for_status()
    corpus = QasperCorpus(Path(args.corpus))
    quotas = {k: int(v) for k, v in (kv.split("=") for kv in args.quotas.split(","))}
    questions = pick_questions(corpus, args.split, quotas, args.seed)
    model = VLLMModelClient(args.base_url, model=args.model, temperature=args.temperature,
                            max_tokens=args.max_tokens, timeout=300)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = []
    with ThreadPoolExecutor(max_workers=args.parallel_questions) as pool, \
            open(out / "rollouts.jsonl", "w") as f:
        for g in pool.map(lambda q: run_question(q, corpus, model, args.group_size), questions):
            groups.append(g)
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{len(groups)}/{len(questions)}] {g['answer_type']:<12} rewards={g['rewards']} "
                  f"({g['seconds']}s)", flush=True)
    summary = summarize(groups, args)
    (out / "summary.md").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
