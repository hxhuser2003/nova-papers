"""GRPO training entrypoint (task 4, training half).

Status: the overall structure (dataset construction, GRPOConfig, reward
wiring, trainer.train()) is real and matches TRL's actual GRPOTrainer
contract (checked against trl's source, not just docs -- see
TRAINING_NOTES.md). `_rollout_func` is a deliberate stub: it is the one
piece that cannot be completed yet.

Known gap (see TRAINING_NOTES.md): rollout_func must return "prompt_ids",
"completion_ids", and "logprobs" that correspond exactly to the training
tokenizer -- not decoded text. backend/vllm_client.py only returns decoded
text today (it was built for agent_loop.py's inference-time loop, not
training), so there is nothing yet that can produce those fields. Trying
to run this script end to end will raise NotImplementedError at that
point rather than silently doing the wrong thing.

trl/datasets are imported lazily (inside main(), not at module level) so
build_dataset()/_reward_func() can be unit-tested on a machine that
doesn't have them installed -- this repo's dev machine doesn't, CARC does.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from backend.qasper_corpus import QasperCorpus
from backend.reward import RolloutSample, composite_reward


def build_dataset_rows(corpus: QasperCorpus, split: str) -> dict[str, list[Any]]:
    """One row per question. The prompt is just the question text --
    agent_loop.build_initial_context assembles the full system prompt and
    the paper's title/abstract at rollout time, not here, so this stays a
    plain dict of columns rather than a pre-rendered chat prompt."""
    questions = corpus.questions_for_split(split)
    return {
        "prompt": [q["question"] for q in questions],
        "paper_id": [q["paper_id"] for q in questions],
        "question_id": [q["question_id"] for q in questions],
        "answers": [q["answers"] for q in questions],
    }


def _rollout_func(prompts: list[str], trainer) -> dict[str, Any]:
    """STUB. See the module docstring and TRAINING_NOTES.md. Needs to:

    1. For each prompt (= question), call backend.agent_loop.run_group()
       via a training-mode ModelClient that captures token ids + logprobs
       from vLLM -- not the decoded-text-only VLLMModelClient used for
       inference/baseline eval.
    2. Convert each AgentTrajectory's rounds into prompt_ids/completion_ids/
       logprobs, plus an env_mask (0 for tokens inside <information>
       blocks, so tool output doesn't get gradient -- TRL's native
       equivalent of Search-R1's info_mask, see TRAINING_NOTES.md).
    3. Return extra fields (answers, evidence_para_ids, question,
       evidence_passages) alongside the required keys; TRL forwards
       anything beyond {"prompt_ids", "completion_ids", "logprobs"} to
       the reward function unchanged.
    """
    raise NotImplementedError(
        "rollout_func needs a token-id/logprob-capturing ModelClient; "
        "backend/vllm_client.py only returns decoded text today. "
        "See TRAINING_NOTES.md's entry on this gap."
    )


def _reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    """Wraps backend.reward.composite_reward for TRL's reward_funcs
    interface. kwargs carries dataset columns plus whatever extra fields
    _rollout_func returns (answers always; question/evidence_passages/
    evidence_para_ids once the stub above is implemented)."""
    answers_batch = kwargs["answers"]
    evidence_batch = kwargs.get("evidence_passages", [[] for _ in completions])
    question_batch = kwargs.get("question", ["" for _ in completions])
    evidence_ids_batch = kwargs.get("evidence_para_ids", [[] for _ in completions])
    rewards = []
    for completion, answers, evidence, question, evidence_ids in zip(
        completions, answers_batch, evidence_batch, question_batch, evidence_ids_batch
    ):
        sample = RolloutSample(raw_output=completion, evidence_para_ids=evidence_ids)
        rewards.append(composite_reward(question, sample, evidence, answers))
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--corpus", default="corpus/qasper")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="checkpoints/grpo")
    parser.add_argument("--num-generations", type=int, default=5)
    args = parser.parse_args()

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    corpus = QasperCorpus(Path(args.corpus))
    dataset = Dataset.from_dict(build_dataset_rows(corpus, args.split))

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        use_vllm=True,
        vllm_mode="colocate",  # single GPU, matches our CARC allocation
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        reward_funcs=_reward_func,
        rollout_func=_rollout_func,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
