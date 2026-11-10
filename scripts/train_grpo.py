"""Single-GPU QASPER GRPO with trainer-synchronized vLLM and LoRA.

Run as a module: python -m scripts.train_grpo. TRL's rollout API is
experimental; the adapter targets trl==1.16.0. See TRAINING_NOTES.md.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from backend.agent_loop import parse_action
from backend.training_metrics import PhaseMetrics
from backend.qasper_corpus import QasperCorpus
from backend.reward import (RolloutSample, RewardWeights, composite_reward,
                            action_reward, correctness_reward, faithfulness_reward)
from backend.token_rollout import ColocatedModelClient, run_token_episode

# Module-level so _reward_func (called by TRL with a fixed (completions, **kwargs)
# signature -- there is no channel to pass extra config through that call) can read
# whatever main() set before trainer.train() starts. Only main() mutates this, and
# training is single-process/synchronous, so there is no concurrent-write hazard.
# Tests set this directly instead of going through argparse.
_WEIGHTS = RewardWeights()


def build_dataset_rows(corpus: QasperCorpus, split: str) -> dict[str, list[Any]]:
    questions = corpus.questions_for_split(split)
    return {
        # The callback receives prompts only, not dataset columns. Include a
        # stable identity: identical question text can occur in several papers.
        # This envelope is resolved before tokenization and never shown to Qwen.
        'prompt': [json.dumps({'question_id': q['question_id'], 'question': q['question']}) for q in questions],
        'paper_id': [q['paper_id'] for q in questions],
        'question_id': [q['question_id'] for q in questions],
        'answers': [q['answers'] for q in questions],
    }


def _rollout_func(prompts: list[str], trainer) -> dict[str, Any]:
    """One result per incoming row: TRL's RepeatSampler already repeats rows.

    Do NOT call run_group here: doing so multiplies the batch a second time
    and misaligns it with rewards. Token prompt IDs contain the actual full
    chat-templated paper/question context, not the dataset's identity envelope.
    """
    corpus = trainer.qasper_corpus
    model = getattr(trainer, 'rollout_model', None) or ColocatedModelClient(trainer)
    fields = ('prompt_ids', 'completion_ids', 'logprobs', 'env_mask',
              'model_outputs', 'evidence_para_ids', 'evidence_passages',
              'final_answer', 'terminated_reason')
    result = {key: [] for key in (*fields, 'question')}
    for prompt in prompts:
        question = corpus.questions[json.loads(prompt)['question_id']]
        trace = run_token_episode(question, corpus, trainer.processing_class, model,
                                  **trainer.rollout_options)
        for key in fields:
            result[key].append(getattr(trace, key))
        result['question'].append(question['question'])
    return result


def _reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    """Score model turns, never the decoded tool-bearing completion stream."""
    size = len(completions)
    def batch(key, default):
        values = kwargs.get(key, [default for _ in range(size)])
        if len(values) != size:
            raise ValueError(f'Reward batch length mismatch: {key}')
        return values
    answers_batch = batch('answers', [])
    if 'answers' not in kwargs:
        raise ValueError('QASPER answers are required')
    evidence_batch = batch('evidence_passages', [])
    questions = batch('question', '')
    ids_batch = batch('evidence_para_ids', [])
    outputs_batch = batch('model_outputs', None)
    finals = batch('final_answer', None)
    weights = _WEIGHTS
    rewards = []
    for i, completion in enumerate(completions):
        outputs = outputs_batch[i] or [completion]
        sample = RolloutSample(raw_output=outputs[-1], evidence_para_ids=ids_batch[i])
        value = composite_reward(questions[i], sample, evidence_batch[i], answers_batch[i], weights=weights)
        # A failed episode has no answer, even if malformed text resembles gold.
        if 'final_answer' in kwargs and finals[i] is None:
            value -= weights.correctness * correctness_reward(sample, answers_batch[i])
            value -= weights.faithfulness * faithfulness_reward(questions[i], sample, evidence_batch[i])
        # Account for repeated searches anywhere in the trajectory, not only
        # the last turn (which is normally an answer, with neutral action score).
        history, scores = [], []
        for output in outputs:
            turn = RolloutSample(output, history_search_keywords=list(history))
            scores.append(action_reward(turn))
            action = parse_action(output)
            if action.kind == 'search':
                history.append(action.content)
        value += weights.action * (sum(scores) / len(scores) - action_reward(sample))
        # history is only populated by a <search> turn above, so a non-empty
        # history means the trajectory searched at least once, regardless of
        # how it ended -- this is the staged-reward schedule's stage-1
        # bootstrap signal (see _staged_weights); 0.0 everywhere else.
        value += weights.retrieval_bonus * (1.0 if history else 0.0)
        rewards.append(value)
    return rewards


def _staged_weights(step: int, stage1_steps: int, final_weights: RewardWeights) -> RewardWeights:
    """R1-Searcher-style two-stage schedule: before stage1_steps, bootstrap
    tool use (drop the outcome-based signals, add a bonus for searching at
    all); from stage1_steps onward, use final_weights unchanged -- whatever
    the existing --zero-*/--evidence-weight ablation flags already resolved
    to, so staging composes with those instead of needing its own branch."""
    if step < stage1_steps:
        return replace(final_weights, correctness=0.0, evidence=0.0, retrieval_bonus=0.5)
    return final_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--corpus', default='corpus/qasper')
    parser.add_argument('--split', default='train')
    parser.add_argument('--output-dir', default='checkpoints/grpo')
    parser.add_argument('--num-generations', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=2)
    parser.add_argument('--max-rounds', type=int, default=5)
    parser.add_argument('--max-completion-length', type=int, default=4096)
    parser.add_argument('--max-round-tokens', type=int, default=500)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.35)
    parser.add_argument('--learning-rate', type=float, default=5e-6,
                        help='R-series tuning: GRPO optimizer learning rate, previously hardcoded and never varied')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='R-series tuning: KL penalty coefficient against the reference policy, '
                             '0.0 (the prior hardcoded default) means no KL constraint at all')
    ablation = parser.add_mutually_exclusive_group()
    ablation.add_argument('--zero-format-reward', action='store_true',
                          help='R1 ablation: set the format reward weight to 0, others unchanged')
    ablation.add_argument('--zero-evidence-reward', action='store_true',
                          help='R2 ablation: set the evidence reward weight to 0, others unchanged')
    ablation.add_argument('--evidence-weight', type=float, default=None,
                          help='Priority-5 tuning: evidence reward weight at a continuous value '
                               '(e.g. 0.2/0.5/1.0), others unchanged -- R2 only tested the on/off '
                               'extremes (0.0 vs the RewardWeights() default of 0.5); mutually '
                               'exclusive with --zero-evidence-reward since that is just this at 0.0')
    parser.add_argument('--search-limit', type=int, default=3,
                        help='R5 training-side ablation: how many passages <search> returns per '
                             'call during rollout collection (QasperCorpus/HybridQasperCorpus '
                             "search_limit, default 3 matches the prior hardcoded value). Vary "
                             "this at training time while keeping inference-time evaluation at "
                             'the default (score_real_rollouts.py/run_baseline.py do not override '
                             "it) to isolate the training-side variable, per the R5/P2 plan in "
                             "TRAINING_NOTES.md.")
    parser.add_argument('--stage1-steps', type=int, default=None,
                        help='Staged (curriculum) reward schedule, R1-Searcher-style: for the '
                             'first N steps, bootstrap tool use (correctness=0, evidence=0, a '
                             'retrieval_bonus for having searched at all) before switching to the '
                             'normal resolved weights for the rest of training. Default None means '
                             'no staging -- identical behavior to every previously-trained '
                             'checkpoint (R1-R10/P1-P5), none of which used this.')
    parser.add_argument('--retriever', choices=('bm25', 'hybrid'), default='bm25',
                        help="R10 training-side ablation: 'hybrid' swaps in HybridQasperCorpus "
                             '(BM25+dense+RRF, already validated inference-side) for rollout '
                             'search() calls during training; requires sentence-transformers '
                             'installed separately (not in requirements-training.txt, matching '
                             "carc_eval_hybrid_retrieval.sh's own ad hoc install)")
    args = parser.parse_args()
    global _WEIGHTS
    if args.zero_format_reward:
        _WEIGHTS = replace(RewardWeights(), format=0.0)
    elif args.zero_evidence_reward:
        _WEIGHTS = replace(RewardWeights(), evidence=0.0)
    elif args.evidence_weight is not None:
        _WEIGHTS = replace(RewardWeights(), evidence=args.evidence_weight)
    # Captured once, before any staging callback starts mutating _WEIGHTS
    # during training -- the end-of-run report must describe what was
    # actually resolved from the CLI flags above, not whatever step the
    # schedule happened to leave _WEIGHTS on when training stopped.
    _final_weights = _WEIGHTS
    from importlib.metadata import version
    if version('trl') != '1.16.0':
        raise RuntimeError('Install trl==1.16.0; custom rollout API must be revalidated for other versions')
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback
    if not torch.cuda.is_available():
        raise RuntimeError('This training entrypoint requires a CUDA GPU')
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('Use an A40/A100/L40S for this vLLM training environment')
    if args.retriever == 'hybrid':
        from backend.hybrid_retrieval import HybridQasperCorpus, SentenceTransformerEmbedder
        corpus = HybridQasperCorpus(Path(args.corpus), SentenceTransformerEmbedder(), search_limit=args.search_limit)
    else:
        corpus = QasperCorpus(Path(args.corpus), search_limit=args.search_limit)
    dataset = Dataset.from_dict(build_dataset_rows(corpus, args.split))
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if not len(dataset):
        raise ValueError('Training split is empty')
    config = GRPOConfig(
        output_dir=args.output_dir, num_generations=args.num_generations,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.num_generations,
        generation_batch_size=args.num_generations, max_steps=args.max_steps,
        max_completion_length=args.max_completion_length,
        use_vllm=True, vllm_mode='colocate', vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_max_model_length=args.max_model_len,
        bf16=True, model_init_kwargs={'dtype': 'bfloat16'},
        learning_rate=args.learning_rate, beta=args.beta, temperature=1.0,
        mask_truncated_completions=False, report_to='none',
        logging_steps=1, save_strategy='no', seed=42,
    )
    metrics = PhaseMetrics(Path(args.output_dir) / 'phase_metrics.jsonl', cuda=torch.cuda)

    class MetricsCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            metrics.start('parameter_update', state.global_step)
        def on_step_end(self, args, state, control, **kwargs):
            metrics.stop()

    class MeasuredGRPOTrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            with metrics.phase('rollout_and_scoring', self.state.global_step):
                return super()._generate_and_score_completions(inputs)

    callbacks = [MetricsCallback()]
    if args.stage1_steps is not None:
        # on_step_begin's own `args` parameter is TRL's GRPOConfig, not this
        # function's argparse Namespace -- it would shadow the outer `args`
        # (which has no .stage1_steps) if referenced by that name inside the
        # method, so the CLI value is captured here as a separate variable.
        _stage1_steps = args.stage1_steps

        class RewardScheduleCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                global _WEIGHTS
                _WEIGHTS = _staged_weights(state.global_step, _stage1_steps, _final_weights)
        callbacks.append(RewardScheduleCallback())

    trainer = MeasuredGRPOTrainer(
        model=args.model, args=config, train_dataset=dataset,
        reward_funcs=_reward_func, rollout_func=_rollout_func, callbacks=callbacks,
        peft_config=LoraConfig(r=8, lora_alpha=16, target_modules='all-linear', task_type='CAUSAL_LM'),
    )
    trainer.qasper_corpus = corpus
    trainer.rollout_options = dict(max_rounds=args.max_rounds,
                                   max_completion_length=args.max_completion_length,
                                   max_round_tokens=args.max_round_tokens,
                                   max_model_len=args.max_model_len)
    before = {name: parameter.detach().cpu().clone()
              for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
    try:
        trainer.train()
    finally:
        metrics.stop()
    changed = [name for name, parameter in trainer.model.named_parameters()
               if name in before and not torch.equal(before[name], parameter.detach().cpu())]
    report = dict(global_steps=trainer.state.global_step, changed_trainable_tensors=len(changed),
                  model=args.model, trl_version=version('trl'), vllm_version=version('vllm'),
                  reward_weights=asdict(_final_weights), stage1_steps=args.stage1_steps,
                  arguments=vars(args))
    if args.stage1_steps is not None:
        report['stage1_weights'] = asdict(_staged_weights(0, args.stage1_steps, _final_weights))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / 'training_verification.json').write_text(json.dumps(report, indent=2))
    if not changed:
        raise RuntimeError('No trainable parameters changed; inspect zero-variance rewards before claiming success')
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
