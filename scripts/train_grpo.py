"""Single-GPU QASPER GRPO with trainer-synchronized vLLM and LoRA.

Run as a module: python -m scripts.train_grpo. TRL's rollout API is
experimental; the adapter targets trl==1.16.0. See TRAINING_NOTES.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.agent_loop import parse_action
from backend.training_metrics import PhaseMetrics
from backend.qasper_corpus import QasperCorpus
from backend.reward import (RolloutSample, RewardWeights, composite_reward,
                            action_reward, correctness_reward, faithfulness_reward)
from backend.token_rollout import ColocatedModelClient, run_token_episode


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
    weights = RewardWeights()
    rewards = []
    for i, completion in enumerate(completions):
        outputs = outputs_batch[i] or [completion]
        sample = RolloutSample(raw_output=outputs[-1], evidence_para_ids=ids_batch[i])
        value = composite_reward(questions[i], sample, evidence_batch[i], answers_batch[i])
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
        rewards.append(value)
    return rewards


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
    args = parser.parse_args()
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
    corpus = QasperCorpus(Path(args.corpus))
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
        learning_rate=5e-6, beta=0.0, temperature=1.0,
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

    trainer = MeasuredGRPOTrainer(
        model=args.model, args=config, train_dataset=dataset,
        reward_funcs=_reward_func, rollout_func=_rollout_func, callbacks=[MetricsCallback()],
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
                  arguments=vars(args))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / 'training_verification.json').write_text(json.dumps(report, indent=2))
    if not changed:
        raise RuntimeError('No trainable parameters changed; inspect zero-variance rewards before claiming success')
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
