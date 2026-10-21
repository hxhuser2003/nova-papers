"""Append-only token trajectories for QASPER GRPO.

Model IDs are never decoded and re-encoded into the training sequence.
Only initial prompts and external chat/tool suffixes are encoded locally.
The state machine intentionally retains inference's current stop rules.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from backend.agent_loop import (
    MAX_CONSECUTIVE_FORMAT_ERRORS, MAX_ROUNDS, build_initial_context,
    parse_action, _execute_tool,
)
from backend.vllm_client import STOP_SEQUENCES


@dataclass
class TokenTrajectory:
    prompt_ids: list[int]
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    env_mask: list[int] = field(default_factory=list)
    model_outputs: list[str] = field(default_factory=list)
    evidence_para_ids: list[str] = field(default_factory=list)
    evidence_passages: list[str] = field(default_factory=list)
    final_answer: str | None = None
    terminated_reason: str = 'max_rounds'


class ColocatedModelClient:
    """Use TRL's own vLLM engine, whose weights GRPOTrainer synchronizes.

    Single process only: variable-length episodes must not enter distributed
    generation collectives independently. TRL v1.16.0 is the tested contract.
    """
    def __init__(self, trainer):
        self.trainer = trainer
        if trainer.accelerator.num_processes != 1:
            raise ValueError('Token rollouts currently require a single training process')
        if trainer.vllm_generation.mode != 'colocate':
            raise ValueError('Training requires trainer-managed colocated vLLM')

    def generate(self, prompt, stop=None, max_tokens=500):
        gen = self.trainer.vllm_generation
        previous_length, previous_kwargs = gen.max_completion_length, gen.generation_kwargs
        try:
            gen.max_completion_length = max_tokens
            gen.generation_kwargs = dict(previous_kwargs, max_tokens=max_tokens,
                                         stop=stop or STOP_SEQUENCES,
                                         include_stop_str_in_output=True)
            _, completions, logprobs, token_ids = gen.generate([prompt], images=None, num_generations=1)
        finally:
            gen.max_completion_length, gen.generation_kwargs = previous_length, previous_kwargs
        ids = completions[0]
        # extract_logprobs returns sampled token first; verify using explicit IDs.
        values = []
        for token, alternatives, scores in zip(ids, token_ids[0], logprobs[0]):
            values.append(scores[alternatives.index(token)])
        return {'text': self.trainer.processing_class.decode(ids, skip_special_tokens=False),
                'token_ids': ids, 'logprobs': values}


def run_token_episode(question, corpus, tokenizer, model, *, max_rounds=MAX_ROUNDS,
                      max_completion_length=4096, max_round_tokens=500,
                      max_model_len=8192):
    if min(max_rounds, max_completion_length, max_round_tokens, max_model_len) <= 0:
        raise ValueError('Token and round budgets must be positive')
    paper = corpus.papers.get(question['paper_id'])
    if paper is None:
        raise ValueError(f"Unknown paper: {question['paper_id']}")
    messages = [{'role': 'user', 'content': build_initial_context(question['question'], paper)}]
    def render():
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    rendered = render()
    initial_ids = tokenizer.encode(rendered, add_special_tokens=False)
    budget = min(max_completion_length, max_model_len - len(initial_ids))
    if budget <= 0:
        raise ValueError('Initial prompt exceeds model context; do not silently truncate the question')
    trace = TokenTrajectory(prompt_ids=initial_ids)
    errors = 0
    for round_index in range(max_rounds):
        remaining = budget - len(trace.completion_ids)
        if remaining <= 0:
            trace.terminated_reason = 'max_length'
            break
        reply = model.generate(trace.prompt_ids + trace.completion_ids,
                               stop=STOP_SEQUENCES, max_tokens=min(max_round_tokens, remaining))
        ids, values = reply['token_ids'], reply['logprobs']
        if not ids or len(ids) != len(values) or len(ids) > min(max_round_tokens, remaining):
            raise ValueError('Invalid generated token/logprob lengths')
        if any(type(i) is not int or i < 0 for i in ids) or any(
            not isinstance(v, (float, int)) or not math.isfinite(v) for v in values
        ):
            raise ValueError('Generated tokens and logprobs must be valid and finite')
        raw = tokenizer.decode(ids, skip_special_tokens=False)
        trace.completion_ids.extend(ids)
        trace.logprobs.extend(values)
        trace.env_mask.extend([1] * len(ids))
        trace.model_outputs.append(raw)
        action = parse_action(raw)
        if action.kind == 'answer':
            trace.final_answer = action.content
            trace.terminated_reason = 'answered'
            break
        errors = errors + 1 if action.kind is None else 0
        if errors >= MAX_CONSECUTIVE_FORMAT_ERRORS:
            trace.terminated_reason = 'repeated_format_error'
            break
        if round_index + 1 == max_rounds:
            break
        if action.kind is None:
            tool_text = f'[error] {action.error}'
        else:
            tool_text = _execute_tool(action, question['paper_id'], corpus)
        # Render only to obtain the new chat suffix. Never replace the exact
        # historical IDs with a freshly tokenized conversation. Fail closed if
        # a template rewrites history (e.g. stripping earlier reasoning).
        content = raw
        eos = getattr(tokenizer, 'eos_token', None)
        if eos and content.endswith(eos):
            content = content[:-len(eos)]
        messages.extend([{'role': 'assistant', 'content': content},
                         {'role': 'user', 'content': f'<information>{tool_text}</information>'}])
        next_rendered = render()
        prefix = rendered + raw
        if not next_rendered.startswith(prefix):
            raise ValueError('Chat template rewrites generated history; append-only rollout is unsafe')
        suffix = tokenizer.encode(next_rendered[len(prefix):], add_special_tokens=False)
        if len(trace.completion_ids) + len(suffix) >= budget:
            trace.terminated_reason = 'max_length'
            break  # Do not insert partial evidence or leave no room to answer.
        trace.completion_ids.extend(suffix)
        trace.logprobs.extend([0.0] * len(suffix))
        trace.env_mask.extend([0] * len(suffix))
        rendered = next_rendered
        if action.kind in ('search', 'read'):
            # Match only actual passage headers, not IDs mentioned inside prose.
            for match in re.finditer(r'^\[(p_\d+)\] \([^\n]*?\) (.*)$', tool_text, re.MULTILINE):
                para_id, passage = match.groups()
                if para_id not in trace.evidence_para_ids:
                    trace.evidence_para_ids.append(para_id)
                trace.evidence_passages.append(passage)
    return trace
