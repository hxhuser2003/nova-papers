from types import SimpleNamespace

import pytest

from backend import token_rollout as rollout
from test_train_grpo import _write_corpus


class CharTokenizer:
    eos_token = '<end>'
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())
    def decode(self, ids, **kwargs):
        return bytes(ids).decode()
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = ''.join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        return text + ('<assistant>' if add_generation_prompt else '')


class Scripted:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.inputs = []
    def generate(self, prompt, stop=None, max_tokens=None):
        self.inputs.append(list(prompt))
        text = next(self.texts)
        ids = list(text.encode())[:max_tokens]
        return {'text': bytes(ids).decode(), 'token_ids': ids, 'logprobs': [-0.25] * len(ids)}


def run(tmp_path, texts, **kwargs):
    corpus = _write_corpus(tmp_path)
    model = Scripted(texts)
    result = rollout.run_token_episode(corpus.questions['q1'], corpus, CharTokenizer(), model, **kwargs)
    return result, model


def test_search_read_answer_keeps_exact_prefix_and_masks_only_external_tokens(tmp_path):
    texts = ['<think>look</think><search>baseline</search>',
             '<think>read</think><read>p_0</read>',
             '<think>done</think><answer>group relative baseline</answer>']
    result, model = run(tmp_path, texts)
    assert result.terminated_reason == 'answered'
    assert result.final_answer == 'group relative baseline'
    assert result.evidence_para_ids == ['p_0']
    assert result.model_outputs == texts
    ids, mask, lp = result.completion_ids, result.env_mask, result.logprobs
    assert len(ids) == len(mask) == len(lp)
    assert bytes(t for t, m in zip(ids, mask) if m).decode() == ''.join(texts)
    assert all(v == 0.0 for v, m in zip(lp, mask) if not m)
    for prompt in model.inputs:
        assert prompt == (result.prompt_ids + ids)[:len(prompt)]
    assert '<user><information>[p_0]' in bytes(t for t, m in zip(ids, mask) if not m).decode()


def test_two_format_errors_stop_without_a_third_generation(tmp_path):
    result, model = run(tmp_path, ['bad', 'still bad'])
    assert result.terminated_reason == 'repeated_format_error'
    assert len(model.inputs) == 2
    assert result.final_answer is None
    assert '[error]' in bytes(result.completion_ids).decode()
    assert result.env_mask[-1] == 1  # no unused tool suffix on terminal round


def test_valid_action_resets_consecutive_error_count(tmp_path):
    result, _ = run(tmp_path, ['bad', '<read>p_0</read>', 'bad', '<answer>Yes</answer>'])
    assert result.terminated_reason == 'answered'


def test_total_budget_counts_environment_and_does_not_credit_unseen_evidence(tmp_path):
    text = '<search>baseline</search>'
    result, model = run(tmp_path, [text], max_completion_length=len(text) + 1)
    assert result.terminated_reason == 'max_length'
    assert len(result.completion_ids) == len(text)
    assert result.evidence_para_ids == []
    assert len(model.inputs) == 1


def test_max_rounds_and_missing_paper(tmp_path):
    result, _ = run(tmp_path, ['<read>p_0</read>'], max_rounds=1)
    assert result.terminated_reason == 'max_rounds'
    assert result.evidence_para_ids == []
    corpus = _write_corpus(tmp_path)
    q = dict(corpus.questions['q1'], paper_id='missing')
    with pytest.raises(ValueError, match='paper'):
        rollout.run_token_episode(q, corpus, CharTokenizer(), Scripted([]))


def test_colocated_adapter_returns_sampled_token_logprob_and_restores_budget():
    gen = SimpleNamespace(mode='colocate', max_completion_length=2048, generation_kwargs={'top_k': 20})
    def generate(prompts, images, num_generations):
        assert prompts == [[1, 2]] and num_generations == 1
        assert gen.max_completion_length == 7
        assert gen.generation_kwargs['include_stop_str_in_output']
        return prompts, [[65]], [[[-0.4, -0.8]]], [[[65, 66]]]
    gen.generate = generate
    trainer = SimpleNamespace(vllm_generation=gen, processing_class=CharTokenizer(),
                              accelerator=SimpleNamespace(num_processes=1))
    value = rollout.ColocatedModelClient(trainer).generate([1, 2], max_tokens=7)
    assert value == {'text': 'A', 'token_ids': [65], 'logprobs': [-0.4]}
    assert gen.max_completion_length == 2048
    assert gen.generation_kwargs == {'top_k': 20}
