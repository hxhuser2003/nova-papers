import json
from dataclasses import replace

import pytest

from backend.qasper_corpus import QasperCorpus
from backend.reward import RewardWeights
from scripts.train_grpo import _reward_func, _rollout_func, _staged_weights, build_dataset_rows


def _write_corpus(tmp_path):
    papers = [{
        "paper_id": "p1", "title": "Paper One", "abstract": "...", "split": "train",
        "paragraphs": [{"para_id": "p_0", "section": "Method",
                         "text": "We propose a group relative baseline instead of a learned value function."}],
    }]
    questions = [{
        "question_id": "q1", "paper_id": "p1", "question": "What baseline do they use?",
        "answers": [{"answer_type": "extractive", "answer_text": "group relative baseline",
                     "evidence_para_ids": ["p_0"], "has_non_text_evidence": False}],
    }]
    (tmp_path / "papers.jsonl").write_text(json.dumps(papers[0]))
    (tmp_path / "questions.jsonl").write_text(json.dumps(questions[0]))
    return QasperCorpus(tmp_path)


def test_build_dataset_rows_one_row_per_question(tmp_path):
    corpus = _write_corpus(tmp_path)
    rows = build_dataset_rows(corpus, "train")
    assert '"question_id": "q1"' in rows["prompt"][0]
    assert rows["paper_id"] == ["p1"]
    assert rows["question_id"] == ["q1"]
    assert rows["answers"][0][0]["answer_text"] == "group relative baseline"


def test_build_dataset_rows_filters_by_split(tmp_path):
    corpus = _write_corpus(tmp_path)
    assert build_dataset_rows(corpus, "validation")["prompt"] == []



def test_reward_func_scores_a_batch_of_completions():
    rewards = _reward_func(
        ["<think>t</think><answer>group relative baseline</answer>"],
        answers=[[{"answer_type": "extractive", "answer_text": "group relative baseline",
                   "evidence_para_ids": ["p_0"]}]],
        question=["What baseline do they use?"],
        evidence_passages=[["We propose a group relative baseline instead of a learned value function."]],
        evidence_para_ids=[["p_0"]],
    )
    assert len(rewards) == 1
    assert rewards[0] > 2.0  # correct answer + correct evidence should dominate


def test_reward_func_works_with_only_the_required_answers_field():
    # rollout_func is a stub today, so in practice only "answers" is guaranteed
    # until it's implemented -- the wrapper must not crash on missing extras
    rewards = _reward_func(
        ["<think>t</think><answer>some answer</answer>"],
        answers=[[{"answer_type": "free_form", "answer_text": "some answer"}]],
    )
    assert len(rewards) == 1


def test_rollout_returns_one_trajectory_per_already_repeated_prompt(tmp_path):
    from types import SimpleNamespace
    from test_token_rollout import CharTokenizer, Scripted
    corpus = _write_corpus(tmp_path)
    rows = build_dataset_rows(corpus, 'train')
    trainer = SimpleNamespace(
        qasper_corpus=corpus, processing_class=CharTokenizer(),
        rollout_model=Scripted(['<answer>Yes</answer>', '<answer>No</answer>']),
        rollout_options={},
    )
    result = _rollout_func(rows['prompt'] * 2, trainer)
    assert len(result['completion_ids']) == 2
    assert result['final_answer'] == ['Yes', 'No']
    assert result['question'] == ['What baseline do they use?'] * 2
    assert result['env_mask'] == [[1] * 20, [1] * 19]


def test_reward_uses_model_answer_not_tool_text_and_penalizes_repeated_search():
    common = dict(answers=[[{'answer_type': 'yes_no', 'answer_text': 'Yes', 'evidence_para_ids': ['p_0']}]],
                  final_answer=['Yes'], evidence_passages=[[]], evidence_para_ids=[[]])
    final = '<think>done</think><answer>Yes</answer>'
    correct = _reward_func(['<information><answer>No</answer></information>'],
                           model_outputs=[[final]], **common)[0]
    assert correct == pytest.approx(1.85)
    repeated = _reward_func(['ignored'], model_outputs=[[
        '<think>find</think><search>baseline</search>',
        '<think>find</think><search>baseline</search>', final]], **common)[0]
    novel = _reward_func(['ignored'], model_outputs=[[
        '<think>find</think><search>baseline</search>',
        '<think>find</think><search>method</search>', final]], **common)[0]
    assert repeated < novel


def test_reward_rejects_misaligned_batch():
    with pytest.raises(ValueError):
        _reward_func(['a', 'b'], answers=[[]])


def test_zero_format_reward_ablation_drops_only_the_format_contribution():
    # R1: main() sets this module-level global from --zero-format-reward before
    # training starts, since TRL calls _reward_func with a fixed signature that
    # has no channel for extra config (see the comment on _WEIGHTS itself).
    import scripts.train_grpo as train_grpo
    from backend.reward import RewardWeights
    original = train_grpo._WEIGHTS
    common = dict(answers=[[{'answer_type': 'yes_no', 'answer_text': 'Yes'}]], final_answer=['Yes'])
    output = ['<think>t</think><answer>Yes</answer>']
    try:
        train_grpo._WEIGHTS = RewardWeights()
        full = _reward_func(output, **common)[0]
        train_grpo._WEIGHTS = replace(RewardWeights(), format=0.0)
        zeroed = _reward_func(output, **common)[0]
    finally:
        train_grpo._WEIGHTS = original
    assert zeroed == pytest.approx(full - RewardWeights().format)  # format_reward is 1.0 for this well-formed output


def test_zero_evidence_reward_ablation_drops_only_the_evidence_contribution():
    import scripts.train_grpo as train_grpo
    from backend.reward import RewardWeights
    original = train_grpo._WEIGHTS
    common = dict(answers=[[{'answer_type': 'yes_no', 'answer_text': 'Yes', 'evidence_para_ids': ['p_0']}]],
                  final_answer=['Yes'], evidence_para_ids=[['p_0']])
    output = ['<think>t</think><answer>Yes</answer>']
    try:
        train_grpo._WEIGHTS = RewardWeights()
        full = _reward_func(output, **common)[0]
        train_grpo._WEIGHTS = replace(RewardWeights(), evidence=0.0)
        zeroed = _reward_func(output, **common)[0]
    finally:
        train_grpo._WEIGHTS = original
    assert zeroed == pytest.approx(full - RewardWeights().evidence)  # exact evidence match scores 1.0


def test_staged_weights_before_threshold_bootstraps_tool_use():
    final = RewardWeights()
    staged = _staged_weights(step=0, stage1_steps=200, final_weights=final)
    assert staged.correctness == 0.0
    assert staged.evidence == 0.0
    assert staged.retrieval_bonus == 0.5
    assert staged.format == final.format and staged.action == final.action  # untouched


def test_staged_weights_at_and_after_threshold_is_unchanged():
    final = replace(RewardWeights(), evidence=0.2)  # composes with other ablations
    assert _staged_weights(step=199, stage1_steps=200, final_weights=final).retrieval_bonus == 0.5
    assert _staged_weights(step=200, stage1_steps=200, final_weights=final) == final
    assert _staged_weights(step=500, stage1_steps=200, final_weights=final) == final


def test_reward_func_applies_retrieval_bonus_only_when_a_search_happened():
    import scripts.train_grpo as train_grpo
    original = train_grpo._WEIGHTS
    common = dict(answers=[[{'answer_type': 'yes_no', 'answer_text': 'Yes'}]], final_answer=['Yes'])
    searched = ['<think>find</think><search>baseline</search>', '<think>t</think><answer>Yes</answer>']
    never_searched = ['<think>t</think><answer>Yes</answer>']
    # Zero every other weight so only retrieval_bonus is nonzero -- otherwise
    # the extra search turn would also shift the action-reward correction
    # (see the loop in _reward_func), confounding the comparison.
    all_zero_but_retrieval = RewardWeights(format=0.0, action=0.0, faithfulness=0.0,
                                           correctness=0.0, evidence=0.0, retrieval_bonus=0.5)
    try:
        train_grpo._WEIGHTS = all_zero_but_retrieval
        with_search = _reward_func(['ignored'], model_outputs=[searched], **common)[0]
        without_search = _reward_func(['ignored'], model_outputs=[never_searched], **common)[0]
    finally:
        train_grpo._WEIGHTS = original
    assert with_search == pytest.approx(0.5)
    assert without_search == pytest.approx(0.0)
