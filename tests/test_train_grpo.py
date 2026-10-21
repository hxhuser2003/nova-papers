import json

import pytest

from backend.qasper_corpus import QasperCorpus
from scripts.train_grpo import _reward_func, _rollout_func, build_dataset_rows


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
