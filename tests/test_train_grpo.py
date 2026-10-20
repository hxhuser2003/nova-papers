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
    assert rows["prompt"] == ["What baseline do they use?"]
    assert rows["paper_id"] == ["p1"]
    assert rows["question_id"] == ["q1"]
    assert rows["answers"][0][0]["answer_text"] == "group relative baseline"


def test_build_dataset_rows_filters_by_split(tmp_path):
    corpus = _write_corpus(tmp_path)
    assert build_dataset_rows(corpus, "validation")["prompt"] == []


def test_rollout_func_is_a_deliberate_stub_not_silently_wrong():
    with pytest.raises(NotImplementedError):
        _rollout_func(["some prompt"], trainer=None)


def test_reward_func_scores_a_batch_of_completions():
    rewards = _reward_func(
        ["<think>t</think>group relative baseline"],
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
        ["<think>t</think>some answer"],
        answers=[[{"answer_type": "free_form", "answer_text": "some answer"}]],
    )
    assert len(rewards) == 1
