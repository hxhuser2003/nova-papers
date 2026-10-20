from backend.qasper_metrics import answer_f1, evidence_f1, gold_text, token_f1


def test_answer_f1_exact_match_is_1():
    answers = [{"answer_type": "extractive", "answer_text": "group relative baseline"}]
    assert answer_f1("group relative baseline", answers) == 1.0


def test_answer_f1_none_prediction_is_0():
    assert answer_f1(None, [{"answer_type": "extractive", "answer_text": "x"}]) == 0.0


def test_answer_f1_empty_answers_list_returns_0_instead_of_crashing():
    # regression: max() over an empty generator used to raise ValueError here,
    # inconsistent with evidence_f1's safe handling of the same situation
    assert answer_f1("some prediction", []) == 0.0


def test_evidence_f1_empty_answers_list_returns_0():
    assert evidence_f1({"p_1"}, []) == 0.0


def test_evidence_f1_exact_match_is_1():
    assert evidence_f1({"p_1", "p_2"}, [{"evidence_para_ids": ["p_1", "p_2"]}]) == 1.0


def test_gold_text_unanswerable_overrides_answer_text():
    assert gold_text({"answer_type": "unanswerable", "answer_text": "ignored"}) == "Unanswerable"


def test_token_f1_is_symmetric_overlap_based():
    assert token_f1("the cat sat", "a cat sat") > 0.5
