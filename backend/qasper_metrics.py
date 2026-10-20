"""QASPER-official-style Answer F1 / Evidence F1, shared by scripts/run_baseline.py
(baseline evaluation) and backend/reward.py (training-time correctness signal) so
there is exactly one implementation to keep in sync, not two copies that can drift.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not pred and not ref:
        return 1.0
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred), same / len(ref)
    return 2 * p * r / (p + r)


def gold_text(answer: dict) -> str:
    if answer.get("answer_type") == "unanswerable":
        return "Unanswerable"
    return answer.get("answer_text", "") or ""


def answer_f1(prediction: str | None, answers: list[dict]) -> float:
    """Best token F1 against any of QASPER's (possibly multiple) annotator answers."""
    if prediction is None:
        return 0.0
    return max(token_f1(prediction, gold_text(a)) for a in answers)


def evidence_f1(predicted: set[str], answers: list[dict]) -> float:
    """Best para_id-set F1 against any annotator's evidence_para_ids."""
    best = 0.0
    for a in answers:
        gold = set(a.get("evidence_para_ids", []))
        if not gold and not predicted:
            score = 1.0
        elif not gold or not predicted:
            score = 0.0
        else:
            same = len(gold & predicted)
            if same == 0:
                score = 0.0
            else:
                p, r = same / len(predicted), same / len(gold)
                score = 2 * p * r / (p + r)
        best = max(best, score)
    return best
