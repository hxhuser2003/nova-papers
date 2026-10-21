"""Check that backend/reward.py scores *real* AgentTrajectory output correctly.

Runs backend.agent_loop.run_episode with scripted (fake) model outputs in the
exact <think>/<search>/<read>/<answer> format the real model produces, converts
each trajectory into a RolloutSample the way TRAINING_NOTES.md proposes
(raw_output = last round's output, evidence_para_ids = paragraphs surfaced),
and compares reward.py's scores with what the baseline evaluation would give
(answer_f1 on trajectory.final_answer).

No GPU, model, or QASPER corpus needed:  python scripts/validate_reward.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agent_loop import AgentTrajectory, run_episode  # noqa: E402
from backend.qasper_metrics import answer_f1  # noqa: E402
from backend.reward import (  # noqa: E402
    RolloutSample,
    composite_reward,
    correctness_reward,
    evidence_reward,
    faithfulness_reward,
    format_reward,
    action_reward,
)

PARA_ID = re.compile(r"\[(p_\d+)\]")

PASSAGES = {
    "p_3": "(Experiments) We compare against strong baselines including BERT and ELMo on SQuAD.",
    "p_7": "(Results) Our model outperforms all baselines on every dataset we evaluate.",
}


class FakeCorpus:
    """Stands in for QasperCorpus: same .papers / .search() / .read() surface."""

    papers = {"paper1": {"title": "A Test Paper", "abstract": "We study things."}}

    def search(self, paper_id: str, query: str) -> str:
        return "\n".join(f"[{pid}] {text}" for pid, text in PASSAGES.items())

    def read(self, paper_id: str, para_id: str) -> str:
        return f"[{para_id}] {PASSAGES.get(para_id, '[error] not found')}"


def scripted(outputs: list[str]):
    it = iter(outputs)
    return lambda messages: next(it)


def to_sample(traj: AgentTrajectory) -> RolloutSample:
    """Conversion proposed in TRAINING_NOTES.md (2026-10-20 interface note)."""
    history, surfaced = [], []
    for step in traj.steps:
        if step.action.kind == "search":
            history.append(step.action.content)
        for pid in PARA_ID.findall(step.tool_result or ""):
            if pid not in surfaced:
                surfaced.append(pid)
    last = traj.steps[-1].model_output if traj.steps else ""
    return RolloutSample(raw_output=last, history_search_keywords=history[:-1], evidence_para_ids=surfaced)


SEARCH = "<think>Look in the experiments.</think>\n<search>baselines compared</search>"
CASES = {
    "1 yes/no, correct": (["<think>The abstract says so.</think>\n<answer>Yes</answer>"],
                         [{"answer_type": "yes_no", "answer_text": "Yes", "evidence_para_ids": ["p_7"]}]),
    "2 extractive, correct": ([SEARCH, "<think>Found it.</think>\n<answer>BERT and ELMo</answer>"],
                              [{"answer_type": "extractive", "answer_text": "BERT and ELMo", "evidence_para_ids": ["p_3"]}]),
    "3 unanswerable, correct": ([SEARCH, "<think>Not in the paper.</think>\n<answer>Unanswerable</answer>"],
                                [{"answer_type": "unanswerable", "answer_text": "", "evidence_para_ids": []}]),
    "4 extractive, wrong": ([SEARCH, "<think>Guess.</think>\n<answer>GPT-2</answer>"],
                            [{"answer_type": "extractive", "answer_text": "BERT and ELMo", "evidence_para_ids": ["p_3"]}]),
    "5 no answer (max rounds)": ([SEARCH] * 5,
                                 [{"answer_type": "extractive", "answer_text": "BERT and ELMo", "evidence_para_ids": ["p_3"]}]),
    "6 format error x2": (["I think the answer is BERT and ELMo.", "BERT and ELMo"],
                          [{"answer_type": "extractive", "answer_text": "BERT and ELMo", "evidence_para_ids": ["p_3"]}]),
}


def main() -> None:
    corpus = FakeCorpus()
    header = f"{'case':<26}{'end':<22}{'eval F1':>8}{'correct':>8}{'format':>7}{'action':>7}{'faith':>7}{'evid':>6}{'TOTAL':>7}"
    print(header)
    print("-" * len(header))
    for name, (outputs, answers) in CASES.items():
        traj = run_episode("q", "Which baselines?", "paper1", corpus, scripted(outputs))
        s = to_sample(traj)
        passages = list(PASSAGES.values())
        row = [
            answer_f1(traj.final_answer, answers),  # what eval/BASELINE.md reports
            correctness_reward(s, answers),
            format_reward(s),
            action_reward(s),
            faithfulness_reward("Which baselines?", s, passages),
            evidence_reward(s, answers),
            composite_reward("Which baselines?", s, passages, answers),
        ]
        print(f"{name:<26}{traj.terminated_reason:<22}" + "".join(f"{v:>7.2f} " for v in row))


if __name__ == "__main__":
    main()
