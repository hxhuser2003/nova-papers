"""Composite reward for RLVR/GRPO training of the paper Q&A agent.

Stage one (see STAGE_ONE.md) only calls an external API and does no training.
This module implements the reward side of the future RLVR loop described in
project_goal.md: for a group of G rollouts sampled from the policy for the
same (question, evidence) pair, score each rollout on five verifiable
signals, then normalize the group into GRPO advantages.

Signals:
  - format: did the rollout emit a well-formed <think>...</think> block?
  - action: does a follow-up <search> query introduce new keywords, or does
    it repeat a query already tried earlier in the trace (a dead loop)?
  - faithfulness: is the answer grounded in the retrieved evidence rather
    than copied verbatim from it, optionally blended with an LLM-as-judge
    relevance score?
  - correctness: does the answer actually match QASPER's gold answer?
    Without this, a rollout can be well-formatted, grounded in *some*
    retrieved passage, and still wrong -- faithfulness alone never checks
    the answer against the gold label, only against what was retrieved.
  - evidence: did the policy's search/read actions land on the paragraphs
    QASPER's annotators actually cited, not just any retrieved text?
"""
from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from backend.agent_loop import ACTION_PATTERN
from backend.qasper_metrics import answer_f1, evidence_f1
from backend.retrieval import tokenize

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)


@dataclass
class RolloutSample:
    """One sampled response from the policy for a given (question, evidence)."""

    raw_output: str
    history_search_keywords: list[str] = field(default_factory=list)
    """Keywords already used by earlier turns/tool calls in this rollout's trace."""
    evidence_para_ids: list[str] = field(default_factory=list)
    """Paragraph ids the policy's search/read actions surfaced over the whole
    trace, compared against QASPER's gold evidence_para_ids for evidence_reward."""


@dataclass
class RewardWeights:
    """Un-normalized weights (do not need to sum to 1): format and action
    are kept small early in training so the policy first learns to follow
    the output contract and stop over-searching. correctness is weighted
    highest since matching the gold answer is the actual training target;
    faithfulness and evidence are supporting signals (grounded-in-retrieval,
    cited-the-right-paragraphs) that correctness alone doesn't check."""

    format: float = 0.2
    action: float = 0.3
    faithfulness: float = 1.0
    correctness: float = 1.5
    evidence: float = 0.5


LlmJudge = Callable[[str, str, Sequence[str]], float]
"""question, answer, evidence passages -> relevance score in [0, 1]."""


def format_reward(sample: RolloutSample) -> float:
    """1.0 for a single non-empty <think> block followed by exactly one
    valid action tag (<search>, <read>, or <answer>) -- the same test
    agent_loop.parse_action itself uses to accept a round, so this never
    gives credit to a round parse_action would reject as a format error.
    0.5 for a <think> block present but malformed (empty, duplicated, or
    with no/multiple action tags after it); 0.0 if <think> is absent."""
    matches = THINK_PATTERN.findall(sample.raw_output)
    if not matches:
        return 0.0
    think_body = matches[0].strip()
    after = sample.raw_output.split("</think>", 1)[-1].strip()
    actions_after = ACTION_PATTERN.findall(after)
    if len(matches) == 1 and think_body and len(actions_after) == 1:
        return 1.0
    return 0.5


def _search_keywords(raw_output: str) -> list[str]:
    match = SEARCH_PATTERN.search(raw_output)
    query = match.group(1) if match else ""
    return list(dict.fromkeys(tokenize(query)))


def action_reward(sample: RolloutSample) -> float:
    """Fraction of a follow-up search's keywords that are new versus the
    trace's history (guards against repeating the same query forever).
    Neutral when the rollout does not request another search."""
    keywords = _search_keywords(sample.raw_output)
    if not keywords:
        return 0.5
    history = set(tokenize(" ".join(sample.history_search_keywords)))
    novel = [k for k in keywords if k not in history]
    return len(novel) / len(keywords)


def _rouge_l_f1(candidate_tokens: Sequence[str], reference_tokens: Sequence[str]) -> float:
    if not candidate_tokens or not reference_tokens:
        return 0.0
    n, m = len(candidate_tokens), len(reference_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if candidate_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    precision, recall = lcs / n, lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _extract_answer(sample: RolloutSample) -> str:
    """Returns the text of a well-formed <answer>...</answer> action, using
    agent_loop.ACTION_PATTERN -- the exact same pattern parse_action uses to
    decide a round is valid -- against the whole raw_output (not just the
    text after </think>, since parse_action doesn't treat <think> specially
    either). Anything parse_action itself would reject (no action tag, more
    than one, or a <search>/<read> tag instead of <answer>) returns "".
    This must stay this strict: a rollout that failed the format entirely
    (e.g. terminated_reason="repeated_format_error") must never be credited
    with "the right answer" just because its leftover raw text happens to
    overlap the gold answer."""
    matches = ACTION_PATTERN.findall(sample.raw_output)
    if len(matches) == 1 and matches[0][0] == "answer":
        return matches[0][1].strip()
    return ""


def faithfulness_reward(
    question: str,
    sample: RolloutSample,
    evidence_passages: Sequence[str],
    judge: LlmJudge | None = None,
    copy_ceiling: float = 0.8,
) -> float:
    """Reward the strongest grounding against any retrieved passage, then
    discount overlap above `copy_ceiling` (near-verbatim copying), blended
    with an optional LLM-as-judge relevance score."""
    answer = _extract_answer(sample)
    answer_tokens = tokenize(answer)
    overlaps = [_rouge_l_f1(answer_tokens, tokenize(passage)) for passage in evidence_passages]
    grounding = max(overlaps) if overlaps else 0.0
    if copy_ceiling < 1 and grounding > copy_ceiling:
        grounding -= (grounding - copy_ceiling) / (1 - copy_ceiling) * grounding
    if judge is None:
        return grounding
    judge_score = judge(question, answer, evidence_passages)
    return 0.5 * grounding + 0.5 * judge_score


def correctness_reward(sample: RolloutSample, answers: Sequence[dict]) -> float:
    """QASPER-official-style Answer F1 between the rollout's answer and the
    best-matching gold answer. This is the one signal that actually checks
    "did it get the right answer" rather than "is it grounded in something
    retrieved" (faithfulness) or "is it well-formed" (format)."""
    return answer_f1(_extract_answer(sample), list(answers))


def evidence_reward(sample: RolloutSample, answers: Sequence[dict]) -> float:
    """QASPER-official-style Evidence F1 between the paragraphs the rollout's
    search/read actions surfaced (sample.evidence_para_ids) and the gold
    evidence_para_ids from QASPER's annotators."""
    return evidence_f1(set(sample.evidence_para_ids), list(answers))


def composite_reward(
    question: str,
    sample: RolloutSample,
    evidence_passages: Sequence[str],
    answers: Sequence[dict],
    judge: LlmJudge | None = None,
    weights: RewardWeights = RewardWeights(),
) -> float:
    """R = w_format*R_format + w_action*R_action + w_faithfulness*R_faithfulness
         + w_correctness*R_correctness + w_evidence*R_evidence.

    With the default weights this ranges over [0, 3.5], not [0, 1] -- the
    weights are deliberately un-normalized (see RewardWeights)."""
    return (
        weights.correctness * correctness_reward(sample, answers)
        + weights.evidence * evidence_reward(sample, answers)
        + weights.format * format_reward(sample)
        + weights.action * action_reward(sample)
        + weights.faithfulness * faithfulness_reward(question, sample, evidence_passages, judge)
    )


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """GRPO group-relative advantage: A_i = (R_i - mean(R)) / std(R).

    Returns zeros when the group has fewer than two samples or zero variance,
    since there is then no relative signal to learn from.
    """
    if len(rewards) < 2:
        return [0.0 for _ in rewards]
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    if std == 0:
        return [0.0 for _ in rewards]
    return [(r - mean) / std for r in rewards]
