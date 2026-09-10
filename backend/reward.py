"""Composite reward for RLVR/GRPO training of the paper Q&A agent.

Stage one (see STAGE_ONE.md) only calls an external API and does no training.
This module implements the reward side of the future RLVR loop described in
project_goal.md: for a group of G rollouts sampled from the policy for the
same (question, evidence) pair, score each rollout on three verifiable
signals, then normalize the group into GRPO advantages.

Signals:
  - format: did the rollout emit a well-formed <think>...</think> block?
  - action: does a follow-up <search> query introduce new keywords, or does
    it repeat a query already tried earlier in the trace (a dead loop)?
  - faithfulness: is the answer grounded in the retrieved evidence rather
    than copied verbatim from it, optionally blended with an LLM-as-judge
    relevance score?
"""
from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from backend.retrieval import tokenize

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)


@dataclass
class RolloutSample:
    """One sampled response from the policy for a given (question, evidence)."""

    raw_output: str
    history_search_keywords: list[str] = field(default_factory=list)
    """Keywords already used by earlier turns/tool calls in this rollout's trace."""


@dataclass
class RewardWeights:
    format: float = 0.2
    action: float = 0.2
    faithfulness: float = 0.6


LlmJudge = Callable[[str, str, Sequence[str]], float]
"""question, answer, evidence passages -> relevance score in [0, 1]."""


def format_reward(sample: RolloutSample) -> float:
    """1.0 for a single non-empty <think> block followed by a non-empty
    answer, 0.5 for a present but malformed/empty block, 0.0 if absent."""
    matches = THINK_PATTERN.findall(sample.raw_output)
    if not matches:
        return 0.0
    think_body = matches[0].strip()
    after = sample.raw_output.split("</think>", 1)[-1].strip()
    if len(matches) == 1 and think_body and after:
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
    answer = sample.raw_output.split("</think>", 1)[-1].strip() or sample.raw_output
    answer_tokens = tokenize(answer)
    overlaps = [_rouge_l_f1(answer_tokens, tokenize(passage)) for passage in evidence_passages]
    grounding = max(overlaps) if overlaps else 0.0
    if copy_ceiling < 1 and grounding > copy_ceiling:
        grounding -= (grounding - copy_ceiling) / (1 - copy_ceiling) * grounding
    if judge is None:
        return grounding
    judge_score = judge(question, answer, evidence_passages)
    return 0.5 * grounding + 0.5 * judge_score


def composite_reward(
    question: str,
    sample: RolloutSample,
    evidence_passages: Sequence[str],
    judge: LlmJudge | None = None,
    weights: RewardWeights = RewardWeights(),
) -> float:
    """R = w_format*R_format + w_action*R_action + w_faithfulness*R_faithfulness."""
    return (
        weights.format * format_reward(sample)
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
