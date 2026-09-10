import pytest

from backend.reward import (
    RolloutSample,
    action_reward,
    composite_reward,
    faithfulness_reward,
    format_reward,
    group_advantages,
)


def test_format_reward_scores_a_well_formed_think_block():
    sample = RolloutSample(raw_output="<think>plan the search</think>Here is the answer [S1].")
    assert format_reward(sample) == 1.0


def test_format_reward_penalizes_missing_think_block():
    sample = RolloutSample(raw_output="Here is the answer with no reasoning shown.")
    assert format_reward(sample) == 0.0


@pytest.mark.parametrize("raw_output", [
    "<think></think>Answer with an empty think block.",
    "<think>reasoning only, no answer follows</think>",
    "<think>first</think><think>second</think>Answer.",
])
def test_format_reward_gives_partial_credit_for_malformed_blocks(raw_output):
    assert format_reward(RolloutSample(raw_output=raw_output)) == 0.5


def test_action_reward_is_neutral_without_a_follow_up_search():
    sample = RolloutSample(raw_output="<think>done</think>Final answer, no more retrieval needed.")
    assert action_reward(sample) == 0.5


def test_action_reward_is_zero_for_an_exact_repeat_query():
    sample = RolloutSample(
        raw_output="<think>try again</think><search>GRPO advantage normalization</search>",
        history_search_keywords=["GRPO advantage normalization"],
    )
    assert action_reward(sample) == 0.0


def test_action_reward_rewards_novel_keywords():
    sample = RolloutSample(
        raw_output="<think>widen the search</think><search>DPO reference policy KL penalty</search>",
        history_search_keywords=["GRPO advantage normalization"],
    )
    assert action_reward(sample) == 1.0


def test_faithfulness_reward_grounds_paraphrase_above_unrelated_answer():
    evidence = ["PPO uses a clipped surrogate objective to constrain the size of policy updates."]
    grounded = RolloutSample(raw_output="The method clips the surrogate objective to limit policy updates. [S1]")
    unrelated = RolloutSample(raw_output="Bananas are a good source of potassium.")
    grounded_score = faithfulness_reward("How does PPO limit updates?", grounded, evidence)
    unrelated_score = faithfulness_reward("How does PPO limit updates?", unrelated, evidence)
    assert grounded_score > unrelated_score


def test_faithfulness_reward_discounts_verbatim_copying():
    passage = "PPO uses a clipped surrogate objective to constrain the size of policy updates."
    verbatim = RolloutSample(raw_output=passage)
    paraphrase = RolloutSample(raw_output="The clipped surrogate objective keeps PPO's policy updates bounded.")
    verbatim_score = faithfulness_reward("Explain PPO.", verbatim, [passage])
    paraphrase_score = faithfulness_reward("Explain PPO.", paraphrase, [passage])
    assert verbatim_score < paraphrase_score


def test_faithfulness_reward_blends_in_an_llm_judge_score():
    evidence = ["Some grounded passage text."]
    sample = RolloutSample(raw_output="An answer only loosely touching the passage.")
    without_judge = faithfulness_reward("Q?", sample, evidence)
    with_judge = faithfulness_reward("Q?", sample, evidence, judge=lambda q, a, e: 1.0)
    assert with_judge > without_judge


def test_composite_reward_combines_all_three_signals():
    sample = RolloutSample(raw_output="<think>plan</think>The clipped objective bounds policy updates. [S1]")
    evidence = ["The clipped objective bounds policy updates."]
    score = composite_reward("Explain PPO.", sample, evidence)
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_group_advantages_normalize_around_zero_mean():
    advantages = group_advantages([1.0, 2.0, 3.0, 4.0])
    assert advantages[0] < 0 < advantages[-1]
    assert pytest.approx(sum(advantages), abs=1e-9) == 0.0


def test_group_advantages_are_zero_without_variance_or_a_group():
    assert group_advantages([5.0]) == [0.0]
    assert group_advantages([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]
