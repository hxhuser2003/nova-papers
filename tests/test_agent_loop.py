import json

from backend.agent_loop import build_initial_context, parse_action, run_episode, run_group
from backend.qasper_corpus import QasperCorpus


def _write_corpus(tmp_path):
    papers = [{
        "paper_id": "p1", "title": "Paper One", "abstract": "...", "split": "train",
        "paragraphs": [
            {"para_id": "p_0", "section": "Intro", "text": "This paper studies clipped surrogate objectives for policy optimization."},
            {"para_id": "p_1", "section": "Method", "text": "We propose a group relative baseline instead of a learned value function."},
        ],
    }]
    (tmp_path / "papers.jsonl").write_text("\n".join(json.dumps(p) for p in papers))
    (tmp_path / "questions.jsonl").write_text("")
    return QasperCorpus(tmp_path)


def _scripted_model(responses):
    """Returns each entry in `responses` in order, one per call, ignoring the messages argument."""
    calls = iter(responses)
    return lambda messages: next(calls)


# --- parse_action ---------------------------------------------------------

def test_parse_action_extracts_a_single_search_action():
    action = parse_action("<think>let's search</think><search>group relative baseline</search>")
    assert action.kind == "search"
    assert action.content == "group relative baseline"


def test_parse_action_extracts_a_single_answer_action():
    action = parse_action("<think>done</think><answer>group relative baseline</answer>")
    assert action.kind == "answer"
    assert action.content == "group relative baseline"


def test_parse_action_flags_missing_action_tag_as_a_format_error():
    action = parse_action("<think>just thinking, no action</think>")
    assert action.kind is None
    assert action.error is not None


def test_parse_action_flags_multiple_action_tags_as_a_format_error():
    action = parse_action("<search>a</search><answer>b</answer>")
    assert action.kind is None


# --- build_initial_context --------------------------------------------------

def test_build_initial_context_includes_title_abstract_question_and_tag_rules():
    paper = {"title": "Paper One", "abstract": "A study of clipped objectives."}
    context = build_initial_context("What baseline do they use?", paper)
    assert "Paper One" in context
    assert "A study of clipped objectives." in context
    assert "What baseline do they use?" in context
    assert "<search>" in context and "<read>" in context and "<answer>" in context  # tag rules present


def test_build_initial_context_warns_about_qasper_placeholder_markers():
    paper = {"title": "Paper One", "abstract": "..."}
    context = build_initial_context("What baseline do they use?", paper)
    assert "BIBREF" in context
    assert "INLINEFORM" in context


def test_build_initial_context_tells_the_model_not_to_repeat_the_same_search():
    paper = {"title": "Paper One", "abstract": "..."}
    context = build_initial_context("What baseline do they use?", paper)
    assert "never repeat the exact same search query" in context


def test_run_episode_gives_the_model_the_papers_title_and_abstract(tmp_path):
    corpus = _write_corpus(tmp_path)
    seen_calls = []

    def spy_model(messages):
        seen_calls.append(list(messages))  # snapshot before run_episode appends more turns
        return "<think>done</think><answer>group relative baseline</answer>"

    run_episode("q1", "What baseline do they use?", "p1", corpus, spy_model)
    first_call = seen_calls[0]
    assert first_call == [{"role": "user", "content": first_call[0]["content"]}]
    assert "Paper One" in first_call[0]["content"]  # title from _write_corpus's fixture


# --- structured turns (not a flattened string) ------------------------------

def test_run_episode_keeps_turns_structured_even_when_the_model_quotes_a_tool_result(tmp_path):
    # the model paraphrasing an earlier <information> block inside its own
    # <think> must not be mistaken for a real tool turn -- there is no
    # string to misparse, since messages are tracked as data throughout
    corpus = _write_corpus(tmp_path)
    seen_calls = []

    def spy_model(messages):
        seen_calls.append([dict(m) for m in messages])  # snapshot before mutation
        if len(seen_calls) == 1:
            return "<think>searching</think><search>baseline</search>"
        return (
            "<think>Earlier I got <information>[p_0] foo</information> which did not help, "
            "trying again</think><answer>group relative baseline</answer>"
        )

    run_episode("q1", "What baseline do they use?", "p1", corpus, spy_model)
    second_call = seen_calls[1]
    # exactly 3 turns: initial question, the round-1 assistant output, the real tool result
    assert [m["role"] for m in second_call] == ["user", "assistant", "user"]
    assert "<information>" not in second_call[1]["content"]  # round 1's own output had no such text
    assert "p_1" in second_call[2]["content"]  # the genuine search result landed in its own user turn


# --- run_episode: happy paths ---------------------------------------------

def test_run_episode_searches_then_answers(tmp_path):
    corpus = _write_corpus(tmp_path)
    model = _scripted_model([
        "<think>look for the baseline</think><search>group relative baseline</search>",
        "<think>found it</think><answer>group relative baseline</answer>",
    ])
    trajectory = run_episode("q1", "What baseline do they use?", "p1", corpus, model)
    assert trajectory.terminated_reason == "answered"
    assert trajectory.final_answer == "group relative baseline"
    assert len(trajectory.steps) == 2
    assert "p_1" in trajectory.steps[0].tool_result  # the search tool result got attached


def test_run_episode_executes_the_read_tool(tmp_path):
    corpus = _write_corpus(tmp_path)
    model = _scripted_model([
        "<think>read it directly</think><read>p_1</read>",
        "<think>done</think><answer>group relative baseline</answer>",
    ])
    trajectory = run_episode("q1", "What baseline do they use?", "p1", corpus, model)
    assert "learned value function" in trajectory.steps[0].tool_result


def test_run_episode_calls_on_step_for_each_round(tmp_path):
    corpus = _write_corpus(tmp_path)
    model = _scripted_model([
        "<think>search</think><search>baseline</search>",
        "<think>done</think><answer>group relative baseline</answer>",
    ])
    seen = []
    run_episode("q1", "What baseline do they use?", "p1", corpus, model, on_step=seen.append)
    assert [s.action.kind for s in seen] == ["search", "answer"]
    assert seen[0].round_index == 1 and seen[1].round_index == 2


# --- run_episode: termination conditions -----------------------------------

def test_run_episode_stops_after_two_consecutive_format_errors(tmp_path):
    corpus = _write_corpus(tmp_path)
    model = _scripted_model(["garbage with no tags", "more garbage"])
    trajectory = run_episode("q1", "What baseline do they use?", "p1", corpus, model)
    assert trajectory.terminated_reason == "repeated_format_error"
    assert len(trajectory.steps) == 2
    assert trajectory.final_answer is None


def test_run_episode_resets_the_format_error_counter_on_a_valid_round(tmp_path):
    corpus = _write_corpus(tmp_path)
    # error, valid search, error, valid answer -- never two *consecutive* errors, so it must finish
    model = _scripted_model([
        "garbage",
        "<think>ok</think><search>baseline</search>",
        "garbage again",
        "<think>done</think><answer>group relative baseline</answer>",
    ])
    trajectory = run_episode("q1", "What baseline do they use?", "p1", corpus, model, max_rounds=4)
    assert trajectory.terminated_reason == "answered"
    assert len(trajectory.steps) == 4


def test_run_episode_stops_at_max_rounds_without_an_answer(tmp_path):
    corpus = _write_corpus(tmp_path)
    model = lambda messages: "<think>keep searching</think><search>baseline</search>"
    trajectory = run_episode("q1", "What baseline do they use?", "p1", corpus, model, max_rounds=3)
    assert trajectory.terminated_reason == "max_rounds"
    assert len(trajectory.steps) == 3
    assert trajectory.final_answer is None


# --- unknown paper_id --------------------------------------------------------

def test_run_episode_on_an_unknown_paper_id_terminates_gracefully_instead_of_raising(tmp_path):
    corpus = _write_corpus(tmp_path)
    model = lambda messages: "<think>done</think><answer>x</answer>"
    trajectory = run_episode("q1", "x", "WRONG_ID", corpus, model)
    assert trajectory.terminated_reason == "paper_not_found"
    assert trajectory.steps == []


def test_run_group_is_not_killed_by_one_bad_paper_id(tmp_path):
    corpus = _write_corpus(tmp_path)
    model = lambda messages: "<think>done</think><answer>x</answer>"
    trajectories = run_group("q1", "x", "WRONG_ID", corpus, model, group_size=5)
    assert len(trajectories) == 5
    assert all(t.terminated_reason == "paper_not_found" for t in trajectories)


# --- run_group --------------------------------------------------------------

def test_run_group_runs_independent_rollouts_concurrently(tmp_path):
    corpus = _write_corpus(tmp_path)
    model = _scripted_model(["<think>done</think><answer>group relative baseline</answer>"] * 5)
    trajectories = run_group("q1", "What baseline do they use?", "p1", corpus, model, group_size=5)
    assert len(trajectories) == 5
    assert all(t.terminated_reason == "answered" for t in trajectories)
