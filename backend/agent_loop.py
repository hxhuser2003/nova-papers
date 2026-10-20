"""The Agent loop: parses the policy's per-round <think>/<search>/<read>/
<answer> output, executes the corresponding tool against a QasperCorpus,
feeds the result back as an <information> block, and repeats until the
model answers or a termination condition trips (see project1's "Tool
Requirements and Action Format v0.1" spec, section 6).

The conversation is kept as a structured list of {"role", "content"} turns
throughout (see ModelClient below), not flattened into one string -- a real
chat-tuned model needs proper turn boundaries to behave correctly, and a
flat string is also ambiguous to split back apart (the model's own <think>
text could itself contain something that looks like a real <information>
block, e.g. while paraphrasing an earlier search result).

The model itself is injected as a plain callable (ModelClient): this loop
has no dependency on how the model is served. See backend.vllm_client for
the real vLLM-backed implementation; a scripted stub is enough to build and
test every other part of the loop without a live model.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from backend.qasper_corpus import QasperCorpus

MAX_ROUNDS = 5
MAX_CONSECUTIVE_FORMAT_ERRORS = 2

ACTION_PATTERN = re.compile(r"<(search|read|answer)>(.*?)</\1>", re.DOTALL)

# Per project1's "Tool Requirements and Action Format v0.1" spec, sections 1-2.
SYSTEM_PROMPT = """You are a research assistant answering a question about a single academic paper, using two tools: search and read.

Each round, first think inside <think>...</think>, then choose exactly one action:
- <search>query</search> to search this paper for relevant passages
- <read>para_id</read> to read the full text of a specific paragraph, e.g. <read>p_12</read>
- <answer>...</answer> to give your final answer and end

If a <search> does not return new, useful information, try a different query with \
different keywords on your next turn -- never repeat the exact same search query twice.

Answer rules: for yes/no questions, answer exactly "Yes" or "No". If the paper does not \
contain the answer, answer "Unanswerable". Otherwise, answer as briefly as possible, \
preferring the paper's own wording.

Passage text may contain placeholders like INLINEFORM, DISPLAYFORM (a removed formula) or \
BIBREF, TABREF, FIGREF (a citation/table/figure marker). These are not part of the paper's \
actual content -- never copy them into your answer.

Example of one turn:
<think>The question asks which datasets were used. This is usually described in the \
experiments section, so I will search for it.</think>
<search>datasets used in experiments</search>"""


def build_initial_context(question: str, paper: dict) -> str:
    """Assembles the system prompt plus the paper's title/abstract/question
    into the model's starting context (spec section 1: "input = question +
    paper_id + paper title and abstract"). Owned here, not left to the
    caller or to whatever later implements the real ModelClient, so the
    two don't each assume the other builds it."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Paper title: {paper['title']}\n"
        f"Abstract: {paper['abstract']}\n\n"
        f"Question: {question}\n"
    )

# A ModelClient takes the conversation-so-far as a list of {"role", "content"}
# turns (role is "user" or "assistant") and returns the model's next round
# of output, which the real client stops generation at the end of (e.g.
# "</search>"). Turns are kept structured end-to-end -- never flattened into
# one string and then re-parsed back apart -- because a flat string is
# ambiguous: nothing stops the model's own <think> text from containing
# something that looks like a real <information> block (e.g. paraphrasing
# an earlier search result), which a string-splitting ModelClient can't
# tell apart from a genuine tool turn.
ModelClient = Callable[[list[dict]], str]


@dataclass
class ParsedAction:
    kind: str | None  # "search" | "read" | "answer" | None (format error)
    content: str = ""
    error: str | None = None


@dataclass
class AgentStep:
    round_index: int
    model_output: str
    action: ParsedAction
    tool_result: str | None
    duration_ms: float


@dataclass
class AgentTrajectory:
    question_id: str
    paper_id: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    terminated_reason: str = "in_progress"  # answered | max_rounds | repeated_format_error | paper_not_found


def parse_action(raw_output: str) -> ParsedAction:
    """Exactly one of <search>/<read>/<answer> must be present; anything
    else (none, more than one, or an unclosed tag) is a format error."""
    matches = ACTION_PATTERN.findall(raw_output)
    if len(matches) != 1:
        return ParsedAction(
            kind=None,
            error=f"expected exactly one action tag (<search>/<read>/<answer>), found {len(matches)}",
        )
    kind, content = matches[0]
    return ParsedAction(kind=kind, content=content.strip())


def _execute_tool(action: ParsedAction, paper_id: str, corpus: QasperCorpus) -> str:
    if action.kind == "search":
        return corpus.search(paper_id, action.content)
    if action.kind == "read":
        return corpus.read(paper_id, action.content)
    raise ValueError(f"_execute_tool called on a non-tool action: {action.kind}")


def run_episode(
    question_id: str,
    question: str,
    paper_id: str,
    corpus: QasperCorpus,
    model: ModelClient,
    max_rounds: int = MAX_ROUNDS,
    on_step: Callable[[AgentStep], None] | None = None,
) -> AgentTrajectory:
    """`on_step`, if given, is called synchronously right after each step is
    appended to the trajectory -- e.g. to push a live "searching..."/
    "reading p_12..." status to a caller streaming progress over SSE. Errors
    raised by `on_step` are not caught: a broken callback should surface,
    not silently swallow a round's progress."""
    trajectory = AgentTrajectory(question_id=question_id, paper_id=paper_id)
    paper = corpus.papers.get(paper_id)
    if paper is None:
        trajectory.terminated_reason = "paper_not_found"
        return trajectory
    messages = [{"role": "user", "content": build_initial_context(question, paper)}]
    consecutive_format_errors = 0

    def record(step: AgentStep) -> None:
        trajectory.steps.append(step)
        if on_step is not None:
            on_step(step)

    for round_index in range(1, max_rounds + 1):
        start = time.monotonic()
        model_output = model(messages)
        duration_ms = (time.monotonic() - start) * 1000
        action = parse_action(model_output)
        messages.append({"role": "assistant", "content": model_output})

        if action.kind is None:
            consecutive_format_errors += 1
            tool_result = f"<information>[error] {action.error}</information>"
            messages.append({"role": "user", "content": tool_result})
            record(AgentStep(round_index, model_output, action, tool_result, duration_ms))
            if consecutive_format_errors >= MAX_CONSECUTIVE_FORMAT_ERRORS:
                trajectory.terminated_reason = "repeated_format_error"
                return trajectory
            continue

        consecutive_format_errors = 0

        if action.kind == "answer":
            record(AgentStep(round_index, model_output, action, None, duration_ms))
            trajectory.final_answer = action.content
            trajectory.terminated_reason = "answered"
            return trajectory

        tool_result = f"<information>{_execute_tool(action, paper_id, corpus)}</information>"
        messages.append({"role": "user", "content": tool_result})
        record(AgentStep(round_index, model_output, action, tool_result, duration_ms))

    trajectory.terminated_reason = "max_rounds"
    return trajectory


def run_group(
    question_id: str,
    question: str,
    paper_id: str,
    corpus: QasperCorpus,
    model: ModelClient,
    group_size: int = 5,
    max_rounds: int = MAX_ROUNDS,
    max_workers: int = 8,
) -> list[AgentTrajectory]:
    """Runs `group_size` independent rollouts for the same (question, paper)
    concurrently -- the GRPO unit of work (same prompt sampled G times,
    with up to ~100 trajectories in flight across a training batch)."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(run_episode, question_id, question, paper_id, corpus, model, max_rounds)
            for _ in range(group_size)
        ]
        return [f.result() for f in futures]
