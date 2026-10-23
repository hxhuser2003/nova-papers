"""Isolated inference ablations; production agent and training remain unchanged."""
from __future__ import annotations
import argparse
import sys
import time
from dataclasses import asdict
from backend.agent_loop import run_episode, build_initial_context, parse_action, AgentStep
from scripts import run_baseline as baseline


def experiment_episode(qid, question, pid, corpus, model, experiment, arm, max_rounds=5):
    meta = dict(search_queries=[], extra_answer_attempted=False, extra_answer_seconds=0.0)
    class SearchProxy:
        papers = corpus.papers
        def search(self, paper_id, query):
            actual = question if experiment == 'first-search' and arm == 'treatment' and not meta['search_queries'] else query
            meta['search_queries'].append(dict(model_query=query, executed_query=actual))
            return corpus.search(paper_id, actual)
        def read(self, paper_id, para_id):
            return corpus.read(paper_id, para_id)
    trace = run_episode(qid, question, pid, SearchProxy(), model, max_rounds=max_rounds)
    if experiment != 'final-answer' or arm != 'treatment' or trace.terminated_reason != 'max_rounds':
        return trace, meta
    messages = [dict(role='user', content=build_initial_context(question, corpus.papers[pid]))]
    for step in trace.steps:
        messages.append(dict(role='assistant', content=step.model_output))
        if step.tool_result is not None:
            messages.append(dict(role='user', content=step.tool_result))
    messages.append(dict(role='user', content='No tool calls remain. Use only the information already provided. Think inside <think>...</think>, then output exactly one <answer>...</answer>. Do not output search or read actions. If the answer cannot be determined, answer Unanswerable.'))
    meta['extra_answer_attempted'] = True
    start = time.monotonic()
    output = model(messages)
    meta['extra_answer_seconds'] = time.monotonic() - start
    action = parse_action(output)
    # Never dispatch a tool from the extra turn, even if the model ignores instructions.
    trace.steps.append(AgentStep(max_rounds+1, output, action, None, meta['extra_answer_seconds']*1000))
    if action.kind == 'answer' and action.content:
        trace.final_answer = action.content
        trace.terminated_reason = 'answered'
    else:
        trace.terminated_reason = 'forced_answer_invalid'
    return trace, meta


class MeteredModel:
    def __init__(self, client):
        self.client = client
        self.usage = []
    def __call__(self, messages):
        c = self.client
        r = c.client.chat.completions.create(model=c.model, messages=messages,
            max_tokens=c.max_tokens, temperature=c.temperature, stop=baseline.STOPS,
            extra_body={'include_stop_str_in_output': True})
        if r.usage is None:
            raise ValueError('Server omitted token usage; cannot compare extra-turn costs')
        self.usage.append(dict(prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens))
        choice = r.choices[0]
        text = choice.message.content
        stop = getattr(choice, 'stop_reason', None)
        if choice.finish_reason == 'stop' and stop in baseline.ACTION_ENDS and not text.endswith(stop):
            text += stop
        return text


def evaluate(q, paper, corpus, client, experiment, arm):
    model = MeteredModel(client)
    start = time.monotonic()
    trace, meta = experiment_episode(q['question_id'], q['question'], q['paper_id'], corpus, model, experiment, arm)
    reads, seen = set(), set()
    for step in trace.steps:
        if step.action.kind == 'read' and step.tool_result is not None:
            reads.add(step.action.content.strip())
        if step.tool_result:
            seen.update(baseline.PARA_ID.findall(step.tool_result))
    return dict(prediction=trace.final_answer, evidence=sorted(reads), evidence_seen=sorted(seen),
        terminated_reason=trace.terminated_reason, rounds=len(trace.steps), latency_s=time.monotonic()-start,
        steps=[asdict(s) for s in trace.steps], experiment=experiment, arm=arm, **meta,
        token_usage=model.usage, prompt_tokens=sum(u['prompt_tokens'] for u in model.usage),
        completion_tokens=sum(u['completion_tokens'] for u in model.usage),
        extra_answer_tokens=model.usage[-1] if meta['extra_answer_attempted'] else None)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--experiment', choices=['first-search','final-answer'], required=True)
    parser.add_argument('--arm', choices=['control','treatment'], required=True)
    args, remaining = parser.parse_known_args()
    baseline.MODES['agent'] = lambda q,p,c,m: evaluate(q,p,c,m,args.experiment,args.arm)
    sys.argv = [sys.argv[0], *remaining]
    baseline.main()


if __name__ == '__main__':
    main()
