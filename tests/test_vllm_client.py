import json

import httpx
import pytest

from backend.agent_loop import parse_action, run_episode
from backend.qasper_corpus import QasperCorpus
from backend.vllm_client import VLLMModelClient


def _client_with_response(status_code=200, json_body=None, raise_on_request=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if raise_on_request is not None:
            raise raise_on_request
        return httpx.Response(status_code, json=json_body)
    transport = httpx.MockTransport(handler)
    return VLLMModelClient(base_url="http://fake-vllm:8000", transport=transport)


def _chat_response(text, finish_reason="stop", stop_reason=None):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish_reason, "stop_reason": stop_reason}]}


def test_forwards_the_messages_list_as_is_with_stop_sequences():
    captured = {}
    messages = [{"role": "user", "content": "SYSTEM\n\nQuestion: q\n"}]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_response("<think>ok</think>", finish_reason="length"))

    client = VLLMModelClient(base_url="http://fake-vllm:8000", transport=httpx.MockTransport(handler))
    client(messages)
    assert captured["json"]["messages"] == messages
    assert captured["json"]["stop"] == ["</search>", "</read>", "</answer>"]
    assert captured["json"]["include_stop_str_in_output"] is True


def test_passes_through_unchanged_when_vllm_already_included_the_stop_string():
    client = _client_with_response(json_body=_chat_response(
        "<think>look it up</think><search>group relative baseline</search>",
        stop_reason="</search>",
    ))
    output = client([{"role": "user", "content": "q"}])
    assert output == "<think>look it up</think><search>group relative baseline</search>"
    action = parse_action(output)
    assert action.kind == "search"
    assert action.content == "group relative baseline"


def test_falls_back_to_reappending_if_vllm_stripped_the_stop_string_anyway():
    client = _client_with_response(json_body=_chat_response(
        "<think>look it up</think><search>group relative baseline",
        stop_reason="</search>",
    ))
    output = client([{"role": "user", "content": "q"}])
    assert output == "<think>look it up</think><search>group relative baseline</search>"
    assert parse_action(output).kind == "search"


def test_does_not_reappend_anything_when_the_model_just_ran_out_of_tokens():
    client = _client_with_response(json_body=_chat_response(
        "<think>rambling with no action tag yet", finish_reason="length", stop_reason=None,
    ))
    output = client([{"role": "user", "content": "q"}])
    assert output == "<think>rambling with no action tag yet"
    assert parse_action(output).kind is None  # correctly unparseable: the round never closed a tag


def test_raises_on_an_http_error_instead_of_returning_malformed_text():
    client = _client_with_response(status_code=500, json_body={"error": "internal error"})
    with pytest.raises(httpx.HTTPStatusError):
        client([{"role": "user", "content": "q"}])


def test_raises_on_a_connection_failure():
    client = _client_with_response(raise_on_request=httpx.ConnectError("connection refused"))
    with pytest.raises(httpx.ConnectError):
        client([{"role": "user", "content": "q"}])


def test_vllm_client_plugs_directly_into_run_episode(tmp_path):
    papers = [{
        "paper_id": "p1", "title": "Paper One", "abstract": "...", "split": "train",
        "paragraphs": [{"para_id": "p_0", "section": "Method",
                         "text": "We propose a group relative baseline instead of a learned value function."}],
    }]
    (tmp_path / "papers.jsonl").write_text(json.dumps(papers[0]))
    (tmp_path / "questions.jsonl").write_text("")
    corpus = QasperCorpus(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        if not any(m["role"] == "user" and "<information>" in m["content"] for m in messages):
            text = "<think>search for it</think><search>baseline</search>"
        else:
            text = "<think>done</think><answer>group relative baseline</answer>"
        return httpx.Response(200, json=_chat_response(text))

    client = VLLMModelClient(base_url="http://fake-vllm:8000", transport=httpx.MockTransport(handler))
    trajectory = run_episode("q1", "What baseline do they use?", "p1", corpus, client)
    assert trajectory.terminated_reason == "answered"
    assert trajectory.final_answer == "group relative baseline"
    assert len(trajectory.steps) == 2


def test_vllm_client_handles_a_model_that_quotes_a_tool_result_in_its_own_thinking(tmp_path):
    # the exact scenario that broke the old flatten-then-reparse design:
    # the model paraphrases an earlier <information> block inside <think>
    papers = [{
        "paper_id": "p1", "title": "Paper One", "abstract": "...", "split": "train",
        "paragraphs": [{"para_id": "p_0", "section": "Method",
                         "text": "We propose a group relative baseline instead of a learned value function."}],
    }]
    (tmp_path / "papers.jsonl").write_text(json.dumps(papers[0]))
    (tmp_path / "questions.jsonl").write_text("")
    corpus = QasperCorpus(tmp_path)

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            text = "<think>searching</think><search>baseline</search>"
        else:
            text = ("<think>Earlier I got <information>[p_0] foo</information> which did not help, "
                     "trying again</think><answer>group relative baseline</answer>")
        return httpx.Response(200, json=_chat_response(text))

    client = VLLMModelClient(base_url="http://fake-vllm:8000", transport=httpx.MockTransport(handler))
    trajectory = run_episode("q1", "What baseline do they use?", "p1", corpus, client)
    assert trajectory.terminated_reason == "answered"
    assert trajectory.final_answer == "group relative baseline"
