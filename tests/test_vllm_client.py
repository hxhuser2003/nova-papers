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


def test_sends_the_context_as_prompt_with_stop_sequences_and_asks_vllm_to_keep_them():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"text": "<think>ok</think>", "finish_reason": "length"}]})

    client = VLLMModelClient(base_url="http://fake-vllm:8000", transport=httpx.MockTransport(handler))
    client("some context")
    assert captured["json"]["prompt"] == "some context"
    assert captured["json"]["stop"] == ["</search>", "</read>", "</answer>"]
    assert captured["json"]["include_stop_str_in_output"] is True


def test_passes_through_unchanged_when_vllm_already_included_the_stop_string():
    # the expected case once include_stop_str_in_output is honored
    client = _client_with_response(json_body={
        "choices": [{"text": "<think>look it up</think><search>group relative baseline</search>",
                     "finish_reason": "stop", "stop_reason": "</search>"}]
    })
    output = client("context")
    assert output == "<think>look it up</think><search>group relative baseline</search>"
    action = parse_action(output)
    assert action.kind == "search"
    assert action.content == "group relative baseline"


def test_falls_back_to_reappending_if_vllm_stripped_the_stop_string_anyway():
    # defends against a server that ignores include_stop_str_in_output
    client = _client_with_response(json_body={
        "choices": [{"text": "<think>look it up</think><search>group relative baseline",
                     "finish_reason": "stop", "stop_reason": "</search>"}]
    })
    output = client("context")
    assert output == "<think>look it up</think><search>group relative baseline</search>"
    assert parse_action(output).kind == "search"


def test_does_not_reappend_anything_when_the_model_just_ran_out_of_tokens():
    client = _client_with_response(json_body={
        "choices": [{"text": "<think>rambling with no action tag yet",
                     "finish_reason": "length", "stop_reason": None}]
    })
    output = client("context")
    assert output == "<think>rambling with no action tag yet"
    # this is correctly unparseable -- the round genuinely never produced a complete action
    assert parse_action(output).kind is None


def test_raises_on_an_http_error_instead_of_returning_malformed_text():
    client = _client_with_response(status_code=500, json_body={"error": "internal error"})
    with pytest.raises(httpx.HTTPStatusError):
        client("context")


def test_raises_on_a_connection_failure():
    client = _client_with_response(raise_on_request=httpx.ConnectError("connection refused"))
    with pytest.raises(httpx.ConnectError):
        client("context")


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
        prompt = json.loads(request.content)["prompt"]
        if "<information>" not in prompt:
            text = "<think>search for it</think><search>baseline</search>"
        else:
            text = "<think>done</think><answer>group relative baseline</answer>"
        return httpx.Response(200, json={"choices": [{"text": text, "finish_reason": "stop", "stop_reason": None}]})

    client = VLLMModelClient(base_url="http://fake-vllm:8000", transport=httpx.MockTransport(handler))
    trajectory = run_episode("q1", "What baseline do they use?", "p1", corpus, client)
    assert trajectory.terminated_reason == "answered"
    assert trajectory.final_answer == "group relative baseline"
    assert len(trajectory.steps) == 2
