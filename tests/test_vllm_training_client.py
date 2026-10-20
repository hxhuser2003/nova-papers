import json

import httpx
import pytest

from backend.vllm_training_client import RolloutModelClient


def _client_with_response(json_body):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=json_body)
    return RolloutModelClient(base_url="http://fake-vllm:8000", model="Qwen/Qwen2.5-3B-Instruct",
                               transport=httpx.MockTransport(handler))


def test_generate_sends_logprobs_and_return_tokens_params():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"text": "ok", "logprobs": {"tokens": [], "token_logprobs": []}}]})

    client = RolloutModelClient(base_url="http://fake-vllm:8000", model="m", transport=httpx.MockTransport(handler))
    client.generate("some prompt", stop=["</search>"])
    assert captured["json"]["prompt"] == "some prompt"
    assert captured["json"]["logprobs"] == 1
    assert captured["json"]["stop"] == ["</search>"]
    assert captured["json"]["include_stop_str_in_output"] is True


def test_generate_parses_token_id_placeholders_into_real_integers():
    client = _client_with_response({
        "choices": [{
            "text": "<search>baseline</search>",
            "logprobs": {"tokens": ["token_id:1234", "token_id:5678", "token_id:91"],
                         "token_logprobs": [-0.1, -0.2, -0.3]},
        }]
    })
    result = client.generate("prompt")
    assert result["token_ids"] == [1234, 5678, 91]
    assert result["logprobs"] == [-0.1, -0.2, -0.3]
    assert result["text"] == "<search>baseline</search>"


def test_generate_handles_a_missing_logprobs_field_without_crashing():
    # defensive: if the server wasn't started with --return-tokens-as-token-ids
    # (or logprobs is otherwise absent), fail with empty lists, not a KeyError
    client = _client_with_response({"choices": [{"text": "ok", "logprobs": None}]})
    result = client.generate("prompt")
    assert result == {"text": "ok", "token_ids": [], "logprobs": []}


def test_generate_raises_on_an_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = RolloutModelClient(base_url="http://fake-vllm:8000", model="m", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        client.generate("prompt")
