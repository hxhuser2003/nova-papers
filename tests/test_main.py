import json

import httpx
from fastapi.testclient import TestClient

from backend.main import create_app


def _scripted_transport(responses):
    calls = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        text = next(calls)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": text}, "finish_reason": "stop", "stop_reason": None}]
        })

    return httpx.MockTransport(handler)


def _events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        kind = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        out.append((kind, data))
    return out


def test_health_reports_the_vllm_config_and_corpus_size():
    app = create_app(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with TestClient(app) as client:
        body = client.get("/api/health").json()
        assert body["model"] == "Qwen/Qwen2.5-3B-Instruct"
        assert body["corpus_ready"] is True
        assert body["paper_count"] == 1585


def test_papers_endpoint_returns_arxiv_links_not_local_pdfs():
    app = create_app(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with TestClient(app) as client:
        papers = client.get("/api/papers").json()["papers"]
        assert len(papers) == 1585
        assert papers[0]["url"].startswith("https://arxiv.org/abs/")


def test_chat_picks_the_right_paper_and_streams_real_round_by_round_status():
    app = create_app(transport=_scripted_transport([
        "<think>search</think><search>seed lexicon</search>",
        "<think>done</think><answer>a vocabulary of positive and negative predicates</answer>",
    ]))
    with TestClient(app) as client:
        with client.stream("POST", "/api/chat", json={"question": "What is the seed lexicon?"}) as resp:
            events = _events("".join(resp.iter_text()))

    kinds = [k for k, _ in events]
    assert kinds == ["status", "status", "status", "sources", "token", "done"]
    assert events[0][1]["step"] == "select_paper"
    assert "Affective Events" in events[0][1]["label"]
    assert events[1][1] == {"step": "search", "label": "Searching: seed lexicon"}
    assert events[2][1]["step"] == "answer"
    assert events[4][1]["text"] == "a vocabulary of positive and negative predicates"


def test_chat_returns_404_when_no_paper_matches(monkeypatch):
    import backend.qasper_corpus

    monkeypatch.setattr(backend.qasper_corpus.QasperCorpus, "find_paper", lambda self, query: None)
    app = create_app(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with TestClient(app) as client:
        resp = client.post("/api/chat", json={"question": "anything"})
    assert resp.status_code == 404


def test_chat_emits_an_error_event_instead_of_hanging_when_the_model_is_unreachable():
    # regression: an httpx.ConnectError from the model call used to propagate
    # out of the SSE generator uncaught, killing the stream with no "error"
    # event ever sent -- the frontend was left showing "Reading..." forever
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    app = create_app(transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        with client.stream("POST", "/api/chat", json={"question": "What is the seed lexicon?"}) as resp:
            events = _events("".join(resp.iter_text()))

    assert [k for k, _ in events][-1] == "error"
    assert events[-1][1]["message"]


def test_chat_emits_an_error_event_instead_of_a_final_answer_on_max_rounds():
    # every round is a valid <search>, so the episode runs out of rounds without answering
    app = create_app(transport=_scripted_transport(
        ["<think>search</think><search>seed lexicon</search>"] * 5
    ))
    with TestClient(app) as client:
        with client.stream("POST", "/api/chat", json={"question": "What is the seed lexicon?"}) as resp:
            events = _events("".join(resp.iter_text()))

    kinds = [k for k, _ in events]
    assert kinds[-1] == "error"
    assert "max_rounds" in events[-1][1]["message"]
