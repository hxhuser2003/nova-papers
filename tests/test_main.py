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
    # The first <search> always runs against the raw question text (see
    # agent_loop's first-search forcing), regardless of the model's own query.
    assert events[1][1] == {"step": "search", "label": "Searching: What is the seed lexicon?"}
    assert events[2][1]["step"] == "answer"
    assert events[4][1]["text"] == "a vocabulary of positive and negative predicates"


def test_chat_sources_are_keyed_by_paragraph_id_so_citations_are_clickable():
    # Regression: _sources_for used to mint an unrelated sequential "S1"/"S2"
    # id the model was never told about (see agent_loop.SYSTEM_PROMPT), so an
    # answer could never contain a bracket tag the frontend's citation regex
    # would match -- "N passages attached" always showed, but no citation was
    # ever clickable (confirmed live: https://github.com/hxhuser2003/nova-papers
    # issue found via a real Colab run on 2026-11-18). Sources must be keyed
    # by the paper's own para_id, the exact bracket the model already sees in
    # every <information> block and is now asked to echo back when citing.
    app = create_app(transport=_scripted_transport([
        "<think>search</think><search>seed lexicon</search>",
        "<think>done</think><answer>The seed lexicon had no impact [p_36].</answer>",
    ]))
    with TestClient(app) as client:
        with client.stream("POST", "/api/chat", json={"question": "What is the seed lexicon?"}) as resp:
            events = _events("".join(resp.iter_text()))

    sources_event = next(data for kind, data in events if kind == "sources")
    citation_ids = [s["citation_id"] for s in sources_event["sources"]]
    # One entry per passage the single <search> call returned (BM25 top-3
    # for this question), not one entry per tool call.
    assert citation_ids == ["p_36", "p_13", "p_14"]
    assert all(cid.startswith("p_") for cid in citation_ids)
    # The id in the model's own answer must resolve to a real returned source.
    answer_citation = events[4][1]["text"]
    assert "[p_36]" in answer_citation
    assert "p_36" in citation_ids


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
    # every round is a valid <search>, including the forced extra turn after
    # round 5 runs out (see agent_loop.FORCE_ANSWER_INSTRUCTION) -- 6 calls total
    app = create_app(transport=_scripted_transport(
        ["<think>search</think><search>seed lexicon</search>"] * 6
    ))
    with TestClient(app) as client:
        with client.stream("POST", "/api/chat", json={"question": "What is the seed lexicon?"}) as resp:
            events = _events("".join(resp.iter_text()))

    kinds = [k for k, _ in events]
    assert kinds[-1] == "error"
    assert "forced_answer_invalid" in events[-1][1]["message"]
