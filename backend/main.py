from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend.chat_admission import ChatAdmission, QueueFull

from backend.agent_loop import AgentStep, AgentTrajectory, run_episode
from backend.config import ROOT, Settings
from backend.qasper_corpus import QasperCorpus
from backend.vllm_client import VLLMModelClient


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def nonempty(cls, value: str):
        if not value.strip():
            raise ValueError("Enter a question.")
        return value.strip()


def event(kind: str, data: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _status_for_step(step: AgentStep) -> dict:
    kind = step.action.kind
    if kind == "search":
        return {"step": "search", "label": f"Searching: {step.action.content}"}
    if kind == "read":
        return {"step": "read", "label": f"Reading paragraph {step.action.content}"}
    if kind == "answer":
        return {"step": "answer", "label": "Drafting answer…"}
    return {"step": "retry", "label": "Reformatting response…"}


PASSAGE_PATTERN = re.compile(r"^\[(p_\d+)\] (.*)$", re.MULTILINE)
# Captures the whole "(section) text..." remainder as the quote, rather than
# trying to additionally strip the "(section)" prefix with a nested-paren-
# aware pattern -- real section names can themselves contain parentheses
# (e.g. "CA (Cause Pairs)"), which a naive "\([^)]*\)" group fails to skip
# past, silently dropping that paragraph's citation entirely.


def _sources_for(trajectory: AgentTrajectory, paper: dict) -> list[dict]:
    """One citation entry per distinct paragraph the agent's search/read
    actions actually surfaced, keyed by the paper's own para_id (e.g. "p_12")
    -- the exact bracket id the model already sees in every <information>
    block and is now asked (see agent_loop.SYSTEM_PROMPT) to echo back when
    citing. Previously this minted an unrelated sequential "S1"/"S2" id the
    model was never told about, so its answers could never contain a bracket
    tag the frontend's citation regex would actually match -- "N passages
    attached" always showed, but no citation was ever clickable. A single
    <search> result can list multiple passages; each becomes its own entry.
    Dedup by para_id (last occurrence wins) since a later <read> of the same
    paragraph returns the full, untruncated text, while an earlier <search>
    hit may have truncated it to ~150 words (see QasperCorpus.search)."""
    sources: dict[str, dict] = {}
    for step in trajectory.steps:
        if step.action.kind not in ("search", "read") or not step.tool_result:
            continue
        body = step.tool_result.removeprefix("<information>").removesuffix("</information>")
        for para_id, quote in PASSAGE_PATTERN.findall(body):
            sources[para_id] = {
                "citation_id": para_id, "title": paper["title"],
                "short_title": paper["title"], "quote": quote,
                "url": f"https://arxiv.org/abs/{paper['paper_id']}",
            }
    return list(sources.values())


def create_app(settings: Settings | None = None, transport=None) -> FastAPI:
    settings = settings or Settings()
    corpus = QasperCorpus(settings.qasper_corpus_dir)
    # One application process only; multiple workers need shared admission.
    admission = ChatAdmission(settings.max_concurrent_chats,
                              settings.max_queued_chats, settings.chat_queue_timeout_seconds)
    logger = logging.getLogger("uvicorn.error")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = VLLMModelClient(
            base_url=settings.vllm_base_url, model=settings.vllm_model,
            timeout=settings.vllm_timeout_seconds, transport=transport,
        )
        app.state.model_client = client
        try:
            yield
        finally:
            client.close()

    app = FastAPI(title="Nova Papers", lifespan=lifespan)

    @app.get("/api/health")
    async def health():
        return {"model": settings.vllm_model, "vllm_base_url": settings.vllm_base_url,
                "corpus_ready": bool(corpus.papers), "paper_count": len(corpus.papers)}

    @app.get("/api/papers")
    async def papers():
        return {"papers": [
            {"id": p["paper_id"], "title": p["title"], "abstract": p["abstract"],
             "url": f"https://arxiv.org/abs/{p['paper_id']}"}
            for p in corpus.papers.values()
        ]}

    @app.post("/api/chat")
    async def chat(body: ChatRequest):
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        def trace(stage, **fields):
            logger.info("chat_timing %s", json.dumps(dict(request_id=request_id, stage=stage,
                        wall_time=time.time(), elapsed_s=time.perf_counter()-started, **fields)))
        trace("request_received")
        if not corpus.papers:
            raise HTTPException(503, "The paper corpus is not ready. Run scripts/prepare_qasper.py and restart.")
        trace("find_paper_start")
        paper_id = corpus.find_paper(body.question)
        trace("find_paper_end")
        if paper_id is None:
            raise HTTPException(404, "Could not find a paper in the corpus related to this question.")
        paper = corpus.papers[paper_id]

        task = None
        stopped = threading.Event()

        async def generate():
            nonlocal task
            acquired = False
            try:
                if admission.semaphore.locked():
                    yield event("status", {"step": "queued", "label": "Server busy — waiting for a slot…"})
                trace("admission_wait_start")
                try:
                    await admission.acquire()
                except (QueueFull, asyncio.TimeoutError) as exc:
                    code = "queue_full" if isinstance(exc, QueueFull) else "queue_timeout"
                    trace(code)
                    yield event("error", {"code": code, "request_id": request_id,
                                          "message": "Server is busy. Please try again shortly."})
                    return
                acquired = True
                trace("admission_acquired")
                async for chunk in answer():
                    yield chunk
            finally:
                stopped.set()
                # Cancelling to_thread does NOT stop the underlying HTTP call.
                # Hold the admission slot until the worker actually returns.
                if acquired:
                    if task is not None and not task.done():
                        def release_when_done(future):
                            if not future.cancelled():
                                future.exception()
                            admission.release()
                            trace("admission_released_after_disconnect")
                        task.add_done_callback(release_when_done)
                    else:
                        admission.release()
                        trace("admission_released")
                trace("stream_closed")

        async def answer():
            nonlocal task
            yield event("status", {"step": "select_paper", "label": f'Reading "{paper["title"]}"…'})

            # run_episode is synchronous and blocking (it makes HTTP calls to
            # vLLM round by round); run it in a worker thread so the event loop
            # stays free to flush each status event as on_step reports it.
            # Thread-safe callbacks feed an async queue without blocking a
            # second executor thread merely to wait for progress.
            loop = asyncio.get_running_loop()
            step_queue = asyncio.Queue()
            call_number = 0

            def timed_model(messages):
                nonlocal call_number
                if stopped.is_set():
                    raise RuntimeError("Client disconnected")
                call_number += 1
                trace("vllm_call_start", call=call_number)
                try:
                    return app.state.model_client(messages)
                except Exception as exc:
                    trace("vllm_call_error", call=call_number, error_type=type(exc).__name__)
                    raise
                finally:
                    trace("vllm_call_end", call=call_number)


            def on_step(step: AgentStep) -> None:
                trace("agent_step_end", action=step.action.kind)
                loop.call_soon_threadsafe(step_queue.put_nowait, step)

            def run() -> AgentTrajectory:
                trace("worker_started")
                try:
                    return run_episode(
                        question_id=request_id, question=body.question, paper_id=paper_id,
                        corpus=corpus, model=timed_model, on_step=on_step,
                    )
                except StopIteration as e:
                    # asyncio.Future.set_exception() refuses a bare StopIteration
                    # (it conflicts with the generator/coroutine protocol), which
                    # corrupts the task below instead of raising cleanly -- the
                    # `except Exception` around `await task` would never run and
                    # the stream hangs forever instead of emitting an "error"
                    # event. Must not cross the thread/Task boundary as-is.
                    raise RuntimeError(str(e)) from e
                finally:
                    trace("worker_finished")
                    loop.call_soon_threadsafe(step_queue.put_nowait, None)

            trace("worker_submitted")
            task = asyncio.create_task(asyncio.to_thread(run))
            while True:
                step = await step_queue.get()
                if step is None:
                    break
                yield event("status", _status_for_step(step))
            try:
                trajectory = await asyncio.shield(task)
            except Exception:
                # The model call failed outright (e.g. vLLM is unreachable or
                # timed out) rather than the agent finishing with no answer --
                # a different failure mode from terminated_reason below, but
                # the user sees the same thing: a graceful error, not a dead
                # connection they're left staring at.
                yield event("error", {"message": "Could not reach the model service. Please try again shortly."})
                return

            if trajectory.terminated_reason != "answered":
                yield event("error", {
                    "message": f"The agent could not produce an answer (ended: {trajectory.terminated_reason})."
                })
                return
            yield event("sources", {"sources": _sources_for(trajectory, paper)})
            yield event("token", {"text": trajectory.final_answer})
            yield event("done", {"label": "Answer complete"})

        return StreamingResponse(generate(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-ID": request_id,
        })

    dist = ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        async def frontend():
            return FileResponse(dist / "index.html")

    return app


app = create_app()
