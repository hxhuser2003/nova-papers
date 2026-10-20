from __future__ import annotations

import asyncio
import json
import queue
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

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


def _sources_for(trajectory: AgentTrajectory, paper: dict) -> list[dict]:
    """One citation entry per search/read step, built from the passages the
    agent actually looked at -- not from <information> markup."""
    sources = []
    for step in trajectory.steps:
        if step.action.kind not in ("search", "read") or not step.tool_result:
            continue
        quote = step.tool_result.removeprefix("<information>").removesuffix("</information>")
        sources.append({
            "citation_id": f"S{len(sources) + 1}", "title": paper["title"],
            "short_title": paper["title"], "quote": quote,
            "url": f"https://arxiv.org/abs/{paper['paper_id']}",
        })
    return sources


def create_app(settings: Settings | None = None, transport=None) -> FastAPI:
    settings = settings or Settings()
    corpus = QasperCorpus(settings.qasper_corpus_dir)

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
        if not corpus.papers:
            raise HTTPException(503, "The paper corpus is not ready. Run scripts/prepare_qasper.py and restart.")
        paper_id = corpus.find_paper(body.question)
        if paper_id is None:
            raise HTTPException(404, "Could not find a paper in the corpus related to this question.")
        paper = corpus.papers[paper_id]

        async def generate():
            yield event("status", {"step": "select_paper", "label": f'Reading "{paper["title"]}"…'})

            # run_episode is synchronous and blocking (it makes HTTP calls to
            # vLLM round by round); run it in a worker thread so the event loop
            # stays free to flush each status event as on_step reports it,
            # rather than only learning about progress after the whole
            # trajectory finishes. queue.Queue is thread-safe for exactly this
            # handoff between the worker thread and the loop.
            step_queue: queue.Queue = queue.Queue()

            def on_step(step: AgentStep) -> None:
                step_queue.put(step)

            def run() -> AgentTrajectory:
                try:
                    return run_episode(
                        question_id=str(uuid.uuid4()), question=body.question, paper_id=paper_id,
                        corpus=corpus, model=app.state.model_client, on_step=on_step,
                    )
                finally:
                    step_queue.put(None)

            task = asyncio.create_task(asyncio.to_thread(run))
            while True:
                step = await asyncio.to_thread(step_queue.get)
                if step is None:
                    break
                yield event("status", _status_for_step(step))
            trajectory = await task

            if trajectory.terminated_reason != "answered":
                yield event("error", {
                    "message": f"The agent could not produce an answer (ended: {trajectory.terminated_reason})."
                })
                return
            yield event("sources", {"sources": _sources_for(trajectory, paper)})
            yield event("token", {"text": trajectory.final_answer})
            yield event("done", {"label": "Answer complete"})

        return StreamingResponse(generate(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    dist = ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        async def frontend():
            return FileResponse(dist / "index.html")

    return app


app = create_app()
