import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend.config import ROOT, Settings
from backend.provider import ProviderError, build_messages, stream_answer
from backend.retrieval import Corpus


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=24000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list, max_length=12)

    @field_validator("question")
    @classmethod
    def nonempty(cls, value: str):
        if not value.strip():
            raise ValueError("Enter a question.")
        return value.strip()


def event(kind: str, data: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None, transport=None) -> FastAPI:
    settings = settings or Settings()
    corpus = Corpus(settings.corpus_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.glm_timeout_seconds, connect=15), transport=transport,
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(title="Nova Papers", lifespan=lifespan)

    @app.get("/api/health")
    async def health():
        return {"model": settings.glm_model, "api_configured": bool(settings.glm_api_key.get_secret_value()),
                "corpus_ready": bool(corpus.chunks), "paper_count": len(corpus.papers)}

    @app.get("/api/papers")
    async def papers():
        return {"papers": corpus.papers}

    @app.get("/api/papers/{paper_id}/pdf")
    async def paper_pdf(paper_id: str):
        paper = next((p for p in corpus.papers if p["id"] == paper_id), None)
        if not paper or not paper.get("filename"):
            raise HTTPException(404, "Paper not found.")
        path = settings.corpus_dir / "pdfs" / paper["filename"]
        if not path.exists():
            raise HTTPException(404, "Local PDF is missing. Run the corpus preparation script.")
        return FileResponse(path, media_type="application/pdf", filename=paper["filename"], content_disposition_type="inline")

    @app.post("/api/chat")
    async def chat(body: ChatRequest):
        if not settings.glm_api_key.get_secret_value():
            raise HTTPException(503, "GLM API key is not configured. Add GLM_API_KEY to the server's .env and restart.")
        if not corpus.chunks:
            raise HTTPException(503, "The paper corpus is not ready. Run scripts/prepare_corpus.py and restart.")
        history = [m.model_dump() for m in body.history]

        async def generate():
            yield event("status", {"step": "search", "label": "Searching papers…"})
            sources = await asyncio.to_thread(corpus.search, body.question, history)
            yield event("status", {"step": "evidence", "label": f"Selected {len(sources)} passages"})
            yield event("sources", {"sources": sources})
            yield event("status", {"step": "answer", "label": "Drafting answer…"})
            stream = stream_answer(app.state.client, settings, build_messages(body.question, history, sources))
            pending = None
            try:
                while True:
                    pending = asyncio.create_task(anext(stream))
                    while not (await asyncio.wait({pending}, timeout=15))[0]:
                        yield ": keep-alive\n\n"
                    try:
                        content = pending.result()
                    except StopAsyncIteration:
                        break
                    yield event("token", {"text": content})
                yield event("done", {"label": "Answer complete"})
            except ProviderError as exc:
                yield event("error", {"message": str(exc)})
            finally:
                if pending is not None:
                    if not pending.done():
                        pending.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await pending
                await stream.aclose()

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
