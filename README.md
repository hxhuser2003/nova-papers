# Nova Papers

Nova is a stage-one SDE demo for grounded RL/LLM paper Q&A. It uses a local corpus of five open arXiv papers, BM25 retrieval, FastAPI SSE, and the OpenAI-compatible GLM-5.3 Chat Completions API. The browser never receives the API key or the model's private reasoning stream.

## Run it

1. Install Python dependencies and prepare the corpus:

   ```bash
   uv sync
   uv run python scripts/prepare_corpus.py
   ```

   The repository includes the resulting `corpus/index.json` and PDFs. Re-run the script only when replacing the paper set.

2. Create `.env` from `.env.example` and set `GLM_API_KEY`.

3. Install and build the frontend:

   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

4. Serve the built app and API together:

   ```bash
   uv run uvicorn backend.main:app --reload --port 8127
   ```

   Open `http://127.0.0.1:8127`. During frontend development, use `npm run dev` in `frontend/`; Vite runs on port 5173 and proxies `/api` to port 8127.

## Verify

```bash
uv run pytest -q
```

The tests cover acronym/title retrieval, follow-up context, streaming parsing, reasoning suppression, and safe provider errors. A real GLM request requires the user's key and is intentionally not made in automated tests.

The interview decisions and explicit stage-one boundary are recorded in [STAGE_ONE.md](STAGE_ONE.md). RLVR/GRPO training remains future scope from `project_goal.md`.
