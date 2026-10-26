# Nova Papers

Nova is a paper Q&A agent over QASPER (1585 NLP papers, 5049 questions): given a question, it picks the right paper, then runs a multi-round `<think>/<search>/<read>/<answer>` loop against that paper's BM25-indexed paragraphs, backed by a self-hosted vLLM model (currently Qwen2.5-3B/7B-Instruct) rather than an external API. The browser never receives the model's private reasoning stream.

The agent is the RLVR/GRPO training target: `backend/agent_loop.py` runs the inference-time loop, `backend/reward.py` scores trajectories, and `scripts/train_grpo.py` trains it with TRL's `GRPOTrainer`. See [TRAINING_NOTES.md](TRAINING_NOTES.md) for the training design/status and [docs/EXPERIMENT_RECORD.md](docs/EXPERIMENT_RECORD.md) for real experiment results.

## Run it

1. Start a vLLM server serving the model (CARC or any CUDA host), e.g.:

   ```bash
   vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000
   ```

2. Install Python dependencies and prepare the corpus:

   ```bash
   uv sync
   uv run python scripts/prepare_qasper.py
   ```

   The repository includes the resulting `corpus/qasper/papers.jsonl` and `questions.jsonl`. Re-run the script only to refresh the dataset.

3. Create `.env` from `.env.example` and point `VLLM_BASE_URL`/`VLLM_MODEL` at the server from step 1.

4. Install and build the frontend:

   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

5. Serve the built app and API together:

   ```bash
   uv run uvicorn backend.main:app --reload --port 8127
   ```

   Open `http://127.0.0.1:8127`. During frontend development, use `npm run dev` in `frontend/`; Vite runs on port 5173 and proxies `/api` to port 8127.

## Verify

```bash
uv run pytest -q
```

The tests cover retrieval, the agent loop's action parsing and termination conditions, the reward function, the GRPO rollout/training wiring, and safe handling of an unreachable model service. They run without a live vLLM server or GPU; a few CARC-only scripts (training, CARC evaluation jobs) are exercised for their testable logic but not end-to-end here — see [TRAINING_NOTES.md](TRAINING_NOTES.md) for what's been verified on real GPU hardware versus what's still pending.

[STAGE_ONE.md](STAGE_ONE.md) records this project's original stage-one scope (a simpler GLM-backed demo over five PDFs); that stage is superseded by the QASPER/vLLM/agent_loop design described above. `project_goal.md` has the broader RLVR/GRPO vision.
