# Stage one: paper Q&A demo

Agreed scope from the design interview:

- A polished React + TypeScript frontend and FastAPI backend for an SDE demo.
- External OpenAI-compatible Chat Completions API at `https://open.bigmodel.cn/api/coding/paas/v4`.
- Model `glm-5.3`, `thinking: {"type": "enabled"}`, `reasoning_effort: "max"`.
- Server-side API key supplied locally; no local model or GPU required.
- A fixed, downloaded RL/LLM paper corpus. No upload flow.
- Questions and conversational follow-ups, with fresh retrieval each turn.
- Stream safe status events and answer text through SSE. Do not expose model reasoning.
- Prompt the model to attach evidence to every factual claim. This is a model convention,
  not a guarantee or an application-level fact checker. Display filenames and passages.
- Visible paper-list panel, loading/empty states, clear API errors.
- No RL training, code execution, authentication, or production quality targets in stage one.

Implementation choices:

- Five versioned papers: PPO, InstructGPT, DPO, DeepSeekMath, and DeepSeek-R1.
- Extract PDF text into a local corpus; BM25 retrieval needs no embedding service.
- Include recent conversation in retrieval and model context; retain evidence with each answer.
- A new conversation resets browser-held history. Conversations are not persisted on the server.
- Single-origin production serving through FastAPI; Vite proxies `/api` during development.

Acceptance demo: ask “What is GRPO?”, then “How does it differ from PPO?”. Show retrieval
events, streamed answers, supporting passages, and the local paper list. Exercise an API
failure and confirm it appears in the interface without exposing credentials. Real provider
verification requires the user's API key; automated checks use a test transport explicitly.

`project_goal.md` remains the broader future vision, including RLVR/GRPO training.
