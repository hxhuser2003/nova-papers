"""Training-time vLLM client: returns real token ids + per-token logprobs
tied exactly to the training tokenizer, which is what GRPOTrainer's
rollout_func needs (see TRAINING_NOTES.md's entry on this gap).
backend.vllm_client.VLLMModelClient only returns decoded text -- it was
built for agent_loop.py's inference-time string-in/string-out ModelClient
contract, which doesn't need this.

Uses the standard, long-established OpenAI-compatible /v1/completions
logprobs shape (choices[0].logprobs.tokens / .token_logprobs), not the
newer /inference/v1/generate "tokens in, tokens out" API -- that
endpoint's exact response envelope is still being actively reshaped
upstream (see open vLLM PRs reworking its logprobs format as of
2026-10), so this sticks to the more stable surface instead.

Requires the vLLM server to be started with --return-tokens-as-token-ids,
which replaces each decoded token string in `logprobs.tokens` with the
literal string "token_id:<N>" so the raw integer id can be recovered
without a second tokenizer round-trip.

NOT YET VERIFIED AGAINST A LIVE SERVER: the request/response shapes here
follow vLLM's documented and source-confirmed behavior, but this hasn't
been run against an actual vLLM instance (no GPU access while writing
this). Treat the parsing logic as "should be right, confirm on CARC."
"""

from __future__ import annotations

import math
import re

import httpx


class RolloutModelClient:
    """Produces token ids + logprobs for one round of generation, given an
    chat-templated string or exact token-ID prefix. Does not own a tokenizer itself:
    the caller (backend.token_rollout) is responsible for rendering messages into `prompt`
    and for tokenizing anything this client doesn't generate itself (e.g.
    the <information> blocks a tool injects between rounds)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 500,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def generate(self, prompt: str | list[int], stop: list[str] | None = None,
                 max_tokens: int | None = None) -> dict:
        """Returns {"text": str, "token_ids": list[int], "logprobs": list[float]},
        one entry per generated token, in order."""
        response = self._client.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": prompt,
                "logprobs": 1,
                "stop": stop or [],
                "include_stop_str_in_output": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            },
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        logprobs = choice.get("logprobs") or {}
        tokens = logprobs.get("tokens") or []
        if not tokens or any(not isinstance(t, str) or not re.fullmatch(r"token_id:\d+", t) for t in tokens):
            raise ValueError("Missing real token IDs; start vLLM with --return-tokens-as-token-ids")
        token_ids = [int(t.split(":", 1)[1]) for t in tokens]
        values = logprobs.get("token_logprobs") or []
        if len(values) != len(token_ids) or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) for v in values
        ):
            raise ValueError("Expected one finite logprob per generated token")
        return {
            "text": choice["text"],
            "token_ids": token_ids,
            "logprobs": values,
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RolloutModelClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
