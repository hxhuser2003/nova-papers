"""Real ModelClient backed by a vLLM OpenAI-compatible completions endpoint
(see backend.agent_loop.ModelClient for the contract this must satisfy:
str context -> str completion).

Not yet exercised against a live server -- 3A (deploying Qwen2.5-3B-Instruct
via vLLM) hasn't landed, so `base_url` has no real value to point at yet.
The request/response shape here follows vLLM's documented OpenAI-compatible
API and is covered by tests against a mocked HTTP transport; swap in the
real base_url once 3A's endpoint exists and nothing else needs to change,
including agent_loop.py, which only depends on the ModelClient contract.
"""

from __future__ import annotations

import httpx

# Generation must stop exactly at the end of one action tag so parse_action()
# sees a complete, well-formed trace -- a round that runs past its action tag
# (e.g. the model keeps narrating after <answer>...</answer>) would otherwise
# produce more than one tag and be flagged as a format error.
STOP_SEQUENCES = ["</search>", "</read>", "</answer>"]


class VLLMModelClient:
    def __init__(
        self,
        base_url: str,
        model: str = "Qwen2.5-3B-Instruct",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 500,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def __call__(self, context: str) -> str:
        response = self._client.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": context,
                "stop": STOP_SEQUENCES,
                # vLLM-specific extension (not in the plain OpenAI API): ask
                # the server to keep the matched stop string in `text`
                # instead of stripping it, so parse_action() sees a complete
                # "<tag>...</tag>" pair. This is the primary fix.
                "include_stop_str_in_output": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            },
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        text = choice["text"]
        # Fallback in case include_stop_str_in_output isn't honored for some
        # reason: reappend the matched stop string ourselves. A finish_reason
        # of "length" (ran out of tokens before closing the tag) is left
        # as-is -- that's a genuinely incomplete, unparseable round, not an
        # artifact of stripping.
        stop_reason = choice.get("stop_reason")
        if choice.get("finish_reason") == "stop" and stop_reason in STOP_SEQUENCES and not text.endswith(stop_reason):
            text += stop_reason
        return text

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "VLLMModelClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
