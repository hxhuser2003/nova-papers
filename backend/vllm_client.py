"""Real ModelClient backed by a vLLM OpenAI-compatible chat completions
endpoint (see backend.agent_loop.ModelClient for the contract this must
satisfy: list[{"role", "content"}] -> str completion).

agent_loop.run_episode already maintains the conversation as a structured
list of turns, so this client just forwards it to vLLM's /v1/chat/completions
-- no reconstruction needed. (An earlier version of this file flattened the
transcript into one string and had this client regex-split it back apart;
that approach had two confirmed bugs -- it could mistake the model's own
<think> text for a real tool result if the model paraphrased one, and its
string-search for the prompt/history boundary could misfire on a paper
title or abstract that happened to contain the same text -- both traceable
to the same root cause: guessing structure back out of a flattened string.
Keeping the structure as data end-to-end removes both bugs rather than
patching either one. Not yet exercised against a live server -- 3A's vLLM
endpoint's exact base_url and model name are still placeholders here.)
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
        model: str = "Qwen/Qwen2.5-3B-Instruct",
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

    def __call__(self, messages: list[dict]) -> str:
        response = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "stop": STOP_SEQUENCES,
                # vLLM-specific extension (not in the plain OpenAI API): ask
                # the server to keep the matched stop string in the output
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
        text = choice["message"]["content"]
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
