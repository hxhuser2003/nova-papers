import json

import httpx
import pytest

from backend.config import Settings
from backend.provider import ProviderError, stream_answer


async def collect(iterator):
    return "".join([part async for part in iterator])


@pytest.mark.anyio
async def test_stream_ignores_reasoning_and_yields_content():
    async def handler(request):
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["thinking"]["type"] == "enabled"
        assert body["reasoning_effort"] == "max"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=(
            'data: {"choices":[{"delta":{"reasoning_content":"secret"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"grounded"}}]}\n\n'
            'data: [DONE]\n\n'
        ).encode())
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        settings = Settings(glm_api_key="test", glm_base_url="https://mock.test")
        assert await collect(stream_answer(client, settings, [{"role": "user", "content": "hi"}])) == "grounded"


@pytest.mark.anyio
async def test_non_200_is_safe_provider_error():
    async def handler(request):
        return httpx.Response(401, text='{"secret":"do not expose"}')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderError, match="authentication failed"):
            await collect(stream_answer(client, Settings(glm_api_key="test"), []))
