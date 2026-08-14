import json

import httpx
import pytest

from local_llm_chat.domain.errors import ProviderUnavailable, ToolUseUnavailable
from local_llm_chat.domain.models import ChatMessageInput, ChatRequest, ToolDefinition
from local_llm_chat.domain.states import MessageRole
from local_llm_chat.infrastructure.llm.vllm_provider import VllmProvider


@pytest.mark.asyncio
async def test_list_models_uses_openai_models_endpoint_and_bearer_key() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"data": [{"id": "deepseek-v4-flash-2bit", "owned_by": "vllm"}]},
        )

    client = httpx.AsyncClient(
        base_url="http://192.168.1.17:18080/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = VllmProvider(
        "http://192.168.1.17:18080", "vllm", api_key="test-key", client=client
    )

    models = await provider.list_models()

    assert captured == {
        "path": "/v1/models",
        "authorization": "Bearer test-key",
    }
    assert models[0].name == "deepseek-v4-flash-2bit"
    assert not models[0].size_is_known
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_chat_maps_request_and_parses_usage() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    client = httpx.AsyncClient(
        base_url="http://192.168.1.17:18080/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = VllmProvider(
        "http://192.168.1.17:18080", "vllm", api_key="test-key", client=client
    )
    request = ChatRequest(
        "deepseek-v4-flash-2bit",
        "system",
        (ChatMessageInput(MessageRole.USER, "hello"),),
        options={"num_ctx": 8192, "num_predict": 120, "temperature": 0.4},
    )

    chunks = [chunk async for chunk in provider.stream_chat(request)]

    assert captured["model"] == "deepseek-v4-flash-2bit"
    assert captured["max_tokens"] == 120
    assert captured["temperature"] == 0.4
    assert "num_ctx" not in captured
    assert captured["stream_options"] == {"include_usage": True}
    assert "Authorization" not in captured
    assert "hello world" == "".join(chunk.content for chunk in chunks)
    assert chunks[-1].done
    assert chunks[-1].prompt_tokens == 3
    assert chunks[-1].output_tokens == 2
    assert chunks[-1].total_duration_ns is not None
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_chat_rejects_tools_before_network_call() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    client = httpx.AsyncClient(
        base_url="http://192.168.1.17:18080/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = VllmProvider(
        "http://192.168.1.17:18080", "vllm", api_key="test-key", client=client
    )
    request = ChatRequest(
        "model",
        "system",
        (),
        tools=(ToolDefinition("read", "read", {"type": "object"}),),
    )

    with pytest.raises(ToolUseUnavailable):
        _ = [chunk async for chunk in provider.stream_chat(request)]
    assert calls == 0
    await client.aclose()


@pytest.mark.asyncio
async def test_list_models_hides_auth_failure_detail() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "secret detail"})

    client = httpx.AsyncClient(
        base_url="http://192.168.1.17:18080/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = VllmProvider(
        "http://192.168.1.17:18080", "vllm", api_key="test-key", client=client
    )

    with pytest.raises(ProviderUnavailable, match="APIキーが設定されていないか、無効") as error:
        await provider.list_models()
    assert "secret detail" not in str(error.value)
    assert "test-key" not in str(error.value)
    await client.aclose()
