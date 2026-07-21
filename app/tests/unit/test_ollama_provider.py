import json

import httpx
import pytest

from local_llm_chat.domain.errors import OllamaUnavailable
from local_llm_chat.domain.models import ChatMessageInput, ChatRequest
from local_llm_chat.domain.states import MessageRole
from local_llm_chat.infrastructure.llm.ollama_provider import OllamaProvider


def test_done_chunk_keeps_ollama_generation_duration_for_speed_calculation() -> None:
    chunk = OllamaProvider._parse_stream_line(
        '{"done":true,"message":{"content":""},"eval_count":20,'
        '"eval_duration":1000000000,"total_duration":2000000000}'
    )

    assert chunk.done
    assert chunk.output_tokens == 20
    assert chunk.generation_duration_ns == 1_000_000_000


def test_parses_tool_call_and_marks_broken_arguments_for_denial() -> None:
    chunk = OllamaProvider._parse_stream_line(
        '{"done":true,"message":{"content":"","tool_calls":['
        '{"function":{"name":"read_allowed_text","arguments":"broken"}}]}}'
    )

    assert chunk.tool_calls[0].name == "read_allowed_text"
    assert chunk.tool_calls[0].arguments == {"__invalid_arguments__": True}


@pytest.mark.asyncio
async def test_stream_chat_sends_response_schema_to_ollama() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=b'{"message":{"content":"{}"},"done":true}\n',
        )

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaProvider(client=client)
    schema: dict[str, object] = {"type": "object"}
    request = ChatRequest(
        "model",
        "system",
        (ChatMessageInput(MessageRole.USER, "hello"),),
        response_format=schema,
    )

    chunks = [chunk async for chunk in provider.stream_chat(request)]

    assert chunks[-1].done
    assert captured["format"] == schema
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_read_error_is_reported_as_ollama_connection_loss() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadError("PRIVATE network detail", request=request)

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaProvider(client=client)
    request = ChatRequest(
        "model",
        "system",
        (ChatMessageInput(MessageRole.USER, "hello"),),
    )

    with pytest.raises(
        OllamaUnavailable, match="Ollamaとの接続が途中で切れました"
    ):
        _ = [chunk async for chunk in provider.stream_chat(request)]
    assert calls == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_model_list_read_error_is_not_misreported_as_invalid_payload() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadError("PRIVATE network detail", request=request)

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaProvider(client=client)

    with pytest.raises(OllamaUnavailable, match="Ollamaに接続できません"):
        await provider.list_models()
    assert calls == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_model_list_reconnects_once_after_stale_transport() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadError("stale keep-alive", request=request)
        return httpx.Response(200, json={"models": []})

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    provider = OllamaProvider(client=client)

    assert await provider.list_models() == []
    assert calls == 2
    await client.aclose()
