import json

import httpx
import pytest

from local_llm_chat.domain.errors import FreeOperationBlocked, ValidationError
from local_llm_chat.domain.memory_candidates import MemoryCandidateRequest
from local_llm_chat.domain.states import MemoryEvidenceMode, MemoryKind
from local_llm_chat.infrastructure.llm.ollama_memory_candidate_extractor import (
    OllamaMemoryCandidateExtractor,
)


def request(content: str = "一番好きなのはチョコアイス") -> MemoryCandidateRequest:
    return MemoryCandidateRequest(
        "conversation-1",
        "branch-1",
        "message-1",
        "conversation-model",
        content,
        "user",
        frozenset({"user"}),
        frozenset({"character-1"}),
    )


@pytest.mark.asyncio
async def test_extractor_rejects_client_whose_real_endpoint_differs() -> None:
    client = httpx.AsyncClient(
        base_url="http://192.168.1.17:11434",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"message": {"content": "{}"}})
        ),
    )

    with pytest.raises(ValueError, match="base URL"):
        OllamaMemoryCandidateExtractor(
            endpoint="http://127.0.0.1:11434",
            client=client,
            cloud_is_disabled=True,
        )

    await client.aclose()


@pytest.mark.asyncio
async def test_extractor_itself_blocks_cloud_before_http_request() -> None:
    calls = 0
    cloud_disabled = False

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"message": {"content": '{"candidates": []}'}}
        )

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    extractor = OllamaMemoryCandidateExtractor(
        client=client, cloud_is_disabled=lambda: cloud_disabled
    )

    with pytest.raises(FreeOperationBlocked):
        await extractor.extract(request())

    assert calls == 0
    cloud_disabled = True
    assert await extractor.extract(request()) == ()
    assert calls == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_extractor_uses_schema_and_parses_strict_candidates() -> None:
    captured: dict[str, object] = {}
    content = "一番好きなのはチョコアイス"
    candidate = {
        "subject_id": "user",
        "kind": "preference",
        "slot": "favorite_food",
        "value": "チョコアイス",
        "evidence_mode": "explicit",
        "evidence_start": 0,
        "evidence_end": len(content),
    }

    async def handler(http_request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(http_request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"candidates": [candidate]}, ensure_ascii=False
                    ),
                },
                "done": True,
            },
        )

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    extractor = OllamaMemoryCandidateExtractor(client=client, cloud_is_disabled=True)

    drafts = await extractor.extract(request(content))

    assert len(drafts) == 1
    assert drafts[0].kind is MemoryKind.PREFERENCE
    assert drafts[0].evidence_mode is MemoryEvidenceMode.EXPLICIT
    assert drafts[0].value == "チョコアイス"
    assert captured["stream"] is False
    assert captured["think"] is False
    assert captured["options"] == {"temperature": 0}
    assert isinstance(captured["format"], dict)
    assert captured["model"] == "conversation-model"
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_content",
    (
        "```json\n{\"candidates\": []}\n```",
        json.dumps({"candidates": [], "comment": "extra"}),
        json.dumps(
            {
                "candidates": [
                    {
                        "subject_id": "user",
                        "kind": "unknown",
                        "slot": "favorite_food",
                        "value": "チョコ",
                        "evidence_mode": "explicit",
                        "evidence_start": 0,
                        "evidence_end": 3,
                    }
                ]
            }
        ),
        json.dumps(
            {
                "candidates": [
                    {
                        "subject_id": "user",
                        "kind": "preference",
                        "slot": "favorite_food",
                        "value": "チョコ",
                        "evidence_mode": "explicit",
                        "evidence_start": 0,
                        "evidence_end": 999,
                    }
                ]
            }
        ),
    ),
)
async def test_ollama_extractor_rejects_nonconforming_output(
    model_content: str,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": model_content}})

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    extractor = OllamaMemoryCandidateExtractor(client=client, cloud_is_disabled=True)

    with pytest.raises(ValidationError):
        await extractor.extract(request())

    await client.aclose()
