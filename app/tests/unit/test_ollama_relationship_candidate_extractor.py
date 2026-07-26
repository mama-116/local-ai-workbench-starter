import json

import httpx
import pytest

from local_llm_chat.domain.errors import FreeOperationBlocked, ValidationError
from local_llm_chat.domain.relationship_profile import (
    EvidenceContext,
    RelationshipCandidateRequest,
    RelationshipMeaning,
)
from local_llm_chat.infrastructure.llm.ollama_relationship_candidate_extractor import (
    OllamaRelationshipCandidateExtractor,
)


def request(content: str = "ありがとう") -> RelationshipCandidateRequest:
    return RelationshipCandidateRequest(
        "conversation-1",
        "branch-1",
        "message-1",
        "model",
        content,
        frozenset({"character-1"}),
    )


@pytest.mark.asyncio
async def test_relationship_extractor_is_local_strict_and_deterministic() -> None:
    captured: dict[str, object] = {}
    content = "ありがとう"

    async def handler(http_request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(http_request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "candidates": [
                                {
                                    "character_id": "character-1",
                                    "meaning": "positive_interaction",
                                    "severity": "low",
                                    "evidence_context": "direct",
                                    "evidence_start": 0,
                                    "evidence_end": len(content),
                                }
                            ]
                        }
                    )
                }
            },
        )

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    extractor = OllamaRelationshipCandidateExtractor(
        client=client, cloud_is_disabled=True
    )

    drafts = await extractor.extract(request(content))

    assert drafts[0].meaning is RelationshipMeaning.POSITIVE_INTERACTION
    assert drafts[0].evidence_context is EvidenceContext.DIRECT
    assert captured["stream"] is False
    assert captured["think"] is False
    assert captured["options"] == {"temperature": 0}
    assert isinstance(captured["format"], dict)
    await client.aclose()


@pytest.mark.asyncio
async def test_relationship_extractor_blocks_cloud_before_request() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"message": {"content": '{"candidates":[]}'}})

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    extractor = OllamaRelationshipCandidateExtractor(
        client=client, cloud_is_disabled=False
    )
    with pytest.raises(FreeOperationBlocked):
        await extractor.extract(request())
    assert calls == 0
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    (
        "```json\n{\"candidates\":[]}\n```",
        '{"candidates":[],"extra":true}',
        json.dumps(
            {
                "candidates": [
                    {
                        "character_id": "outside-cast",
                        "meaning": "positive_interaction",
                        "severity": "low",
                        "evidence_context": "direct",
                        "evidence_start": 0,
                        "evidence_end": 2,
                    }
                ]
            }
        ),
        json.dumps(
            {
                "candidates": [
                    {
                        "character_id": "character-1",
                        "meaning": "reset",
                        "severity": "low",
                        "evidence_context": "direct",
                        "evidence_start": 0,
                        "evidence_end": 999,
                    }
                ]
            }
        ),
    ),
)
async def test_relationship_extractor_rejects_nonconforming_output(
    output: str,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": output}})

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    extractor = OllamaRelationshipCandidateExtractor(
        client=client, cloud_is_disabled=True
    )
    with pytest.raises(ValidationError):
        await extractor.extract(request())
    await client.aclose()
