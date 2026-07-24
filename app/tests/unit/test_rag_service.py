from __future__ import annotations

import pytest

from local_llm_chat.application.services.rag_service import RagService
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import (
    DocumentChunk,
    DocumentRecord,
    MessageCitation,
    MessageRagUsage,
    RagCitation,
    RagSearchResult,
)


class FakeRagRepository:
    def __init__(self) -> None:
        self.saved_document: DocumentRecord | None = None
        self.saved_chunks: tuple[DocumentChunk, ...] = ()
        self.search_results: list[RagSearchResult] = []

    async def save_document(
        self, document: DocumentRecord, chunks: tuple[DocumentChunk, ...]
    ) -> DocumentRecord:
        self.saved_document = document
        self.saved_chunks = chunks
        return document

    async def search_chunks(
        self,
        query: str,
        top_k: int,
        document_ids: tuple[str, ...] | None = None,
    ) -> list[RagSearchResult]:
        return self.search_results[:top_k]

    async def list_documents(self) -> list[DocumentRecord]:
        return [self.saved_document] if self.saved_document is not None else []

    async def set_conversation_documents(
        self, conversation_id: str, document_ids: tuple[str, ...]
    ) -> None:
        return None

    async def list_conversation_documents(
        self, conversation_id: str
    ) -> list[DocumentRecord]:
        return []

    async def save_run_rag_usage(
        self,
        run_id: str,
        selected_document_count: int,
        results: tuple[RagSearchResult, ...],
    ) -> None:
        return None

    async def list_message_rag_usage(
        self, message_ids: list[str]
    ) -> dict[str, MessageRagUsage]:
        return {}


class FakeSemanticIndex:
    def __init__(
        self, results: list[RagSearchResult], *, fail: bool = False
    ) -> None:
        self.results = results
        self.fail = fail
        self.changed = False

    async def documents_changed(self) -> None:
        self.changed = True

    async def search(
        self,
        query: str,
        top_k: int,
        document_ids: tuple[str, ...] | None = None,
    ) -> list[RagSearchResult]:
        if self.fail:
            raise RuntimeError("embedding endpoint unavailable")
        return self.results[:top_k]


@pytest.mark.asyncio
async def test_registers_utf8_text_as_bounded_overlapping_chunks() -> None:
    repository = FakeRagRepository()
    service = RagService(repository)
    content = ("あ" * 900).encode()

    document = await service.register_document("notes.md", content)

    assert document.title == "notes.md"
    assert repository.saved_document == document
    assert len(repository.saved_chunks) == 2
    assert len(repository.saved_chunks[0].content) == 800
    assert repository.saved_chunks[1].start_offset == 680
    assert repository.saved_chunks[1].end_offset == 900


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        ("empty.txt", b" \n", "空"),
        ("binary.txt", b"\xff\xfe", "UTF-8"),
        ("manual.pdf", b"content", "対応"),
        ("../secret.txt", b"content", "ファイル名"),
    ],
)
async def test_rejects_invalid_document_before_saving(
    filename: str, payload: bytes, message: str
) -> None:
    repository = FakeRagRepository()
    service = RagService(repository)

    with pytest.raises(ValidationError, match=message):
        await service.register_document(filename, payload)

    assert repository.saved_document is None


@pytest.mark.asyncio
async def test_rejects_document_larger_than_five_mib() -> None:
    repository = FakeRagRepository()
    service = RagService(repository)

    with pytest.raises(ValidationError, match="5MiB"):
        await service.register_document(
            "huge.txt", b"a" * (5 * 1024 * 1024 + 1)
        )

    assert repository.saved_document is None


@pytest.mark.asyncio
async def test_rejects_empty_search_query() -> None:
    service = RagService(FakeRagRepository())

    with pytest.raises(ValidationError, match="検索語"):
        await service.search("   ")


@pytest.mark.asyncio
async def test_hybrid_search_uses_rrf_and_semantic_failure_keeps_lexical() -> None:
    repository = FakeRagRepository()
    lexical = RagSearchResult(
        "語句一致",
        citation=_citation("lexical", "語句.md"),
        score=3.0,
    )
    semantic = RagSearchResult(
        "意味一致",
        citation=_citation("semantic", "意味.md"),
        score=0.9,
    )

    repository.search_results = [lexical]
    service = RagService(repository, FakeSemanticIndex([semantic]))
    results = await service.search("言い換え")

    assert {result.citation.chunk_id for result in results} == {
        "lexical",
        "semantic",
    }
    assert all(result.score > 0 for result in results)

    fallback = RagService(repository, FakeSemanticIndex([], fail=True))
    assert await fallback.search("言い換え") == [lexical]


def _citation(chunk_id: str, title: str) -> RagCitation:
    return RagCitation("document", title, chunk_id, 0, 4)
