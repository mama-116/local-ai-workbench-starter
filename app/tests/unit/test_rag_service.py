from __future__ import annotations

import pytest

from local_llm_chat.application.services.rag_service import RagService
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import (
    DocumentChunk,
    DocumentRecord,
    MessageCitation,
    MessageRagUsage,
    RagSearchResult,
)


class FakeRagRepository:
    def __init__(self) -> None:
        self.saved_document: DocumentRecord | None = None
        self.saved_chunks: tuple[DocumentChunk, ...] = ()

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
        return []

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
