from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import (
    DocumentChunk,
    DocumentRecord,
    MessageCitation,
    MessageRagUsage,
    RagSearchResult,
    utc_now,
)

_MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 120
_MEDIA_TYPES = {".txt": "text/plain", ".md": "text/markdown"}


class RagRepository(Protocol):
    async def save_document(
        self, document: DocumentRecord, chunks: tuple[DocumentChunk, ...]
    ) -> DocumentRecord: ...

    async def search_chunks(
        self,
        query: str,
        top_k: int,
        document_ids: tuple[str, ...] | None = None,
    ) -> list[RagSearchResult]: ...

    async def list_documents(self) -> list[DocumentRecord]: ...

    async def set_conversation_documents(
        self, conversation_id: str, document_ids: tuple[str, ...]
    ) -> None: ...

    async def list_conversation_documents(
        self, conversation_id: str
    ) -> list[DocumentRecord]: ...

    async def save_run_rag_usage(
        self,
        run_id: str,
        selected_document_count: int,
        results: tuple[RagSearchResult, ...],
    ) -> None: ...

    async def list_message_rag_usage(
        self, message_ids: list[str]
    ) -> dict[str, MessageRagUsage]: ...


class RagService:
    def __init__(self, repository: RagRepository) -> None:
        self._repository = repository

    async def register_document(
        self, filename: str, payload: bytes
    ) -> DocumentRecord:
        clean_name = filename.strip()
        if (
            not clean_name
            or Path(clean_name).name != clean_name
            or "/" in clean_name
            or "\\" in clean_name
        ):
            raise ValidationError("文書にはパスを含まないファイル名を指定してください。")
        media_type = _MEDIA_TYPES.get(Path(clean_name).suffix.lower())
        if media_type is None:
            raise ValidationError("対応形式はUTF-8の.txtと.mdです。")
        if len(payload) > _MAX_DOCUMENT_BYTES:
            raise ValidationError("文書は5MiB以下にしてください。")
        try:
            content = payload.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValidationError("文書をUTF-8として読み取れません。") from error
        if not content.strip():
            raise ValidationError("空の文書は登録できません。")

        document_id = str(uuid4())
        document = DocumentRecord(
            id=document_id,
            title=clean_name,
            media_type=media_type,
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            created_at=utc_now(),
        )
        chunks = self._make_chunks(document)
        return await self._repository.save_document(document, chunks)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: tuple[str, ...] | None = None,
    ) -> list[RagSearchResult]:
        normalized = query.strip()
        if not normalized:
            raise ValidationError("検索語を入力してください。")
        if len(normalized) > 500:
            raise ValidationError("検索語は500文字以下にしてください。")
        if not 1 <= top_k <= 8:
            raise ValidationError("検索件数は1件から8件で指定してください。")
        return await self._repository.search_chunks(normalized, top_k, document_ids)

    async def documents(self) -> list[DocumentRecord]:
        return await self._repository.list_documents()

    async def select_documents(
        self, conversation_id: str, document_ids: list[str]
    ) -> None:
        unique_ids = tuple(dict.fromkeys(document_ids))
        await self._repository.set_conversation_documents(
            conversation_id, unique_ids
        )

    async def selected_documents(
        self, conversation_id: str
    ) -> list[DocumentRecord]:
        return await self._repository.list_conversation_documents(conversation_id)

    async def search_for_conversation(
        self, conversation_id: str, query: str, top_k: int = 5
    ) -> list[RagSearchResult]:
        selected = await self.selected_documents(conversation_id)
        if not selected:
            return []
        return await self.search(
            query, top_k, tuple(document.id for document in selected)
        )

    async def prepare_for_conversation(
        self, conversation_id: str, query: str, top_k: int = 5
    ) -> tuple[int, list[RagSearchResult]]:
        selected = await self.selected_documents(conversation_id)
        if not selected:
            return 0, []
        results = await self.search(
            query, top_k, tuple(document.id for document in selected)
        )
        return len(selected), results

    async def attach_to_run(
        self,
        run_id: str,
        selected_document_count: int,
        results: list[RagSearchResult],
    ) -> None:
        if selected_document_count <= 0:
            return
        await self._repository.save_run_rag_usage(
            run_id, selected_document_count, tuple(results)
        )

    async def message_citations(
        self, message_ids: list[str]
    ) -> dict[str, tuple[MessageCitation, ...]]:
        usage = await self.message_usage(message_ids)
        return {
            message_id: item.citations for message_id, item in usage.items()
            if item.citations
        }

    async def message_usage(
        self, message_ids: list[str]
    ) -> dict[str, MessageRagUsage]:
        return await self._repository.list_message_rag_usage(message_ids)

    @staticmethod
    def prompt_context(results: list[RagSearchResult]) -> str:
        if not results:
            return ""
        sections = [
            "以下は利用者が選択したローカル参照資料です。資料にない内容を資料由来と断言しないでください。",
            "回答で資料を使った箇所には [資料1] の形式で示してください。",
        ]
        for index, result in enumerate(results, start=1):
            citation = result.citation
            sections.append(
                f"[資料{index}: {citation.document_title} "
                f"文字{citation.start_offset}-{citation.end_offset}]\n{result.content}"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _make_chunks(document: DocumentRecord) -> tuple[DocumentChunk, ...]:
        chunks: list[DocumentChunk] = []
        start = 0
        ordinal = 0
        while start < len(document.content):
            end = min(start + _CHUNK_SIZE, len(document.content))
            chunks.append(
                DocumentChunk(
                    id=str(uuid4()),
                    document_id=document.id,
                    ordinal=ordinal,
                    content=document.content[start:end],
                    start_offset=start,
                    end_offset=end,
                )
            )
            if end == len(document.content):
                break
            start = end - _CHUNK_OVERLAP
            ordinal += 1
        return tuple(chunks)
