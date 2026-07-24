from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.explicit_memory import (
    MAX_EXPLICIT_MEMORY_VALUE_CHARACTERS,
    ExplicitMemoryDecision,
    ExplicitMemoryEvent,
    ExplicitMemoryReviewItem,
)
from local_llm_chat.domain.ports.repositories import AppRepository


@dataclass(frozen=True, slots=True)
class RememberExplicitMemoryRequest:
    request_id: str
    conversation_id: str
    branch_id: str
    source_message_id: str
    expected_character_id: str
    value: str


@dataclass(frozen=True, slots=True)
class UndoExplicitMemoryRequest:
    conversation_id: str
    branch_id: str
    target_event_id: str
    expected_character_id: str


class ExplicitMemoryService:
    def __init__(
        self,
        repository: AppRepository,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._now = now or _utc_now

    async def remember(
        self, request: RememberExplicitMemoryRequest
    ) -> ExplicitMemoryEvent:
        value = request.value.strip()
        if not value or len(value) > MAX_EXPLICIT_MEMORY_VALUE_CHARACTERS:
            raise ValidationError(
                "明示記憶は1文字以上200文字以内で入力してください。"
            )
        try:
            return await self._repository.remember_explicit_memory(
                request.request_id,
                request.conversation_id,
                request.branch_id,
                request.source_message_id,
                request.expected_character_id,
                value,
                self._now(),
            )
        except ValidationError as error:
            raise ValidationError(
                "保存条件が変わりました。会話、分岐、キャラクターを確認してください。"
            ) from error

    async def undo(
        self, request: UndoExplicitMemoryRequest
    ) -> ExplicitMemoryDecision:
        try:
            return await self._repository.undo_explicit_memory(
                str(uuid4()),
                request.conversation_id,
                request.branch_id,
                request.target_event_id,
                request.expected_character_id,
                self._now(),
            )
        except ValidationError as error:
            raise ValidationError(
                "元に戻せませんでした。会話、分岐、キャラクターを確認してください。"
            ) from error

    async def list_items(
        self,
        conversation_id: str,
        branch_id: str,
        character_id: str,
    ) -> tuple[ExplicitMemoryReviewItem, ...]:
        return await self._repository.project_explicit_memory(
            conversation_id, branch_id, character_id
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)
