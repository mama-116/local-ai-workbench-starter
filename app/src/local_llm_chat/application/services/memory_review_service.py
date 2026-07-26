from __future__ import annotations

from dataclasses import dataclass

from local_llm_chat.domain.explicit_memory import ExplicitMemoryReviewItem
from local_llm_chat.application.services.memory_decision_service import (
    MemoryDecisionRequest,
    MemoryDecisionService,
)
from local_llm_chat.domain.canonical_memory import (
    CanonicalMemoryReviewItem,
    MemoryApprovalDecision,
)
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.states import MemoryApprovalState, MemoryKind


@dataclass(frozen=True, slots=True)
class MemoryReviewSnapshot:
    pending_confirmation: tuple[CanonicalMemoryReviewItem, ...]
    undoable: tuple[CanonicalMemoryReviewItem, ...]
    explicit_undoable: tuple[ExplicitMemoryReviewItem, ...] = ()


class MemoryReviewService:
    def __init__(self, repository: AppRepository) -> None:
        self._repository = repository
        self._decisions = MemoryDecisionService(repository)

    async def list_items(
        self, conversation_id: str, branch_id: str
    ) -> MemoryReviewSnapshot:
        conversation = await self._repository.get_conversation(conversation_id)
        character = await self._repository.get_character_version(
            conversation.character_version_id
        )
        items = await self._repository.list_canonical_memory_review_items(
            conversation_id, branch_id
        )
        explicit_items = await self._repository.project_explicit_memory(
            conversation_id, branch_id, character.character_id
        )
        safety_keys = {
            (item.source_message_id, item.value.strip().casefold())
            for item in items
            if item.kind is MemoryKind.SAFETY_CONSTRAINT and item.is_active
        }
        visible_explicit_items = tuple(
            item
            for item in explicit_items
            if (item.source_message_id, item.value.strip().casefold())
            not in safety_keys
        )
        explicit_keys = {
            (item.source_message_id, item.value.strip().casefold())
            for item in visible_explicit_items
        }
        return MemoryReviewSnapshot(
            pending_confirmation=tuple(
                item
                for item in items
                if item.approval is MemoryApprovalState.PENDING_CONFIRMATION
                and item.is_active
                and (
                    item.source_message_id,
                    item.value.strip().casefold(),
                )
                not in explicit_keys
            ),
            undoable=tuple(
                item
                for item in items
                if item.approval
                in {
                    MemoryApprovalState.AUTO_SAVED,
                    MemoryApprovalState.CONFIRMED,
                }
                and item.is_active
                and (
                    item.source_message_id,
                    item.value.strip().casefold(),
                )
                not in explicit_keys
            ),
            explicit_undoable=visible_explicit_items,
        )

    async def decide(
        self, request: MemoryDecisionRequest
    ) -> MemoryApprovalDecision:
        return await self._decisions.decide(request)
