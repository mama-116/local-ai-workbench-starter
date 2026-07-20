from __future__ import annotations

from dataclasses import dataclass

from local_llm_chat.application.services.memory_decision_service import (
    MemoryDecisionRequest,
    MemoryDecisionService,
)
from local_llm_chat.domain.canonical_memory import (
    CanonicalMemoryReviewItem,
    MemoryApprovalDecision,
)
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.states import MemoryApprovalState


@dataclass(frozen=True, slots=True)
class MemoryReviewSnapshot:
    pending_confirmation: tuple[CanonicalMemoryReviewItem, ...]
    undoable: tuple[CanonicalMemoryReviewItem, ...]


class MemoryReviewService:
    def __init__(self, repository: AppRepository) -> None:
        self._repository = repository
        self._decisions = MemoryDecisionService(repository)

    async def list_items(
        self, conversation_id: str, branch_id: str
    ) -> MemoryReviewSnapshot:
        items = await self._repository.list_canonical_memory_review_items(
            conversation_id, branch_id
        )
        return MemoryReviewSnapshot(
            pending_confirmation=tuple(
                item
                for item in items
                if item.approval is MemoryApprovalState.PENDING_CONFIRMATION
                and item.is_active
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
            ),
        )

    async def decide(
        self, request: MemoryDecisionRequest
    ) -> MemoryApprovalDecision:
        return await self._decisions.decide(request)
