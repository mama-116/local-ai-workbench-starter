from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid5

from local_llm_chat.domain.canonical_memory import MemoryApprovalDecision
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.states import MemoryApprovalState


_MEMORY_DECISION_NAMESPACE = UUID("e0c45c4d-d468-47f4-a7fe-0ae71eb95fe9")


class MemoryDecisionAction(str, Enum):
    CONFIRM = "confirm"
    REJECT = "reject"
    UNDO = "undo"


_ACTION_STATES = {
    MemoryDecisionAction.CONFIRM: MemoryApprovalState.CONFIRMED,
    MemoryDecisionAction.REJECT: MemoryApprovalState.REJECTED,
    MemoryDecisionAction.UNDO: MemoryApprovalState.UNDONE,
}


@dataclass(frozen=True, slots=True)
class MemoryDecisionRequest:
    conversation_id: str
    branch_id: str
    target_event_id: str
    action: MemoryDecisionAction


class MemoryDecisionService:
    def __init__(
        self,
        repository: AppRepository,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock or (lambda: datetime.now(UTC))

    async def decide(
        self, request: MemoryDecisionRequest
    ) -> MemoryApprovalDecision:
        if any(
            not value.strip()
            for value in (
                request.conversation_id,
                request.branch_id,
                request.target_event_id,
            )
        ):
            raise ValidationError("memory decision identifiers must not be blank")
        state = _ACTION_STATES[request.action]
        identity = "\x1f".join(
            (request.conversation_id, request.target_event_id, state.value)
        )
        decision_id = str(uuid5(_MEMORY_DECISION_NAMESPACE, identity))
        return await self._repository.decide_canonical_memory(
            request.conversation_id,
            request.branch_id,
            request.target_event_id,
            state,
            decision_id,
            self._clock(),
        )
