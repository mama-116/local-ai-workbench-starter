from __future__ import annotations

from local_llm_chat.domain.group_turns import (
    TurnBatch,
    TurnBatchDraft,
    validate_turn_batch_draft,
)
from local_llm_chat.domain.models import RunSession
from local_llm_chat.domain.ports.repositories import AppRepository


class TurnBatchService:
    def __init__(self, repository: AppRepository) -> None:
        self._repository = repository

    async def finish(
        self, session: RunSession, draft: TurnBatchDraft
    ) -> TurnBatch:
        validate_turn_batch_draft(draft)
        return await self._repository.finish_turn_batch(session, draft)
