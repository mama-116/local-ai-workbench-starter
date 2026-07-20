from __future__ import annotations

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.group_turns import (
    MAX_FORMAL_CHARACTERS,
    ConversationCast,
)
from local_llm_chat.domain.ports.repositories import AppRepository


class ConversationCastService:
    def __init__(self, repository: AppRepository) -> None:
        self._repository = repository

    async def get(self, conversation_id: str) -> ConversationCast:
        return await self._repository.get_conversation_cast(conversation_id)

    async def set_cast(
        self, conversation_id: str, character_version_ids: tuple[str, ...]
    ) -> ConversationCast:
        if not 1 <= len(character_version_ids) <= MAX_FORMAL_CHARACTERS:
            raise ValidationError("formal cast must contain 1 to 5 characters")
        if len(character_version_ids) != len(set(character_version_ids)) or any(
            not version_id.strip() or version_id != version_id.strip()
            for version_id in character_version_ids
        ):
            raise ValidationError(
                "formal cast must contain unique non-blank character version ids"
            )
        return await self._repository.set_conversation_cast(
            conversation_id, character_version_ids
        )
