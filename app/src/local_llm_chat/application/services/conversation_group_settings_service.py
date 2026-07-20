from __future__ import annotations

from local_llm_chat.domain.group_turns import (
    ConversationGroupSettings,
    validate_conversation_group_settings,
)
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.states import TurnMode


class ConversationGroupSettingsService:
    def __init__(self, repository: AppRepository) -> None:
        self._repository = repository

    async def get(self, conversation_id: str) -> ConversationGroupSettings:
        return await self._repository.get_conversation_group_settings(
            conversation_id
        )

    async def configure(
        self,
        conversation_id: str,
        *,
        enabled: bool,
        mode: TurnMode,
        spotlight_character_id: str | None = None,
    ) -> ConversationGroupSettings:
        validate_conversation_group_settings(
            enabled, mode, spotlight_character_id
        )
        return await self._repository.set_conversation_group_settings(
            conversation_id,
            enabled,
            mode,
            spotlight_character_id,
        )
