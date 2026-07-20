from __future__ import annotations

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.group_turns import (
    MAX_FORMAL_CHARACTERS,
    ConversationGroupConfiguration,
    validate_conversation_group_settings,
)
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.states import TurnMode


class ConversationGroupConfigurationService:
    def __init__(self, repository: AppRepository) -> None:
        self._repository = repository

    async def get(self, conversation_id: str) -> ConversationGroupConfiguration:
        return await self._repository.get_conversation_group_configuration(
            conversation_id
        )

    async def save(
        self,
        conversation_id: str,
        *,
        character_version_ids: tuple[str, ...],
        enabled: bool,
        mode: TurnMode,
        spotlight_character_id: str | None = None,
    ) -> ConversationGroupConfiguration:
        if not 1 <= len(character_version_ids) <= MAX_FORMAL_CHARACTERS:
            raise ValidationError("formal cast must contain 1 to 5 characters")
        if len(character_version_ids) != len(set(character_version_ids)) or any(
            not version_id.strip() or version_id != version_id.strip()
            for version_id in character_version_ids
        ):
            raise ValidationError(
                "formal cast must contain unique non-blank character version ids"
            )
        if not enabled and len(character_version_ids) != 1:
            raise ValidationError(
                "disabled group conversation must keep exactly one character"
            )
        validate_conversation_group_settings(
            enabled, mode, spotlight_character_id
        )
        return await self._repository.set_conversation_group_configuration(
            conversation_id,
            character_version_ids,
            enabled,
            mode,
            spotlight_character_id,
        )
