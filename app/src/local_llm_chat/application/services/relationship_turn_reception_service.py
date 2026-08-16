from __future__ import annotations

from dataclasses import dataclass

from local_llm_chat.application.services.relationship_profile_service import (
    RelationshipProfileService,
)
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.ports.relationship_candidate_extractor import (
    RelationshipCandidateExtractor,
)
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.relationship_profile import (
    RelationshipCandidateRequest,
    RelationshipEvent,
    RelationshipMeaning,
)
from local_llm_chat.domain.states import MessageRole, MessageState


@dataclass(frozen=True, slots=True)
class RelationshipTurnReception:
    event_ids: tuple[str, ...]
    meanings: tuple[RelationshipMeaning, ...]


class RelationshipTurnReceptionService:
    def __init__(
        self,
        repository: AppRepository,
        profiles: RelationshipProfileService,
        extractor: RelationshipCandidateExtractor,
    ) -> None:
        self._repository = repository
        self._profiles = profiles
        self._extractor = extractor

    async def capture(
        self,
        *,
        conversation_id: str,
        branch_id: str,
        source_message_id: str,
        character_ids: tuple[str, ...],
        character_names: dict[str, str] | None = None,
    ) -> RelationshipTurnReception:
        source = await self._repository.get_message(source_message_id)
        if (
            source.conversation_id != conversation_id
            or source.role is not MessageRole.USER
            or source.state is not MessageState.COMPLETED
        ):
            raise ValidationError("関係同期の出典発言が不正です。")
        event_ids = await self._profiles.capture_relationship_candidates(
            request=RelationshipCandidateRequest(
                conversation_id=conversation_id,
                branch_id=branch_id,
                source_message_id=source_message_id,
                model_name="dedicated-loopback",
                content=source.content,
                allowed_character_ids=frozenset(character_ids),
                allowed_characters=tuple(
                    (character_id, (character_names or {}).get(character_id, ""))
                    for character_id in character_ids
                ),
            ),
            extractor=self._extractor,
            recorded_at=source.created_at,
        )
        continuity = await self._repository.get_continuity_for_conversation(
            conversation_id
        )
        by_id: dict[str, RelationshipEvent] = {}
        for character_id in character_ids:
            events = await self._repository.list_relationship_events(
                continuity.id,
                continuity.user_profile_id,
                character_id,
                include_unapplied=True,
            )
            by_id.update((event.id, event) for event in events)
        return RelationshipTurnReception(
            event_ids=event_ids,
            meanings=tuple(
                by_id[event_id].meaning
                for event_id in event_ids
                if event_id in by_id
            ),
        )
