from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from local_llm_chat.domain.canonical_memory import CanonicalMemoryFact
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.group_turns import (
    MAX_SHARED_MEMORY_CONTEXT_CHARACTERS,
    MAX_SHARED_MEMORY_FACTS,
    TurnBatchCharacter,
    TurnBatchDraft,
    TurnBatchGenerationRequest,
    TurnBatchMemoryFact,
    validate_turn_batch_draft,
    validate_turn_batch_generation_request,
)
from local_llm_chat.domain.models import ChatMessageInput, RunSession
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.ports.turn_batch_generator import (
    TurnBatchGenerator,
    TurnBatchStreamCallback,
    no_turn_batch_stream_update,
)
from local_llm_chat.domain.states import TurnMode
from local_llm_chat.domain.states import MemoryKind


TURN_BATCH_PROMPT_VERSION = "group-turn-v3"


class TurnBatchGenerationService:
    def __init__(
        self,
        repository: AppRepository,
        generator: TurnBatchGenerator,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._repository = repository
        self._generator = generator
        self._clock = clock

    async def generate(
        self,
        session: RunSession,
        on_update: TurnBatchStreamCallback = no_turn_batch_stream_update,
    ) -> TurnBatchDraft:
        conversation = await self._repository.get_conversation(
            session.run.conversation_id
        )
        if session.user_message.conversation_id != conversation.id:
            raise ValidationError("turn batch source message belongs to another conversation")
        settings = await self._repository.get_conversation_group_settings(
            conversation.id
        )
        if not settings.enabled:
            raise ValidationError("group generation is disabled for this conversation")
        cast = await self._repository.get_conversation_cast(conversation.id)
        characters: list[TurnBatchCharacter] = []
        for member in cast.members:
            version = await self._repository.get_character_version(
                member.character_version_id
            )
            if version.character_id != member.character_id:
                raise ValidationError("registered cast identity is inconsistent")
            characters.append(
                TurnBatchCharacter(
                    version.character_id,
                    version.id,
                    member.display_name,
                    version.system_prompt,
                )
            )
        profile = await self._repository.get_model_profile(
            conversation.model_profile_id
        )
        context = await self._repository.context_to_message(session.user_message.id)
        shared_memory = await self._shared_memory_facts(
            conversation.id,
            session.branch_id,
            session.user_message.id,
            tuple(character.character_id for character in characters),
        )
        request = TurnBatchGenerationRequest(
            provider_name=profile.provider,
            model_name=profile.model_name,
            mode=settings.mode,
            formal_characters=tuple(characters),
            messages=tuple(
                ChatMessageInput(message.role, message.content)
                for message in context
                if message.content
            ),
            options=dict(profile.parameters),
            prompt_version=TURN_BATCH_PROMPT_VERSION,
            spotlight_character_id=settings.spotlight_character_id,
            shared_memory_facts=shared_memory,
        )
        validate_turn_batch_generation_request(request)
        response_started = self._clock()
        draft = await self._generator.generate(request, on_update)
        draft = replace(
            draft,
            response_duration_ms=max(
                0, round((self._clock() - response_started) * 1000)
            ),
        )
        validate_turn_batch_draft(draft)
        expected_character_ids = tuple(
            character.character_id for character in request.formal_characters
        )
        if (
            draft.mode is not settings.mode
            or draft.formal_character_ids != expected_character_ids
            or draft.guest_ids
            or draft.prompt_version != request.prompt_version
            or draft.spotlight_character_id != settings.spotlight_character_id
        ):
            raise ValidationError("turn batch generator violated its request contract")
        return draft

    async def _shared_memory_facts(
        self,
        conversation_id: str,
        branch_id: str,
        current_source_message_id: str,
        character_ids: tuple[str, ...],
    ) -> tuple[TurnBatchMemoryFact, ...]:
        projected: list[tuple[CanonicalMemoryFact, ...]] = []
        for character_id in character_ids:
            projected.append(
                await self._repository.project_canonical_memory(
                    conversation_id,
                    branch_id,
                    character_id,
                    current_source_message_id=current_source_message_id,
                )
            )
        projections = tuple(projected)
        common_ids = set(fact.event_id for fact in projections[0])
        for projection in projections[1:]:
            common_ids.intersection_update(fact.event_id for fact in projection)

        selected: list[TurnBatchMemoryFact] = []
        selected_characters = 0
        for fact in projections[0]:
            if fact.event_id not in common_ids:
                continue
            item = TurnBatchMemoryFact(
                event_id=fact.event_id,
                subject_id=fact.subject_id,
                kind=fact.kind,
                slot=fact.slot,
                value=fact.value,
                source_message_id=fact.source_message_id,
            )
            item_characters = self._memory_fact_characters(item)
            exceeds_limit = (
                len(selected) >= MAX_SHARED_MEMORY_FACTS
                or selected_characters + item_characters
                > MAX_SHARED_MEMORY_CONTEXT_CHARACTERS
            )
            if exceeds_limit:
                if item.kind is MemoryKind.SAFETY_CONSTRAINT:
                    raise ValidationError(
                        "shared safety memory exceeds the group generation limit"
                    )
                continue
            selected.append(item)
            selected_characters += item_characters
        return tuple(selected)

    @staticmethod
    def _memory_fact_characters(fact: TurnBatchMemoryFact) -> int:
        return sum(
            len(value)
            for value in (
                fact.event_id,
                fact.subject_id,
                fact.kind.value,
                fact.slot,
                fact.value,
                fact.source_message_id,
            )
        )
