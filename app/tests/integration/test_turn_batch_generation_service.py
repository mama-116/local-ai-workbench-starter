from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from local_llm_chat.application.services.conversation_cast_service import (
    ConversationCastService,
)
from local_llm_chat.application.services.conversation_group_settings_service import (
    ConversationGroupSettingsService,
)
from local_llm_chat.application.services.turn_batch_generation_service import (
    TurnBatchGenerationService,
)
from local_llm_chat.domain.group_turns import (
    TurnBatchDraft,
    TurnBatchGenerationRequest,
    TurnSegmentDraft,
)
from local_llm_chat.domain.canonical_memory import CanonicalMemoryEvent
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.states import (
    MemoryApprovalState,
    MemoryCardinality,
    MemoryKind,
    TurnBatchState,
    TurnMode,
    TurnRepairState,
    TurnSpeakerKind,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


async def no_update(_: str) -> None:
    return None


class RecordingGenerator:
    def __init__(self) -> None:
        self.request: TurnBatchGenerationRequest | None = None

    async def generate(
        self,
        request: TurnBatchGenerationRequest,
        on_update: Callable[[str], Awaitable[None]] = no_update,
    ) -> TurnBatchDraft:
        del on_update
        self.request = request
        first = request.formal_characters[0]
        return TurnBatchDraft(
            mode=request.mode,
            formal_character_ids=tuple(
                character.character_id for character in request.formal_characters
            ),
            guest_ids=(),
            prompt_version=request.prompt_version,
            state=TurnBatchState.COMPLETED,
            repair_state=TurnRepairState.NOT_NEEDED,
            segments=(
                TurnSegmentDraft(
                    TurnSpeakerKind.CHARACTER,
                    first.character_id,
                    first.display_name,
                    "response",
                ),
            ),
            spotlight_character_id=request.spotlight_character_id,
        )


class ForgedCastGenerator(RecordingGenerator):
    async def generate(
        self,
        request: TurnBatchGenerationRequest,
        on_update: Callable[[str], Awaitable[None]] = no_update,
    ) -> TurnBatchDraft:
        draft = await super().generate(request, on_update)
        return replace(draft, formal_character_ids=("forged-character",))


@pytest.mark.asyncio
async def test_service_builds_one_generation_request_from_registered_cast(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    first = await repository.ensure_default_character()
    second = await repository.create_character_version("Second", "second prompt")
    profile = await repository.ensure_model_profile(
        "ollama-local", "group-model", {"num_ctx": 8192, "temperature": 0.6}
    )
    conversation = await repository.create_conversation(
        "group", first.id, profile.id
    )
    await ConversationCastService(repository).set_cast(
        conversation.id, (second.id, first.id)
    )
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.STORY
    )
    session = await repository.start_send(conversation.id, "What happens next?")
    generator = RecordingGenerator()

    draft = await TurnBatchGenerationService(repository, generator).generate(session)

    assert draft.formal_character_ids == (second.character_id, first.character_id)
    assert generator.request is not None
    assert generator.request.provider_name == "ollama-local"
    assert generator.request.model_name == "group-model"
    assert generator.request.options == {"num_ctx": 8192, "temperature": 0.6}
    assert tuple(
        character.character_version_id
        for character in generator.request.formal_characters
    ) == (second.id, first.id)
    assert tuple(
        character.system_prompt for character in generator.request.formal_characters
    ) == ("second prompt", first.system_prompt)
    assert generator.request.messages[-1].content == "What happens next?"
    assert generator.request.mode is TurnMode.STORY


@pytest.mark.asyncio
async def test_service_measures_group_generation_response_duration(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "group-model", {}
    )
    conversation = await repository.create_conversation(
        "group", character.id, profile.id
    )
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.STORY
    )
    session = await repository.start_send(conversation.id, "Continue")
    clock_values = iter((10.0, 11.25))

    draft = await TurnBatchGenerationService(
        repository,
        RecordingGenerator(),
        clock=lambda: next(clock_values),
    ).generate(session)

    assert draft.response_duration_ms == 1_250


@pytest.mark.asyncio
async def test_service_injects_only_memory_known_by_the_entire_cast(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    first = await repository.create_character_version("Alice", "alice prompt")
    second = await repository.create_character_version("Bob", "bob prompt")
    profile = await repository.ensure_model_profile("ollama-local", "group-model", {})
    conversation = await repository.create_conversation(
        "group", first.id, profile.id
    )
    await ConversationCastService(repository).set_cast(
        conversation.id, (first.id, second.id)
    )
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.STORY
    )
    session = await repository.start_send(conversation.id, "I cannot eat berries.")

    def event(
        identifier: str,
        kind: MemoryKind,
        value: str,
        known_by: frozenset[str],
        approval: MemoryApprovalState = MemoryApprovalState.AUTO_SAVED,
    ) -> CanonicalMemoryEvent:
        return CanonicalMemoryEvent(
            id=identifier,
            conversation_id=conversation.id,
            branch_id=session.branch_id,
            subject_id="user",
            kind=kind,
            slot="food_allergy" if kind is MemoryKind.SAFETY_CONSTRAINT else "fact",
            value=value,
            cardinality=MemoryCardinality.MULTIPLE,
            approval=approval,
            source_message_id=session.user_message.id,
            known_by_character_ids=known_by,
            supersedes_event_id=None,
            effective_at=session.user_message.created_at,
            recorded_at=session.user_message.created_at + timedelta(microseconds=1),
        )

    await repository.append_canonical_memory_events(
        (
            event("shared-preference", MemoryKind.PREFERENCE, "chocolate", frozenset()),
            event(
                "shared-current-safety",
                MemoryKind.SAFETY_CONSTRAINT,
                "berries",
                frozenset(),
                MemoryApprovalState.PENDING_CONFIRMATION,
            ),
            event(
                "alice-secret",
                MemoryKind.GOAL,
                "do not tell Bob",
                frozenset({first.character_id}),
            ),
        )
    )
    generator = RecordingGenerator()

    await TurnBatchGenerationService(repository, generator).generate(session)

    assert generator.request is not None
    assert [fact.event_id for fact in generator.request.shared_memory_facts] == [
        "shared-current-safety",
        "shared-preference",
    ]
    assert all(
        fact.event_id != "alice-secret"
        for fact in generator.request.shared_memory_facts
    )
    assert generator.request.prompt_version == "group-turn-v2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "must_reject"),
    (
        (MemoryKind.PREFERENCE, False),
        (MemoryKind.SAFETY_CONSTRAINT, True),
    ),
)
async def test_service_bounds_shared_memory_without_silently_dropping_safety(
    tmp_path: Path, kind: MemoryKind, must_reject: bool
) -> None:
    repository = SQLiteAppRepository(tmp_path / f"{kind.value}.sqlite3")
    await repository.initialize()
    character = await repository.create_character_version("Alice", "alice prompt")
    profile = await repository.ensure_model_profile("ollama-local", "group-model", {})
    conversation = await repository.create_conversation(
        "group", character.id, profile.id
    )
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.STORY
    )
    session = await repository.start_send(conversation.id, "Remember these facts.")
    await repository.append_canonical_memory_events(
        tuple(
            CanonicalMemoryEvent(
                id=f"fact-{position:02d}",
                conversation_id=conversation.id,
                branch_id=session.branch_id,
                subject_id="user",
                kind=kind,
                slot="bounded_fact",
                value=f"value-{position:02d}",
                cardinality=MemoryCardinality.MULTIPLE,
                approval=MemoryApprovalState.AUTO_SAVED,
                source_message_id=session.user_message.id,
                known_by_character_ids=frozenset(),
                supersedes_event_id=None,
                effective_at=session.user_message.created_at,
                recorded_at=session.user_message.created_at
                + timedelta(microseconds=position),
            )
            for position in range(33)
        )
    )
    generator = RecordingGenerator()
    service = TurnBatchGenerationService(repository, generator)

    if must_reject:
        with pytest.raises(ValidationError, match="shared safety memory"):
            await service.generate(session)
        assert generator.request is None
    else:
        await service.generate(session)
        assert generator.request is not None
        assert len(generator.request.shared_memory_facts) == 32


@pytest.mark.asyncio
async def test_service_rejects_generator_output_for_another_cast(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "group-model", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "group", character.id, profile.id
    )
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.STORY
    )
    session = await repository.start_send(conversation.id, "Continue")

    with pytest.raises(ValidationError):
        await TurnBatchGenerationService(
            repository, ForgedCastGenerator()
        ).generate(session)


@pytest.mark.asyncio
async def test_service_refuses_group_generation_while_disabled(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "group-model", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "single", character.id, profile.id
    )
    session = await repository.start_send(conversation.id, "Continue")
    generator = RecordingGenerator()

    with pytest.raises(ValidationError):
        await TurnBatchGenerationService(repository, generator).generate(session)

    assert generator.request is None
