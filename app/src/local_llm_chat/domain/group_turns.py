from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import ChatMessageInput
from local_llm_chat.domain.states import (
    MemoryKind,
    TurnBatchState,
    TurnMode,
    TurnRepairState,
    TurnSpeakerKind,
)


MAX_FORMAL_CHARACTERS = 5
MAX_SCENE_GUESTS = 5
MAX_TURN_SEGMENTS = 32
MAX_TURN_SEGMENT_CHARACTERS = 8_000
MAX_SPEAKER_DISPLAY_NAME_CHARACTERS = 100
MAX_TURN_GENERATION_CONTEXT_CHARACTERS = 200_000
MAX_SHARED_MEMORY_FACTS = 32
MAX_SHARED_MEMORY_CONTEXT_CHARACTERS = 16_000


@dataclass(frozen=True, slots=True)
class FormalCastMember:
    character_id: str
    character_version_id: str
    display_name: str
    position: int


@dataclass(frozen=True, slots=True)
class ConversationCast:
    conversation_id: str
    members: tuple[FormalCastMember, ...]


@dataclass(frozen=True, slots=True)
class ConversationGroupSettings:
    conversation_id: str
    enabled: bool
    mode: TurnMode
    spotlight_character_id: str | None
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationGroupConfiguration:
    cast: ConversationCast
    settings: ConversationGroupSettings


@dataclass(frozen=True, slots=True)
class TurnBatchCharacter:
    character_id: str
    character_version_id: str
    display_name: str
    system_prompt: str


@dataclass(frozen=True, slots=True)
class TurnBatchMemoryFact:
    event_id: str
    subject_id: str
    kind: MemoryKind
    slot: str
    value: str
    source_message_id: str


@dataclass(frozen=True, slots=True)
class TurnBatchGenerationRequest:
    provider_name: str
    model_name: str
    mode: TurnMode
    formal_characters: tuple[TurnBatchCharacter, ...]
    messages: tuple[ChatMessageInput, ...]
    options: dict[str, Any]
    prompt_version: str
    spotlight_character_id: str | None = None
    shared_memory_facts: tuple[TurnBatchMemoryFact, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnSegmentDraft:
    speaker_kind: TurnSpeakerKind
    speaker_id: str | None
    display_name: str
    content: str


@dataclass(frozen=True, slots=True)
class TurnBatchDraft:
    mode: TurnMode
    formal_character_ids: tuple[str, ...]
    guest_ids: tuple[str, ...]
    prompt_version: str
    state: TurnBatchState
    repair_state: TurnRepairState
    segments: tuple[TurnSegmentDraft, ...]
    error_code: str | None = None
    spotlight_character_id: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_duration_ns: int | None = None
    generation_duration_ns: int | None = None
    response_duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class TurnSegment:
    id: str
    turn_batch_id: str
    position: int
    speaker_kind: TurnSpeakerKind
    speaker_id: str | None
    display_name: str
    content: str


@dataclass(frozen=True, slots=True)
class TurnBatch:
    id: str
    conversation_id: str
    branch_id: str
    source_message_id: str
    response_message_id: str
    run_id: str
    model: str
    mode: TurnMode
    formal_character_ids: tuple[str, ...]
    guest_ids: tuple[str, ...]
    prompt_version: str
    state: TurnBatchState
    repair_state: TurnRepairState
    error_code: str | None
    spotlight_character_id: str | None
    segments: tuple[TurnSegment, ...]
    created_at: datetime


def validate_turn_batch_draft(draft: TurnBatchDraft) -> None:
    _validate_ids(
        draft.formal_character_ids,
        minimum=1,
        maximum=MAX_FORMAL_CHARACTERS,
        label="formal characters",
    )
    _validate_ids(
        draft.guest_ids,
        minimum=0,
        maximum=MAX_SCENE_GUESTS,
        label="scene guests",
    )
    if set(draft.formal_character_ids).intersection(draft.guest_ids):
        raise ValidationError("formal characters and scene guests must be distinct")
    validate_group_mode(draft.mode, draft.spotlight_character_id)
    if (
        draft.spotlight_character_id is not None
        and draft.spotlight_character_id not in draft.formal_character_ids
    ):
        raise ValidationError("turn batch spotlight character is outside the formal cast")
    if not draft.prompt_version.strip():
        raise ValidationError("turn batch prompt version must not be blank")
    if not draft.segments or len(draft.segments) > MAX_TURN_SEGMENTS:
        raise ValidationError("turn batch must contain 1 to 32 segments")
    if draft.state is TurnBatchState.COMPLETED and draft.error_code is not None:
        raise ValidationError("completed turn batch must not have an error")
    if draft.state is TurnBatchState.PARTIAL and not (
        draft.error_code and draft.error_code.strip()
    ):
        raise ValidationError("partial turn batch must have an error code")
    if (
        draft.repair_state is TurnRepairState.FAILED
        and draft.state is not TurnBatchState.PARTIAL
    ):
        raise ValidationError("failed repair requires a partial turn batch")
    if (
        draft.repair_state is TurnRepairState.SUCCEEDED
        and draft.state is not TurnBatchState.COMPLETED
    ):
        raise ValidationError("successful repair requires a completed turn batch")
    for value in (
        draft.prompt_tokens,
        draft.output_tokens,
        draft.total_duration_ns,
        draft.generation_duration_ns,
        draft.response_duration_ms,
    ):
        if value is not None and value < 0:
            raise ValidationError("turn batch performance metrics must not be negative")

    formal_ids = set(draft.formal_character_ids)
    guest_ids = set(draft.guest_ids)
    for segment in draft.segments:
        display_name = segment.display_name.strip()
        content = segment.content.strip()
        if not display_name or len(display_name) > MAX_SPEAKER_DISPLAY_NAME_CHARACTERS:
            raise ValidationError("turn segment display name is invalid")
        if not content or len(content) > MAX_TURN_SEGMENT_CHARACTERS:
            raise ValidationError("turn segment content is invalid")
        if segment.speaker_kind is TurnSpeakerKind.CHARACTER:
            if segment.speaker_id not in formal_ids:
                raise ValidationError("turn segment character is outside the formal cast")
        elif segment.speaker_kind is TurnSpeakerKind.GUEST:
            if segment.speaker_id not in guest_ids:
                raise ValidationError("turn segment guest is outside the scene guests")
        elif segment.speaker_id is not None:
            raise ValidationError("narrator and unresolved speakers must not have an id")

    if draft.mode is TurnMode.ROUND_TABLE:
        _validate_round_table_draft(draft)


def _validate_round_table_draft(draft: TurnBatchDraft) -> None:
    if any(
        segment.speaker_kind
        not in (TurnSpeakerKind.CHARACTER, TurnSpeakerKind.NARRATOR)
        for segment in draft.segments
    ):
        raise ValidationError(
            "round table permits only character and narrator segments"
        )

    character_ids = tuple(
        segment.speaker_id
        for segment in draft.segments
        if segment.speaker_kind is TurnSpeakerKind.CHARACTER
        and segment.speaker_id is not None
    )
    if draft.state is TurnBatchState.COMPLETED:
        if character_ids != draft.formal_character_ids:
            raise ValidationError(
                "completed round table must match the formal cast order exactly"
            )
        return

    expected_prefix = draft.formal_character_ids[: len(character_ids)]
    if not character_ids or character_ids != expected_prefix:
        raise ValidationError(
            "partial round table must match a non-empty prefix of the formal cast"
        )


def validate_turn_batch_generation_request(
    request: TurnBatchGenerationRequest,
) -> None:
    if not request.provider_name.strip() or not request.model_name.strip():
        raise ValidationError("turn batch provider and model must not be blank")
    if not request.prompt_version.strip():
        raise ValidationError("turn batch prompt version must not be blank")
    if not 1 <= len(request.formal_characters) <= MAX_FORMAL_CHARACTERS:
        raise ValidationError("turn batch generation requires 1 to 5 characters")
    character_ids = tuple(
        character.character_id for character in request.formal_characters
    )
    version_ids = tuple(
        character.character_version_id for character in request.formal_characters
    )
    if len(character_ids) != len(set(character_ids)) or len(version_ids) != len(
        set(version_ids)
    ):
        raise ValidationError("turn batch generation characters must be unique")
    validate_group_mode(request.mode, request.spotlight_character_id)
    if (
        request.spotlight_character_id is not None
        and request.spotlight_character_id not in character_ids
    ):
        raise ValidationError("spotlight character must belong to the formal cast")
    for character in request.formal_characters:
        if any(
            not value.strip()
            for value in (
                character.character_id,
                character.character_version_id,
                character.display_name,
                character.system_prompt,
            )
        ):
            raise ValidationError("turn batch generation character is incomplete")
        if len(character.display_name) > MAX_SPEAKER_DISPLAY_NAME_CHARACTERS:
            raise ValidationError("turn batch generation character name is too long")
    if len(request.shared_memory_facts) > MAX_SHARED_MEMORY_FACTS:
        raise ValidationError("turn batch shared memory contains too many facts")
    memory_ids: set[str] = set()
    non_safety_seen = False
    memory_characters = 0
    for fact in request.shared_memory_facts:
        values = (
            fact.event_id,
            fact.subject_id,
            fact.slot,
            fact.value,
            fact.source_message_id,
        )
        if any(not value.strip() for value in values):
            raise ValidationError("turn batch shared memory fact is incomplete")
        if fact.event_id in memory_ids:
            raise ValidationError("turn batch shared memory facts must be unique")
        memory_ids.add(fact.event_id)
        if fact.kind is MemoryKind.SAFETY_CONSTRAINT:
            if non_safety_seen:
                raise ValidationError("turn batch safety memory must come first")
        else:
            non_safety_seen = True
        memory_characters += sum(len(value) for value in values) + len(fact.kind.value)
    if memory_characters > MAX_SHARED_MEMORY_CONTEXT_CHARACTERS:
        raise ValidationError("turn batch shared memory context is too large")
    if not request.messages or not request.messages[-1].content.strip():
        raise ValidationError("turn batch generation context must not be empty")
    total_input_characters = sum(
        len(message.content) for message in request.messages
    ) + sum(
        len(character.display_name) + len(character.system_prompt)
        for character in request.formal_characters
    ) + memory_characters
    if total_input_characters > MAX_TURN_GENERATION_CONTEXT_CHARACTERS:
        raise ValidationError("turn batch generation context is too large")


def validate_conversation_group_settings(
    enabled: bool,
    mode: TurnMode,
    spotlight_character_id: str | None,
) -> None:
    if not enabled:
        if mode is not TurnMode.STORY or spotlight_character_id is not None:
            raise ValidationError("disabled group conversation must use story defaults")
        return
    validate_group_mode(mode, spotlight_character_id)


def validate_group_mode(
    mode: TurnMode, spotlight_character_id: str | None
) -> None:
    if mode is TurnMode.SPOTLIGHT:
        if spotlight_character_id is None or not spotlight_character_id.strip():
            raise ValidationError("spotlight mode requires a spotlight character")
    elif spotlight_character_id is not None:
        raise ValidationError("only spotlight mode may have a spotlight character")


def _validate_ids(
    values: tuple[str, ...], *, minimum: int, maximum: int, label: str
) -> None:
    if not minimum <= len(values) <= maximum:
        raise ValidationError(f"{label} count is outside its allowed range")
    if len(values) != len(set(values)) or any(
        not value.strip() or value != value.strip() for value in values
    ):
        raise ValidationError(f"{label} must contain unique non-blank ids")
