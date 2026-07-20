from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.group_turns import (
    TurnBatchDraft,
    TurnSegmentDraft,
    validate_turn_batch_draft,
)
from local_llm_chat.domain.states import (
    TurnBatchState,
    TurnMode,
    TurnRepairState,
    TurnSpeakerKind,
)


FORMAL_CHARACTER_IDS = ("character-a", "character-b", "character-c")


def character(character_id: str) -> TurnSegmentDraft:
    return TurnSegmentDraft(
        speaker_kind=TurnSpeakerKind.CHARACTER,
        speaker_id=character_id,
        display_name=character_id,
        content=f"{character_id}の発言",
    )


def narrator() -> TurnSegmentDraft:
    return TurnSegmentDraft(
        speaker_kind=TurnSpeakerKind.NARRATOR,
        speaker_id=None,
        display_name="ナレーター",
        content="場面が移る。",
    )


def round_table_draft() -> TurnBatchDraft:
    return TurnBatchDraft(
        mode=TurnMode.ROUND_TABLE,
        formal_character_ids=FORMAL_CHARACTER_IDS,
        guest_ids=(),
        prompt_version="group-turn-v2",
        state=TurnBatchState.COMPLETED,
        repair_state=TurnRepairState.NOT_NEEDED,
        segments=tuple(character(value) for value in FORMAL_CHARACTER_IDS),
    )


def test_completed_round_table_accepts_exact_cast_order_with_narration() -> None:
    draft = replace(
        round_table_draft(),
        segments=(
            character("character-a"),
            narrator(),
            character("character-b"),
            character("character-c"),
        ),
    )

    validate_turn_batch_draft(draft)


@pytest.mark.parametrize(
    "segments",
    [
        (character("character-a"), character("character-b")),
        (
            character("character-a"),
            character("character-b"),
            character("character-b"),
            character("character-c"),
        ),
        tuple(character(value) for value in reversed(FORMAL_CHARACTER_IDS)),
    ],
    ids=("missing", "duplicate", "wrong-order"),
)
def test_completed_round_table_rejects_character_sequence_that_is_not_exact_cast(
    segments: tuple[TurnSegmentDraft, ...],
) -> None:
    with pytest.raises(
        ValidationError,
        match="completed round table must match the formal cast order exactly",
    ):
        validate_turn_batch_draft(replace(round_table_draft(), segments=segments))


@pytest.mark.parametrize(
    "speaker_kind,speaker_id,guest_ids",
    [
        (TurnSpeakerKind.UNRESOLVED, None, ()),
        (TurnSpeakerKind.GUEST, "guest-a", ("guest-a",)),
    ],
    ids=("unresolved", "guest"),
)
def test_completed_round_table_rejects_non_cast_speakers(
    speaker_kind: TurnSpeakerKind,
    speaker_id: str | None,
    guest_ids: tuple[str, ...],
) -> None:
    extra = TurnSegmentDraft(
        speaker_kind=speaker_kind,
        speaker_id=speaker_id,
        display_name="追加話者",
        content="割り込み発言",
    )
    draft = replace(
        round_table_draft(),
        guest_ids=guest_ids,
        segments=(*round_table_draft().segments, extra),
    )

    with pytest.raises(
        ValidationError,
        match="round table permits only character and narrator segments",
    ):
        validate_turn_batch_draft(draft)


def test_partial_round_table_accepts_non_empty_cast_prefix_with_narration() -> None:
    draft = replace(
        round_table_draft(),
        state=TurnBatchState.PARTIAL,
        repair_state=TurnRepairState.FAILED,
        error_code="stream_interrupted",
        segments=(character("character-a"), narrator(), character("character-b")),
    )

    validate_turn_batch_draft(draft)


@pytest.mark.parametrize(
    "segments",
    [
        (narrator(),),
        (character("character-a"), character("character-c")),
        (character("character-b"),),
    ],
    ids=("no-character", "skipped-character", "wrong-first-character"),
)
def test_partial_round_table_rejects_sequence_that_is_not_non_empty_cast_prefix(
    segments: tuple[TurnSegmentDraft, ...],
) -> None:
    draft = replace(
        round_table_draft(),
        state=TurnBatchState.PARTIAL,
        repair_state=TurnRepairState.FAILED,
        error_code="stream_interrupted",
        segments=segments,
    )

    with pytest.raises(
        ValidationError,
        match="partial round table must match a non-empty prefix of the formal cast",
    ):
        validate_turn_batch_draft(draft)


NEGATIVE_METRIC_MUTATIONS: tuple[
    Callable[[TurnBatchDraft], TurnBatchDraft], ...
] = (
    lambda draft: replace(draft, prompt_tokens=-1),
    lambda draft: replace(draft, output_tokens=-1),
    lambda draft: replace(draft, total_duration_ns=-1),
    lambda draft: replace(draft, generation_duration_ns=-1),
    lambda draft: replace(draft, response_duration_ms=-1),
)


@pytest.mark.parametrize("mutate", NEGATIVE_METRIC_MUTATIONS)
def test_turn_batch_rejects_negative_performance_metric(
    mutate: Callable[[TurnBatchDraft], TurnBatchDraft],
) -> None:
    with pytest.raises(
        ValidationError,
        match="turn batch performance metrics must not be negative",
    ):
        validate_turn_batch_draft(mutate(round_table_draft()))
