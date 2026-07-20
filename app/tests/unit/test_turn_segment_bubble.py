from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, cast

import flet as ft

from local_llm_chat.domain.group_turns import TurnBatch, TurnSegment
from local_llm_chat.domain.models import Translation, utc_now
from local_llm_chat.domain.states import (
    TranslationState,
    TurnBatchState,
    TurnMode,
    TurnRepairState,
    TurnSpeakerKind,
)
from local_llm_chat.presentation.components.turn_segment_bubble import (
    TurnSegmentBubble,
)


def texts(control: object) -> Iterator[str]:
    if isinstance(control, str):
        yield control
        return
    if isinstance(control, ft.Text):
        yield control.value
    content = getattr(control, "content", None)
    if content is not None:
        yield from texts(content)
    controls = getattr(control, "controls", None)
    if controls is not None:
        for child in controls:
            yield from texts(child)


def replay_buttons(control: object) -> Iterator[ft.IconButton]:
    if isinstance(control, ft.IconButton) and control.icon == ft.Icons.REPLAY_ROUNDED:
        yield control
    content = getattr(control, "content", None)
    if content is not None:
        yield from replay_buttons(content)
    controls = getattr(control, "controls", None)
    if controls is not None:
        for child in controls:
            yield from replay_buttons(child)


def translation_buttons(control: object) -> Iterator[ft.IconButton]:
    if isinstance(control, ft.IconButton) and control.icon == ft.Icons.TRANSLATE_ROUNDED:
        yield control
    content = getattr(control, "content", None)
    if content is not None:
        yield from translation_buttons(content)
    controls = getattr(control, "controls", None)
    if controls is not None:
        for child in controls:
            yield from translation_buttons(child)


def batch(state: TurnBatchState = TurnBatchState.COMPLETED) -> TurnBatch:
    now = utc_now()
    return TurnBatch(
        id="batch-1",
        conversation_id="conversation-1",
        branch_id="branch-1",
        source_message_id="user-1",
        response_message_id="assistant-1",
        run_id="run-1",
        model="model",
        mode=TurnMode.STORY,
        formal_character_ids=("character-1",),
        guest_ids=(),
        prompt_version="group-turn-v1",
        state=state,
        repair_state=(
            TurnRepairState.FAILED
            if state is TurnBatchState.PARTIAL
            else TurnRepairState.NOT_NEEDED
        ),
        error_code=(
            "structured_output_truncated"
            if state is TurnBatchState.PARTIAL
            else None
        ),
        spotlight_character_id=None,
        segments=(),
        created_at=now,
    )


def segment(kind: TurnSpeakerKind, speaker_id: str | None = None) -> TurnSegment:
    return TurnSegment(
        id="segment-1",
        turn_batch_id="batch-1",
        position=0,
        speaker_kind=kind,
        speaker_id=speaker_id,
        display_name="先輩",
        content="俺はこう思う。",
    )


def translation(state: TranslationState = TranslationState.COMPLETED) -> Translation:
    now = utc_now()
    return Translation(
        id="translation-1",
        message_id="assistant-1",
        source_hash="hash",
        target_language="ja",
        provider="ollama-local",
        model="translation-model",
        content="全体の日本語訳です。" if state is TranslationState.COMPLETED else "",
        state=state,
        created_at=now,
        completed_at=now if state is TranslationState.COMPLETED else None,
    )


def test_character_segment_has_named_independent_bubble() -> None:
    copy = "\n".join(
        texts(
            TurnSegmentBubble(
                segment(TurnSpeakerKind.CHARACTER, "character-1"),
                batch(),
                utc_now(),
            )
        )
    )

    assert "先輩" in copy
    assert "俺はこう思う。" in copy
    assert "AI" not in copy


def test_unresolved_and_partial_state_are_explicit() -> None:
    copy = "\n".join(
        texts(
            TurnSegmentBubble(
                segment(TurnSpeakerKind.UNRESOLVED),
                batch(TurnBatchState.PARTIAL),
                utc_now(),
                is_batch_end=True,
            )
        )
    )

    assert "未解決話者" in copy
    assert "一部のみ" in copy


def test_regenerate_action_is_rendered_once_at_batch_end() -> None:
    calls = 0

    def regenerate() -> None:
        nonlocal calls
        calls += 1

    first = TurnSegmentBubble(
        segment(TurnSpeakerKind.CHARACTER, "character-1"),
        batch(),
        utc_now(),
        is_batch_end=False,
        on_regenerate=regenerate,
    )
    last = TurnSegmentBubble(
        segment(TurnSpeakerKind.CHARACTER, "character-1"),
        batch(),
        utc_now(),
        is_batch_end=True,
        on_regenerate=regenerate,
    )

    assert list(replay_buttons(first)) == []
    buttons = list(replay_buttons(last))
    assert len(buttons) == 1
    on_click = cast(Callable[[], Any], buttons[0].on_click)
    on_click()
    on_click()
    assert calls == 1


def test_partial_batch_has_only_one_regenerate_action_at_its_end() -> None:
    first = TurnSegmentBubble(
        segment(TurnSpeakerKind.CHARACTER, "character-1"),
        batch(TurnBatchState.PARTIAL),
        utc_now(),
        is_batch_end=False,
        on_regenerate=lambda: None,
    )
    last = TurnSegmentBubble(
        segment(TurnSpeakerKind.CHARACTER, "character-1"),
        batch(TurnBatchState.PARTIAL),
        utc_now(),
        is_batch_end=True,
        on_regenerate=lambda: None,
    )

    assert len(list(replay_buttons(first))) == 0
    assert len(list(replay_buttons(last))) == 1


def test_manual_translation_action_is_rendered_once_at_batch_end() -> None:
    first = TurnSegmentBubble(
        segment(TurnSpeakerKind.CHARACTER, "character-1"),
        batch(),
        utc_now(),
        on_translate=lambda: None,
    )
    last = TurnSegmentBubble(
        segment(TurnSpeakerKind.CHARACTER, "character-1"),
        batch(),
        utc_now(),
        is_batch_end=True,
        on_translate=lambda: None,
    )

    assert list(translation_buttons(first)) == []
    assert len(list(translation_buttons(last))) == 1


def test_completed_translation_is_displayed_only_at_batch_end() -> None:
    translated = translation()
    first = TurnSegmentBubble(
        segment(TurnSpeakerKind.CHARACTER, "character-1"),
        batch(),
        utc_now(),
        translation=translated,
    )
    last = TurnSegmentBubble(
        segment(TurnSpeakerKind.CHARACTER, "character-1"),
        batch(),
        utc_now(),
        is_batch_end=True,
        translation=translated,
    )

    assert "全体の日本語訳です。" not in "\n".join(texts(first))
    assert "全体の日本語訳です。" in "\n".join(texts(last))
