from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from local_llm_chat.application.services.conversation_cast_service import (
    ConversationCastService,
)
from local_llm_chat.application.services.conversation_group_settings_service import (
    ConversationGroupSettingsService,
)
from local_llm_chat.application.services.turn_batch_service import TurnBatchService
from local_llm_chat.domain.errors import PersistenceError, ValidationError
from local_llm_chat.domain.group_turns import (
    TurnBatchDraft,
    TurnSegmentDraft,
)
from local_llm_chat.domain.states import (
    MessageState,
    RunState,
    TurnBatchState,
    TurnMode,
    TurnRepairState,
    TurnSpeakerKind,
)
from local_llm_chat.domain.models import CharacterVersion, Conversation, RunSession
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


async def make_session(
    repository: SQLiteAppRepository,
) -> tuple[Conversation, RunSession, CharacterVersion, CharacterVersion]:
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "group-model", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "group", character.id, profile.id
    )
    second = await repository.create_character_version("second", "second prompt")
    await ConversationCastService(repository).set_cast(
        conversation.id, (character.id, second.id)
    )
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.STORY
    )
    session = await repository.start_send(conversation.id, "みんな、どう思う？")
    return conversation, session, character, second


def completed_draft(character_id: str, second_character_id: str) -> TurnBatchDraft:
    return TurnBatchDraft(
        mode=TurnMode.STORY,
        formal_character_ids=(character_id, second_character_id),
        guest_ids=(),
        prompt_version="group-turn-v1",
        state=TurnBatchState.COMPLETED,
        repair_state=TurnRepairState.NOT_NEEDED,
        segments=(
            TurnSegmentDraft(
                TurnSpeakerKind.CHARACTER,
                character_id,
                "先輩",
                "僕は賛成だよ。",
            ),
            TurnSegmentDraft(
                TurnSpeakerKind.NARRATOR,
                None,
                "ナレーター",
                "田中は少し考え込んだ。",
            ),
            TurnSegmentDraft(
                TurnSpeakerKind.CHARACTER,
                second_character_id,
                "田中",
                "僕も試してみたい。",
            ),
        ),
    )


def too_many_formal_characters(draft: TurnBatchDraft) -> TurnBatchDraft:
    return replace(
        draft, formal_character_ids=tuple(f"character-{i}" for i in range(6))
    )


def too_many_guests(draft: TurnBatchDraft) -> TurnBatchDraft:
    return replace(draft, guest_ids=tuple(f"guest-{i}" for i in range(6)))


def no_segments(draft: TurnBatchDraft) -> TurnBatchDraft:
    return replace(draft, segments=())


def speaker_outside_cast(draft: TurnBatchDraft) -> TurnBatchDraft:
    return replace(
        draft,
        segments=(
            TurnSegmentDraft(
                TurnSpeakerKind.CHARACTER,
                "outside-cast",
                "部外者",
                "勝手に参加するよ。",
            ),
        ),
    )


INVALID_DRAFT_MUTATIONS: tuple[
    Callable[[TurnBatchDraft], TurnBatchDraft], ...
] = (
    too_many_formal_characters,
    too_many_guests,
    no_segments,
    speaker_outside_cast,
)


@pytest.mark.asyncio
async def test_completed_batch_preserves_order_and_restarts_as_one_unit(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation, session, character, second = await make_session(repository)
    service = TurnBatchService(repository)

    saved = await service.finish(
        session, completed_draft(character.character_id, second.character_id)
    )

    assert saved.conversation_id == conversation.id
    assert saved.source_message_id == session.user_message.id
    assert [segment.position for segment in saved.segments] == [0, 1, 2]
    assert [segment.display_name for segment in saved.segments] == [
        "先輩",
        "ナレーター",
        "田中",
    ]
    assert saved.state is TurnBatchState.COMPLETED
    assert saved.error_code is None

    restarted = SQLiteAppRepository(database_path)
    restored = await restarted.get_turn_batch_for_response(
        session.assistant_message.id
    )
    assert restored == saved
    response = await restarted.get_message(session.assistant_message.id)
    assert response.state is MessageState.COMPLETED
    assert response.content == (
        "先輩: 僕は賛成だよ。\n\n"
        "ナレーター: 田中は少し考え込んだ。\n\n"
        "田中: 僕も試してみたい。"
    )


@pytest.mark.asyncio
async def test_completed_batch_persists_group_performance_metrics(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation, session, character, second = await make_session(repository)
    draft = replace(
        completed_draft(character.character_id, second.character_id),
        prompt_tokens=100,
        output_tokens=50,
        total_duration_ns=4_000_000_000,
        generation_duration_ns=2_000_000_000,
        response_duration_ms=4_500,
    )

    await TurnBatchService(repository).finish(session, draft)

    restarted = SQLiteAppRepository(database_path)
    latest = await restarted.get_latest_telemetry(conversation.id)
    assert latest is not None
    assert latest.run.prompt_tokens == 100
    assert latest.run.output_tokens == 50
    assert latest.run.total_duration_ns == 4_000_000_000
    assert latest.run.generation_duration_ns == 2_000_000_000
    assert latest.run.response_duration_ms == 4_500
    assert latest.tokens_per_second == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_partial_batch_keeps_only_validated_segments_and_failure_state(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    _, session, character, second = await make_session(repository)
    service = TurnBatchService(repository)
    draft = replace(
        completed_draft(character.character_id, second.character_id),
        state=TurnBatchState.PARTIAL,
        repair_state=TurnRepairState.FAILED,
        error_code="structured_output_truncated",
        segments=completed_draft(
            character.character_id, second.character_id
        ).segments[:2],
    )

    saved = await service.finish(session, draft)

    assert saved.state is TurnBatchState.PARTIAL
    assert len(saved.segments) == 2
    assert saved.error_code == "structured_output_truncated"
    response = await repository.get_message(session.assistant_message.id)
    assert response.state is MessageState.COMPLETED
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        assert connection.execute(
            "SELECT state, error_code FROM runs WHERE id = ?", (session.run.id,)
        ).fetchone() == (RunState.FAILED.value, "structured_output_truncated")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate_draft",
    INVALID_DRAFT_MUTATIONS,
)
async def test_invalid_batch_is_rejected_before_any_persistence(
    tmp_path: Path,
    mutate_draft: Callable[[TurnBatchDraft], TurnBatchDraft],
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    _, session, character, second = await make_session(repository)
    service = TurnBatchService(repository)
    draft = mutate_draft(
        completed_draft(character.character_id, second.character_id)
    )

    with pytest.raises(ValidationError):
        await service.finish(session, draft)

    assert await repository.get_message(session.assistant_message.id) == (
        session.assistant_message
    )
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM turn_batches").fetchone() == (
            0,
        )


@pytest.mark.asyncio
async def test_incomplete_completed_round_table_is_rejected_before_persistence(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation, session, character, second = await make_session(repository)
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.ROUND_TABLE
    )
    draft = replace(
        completed_draft(character.character_id, second.character_id),
        mode=TurnMode.ROUND_TABLE,
        segments=(
            TurnSegmentDraft(
                TurnSpeakerKind.CHARACTER,
                second.character_id,
                "second",
                "secondだけの発言",
            ),
        ),
    )

    with pytest.raises(
        ValidationError,
        match="completed round table must match the formal cast order exactly",
    ):
        await TurnBatchService(repository).finish(session, draft)

    assert await repository.get_message(session.assistant_message.id) == (
        session.assistant_message
    )
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM turn_batches").fetchone() == (
            0,
        )


@pytest.mark.asyncio
async def test_segment_write_failure_rolls_back_batch_message_and_run(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    _, session, character, second = await make_session(repository)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_second_turn_segment BEFORE INSERT ON turn_segments "
            "WHEN NEW.position = 1 BEGIN SELECT RAISE(ABORT, 'forced'); END"
        )
    service = TurnBatchService(repository)
    draft = replace(
        completed_draft(character.character_id, second.character_id),
        prompt_tokens=100,
        output_tokens=50,
        total_duration_ns=4_000_000_000,
        generation_duration_ns=2_000_000_000,
        response_duration_ms=4_500,
    )

    with pytest.raises(PersistenceError):
        await service.finish(session, draft)

    response = await repository.get_message(session.assistant_message.id)
    assert response.state is MessageState.PENDING
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM turn_batches").fetchone() == (
            0,
        )
        assert connection.execute("SELECT COUNT(*) FROM turn_segments").fetchone() == (
            0,
        )
        assert connection.execute(
            """
            SELECT state, prompt_tokens, output_tokens, total_duration_ns,
                   generation_duration_ns, response_duration_ms
            FROM runs WHERE id = ?
            """,
            (session.run.id,),
        ).fetchone() == (RunState.PENDING.value, None, None, None, None, None)


@pytest.mark.asyncio
async def test_batch_rejects_a_branch_from_another_conversation(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation, session, character, second = await make_session(repository)
    other = await repository.create_conversation(
        "other", character.id, conversation.model_profile_id
    )
    forged_session = replace(session, branch_id=other.active_branch_id)

    with pytest.raises(ValidationError):
        await TurnBatchService(repository).finish(
            forged_session,
            completed_draft(character.character_id, second.character_id),
        )

    assert (await repository.get_message(session.assistant_message.id)).state is (
        MessageState.PENDING
    )


@pytest.mark.asyncio
async def test_batch_rejects_a_stale_or_reordered_formal_cast_atomically(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    _, session, character, second = await make_session(repository)
    draft = completed_draft(character.character_id, second.character_id)
    reordered = replace(
        draft,
        formal_character_ids=(second.character_id, character.character_id),
    )

    with pytest.raises(ValidationError):
        await TurnBatchService(repository).finish(session, reordered)

    assert (await repository.get_message(session.assistant_message.id)).state is (
        MessageState.PENDING
    )
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM turn_batches").fetchone() == (
            0,
        )
        assert connection.execute(
            "SELECT state FROM runs WHERE id = ?", (session.run.id,)
        ).fetchone() == (RunState.PENDING.value,)


@pytest.mark.asyncio
async def test_batch_rejects_a_stale_mode_before_any_persistence(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation, session, character, second = await make_session(repository)
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.ROUND_TABLE
    )

    with pytest.raises(ValidationError):
        await TurnBatchService(repository).finish(
            session, completed_draft(character.character_id, second.character_id)
        )

    assert (await repository.get_message(session.assistant_message.id)).state is (
        MessageState.PENDING
    )
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM turn_batches").fetchone() == (
            0,
        )


@pytest.mark.asyncio
async def test_spotlight_target_is_saved_and_restored_with_the_batch(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation, session, character, second = await make_session(repository)
    await ConversationGroupSettingsService(repository).configure(
        conversation.id,
        enabled=True,
        mode=TurnMode.SPOTLIGHT,
        spotlight_character_id=second.character_id,
    )
    draft = replace(
        completed_draft(character.character_id, second.character_id),
        mode=TurnMode.SPOTLIGHT,
        spotlight_character_id=second.character_id,
    )

    saved = await TurnBatchService(repository).finish(session, draft)

    assert saved.spotlight_character_id == second.character_id
    restarted = SQLiteAppRepository(database_path)
    assert await restarted.get_turn_batch_for_response(
        session.assistant_message.id
    ) == saved
