import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from local_llm_chat.domain.canonical_memory import (
    CanonicalMemoryFact,
    CanonicalMemoryEvent,
    MemoryApprovalDecision,
)
from local_llm_chat.domain.errors import PersistenceError, ValidationError
from local_llm_chat.domain.states import (
    MemoryApprovalState,
    MemoryCardinality,
    MemoryKind,
    MessageState,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


BASE_TIME = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def memory_event(
    identifier: str,
    *,
    conversation_id: str,
    branch_id: str,
    source_message_id: str,
    value: str,
    kind: MemoryKind = MemoryKind.PREFERENCE,
    slot: str = "favorite_ice_cream",
    cardinality: MemoryCardinality = MemoryCardinality.SINGLE,
    approval: MemoryApprovalState = MemoryApprovalState.AUTO_SAVED,
    known_by: frozenset[str] = frozenset(),
    supersedes_event_id: str | None = None,
    minutes: int = 0,
) -> CanonicalMemoryEvent:
    recorded_at = BASE_TIME + timedelta(minutes=minutes)
    return CanonicalMemoryEvent(
        id=identifier,
        conversation_id=conversation_id,
        branch_id=branch_id,
        subject_id="user",
        kind=kind,
        slot=slot,
        value=value,
        cardinality=cardinality,
        approval=approval,
        source_message_id=source_message_id,
        known_by_character_ids=known_by,
        supersedes_event_id=supersedes_event_id,
        effective_at=recorded_at,
        recorded_at=recorded_at,
    )


@pytest.mark.asyncio
async def test_canonical_memory_reopens_with_preference_undo_secret_and_branches(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    alice = await repository.create_character_version("アリス", "アリスとして話す")
    bob = await repository.create_character_version("ボブ", "ボブとして話す")
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "正史記憶", alice.id, profile.id
    )
    root_branch_id = conversation.active_branch_id
    root = await repository.start_send(conversation.id, "アイスの話")
    await repository.finish_response(root, "覚えておくね", MessageState.COMPLETED)

    branch_a = await repository.start_rewrite(
        conversation.id, root.user_message.id, "今はチョコが好き"
    )
    await repository.finish_response(branch_a, "更新したよ", MessageState.COMPLETED)
    await repository.activate_branch(conversation.id, root_branch_id)
    branch_b = await repository.start_rewrite(
        conversation.id, root.user_message.id, "前の話を続けよう"
    )
    await repository.finish_response(branch_b, "続けよう", MessageState.COMPLETED)

    strawberry = memory_event(
        "memory-strawberry",
        conversation_id=conversation.id,
        branch_id=root_branch_id,
        source_message_id=root.user_message.id,
        value="ストロベリーアイス",
    )
    chocolate = memory_event(
        "memory-chocolate",
        conversation_id=conversation.id,
        branch_id=branch_a.branch_id,
        source_message_id=branch_a.user_message.id,
        value="チョコアイス",
        minutes=3,
    )
    allergy = memory_event(
        "memory-allergy",
        conversation_id=conversation.id,
        branch_id=root_branch_id,
        source_message_id=root.user_message.id,
        value="ストロベリー",
        kind=MemoryKind.SAFETY_CONSTRAINT,
        slot="food_allergy",
        approval=MemoryApprovalState.PENDING_CONFIRMATION,
        minutes=1,
    )
    secret = memory_event(
        "memory-secret",
        conversation_id=conversation.id,
        branch_id=root_branch_id,
        source_message_id=root.user_message.id,
        value="本当は部活を辞めたい",
        kind=MemoryKind.GOAL,
        slot="club_membership_intent",
        known_by=frozenset({alice.character_id}),
        minutes=2,
    )
    temporary = memory_event(
        "memory-temporary",
        conversation_id=conversation.id,
        branch_id=root_branch_id,
        source_message_id=root.user_message.id,
        value="ミントアイス",
        slot="liked_ice_cream",
        cardinality=MemoryCardinality.MULTIPLE,
        minutes=2,
    )
    undo = MemoryApprovalDecision(
        id="decision-undo-temporary",
        target_event_id=temporary.id,
        state=MemoryApprovalState.UNDONE,
        source_message_id=root.user_message.id,
        recorded_at=BASE_TIME + timedelta(minutes=4),
    )

    for item in (strawberry, chocolate, allergy, secret, temporary):
        await repository.append_canonical_memory_event(item)
    await repository.append_memory_approval_decision(undo)

    async def projections(
        target: SQLiteAppRepository,
    ) -> tuple[tuple[CanonicalMemoryFact, ...], ...]:
        return (
            await target.project_canonical_memory(
                conversation.id,
                root_branch_id,
                alice.character_id,
                current_source_message_id=root.user_message.id,
            ),
            await target.project_canonical_memory(
                conversation.id, branch_a.branch_id, alice.character_id
            ),
            await target.project_canonical_memory(
                conversation.id, branch_a.branch_id, bob.character_id
            ),
            await target.project_canonical_memory(
                conversation.id, branch_b.branch_id, alice.character_id
            ),
        )

    before_restart = await projections(repository)
    reopened = SQLiteAppRepository(database_path)
    await reopened.initialize()
    after_restart = await projections(reopened)

    assert after_restart == before_restart
    root_current, branch_a_alice, branch_a_bob, branch_b_alice = after_restart
    assert root_current[0].value == "ストロベリー"
    assert {fact.value for fact in root_current[1:]} == {
        "ストロベリーアイス",
        "本当は部活を辞めたい",
    }
    assert [fact.value for fact in branch_a_alice] == ["チョコアイス"]
    assert [fact.value for fact in branch_a_bob] == ["チョコアイス"]
    assert branch_b_alice == ()

    await reopened.append_canonical_memory_event(strawberry)
    await reopened.append_memory_approval_decision(undo)

    confirm = MemoryApprovalDecision(
        id="decision-confirm-allergy",
        target_event_id=allergy.id,
        state=MemoryApprovalState.CONFIRMED,
        source_message_id=root.user_message.id,
        recorded_at=BASE_TIME + timedelta(minutes=5),
    )
    reject = MemoryApprovalDecision(
        id="decision-reject-allergy",
        target_event_id=allergy.id,
        state=MemoryApprovalState.REJECTED,
        source_message_id=root.user_message.id,
        recorded_at=BASE_TIME + timedelta(minutes=5),
    )
    competitor = SQLiteAppRepository(database_path)
    await competitor.initialize()
    decision_results = await asyncio.gather(
        reopened.append_memory_approval_decision(confirm),
        competitor.append_memory_approval_decision(reject),
        return_exceptions=True,
    )
    assert sum(result is None for result in decision_results) == 1
    decision_errors = [
        result for result in decision_results if isinstance(result, Exception)
    ]
    assert len(decision_errors) == 1
    assert isinstance(decision_errors[0], (PersistenceError, ValidationError))

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM canonical_memory_events"
        ).fetchone() == (5,)
        assert connection.execute(
            "SELECT COUNT(*) FROM canonical_memory_decisions"
        ).fetchone() == (2,)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE canonical_memory_events SET value = '改ざん' "
                "WHERE id = 'memory-strawberry'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "INSERT INTO canonical_memory_event_knowledge(event_id, character_id) "
                "VALUES(?, ?)",
                (secret.id, bob.character_id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE canonical_memory_decisions SET state = 'confirmed' "
                "WHERE id = 'decision-undo-temporary'"
            )
        connection.execute(
            """
            INSERT INTO canonical_memory_events(
                id, conversation_id, branch_id, subject_id, kind, slot, value,
                cardinality, approval, knowledge_count, source_message_id,
                supersedes_event_id, effective_at, recorded_at
            ) VALUES(?, ?, ?, 'user', 'goal', 'corrupt_scope', '破損データ',
                     'single', 'auto_saved', 1, ?, NULL, ?, ?)
            """,
            (
                "memory-corrupt-scope",
                conversation.id,
                root_branch_id,
                root.user_message.id,
                BASE_TIME.isoformat(),
                BASE_TIME.isoformat(),
            ),
        )

    with pytest.raises(PersistenceError, match="知識範囲が壊れています"):
        await reopened.project_canonical_memory(
            conversation.id, root_branch_id, alice.character_id
        )
