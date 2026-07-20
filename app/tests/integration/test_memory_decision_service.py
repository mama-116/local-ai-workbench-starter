import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_llm_chat.application.services.memory_decision_service import (
    MemoryDecisionAction,
    MemoryDecisionRequest,
    MemoryDecisionService,
)
from local_llm_chat.domain.canonical_memory import (
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


DECISION_TIME = datetime(2026, 7, 18, 15, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_memory_decisions_confirm_reject_undo_retry_and_reopen(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile("ollama-local", "test-local", {})
    conversation = await repository.create_conversation("記憶判断", character.id, profile.id)
    session = await repository.start_send(conversation.id, "記憶を確認する")
    await repository.finish_response(session, "確認しよう", MessageState.COMPLETED)

    def memory_event(
        identifier: str, approval: MemoryApprovalState, value: str
    ) -> CanonicalMemoryEvent:
        return CanonicalMemoryEvent(
            id=identifier,
            conversation_id=conversation.id,
            branch_id=conversation.active_branch_id,
            subject_id="user",
            kind=MemoryKind.GOAL,
            slot="goals",
            value=value,
            cardinality=MemoryCardinality.MULTIPLE,
            approval=approval,
            source_message_id=session.user_message.id,
            known_by_character_ids=frozenset(),
            supersedes_event_id=None,
            effective_at=session.user_message.created_at,
            recorded_at=session.user_message.created_at,
        )

    pending_confirm = memory_event(
        "pending-confirm", MemoryApprovalState.PENDING_CONFIRMATION, "小説を書く"
    )
    pending_reject = memory_event(
        "pending-reject", MemoryApprovalState.PENDING_CONFIRMATION, "漫画を描く"
    )
    auto_saved = memory_event(
        "auto-saved", MemoryApprovalState.AUTO_SAVED, "毎日散歩する"
    )
    await repository.append_canonical_memory_events(
        (pending_confirm, pending_reject, auto_saved)
    )
    service = MemoryDecisionService(repository, clock=lambda: DECISION_TIME)

    confirm_request = MemoryDecisionRequest(
        conversation.id,
        conversation.active_branch_id,
        pending_confirm.id,
        MemoryDecisionAction.CONFIRM,
    )
    confirmed = await service.decide(confirm_request)
    confirmed_retry = await service.decide(confirm_request)
    rejected = await service.decide(
        MemoryDecisionRequest(
            conversation.id,
            conversation.active_branch_id,
            pending_reject.id,
            MemoryDecisionAction.REJECT,
        )
    )
    undone = await service.decide(
        MemoryDecisionRequest(
            conversation.id,
            conversation.active_branch_id,
            auto_saved.id,
            MemoryDecisionAction.UNDO,
        )
    )

    assert confirmed_retry == confirmed
    assert confirmed.state is MemoryApprovalState.CONFIRMED
    assert rejected.state is MemoryApprovalState.REJECTED
    assert undone.state is MemoryApprovalState.UNDONE
    projected = await repository.project_canonical_memory(
        conversation.id, conversation.active_branch_id, character.character_id
    )
    assert [fact.value for fact in projected] == ["小説を書く"]

    with pytest.raises(ValidationError):
        await service.decide(
            MemoryDecisionRequest(
                conversation.id,
                "not-this-branch",
                pending_confirm.id,
                MemoryDecisionAction.REJECT,
            )
        )

    await service.decide(
        MemoryDecisionRequest(
            conversation.id,
            conversation.active_branch_id,
            pending_confirm.id,
            MemoryDecisionAction.UNDO,
        )
    )
    with pytest.raises(ValidationError, match="stale or conflicting"):
        await service.decide(confirm_request)

    reopened = SQLiteAppRepository(database_path)
    await reopened.initialize()
    assert await reopened.project_canonical_memory(
        conversation.id, conversation.active_branch_id, character.character_id
    ) == ()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM canonical_memory_decisions"
        ).fetchone() == (4,)


@pytest.mark.asyncio
async def test_concurrent_confirm_and_reject_allow_only_one_decision(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile("ollama-local", "test-local", {})
    conversation = await repository.create_conversation("同時判断", character.id, profile.id)
    session = await repository.start_send(conversation.id, "目標は小説を書くこと")
    await repository.finish_response(session, "確認する", MessageState.COMPLETED)
    pending = CanonicalMemoryEvent(
        id="pending-concurrent",
        conversation_id=conversation.id,
        branch_id=conversation.active_branch_id,
        subject_id="user",
        kind=MemoryKind.GOAL,
        slot="goals",
        value="小説を書く",
        cardinality=MemoryCardinality.MULTIPLE,
        approval=MemoryApprovalState.PENDING_CONFIRMATION,
        source_message_id=session.user_message.id,
        known_by_character_ids=frozenset(),
        supersedes_event_id=None,
        effective_at=session.user_message.created_at,
        recorded_at=session.user_message.created_at,
    )
    await repository.append_canonical_memory_event(pending)
    competitor = SQLiteAppRepository(database_path)
    await competitor.initialize()
    confirm = MemoryDecisionService(repository, clock=lambda: DECISION_TIME)
    reject = MemoryDecisionService(competitor, clock=lambda: DECISION_TIME)

    results = await asyncio.gather(
        confirm.decide(
            MemoryDecisionRequest(
                conversation.id,
                conversation.active_branch_id,
                pending.id,
                MemoryDecisionAction.CONFIRM,
            )
        ),
        reject.decide(
            MemoryDecisionRequest(
                conversation.id,
                conversation.active_branch_id,
                pending.id,
                MemoryDecisionAction.REJECT,
            )
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(item, MemoryApprovalDecision) for item in results) == 1
    assert sum(
        isinstance(item, (PersistenceError, ValidationError)) for item in results
    ) == 1
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM canonical_memory_decisions"
        ).fetchone() == (1,)


@pytest.mark.asyncio
async def test_undo_rejects_notification_after_fact_was_superseded(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile("ollama-local", "test-local", {})
    conversation = await repository.create_conversation("期限切れUndo", character.id, profile.id)
    first = await repository.start_send(conversation.id, "一番好きなのは苺")
    await repository.finish_response(first, "覚えた", MessageState.COMPLETED)
    second = await repository.start_send(conversation.id, "一番好きなのはチョコ")
    await repository.finish_response(second, "更新した", MessageState.COMPLETED)

    def preference(identifier: str, value: str) -> CanonicalMemoryEvent:
        message = first.user_message if identifier == "old" else second.user_message
        return CanonicalMemoryEvent(
            id=identifier,
            conversation_id=conversation.id,
            branch_id=conversation.active_branch_id,
            subject_id="user",
            kind=MemoryKind.PREFERENCE,
            slot="favorite_food",
            value=value,
            cardinality=MemoryCardinality.SINGLE,
            approval=MemoryApprovalState.AUTO_SAVED,
            source_message_id=message.id,
            known_by_character_ids=frozenset(),
            supersedes_event_id=None,
            effective_at=message.created_at,
            recorded_at=message.created_at,
        )

    old = preference("old", "苺")
    new = preference("new", "チョコ")
    await repository.append_captured_memory_events((old,))
    await repository.append_captured_memory_events((new,))
    service = MemoryDecisionService(repository, clock=lambda: DECISION_TIME)

    with pytest.raises(ValidationError, match="no longer active"):
        await service.decide(
            MemoryDecisionRequest(
                conversation.id,
                conversation.active_branch_id,
                old.id,
                MemoryDecisionAction.UNDO,
            )
        )

    active = await repository.project_canonical_memory(
        conversation.id, conversation.active_branch_id, character.character_id
    )
    assert [fact.value for fact in active] == ["チョコ"]
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM canonical_memory_decisions"
        ).fetchone() == (0,)
