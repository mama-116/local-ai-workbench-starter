from __future__ import annotations

from pathlib import Path

import pytest

from local_llm_chat.application.services.memory_decision_service import (
    MemoryDecisionAction,
    MemoryDecisionRequest,
)
from local_llm_chat.application.services.memory_review_service import (
    MemoryReviewService,
)
from local_llm_chat.domain.canonical_memory import CanonicalMemoryEvent
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.states import (
    MemoryApprovalState,
    MemoryCardinality,
    MemoryKind,
    MessageState,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


@pytest.mark.asyncio
async def test_review_lists_only_currently_actionable_memory_and_refreshes_decisions(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation(
        "review", character.id, profile.id
    )
    session = await repository.start_send(conversation.id, "remember")

    def event(identifier: str, approval: MemoryApprovalState) -> CanonicalMemoryEvent:
        return CanonicalMemoryEvent(
            id=identifier,
            conversation_id=conversation.id,
            branch_id=session.branch_id,
            subject_id="user",
            kind=MemoryKind.PREFERENCE,
            slot="favorite_food",
            value=identifier,
            cardinality=MemoryCardinality.MULTIPLE,
            approval=approval,
            source_message_id=session.user_message.id,
            known_by_character_ids=frozenset({character.character_id}),
            supersedes_event_id=None,
            effective_at=session.user_message.created_at,
            recorded_at=session.user_message.created_at,
        )

    await repository.append_canonical_memory_events(
        (
            event("pending", MemoryApprovalState.PENDING_CONFIRMATION),
            event("automatic", MemoryApprovalState.AUTO_SAVED),
        )
    )
    service = MemoryReviewService(repository)

    first = await service.list_items(conversation.id, session.branch_id)

    assert [item.event_id for item in first.pending_confirmation] == ["pending"]
    assert [item.event_id for item in first.undoable] == ["automatic"]
    assert first.pending_confirmation[0].known_by_character_ids == frozenset(
        {character.character_id}
    )

    await service.decide(
        MemoryDecisionRequest(
            conversation.id,
            session.branch_id,
            "pending",
            MemoryDecisionAction.REJECT,
        )
    )
    await service.decide(
        MemoryDecisionRequest(
            conversation.id,
            session.branch_id,
            "automatic",
            MemoryDecisionAction.UNDO,
        )
    )

    refreshed = await service.list_items(conversation.id, session.branch_id)
    assert refreshed.pending_confirmation == ()
    assert refreshed.undoable == ()


@pytest.mark.asyncio
async def test_review_rejects_a_branch_outside_the_conversation(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile("ollama-local", "model", {})
    first = await repository.create_conversation("first", character.id, profile.id)
    second = await repository.create_conversation("second", character.id, profile.id)

    with pytest.raises(ValidationError, match="branch"):
        await MemoryReviewService(repository).list_items(
            first.id, second.active_branch_id
        )


@pytest.mark.asyncio
async def test_review_never_mixes_sibling_branch_memory(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation(
        "branches", character.id, profile.id
    )
    root = await repository.start_send(conversation.id, "root")
    await repository.finish_response(root, "done", MessageState.COMPLETED)
    root_branch_id = root.branch_id
    child = await repository.start_rewrite(
        conversation.id, root.user_message.id, "child"
    )
    await repository.finish_response(child, "done", MessageState.COMPLETED)

    def event(identifier: str, branch_id: str, source_id: str) -> CanonicalMemoryEvent:
        source = root.user_message if source_id == root.user_message.id else child.user_message
        return CanonicalMemoryEvent(
            id=identifier,
            conversation_id=conversation.id,
            branch_id=branch_id,
            subject_id="user",
            kind=MemoryKind.PREFERENCE,
            slot="liked_food",
            value=identifier,
            cardinality=MemoryCardinality.MULTIPLE,
            approval=MemoryApprovalState.AUTO_SAVED,
            source_message_id=source_id,
            known_by_character_ids=frozenset(),
            supersedes_event_id=None,
            effective_at=source.created_at,
            recorded_at=source.created_at,
        )

    await repository.append_canonical_memory_events(
        (
            event("root-only", root_branch_id, root.user_message.id),
            event("child-only", child.branch_id, child.user_message.id),
        )
    )
    service = MemoryReviewService(repository)

    root_items = await service.list_items(conversation.id, root_branch_id)
    child_items = await service.list_items(conversation.id, child.branch_id)

    assert [item.event_id for item in root_items.undoable] == ["root-only"]
    assert [item.event_id for item in child_items.undoable] == ["child-only"]
