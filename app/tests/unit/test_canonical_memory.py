from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from local_llm_chat.domain.canonical_memory import (
    CanonicalMemoryEvent,
    CanonicalMemoryLedger,
    MemoryApprovalDecision,
    MemoryProjectionQuery,
)
from local_llm_chat.domain.states import (
    MemoryApprovalState,
    MemoryCardinality,
    MemoryFactState,
    MemoryKind,
)


BASE_TIME = datetime(2026, 7, 18, 9, 0, tzinfo=UTC)


def event(
    identifier: str,
    *,
    branch_id: str = "branch-root",
    kind: MemoryKind = MemoryKind.PREFERENCE,
    slot: str = "favorite_ice_cream",
    value: str,
    approval: MemoryApprovalState = MemoryApprovalState.AUTO_SAVED,
    known_by: frozenset[str] = frozenset(),
    supersedes_event_id: str | None = None,
    minutes: int = 0,
    source_message_id: str | None = None,
) -> CanonicalMemoryEvent:
    recorded_at = BASE_TIME + timedelta(minutes=minutes)
    return CanonicalMemoryEvent(
        id=identifier,
        conversation_id="conversation-1",
        branch_id=branch_id,
        subject_id="user",
        kind=kind,
        slot=slot,
        value=value,
        cardinality=MemoryCardinality.SINGLE,
        approval=approval,
        source_message_id=source_message_id or f"message-{identifier}",
        known_by_character_ids=known_by,
        supersedes_event_id=supersedes_event_id,
        effective_at=recorded_at,
        recorded_at=recorded_at,
    )


def query(
    *,
    branch_lineage: tuple[str, ...] = ("branch-root",),
    speaker_id: str = "character-alice",
    current_source_message_id: str | None = None,
    include_historical: bool = False,
) -> MemoryProjectionQuery:
    return MemoryProjectionQuery(
        conversation_id="conversation-1",
        branch_lineage=branch_lineage,
        speaker_character_id=speaker_id,
        current_source_message_id=current_source_message_id,
        include_historical=include_historical,
    )


def test_changed_preference_keeps_history_without_using_it_as_current() -> None:
    strawberry = event("preference-1", value="ストロベリーアイス")
    chocolate = event(
        "preference-2",
        value="チョコアイス",
        supersedes_event_id=strawberry.id,
        minutes=5,
    )
    ledger = CanonicalMemoryLedger((strawberry, chocolate))

    current = ledger.project(query())
    with_history = ledger.project(query(include_historical=True))

    assert [(fact.value, fact.state) for fact in current] == [
        ("チョコアイス", MemoryFactState.ACTIVE)
    ]
    assert [(fact.value, fact.state) for fact in with_history] == [
        ("チョコアイス", MemoryFactState.ACTIVE),
        ("ストロベリーアイス", MemoryFactState.HISTORICAL),
    ]


def test_pending_allergy_applies_only_to_its_current_response() -> None:
    current_favorite = event("preference-1", value="チョコアイス")
    allergy = event(
        "allergy-1",
        kind=MemoryKind.SAFETY_CONSTRAINT,
        slot="food_allergy",
        value="ストロベリー",
        approval=MemoryApprovalState.PENDING_CONFIRMATION,
        source_message_id="message-current",
    )
    ledger = CanonicalMemoryLedger((current_favorite, allergy))

    current_response = ledger.project(
        query(current_source_message_id="message-current")
    )
    later_response = ledger.project(query())

    assert [(fact.kind, fact.value) for fact in current_response] == [
        (MemoryKind.SAFETY_CONSTRAINT, "ストロベリー"),
        (MemoryKind.PREFERENCE, "チョコアイス"),
    ]
    assert [fact.value for fact in later_response] == ["チョコアイス"]


def test_secret_is_visible_only_to_characters_in_its_knowledge_scope() -> None:
    secret = event(
        "secret-1",
        kind=MemoryKind.GOAL,
        slot="club_membership_intent",
        value="本当は部活を辞めたい",
        known_by=frozenset({"character-alice"}),
    )
    ledger = CanonicalMemoryLedger((secret,))

    assert [fact.value for fact in ledger.project(query(speaker_id="character-alice"))] == [
        "本当は部活を辞めたい"
    ]
    assert ledger.project(query(speaker_id="character-bob")) == ()


def test_undo_is_an_append_only_decision_and_hides_the_auto_saved_fact() -> None:
    auto_saved = event("preference-undo", value="ストロベリーアイス")
    undo = MemoryApprovalDecision(
        id="decision-undo",
        target_event_id=auto_saved.id,
        state=MemoryApprovalState.UNDONE,
        source_message_id="message-undo",
        recorded_at=BASE_TIME + timedelta(minutes=1),
    )

    assert CanonicalMemoryLedger((auto_saved,)).project(query())
    assert CanonicalMemoryLedger((auto_saved,), (undo,)).project(query()) == ()


def test_undoing_historical_middle_value_does_not_resurrect_older_value() -> None:
    strawberry = event("preference-1", value="ストロベリーアイス")
    chocolate = event(
        "preference-2",
        value="チョコアイス",
        supersedes_event_id=strawberry.id,
        minutes=1,
    )
    vanilla = event(
        "preference-3",
        value="バニラアイス",
        supersedes_event_id=chocolate.id,
        minutes=2,
    )
    undo_middle = MemoryApprovalDecision(
        id="decision-undo-middle",
        target_event_id=chocolate.id,
        state=MemoryApprovalState.UNDONE,
        source_message_id="message-undo-middle",
        recorded_at=BASE_TIME + timedelta(minutes=3),
    )

    projected = CanonicalMemoryLedger(
        (strawberry, chocolate, vanilla), (undo_middle,)
    ).project(query(include_historical=True))

    assert [(fact.value, fact.state) for fact in projected] == [
        ("バニラアイス", MemoryFactState.ACTIVE),
        ("ストロベリーアイス", MemoryFactState.HISTORICAL),
    ]


def test_sibling_branch_change_does_not_replace_parent_fact_elsewhere() -> None:
    root = event("root-preference", value="ストロベリーアイス")
    branch_a_change = event(
        "branch-a-preference",
        branch_id="branch-a",
        value="チョコアイス",
        supersedes_event_id=root.id,
        minutes=5,
    )
    ledger = CanonicalMemoryLedger((root, branch_a_change))

    branch_a = ledger.project(query(branch_lineage=("branch-root", "branch-a")))
    branch_b = ledger.project(query(branch_lineage=("branch-root", "branch-b")))

    assert [fact.value for fact in branch_a] == ["チョコアイス"]
    assert [fact.value for fact in branch_b] == ["ストロベリーアイス"]


def test_rejects_cross_subject_supersede_that_would_corrupt_history() -> None:
    user_fact = event("user-fact", value="チョコアイス")
    invalid = replace(
        user_fact,
        id="other-subject-fact",
        subject_id="character-alice",
        supersedes_event_id=user_fact.id,
    )

    with pytest.raises(ValueError, match="same subject, kind, slot, and cardinality"):
        CanonicalMemoryLedger((user_fact, invalid))
