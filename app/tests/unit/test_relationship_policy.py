from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.relationship_profile import (
    EvidenceContext,
    RelationshipApproval,
    RelationshipCandidate,
    RelationshipEvent,
    RelationshipMeaning,
    RelationshipSeverity,
    revalidate_relationship_candidate,
    reduce_relationship_events,
)


NOW = datetime(2026, 7, 26, tzinfo=UTC)


def event(
    event_id: str,
    meaning: RelationshipMeaning,
    severity: RelationshipSeverity = RelationshipSeverity.LOW,
    approval: RelationshipApproval = RelationshipApproval.AUTO_APPLIED,
) -> RelationshipEvent:
    return RelationshipEvent(
        id=event_id,
        continuity_id="continuity-1",
        user_profile_id="profile-1",
        character_id="alice",
        source_conversation_id="conversation-1",
        source_branch_id="branch-1",
        source_message_id="message-1",
        meaning=meaning,
        severity=severity,
        evidence_context=EvidenceContext.DIRECT,
        evidence_start=0,
        evidence_end=4,
        reason="固定シナリオ",
        approval=approval,
        policy_version="relationship-v1",
        known_by_character_ids=("alice",),
        relationship_definition_id=None,
        assignment_state=None,
        role=None,
        recorded_at=NOW,
    )


def candidate(
    context: EvidenceContext,
    meaning: RelationshipMeaning = RelationshipMeaning.BOUNDARY_VIOLATION,
    **overrides: object,
) -> RelationshipCandidate:
    values: dict[str, object] = {
        "event_id": "candidate-1",
        "character_id": "alice",
        "meaning": meaning,
        "severity": RelationshipSeverity.LOW,
        "evidence_context": context,
        "evidence_start": 0,
        "evidence_end": len("最低だ"),
        "evidence_text": "最低だ",
    }
    values.update(overrides)
    return RelationshipCandidate(**values)  # type: ignore[arg-type]


def test_reducer_is_deterministic_idempotent_and_bounded() -> None:
    violation = event(
        "event-1",
        RelationshipMeaning.BOUNDARY_VIOLATION,
        RelationshipSeverity.HIGH,
    )
    repair = event(
        "event-2", RelationshipMeaning.REPAIR, RelationshipSeverity.HIGH
    )

    result = reduce_relationship_events((violation, violation, repair))

    assert result.affinity == 48
    assert result.trust == 28
    assert result.tension == 4
    assert result.applied_event_ids == ("event-1", "event-2")


def test_pending_rejected_and_undone_events_do_not_change_metrics() -> None:
    events = tuple(
        event(
            f"event-{approval.value}",
            RelationshipMeaning.REPEATED_BOUNDARY_VIOLATION,
            RelationshipSeverity.HIGH,
            approval,
        )
        for approval in (
            RelationshipApproval.PENDING_CONFIRMATION,
            RelationshipApproval.REJECTED,
            RelationshipApproval.UNDONE,
        )
    )

    result = reduce_relationship_events(events)

    assert (result.affinity, result.trust, result.tension) == (50, 30, 0)
    assert result.applied_event_ids == ()


def test_reset_returns_to_initial_values_without_deleting_history() -> None:
    result = reduce_relationship_events(
        (
            event("conflict", RelationshipMeaning.CONFLICT),
            event("reset", RelationshipMeaning.RESET),
            event("positive", RelationshipMeaning.POSITIVE_INTERACTION),
        )
    )

    assert (result.affinity, result.trust, result.tension) == (51, 31, 0)
    assert result.applied_event_ids == ("conflict", "reset", "positive")


def test_unknown_policy_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        reduce_relationship_events(
            (
                replace(
                    event("event-1", RelationshipMeaning.POSITIVE_INTERACTION),
                    policy_version="model-picked-v99",
                ),
            )
        )


@pytest.mark.parametrize(
    "context",
    (
        EvidenceContext.QUOTED,
        EvidenceContext.HYPOTHETICAL,
        EvidenceContext.NARRATIVE,
        EvidenceContext.THIRD_PARTY,
        EvidenceContext.UNKNOWN,
    ),
)
def test_non_direct_context_is_never_saved_as_boundary_violation(
    context: EvidenceContext,
) -> None:
    assert (
        revalidate_relationship_candidate(
            candidate=candidate(context),
            source_text="最低だ、と彼は言った",
            expected_character_id="alice",
        )
        is None
    )


def test_agreed_roleplay_and_reasoned_conflict_are_not_boundary_violations() -> None:
    item = candidate(
        EvidenceContext.ROLEPLAY,
        conflict_has_reason=True,
        roleplay_active=True,
    )

    assert (
        revalidate_relationship_candidate(
            candidate=item,
            source_text="最低だ、役を演じて",
            expected_character_id="alice",
        )
        is RelationshipMeaning.CONFLICT
    )


def test_direct_uncontextualized_insult_can_be_a_boundary_violation() -> None:
    assert (
        revalidate_relationship_candidate(
            candidate=candidate(EvidenceContext.DIRECT),
            source_text="最低だ",
            expected_character_id="alice",
        )
        is RelationshipMeaning.BOUNDARY_VIOLATION
    )


def test_repeated_violation_requires_a_previously_presented_boundary() -> None:
    item = candidate(
        EvidenceContext.DIRECT,
        RelationshipMeaning.REPEATED_BOUNDARY_VIOLATION,
    )
    assert (
        revalidate_relationship_candidate(
            candidate=item,
            source_text="最低だ",
            expected_character_id="alice",
        )
        is None
    )
    assert (
        revalidate_relationship_candidate(
            candidate=replace(item, boundary_previously_set=True),
            source_text="最低だ",
            expected_character_id="alice",
        )
        is RelationshipMeaning.REPEATED_BOUNDARY_VIOLATION
    )


def test_apology_is_a_repair_event_and_does_not_erase_past_violation() -> None:
    item = candidate(
        EvidenceContext.DIRECT,
        RelationshipMeaning.REPAIR,
        is_apology=True,
    )
    assert (
        revalidate_relationship_candidate(
            candidate=item,
            source_text="最低だ",
            expected_character_id="alice",
        )
        is RelationshipMeaning.REPAIR
    )
    result = reduce_relationship_events(
        (
            event(
                "violation",
                RelationshipMeaning.BOUNDARY_VIOLATION,
                RelationshipSeverity.HIGH,
            ),
            event("apology", RelationshipMeaning.REPAIR),
        )
    )
    assert result.applied_event_ids == ("violation", "apology")
    assert result.trust < 30


def test_candidate_evidence_must_match_the_persisted_source_exactly() -> None:
    assert (
        revalidate_relationship_candidate(
            candidate=candidate(EvidenceContext.DIRECT),
            source_text="別の本文",
            expected_character_id="alice",
        )
        is None
    )
