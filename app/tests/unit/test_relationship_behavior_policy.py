from __future__ import annotations

import json

from local_llm_chat.domain.relationship_behavior import (
    DEFAULT_RELATIONSHIP_STYLE,
    RelationshipConflictResponse,
    RelationshipExpressiveness,
    RelationshipPace,
    RelationshipPriority,
    RelationshipStyle,
    build_minimal_behavior_envelope,
    is_registered_auto_apply_evidence,
    resolve_relationship_delta,
    should_auto_apply_relationship_event,
)
from local_llm_chat.domain.relationship_profile import (
    EvidenceContext,
    RelationshipMeaning,
    RelationshipMetrics,
    RelationshipSeverity,
)


def test_default_style_preserves_relationship_v1_base_delta() -> None:
    delta = resolve_relationship_delta(
        RelationshipMeaning.KEPT_COMMITMENT,
        RelationshipSeverity.LOW,
        DEFAULT_RELATIONSHIP_STYLE,
    )

    assert (delta.affinity, delta.trust, delta.tension) == (1, 2, -1)


def test_character_style_changes_weight_without_breaking_global_caps() -> None:
    commitment_focused = RelationshipStyle(
        attachment_pace=RelationshipPace.QUICK,
        expressiveness=RelationshipExpressiveness.RESERVED,
        priority=RelationshipPriority.COMMITMENTS,
        conflict_response=RelationshipConflictResponse.DIRECT,
        recovery_pace=RelationshipPace.STANDARD,
    )

    low = resolve_relationship_delta(
        RelationshipMeaning.KEPT_COMMITMENT,
        RelationshipSeverity.LOW,
        commitment_focused,
    )
    high = resolve_relationship_delta(
        RelationshipMeaning.KEPT_COMMITMENT,
        RelationshipSeverity.HIGH,
        commitment_focused,
    )

    assert (low.affinity, low.trust, low.tension) == (2, 3, -1)
    assert abs(high.affinity) <= 5
    assert abs(high.trust) <= 5
    assert abs(high.tension) <= 10


def test_recovery_style_changes_repair_gradually_without_erasing_harm() -> None:
    quick_recovery = RelationshipStyle(
        attachment_pace=RelationshipPace.STANDARD,
        expressiveness=RelationshipExpressiveness.EXPRESSIVE,
        priority=RelationshipPriority.WORDS,
        conflict_response=RelationshipConflictResponse.REPAIR_SEEKING,
        recovery_pace=RelationshipPace.QUICK,
    )

    delta = resolve_relationship_delta(
        RelationshipMeaning.REPAIR,
        RelationshipSeverity.LOW,
        quick_recovery,
    )

    assert (delta.affinity, delta.trust, delta.tension) == (1, 2, -3)


def test_only_low_direct_positive_events_auto_apply_and_saturate() -> None:
    assert should_auto_apply_relationship_event(
        meaning=RelationshipMeaning.KEPT_COMMITMENT,
        severity=RelationshipSeverity.LOW,
        evidence_context=EvidenceContext.DIRECT,
        current_affinity=50,
        prior_low_positive_count=0,
    )
    assert not should_auto_apply_relationship_event(
        meaning=RelationshipMeaning.CONFLICT,
        severity=RelationshipSeverity.LOW,
        evidence_context=EvidenceContext.DIRECT,
        current_affinity=50,
        prior_low_positive_count=0,
    )
    assert not should_auto_apply_relationship_event(
        meaning=RelationshipMeaning.POSITIVE_INTERACTION,
        severity=RelationshipSeverity.MEDIUM,
        evidence_context=EvidenceContext.DIRECT,
        current_affinity=50,
        prior_low_positive_count=0,
    )
    assert not should_auto_apply_relationship_event(
        meaning=RelationshipMeaning.POSITIVE_INTERACTION,
        severity=RelationshipSeverity.LOW,
        evidence_context=EvidenceContext.QUOTED,
        current_affinity=50,
        prior_low_positive_count=0,
    )
    assert should_auto_apply_relationship_event(
        meaning=RelationshipMeaning.POSITIVE_INTERACTION,
        severity=RelationshipSeverity.LOW,
        evidence_context=EvidenceContext.DIRECT,
        current_affinity=80,
        prior_low_positive_count=2,
    )
    assert not should_auto_apply_relationship_event(
        meaning=RelationshipMeaning.POSITIVE_INTERACTION,
        severity=RelationshipSeverity.LOW,
        evidence_context=EvidenceContext.DIRECT,
        current_affinity=80,
        prior_low_positive_count=3,
    )
    assert not should_auto_apply_relationship_event(
        meaning=RelationshipMeaning.POSITIVE_INTERACTION,
        severity=RelationshipSeverity.LOW,
        evidence_context=EvidenceContext.DIRECT,
        current_affinity=90,
        prior_low_positive_count=4,
    )


def test_minimal_lan_envelope_has_fixed_anonymous_fields() -> None:
    style = RelationshipStyle(
        attachment_pace=RelationshipPace.SLOW,
        expressiveness=RelationshipExpressiveness.RESERVED,
        priority=RelationshipPriority.BOUNDARIES,
        conflict_response=RelationshipConflictResponse.WITHDRAW,
        recovery_pace=RelationshipPace.SLOW,
    )
    payload = build_minimal_behavior_envelope(
        metrics=RelationshipMetrics(54, 38, 4, ()),
        turn_reception=RelationshipMeaning.KEPT_COMMITMENT,
        style=style,
    )

    assert payload == {
        "version": "relationship-behavior-v1",
        "affinity_band": "neutral",
        "trust_band": "developing",
        "tension_band": "calm",
        "turn_reception": "kept_commitment",
        "expressiveness": "reserved",
        "conflict_response": "withdraw",
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    for forbidden in (
        "character_id",
        "profile",
        "continuity",
        "event_id",
        "source",
        "reason",
        "マイルド",
    ):
        assert forbidden not in encoded


def test_distinct_character_styles_produce_distinct_behavior_envelopes() -> None:
    reserved = DEFAULT_RELATIONSHIP_STYLE
    expressive = RelationshipStyle(
        attachment_pace=RelationshipPace.QUICK,
        expressiveness=RelationshipExpressiveness.EXPRESSIVE,
        priority=RelationshipPriority.WORDS,
        conflict_response=RelationshipConflictResponse.REPAIR_SEEKING,
        recovery_pace=RelationshipPace.QUICK,
    )
    metrics = RelationshipMetrics(65, 45, 15, ())

    reserved_payload = build_minimal_behavior_envelope(
        metrics=metrics,
        turn_reception=RelationshipMeaning.POSITIVE_INTERACTION,
        style=reserved,
    )
    expressive_payload = build_minimal_behavior_envelope(
        metrics=metrics,
        turn_reception=RelationshipMeaning.POSITIVE_INTERACTION,
        style=expressive,
    )

    assert reserved_payload != expressive_payload
    assert reserved_payload["affinity_band"] == expressive_payload["affinity_band"]


def test_auto_apply_cadence_produces_five_changes_per_ten_meaningful_turns() -> None:
    decisions = [
        should_auto_apply_relationship_event(
            meaning=RelationshipMeaning.KEPT_COMMITMENT,
            severity=RelationshipSeverity.LOW,
            evidence_context=EvidenceContext.DIRECT,
            current_affinity=50,
            prior_low_positive_count=0,
            prior_meaningful_count=index,
        )
        for index in range(10)
    ]

    assert decisions.count(True) == 5


def test_auto_apply_requires_registered_semantic_evidence() -> None:
    assert is_registered_auto_apply_evidence(
        RelationshipMeaning.KEPT_COMMITMENT, "約束どおり戻ったよ"
    )
    assert is_registered_auto_apply_evidence(
        RelationshipMeaning.REPAIR, "さっきはごめん。繰り返さない"
    )
    assert not is_registered_auto_apply_evidence(
        RelationshipMeaning.KEPT_COMMITMENT, "今日は晴れている"
    )
