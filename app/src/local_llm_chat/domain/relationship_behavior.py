from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from local_llm_chat.domain.relationship_profile import (
    EvidenceContext,
    RelationshipMeaning,
    RelationshipMetrics,
    RelationshipSeverity,
)


MAX_AFFINITY_OR_TRUST_DELTA = 5
MAX_TENSION_DELTA = 10
BEHAVIOR_ENVELOPE_VERSION = "relationship-behavior-v1"


class RelationshipPace(StrEnum):
    SLOW = "slow"
    STANDARD = "standard"
    QUICK = "quick"


class RelationshipExpressiveness(StrEnum):
    RESERVED = "reserved"
    BALANCED = "balanced"
    EXPRESSIVE = "expressive"


class RelationshipPriority(StrEnum):
    WORDS = "words"
    COMMITMENTS = "commitments"
    BOUNDARIES = "boundaries"
    SHARED_EXPERIENCE = "shared_experience"


class RelationshipConflictResponse(StrEnum):
    WITHDRAW = "withdraw"
    DIRECT = "direct"
    REPAIR_SEEKING = "repair_seeking"


@dataclass(frozen=True, slots=True)
class RelationshipStyle:
    attachment_pace: RelationshipPace
    expressiveness: RelationshipExpressiveness
    priority: RelationshipPriority
    conflict_response: RelationshipConflictResponse
    recovery_pace: RelationshipPace


@dataclass(frozen=True, slots=True)
class RelationshipDelta:
    affinity: int
    trust: int
    tension: int


DEFAULT_RELATIONSHIP_STYLE = RelationshipStyle(
    attachment_pace=RelationshipPace.STANDARD,
    expressiveness=RelationshipExpressiveness.BALANCED,
    priority=RelationshipPriority.SHARED_EXPERIENCE,
    conflict_response=RelationshipConflictResponse.DIRECT,
    recovery_pace=RelationshipPace.STANDARD,
)

_SEVERITY_INDEX = {
    RelationshipSeverity.LOW: 0,
    RelationshipSeverity.MEDIUM: 1,
    RelationshipSeverity.HIGH: 2,
}

_BASE_DELTA_TABLE: dict[
    RelationshipMeaning, tuple[tuple[int, int, int], ...]
] = {
    RelationshipMeaning.POSITIVE_INTERACTION: ((1, 1, -1), (2, 2, -2), (5, 4, -4)),
    RelationshipMeaning.KEPT_COMMITMENT: ((1, 2, -1), (3, 4, -2), (5, 5, -4)),
    RelationshipMeaning.RESPECTED_BOUNDARY: ((1, 2, -2), (3, 4, -4), (5, 5, -6)),
    RelationshipMeaning.CONFLICT: ((-1, -1, 3), (-3, -2, 6), (-5, -5, 10)),
    RelationshipMeaning.BOUNDARY_VIOLATION: (
        (-2, -3, 4),
        (-4, -5, 7),
        (-5, -5, 10),
    ),
    RelationshipMeaning.REPEATED_BOUNDARY_VIOLATION: (
        (-3, -4, 6),
        (-5, -5, 9),
        (-5, -5, 10),
    ),
    RelationshipMeaning.REPAIR: ((1, 1, -2), (2, 2, -4), (3, 3, -6)),
    RelationshipMeaning.RELATIONSHIP_SET: ((0, 0, 0),) * 3,
    RelationshipMeaning.RELATIONSHIP_RETIRED: ((0, 0, 0),) * 3,
}

_AUTO_APPLY_MEANINGS = frozenset(
    {
        RelationshipMeaning.POSITIVE_INTERACTION,
        RelationshipMeaning.KEPT_COMMITMENT,
        RelationshipMeaning.RESPECTED_BOUNDARY,
        RelationshipMeaning.REPAIR,
    }
)
_AUTO_APPLY_EVIDENCE_MARKERS = {
    RelationshipMeaning.POSITIVE_INTERACTION: (
        "ありがとう",
        "うれしい",
        "嬉しい",
        "助かった",
        "感謝",
    ),
    RelationshipMeaning.KEPT_COMMITMENT: (
        "約束どおり",
        "約束通り",
        "言ったとおり",
        "言った通り",
    ),
    RelationshipMeaning.RESPECTED_BOUNDARY: (
        "嫌ならやめ",
        "無理しなくていい",
        "無理しないで",
        "境界を尊重",
    ),
    RelationshipMeaning.REPAIR: (
        "ごめん",
        "すみません",
        "申し訳",
        "謝る",
        "繰り返さない",
    ),
}


def base_relationship_delta(
    meaning: RelationshipMeaning, severity: RelationshipSeverity
) -> RelationshipDelta:
    raw = _BASE_DELTA_TABLE[meaning][_SEVERITY_INDEX[severity]]
    return RelationshipDelta(*raw)


def resolve_relationship_delta(
    meaning: RelationshipMeaning,
    severity: RelationshipSeverity,
    style: RelationshipStyle,
) -> RelationshipDelta:
    base = base_relationship_delta(meaning, severity)
    affinity = base.affinity
    trust = base.trust
    tension = base.tension

    if affinity > 0 and style.attachment_pace is RelationshipPace.QUICK:
        affinity += 1
    if (
        meaning is RelationshipMeaning.POSITIVE_INTERACTION
        and style.priority is RelationshipPriority.WORDS
    ):
        affinity += 1
    if (
        meaning is RelationshipMeaning.KEPT_COMMITMENT
        and style.priority is RelationshipPriority.COMMITMENTS
    ):
        trust += 1
    if (
        meaning is RelationshipMeaning.RESPECTED_BOUNDARY
        and style.priority is RelationshipPriority.BOUNDARIES
    ):
        trust += 1
    if meaning is RelationshipMeaning.REPAIR:
        if style.recovery_pace is RelationshipPace.QUICK:
            trust += 1
            tension -= 1
        elif style.recovery_pace is RelationshipPace.SLOW:
            trust = max(0, trust - 1)
            tension = min(0, tension + 1)

    return RelationshipDelta(
        affinity=_bounded(
            affinity, -MAX_AFFINITY_OR_TRUST_DELTA, MAX_AFFINITY_OR_TRUST_DELTA
        ),
        trust=_bounded(
            trust, -MAX_AFFINITY_OR_TRUST_DELTA, MAX_AFFINITY_OR_TRUST_DELTA
        ),
        tension=_bounded(tension, -MAX_TENSION_DELTA, MAX_TENSION_DELTA),
    )


def should_auto_apply_relationship_event(
    *,
    meaning: RelationshipMeaning,
    severity: RelationshipSeverity,
    evidence_context: EvidenceContext,
    current_affinity: int,
    prior_low_positive_count: int,
    prior_meaningful_count: int = 0,
) -> bool:
    if (
        meaning not in _AUTO_APPLY_MEANINGS
        or severity is not RelationshipSeverity.LOW
        or evidence_context is not EvidenceContext.DIRECT
    ):
        return False
    if prior_meaningful_count % 2 != 0:
        return False
    if meaning is not RelationshipMeaning.POSITIVE_INTERACTION:
        return True
    if current_affinity >= 90:
        return False
    if current_affinity >= 80:
        return prior_low_positive_count % 2 == 0
    return True


def is_registered_auto_apply_evidence(
    meaning: RelationshipMeaning, evidence: str
) -> bool:
    normalized = " ".join(evidence.casefold().split())
    markers = _AUTO_APPLY_EVIDENCE_MARKERS.get(meaning, ())
    return any(marker in normalized for marker in markers)


def build_minimal_behavior_envelope(
    *,
    metrics: RelationshipMetrics,
    turn_reception: RelationshipMeaning | None,
    style: RelationshipStyle,
) -> dict[str, str]:
    return {
        "version": BEHAVIOR_ENVELOPE_VERSION,
        "affinity_band": _band(metrics.affinity, (35, 65, 85), ("cool", "neutral", "warm", "close")),
        "trust_band": _band(
            metrics.trust, (35, 70), ("cautious", "developing", "reliable")
        ),
        "tension_band": _band(
            metrics.tension, (25, 60), ("calm", "strained", "guarded")
        ),
        "turn_reception": turn_reception.value if turn_reception is not None else "none",
        "expressiveness": style.expressiveness.value,
        "conflict_response": style.conflict_response.value,
    }


def _band(
    value: int, thresholds: tuple[int, ...], labels: tuple[str, ...]
) -> str:
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return labels[index]
    return labels[-1]


def _bounded(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
