from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from local_llm_chat.domain.errors import ValidationError


RELATIONSHIP_POLICY_VERSION = "relationship-v1"
INITIAL_AFFINITY = 50
INITIAL_TRUST = 30
INITIAL_TENSION = 0
MAX_AFFINITY_OR_TRUST_DELTA = 5
MAX_TENSION_DELTA = 10


class ProfileOrigin(StrEnum):
    USER_ASSERTED = "user_asserted"
    AI_AUTO_SAVED = "ai_auto_saved"
    AI_PROPOSED = "ai_proposed"


class ProfileApproval(StrEnum):
    AUTO_SAVED = "auto_saved"
    CONFIRMED = "confirmed"
    PENDING_CONFIRMATION = "pending_confirmation"
    REJECTED = "rejected"
    UNDONE = "undone"


class ProfileUsageState(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


class ProfileScope(StrEnum):
    PROFILE_ONLY = "profile_only"
    CONTINUITY = "continuity"
    SELECTED_CHARACTERS = "selected_characters"
    CONTINUITY_CAST = "continuity_cast"


class LedgerActor(StrEnum):
    USER = "user"
    AI = "ai"
    SYSTEM = "system"


class RelationshipMeaning(StrEnum):
    POSITIVE_INTERACTION = "positive_interaction"
    KEPT_COMMITMENT = "kept_commitment"
    RESPECTED_BOUNDARY = "respected_boundary"
    CONFLICT = "conflict"
    BOUNDARY_VIOLATION = "boundary_violation"
    REPEATED_BOUNDARY_VIOLATION = "repeated_boundary_violation"
    REPAIR = "repair"
    RELATIONSHIP_SET = "relationship_set"
    RELATIONSHIP_RETIRED = "relationship_retired"
    RESET = "reset"


class RelationshipSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RelationshipApproval(StrEnum):
    AUTO_APPLIED = "auto_applied"
    CONFIRMED = "confirmed"
    PENDING_CONFIRMATION = "pending_confirmation"
    REJECTED = "rejected"
    UNDONE = "undone"


class EvidenceContext(StrEnum):
    DIRECT = "direct"
    QUOTED = "quoted"
    HYPOTHETICAL = "hypothetical"
    ROLEPLAY = "roleplay"
    NARRATIVE = "narrative"
    THIRD_PARTY = "third_party"
    UNKNOWN = "unknown"


class RelationshipDirection(StrEnum):
    SYMMETRIC = "symmetric"
    DIRECTED = "directed"


class RelationshipAssignmentState(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    HISTORICAL = "historical"
    DISABLED = "disabled"


class RelationshipInterpretationState(StrEnum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    RECOMPUTING = "recomputing"


@dataclass(frozen=True, slots=True)
class Continuity:
    id: str
    display_name: str
    user_profile_id: str
    created_at: datetime
    archived_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class UserProfile:
    id: str
    display_name: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ProfileEvent:
    id: str
    user_profile_id: str
    item_kind: str
    item_name: str
    value: str
    origin: ProfileOrigin
    approval: ProfileApproval
    scope: ProfileScope
    known_by_character_ids: tuple[str, ...]
    source_conversation_id: str | None
    source_branch_id: str | None
    source_message_id: str | None
    manual_operation_id: str | None
    supersedes_event_id: str | None
    effective_at: datetime
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class ProfileItem:
    event_id: str
    user_profile_id: str
    item_kind: str
    item_name: str
    value: str
    origin: ProfileOrigin
    approval: ProfileApproval
    usage: ProfileUsageState
    scope: ProfileScope
    known_by_character_ids: tuple[str, ...]
    source_conversation_id: str | None
    source_message_id: str | None
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class ProfilePurgeReceipt:
    request_id: str
    user_profile_id: str
    deleted_event_count: int
    deleted_relationship_event_count: int
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class RelationshipDefinition:
    id: str
    category: str
    group_name: str
    display_name: str
    direction: RelationshipDirection
    role_a: str | None
    role_b: str | None
    caution_tags: tuple[str, ...]
    archived_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RelationshipEvent:
    id: str
    continuity_id: str
    user_profile_id: str
    character_id: str
    source_conversation_id: str | None
    source_branch_id: str | None
    source_message_id: str | None
    meaning: RelationshipMeaning
    severity: RelationshipSeverity
    evidence_context: EvidenceContext
    evidence_start: int | None
    evidence_end: int | None
    reason: str
    approval: RelationshipApproval
    policy_version: str
    known_by_character_ids: tuple[str, ...]
    relationship_definition_id: str | None
    assignment_state: RelationshipAssignmentState | None
    role: str | None
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class RelationshipMetrics:
    affinity: int = INITIAL_AFFINITY
    trust: int = INITIAL_TRUST
    tension: int = INITIAL_TENSION
    applied_event_ids: tuple[str, ...] = ()
    policy_version: str = RELATIONSHIP_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class RelationshipInterpretation:
    id: str
    continuity_id: str
    user_profile_id: str
    character_id: str
    character_version_id: str
    relationship_definition_ids: tuple[str, ...]
    summary: str
    evidence_event_ids: tuple[str, ...]
    state: RelationshipInterpretationState
    generated_at: datetime


@dataclass(frozen=True, slots=True)
class RelationshipCandidate:
    event_id: str
    character_id: str
    meaning: RelationshipMeaning
    severity: RelationshipSeverity
    evidence_context: EvidenceContext
    evidence_start: int
    evidence_end: int
    evidence_text: str
    conflict_has_reason: bool = False
    roleplay_active: bool = False
    boundary_previously_set: bool = False
    is_apology: bool = False
    is_agreed_repair: bool = False


@dataclass(frozen=True, slots=True)
class RelationshipContext:
    continuity_id: str
    user_profile_id: str
    character_ids: tuple[str, ...]
    profile_items: tuple[ProfileItem, ...]
    metrics: tuple[tuple[str, RelationshipMetrics], ...]
    interpretations: tuple[RelationshipInterpretation, ...]
    recent_events: tuple[RelationshipEvent, ...]


_SEVERITY_INDEX = {
    RelationshipSeverity.LOW: 0,
    RelationshipSeverity.MEDIUM: 1,
    RelationshipSeverity.HIGH: 2,
}

_DELTA_TABLE: dict[
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


def reduce_relationship_events(
    events: tuple[RelationshipEvent, ...],
) -> RelationshipMetrics:
    affinity = INITIAL_AFFINITY
    trust = INITIAL_TRUST
    tension = INITIAL_TENSION
    applied: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event.id in seen:
            continue
        seen.add(event.id)
        if event.policy_version != RELATIONSHIP_POLICY_VERSION:
            raise ValidationError("unknown relationship reducer policy version")
        if event.approval not in {
            RelationshipApproval.AUTO_APPLIED,
            RelationshipApproval.CONFIRMED,
        }:
            continue
        if event.meaning is RelationshipMeaning.RESET:
            affinity = INITIAL_AFFINITY
            trust = INITIAL_TRUST
            tension = INITIAL_TENSION
            applied.append(event.id)
            continue
        delta = _DELTA_TABLE[event.meaning][_SEVERITY_INDEX[event.severity]]
        affinity_delta = _bounded(
            delta[0], -MAX_AFFINITY_OR_TRUST_DELTA, MAX_AFFINITY_OR_TRUST_DELTA
        )
        trust_delta = _bounded(
            delta[1], -MAX_AFFINITY_OR_TRUST_DELTA, MAX_AFFINITY_OR_TRUST_DELTA
        )
        tension_delta = _bounded(
            delta[2], -MAX_TENSION_DELTA, MAX_TENSION_DELTA
        )
        affinity = _bounded(affinity + affinity_delta, 0, 100)
        trust = _bounded(trust + trust_delta, 0, 100)
        tension = _bounded(tension + tension_delta, 0, 100)
        applied.append(event.id)
    return RelationshipMetrics(
        affinity=affinity,
        trust=trust,
        tension=tension,
        applied_event_ids=tuple(applied),
    )


def revalidate_relationship_candidate(
    *,
    candidate: RelationshipCandidate,
    source_text: str,
    expected_character_id: str,
) -> RelationshipMeaning | None:
    if (
        candidate.character_id != expected_character_id
        or candidate.evidence_start < 0
        or candidate.evidence_end <= candidate.evidence_start
        or candidate.evidence_end > len(source_text)
        or source_text[candidate.evidence_start : candidate.evidence_end]
        != candidate.evidence_text
    ):
        return None
    if candidate.evidence_context in {
        EvidenceContext.QUOTED,
        EvidenceContext.HYPOTHETICAL,
        EvidenceContext.NARRATIVE,
        EvidenceContext.THIRD_PARTY,
        EvidenceContext.UNKNOWN,
    }:
        return None
    if candidate.evidence_context is EvidenceContext.ROLEPLAY:
        return (
            RelationshipMeaning.CONFLICT
            if candidate.roleplay_active and candidate.conflict_has_reason
            else None
        )
    if candidate.meaning is RelationshipMeaning.REPEATED_BOUNDARY_VIOLATION:
        return (
            candidate.meaning
            if candidate.boundary_previously_set
            and not candidate.conflict_has_reason
            and not candidate.roleplay_active
            else None
        )
    if candidate.meaning is RelationshipMeaning.BOUNDARY_VIOLATION:
        return (
            candidate.meaning
            if not candidate.conflict_has_reason and not candidate.roleplay_active
            else None
        )
    if candidate.meaning is RelationshipMeaning.REPAIR:
        return (
            candidate.meaning
            if candidate.is_apology or candidate.is_agreed_repair
            else None
        )
    return candidate.meaning


def _bounded(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
