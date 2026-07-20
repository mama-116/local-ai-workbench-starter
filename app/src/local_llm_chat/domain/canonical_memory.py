from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from local_llm_chat.domain.states import (
    MemoryApprovalState,
    MemoryCardinality,
    MemoryFactState,
    MemoryKind,
)


_PERSISTED_APPROVAL_STATES = frozenset(
    {MemoryApprovalState.AUTO_SAVED, MemoryApprovalState.CONFIRMED}
)


def transitive_superseded_event_ids(
    visible_events: tuple[CanonicalMemoryEvent, ...],
    all_events: tuple[CanonicalMemoryEvent, ...],
) -> frozenset[str]:
    events_by_id = {item.id: item for item in all_events}
    superseded: set[str] = set()
    pending_targets = [
        item.supersedes_event_id
        for item in visible_events
        if item.supersedes_event_id is not None
    ]
    while pending_targets:
        target_id = pending_targets.pop()
        if target_id in superseded:
            continue
        superseded.add(target_id)
        target = events_by_id.get(target_id)
        if target is not None and target.supersedes_event_id is not None:
            pending_targets.append(target.supersedes_event_id)
    return frozenset(superseded)


@dataclass(frozen=True, slots=True)
class CanonicalMemoryEvent:
    id: str
    conversation_id: str
    branch_id: str
    subject_id: str
    kind: MemoryKind
    slot: str
    value: str
    cardinality: MemoryCardinality
    approval: MemoryApprovalState
    source_message_id: str
    known_by_character_ids: frozenset[str]
    supersedes_event_id: str | None
    effective_at: datetime
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryApprovalDecision:
    id: str
    target_event_id: str
    state: MemoryApprovalState
    source_message_id: str
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryProjectionQuery:
    conversation_id: str
    branch_lineage: tuple[str, ...]
    speaker_character_id: str
    current_source_message_id: str | None = None
    include_historical: bool = False
    visible_source_message_ids: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class CanonicalMemoryFact:
    event_id: str
    subject_id: str
    kind: MemoryKind
    slot: str
    value: str
    state: MemoryFactState
    source_message_id: str
    effective_at: datetime


@dataclass(frozen=True, slots=True)
class CanonicalMemoryReviewItem:
    event_id: str
    subject_id: str
    kind: MemoryKind
    slot: str
    value: str
    approval: MemoryApprovalState
    source_message_id: str
    known_by_character_ids: frozenset[str]
    effective_at: datetime
    is_active: bool


class CanonicalMemoryLedger:
    """Projects immutable, source-backed events without crossing knowledge or branches."""

    def __init__(
        self,
        events: tuple[CanonicalMemoryEvent, ...],
        decisions: tuple[MemoryApprovalDecision, ...] = (),
    ) -> None:
        self._events = tuple(sorted(events, key=lambda item: (item.recorded_at, item.id)))
        # Decisions are an append-only stream. Preserve repository sequence even if
        # the wall clock moves backwards; timestamps are evidence, not ordering keys.
        self._decisions = decisions
        self._validate_events()
        self._decision_states = self._validate_and_reduce_decisions()

    def project(self, query: MemoryProjectionQuery) -> tuple[CanonicalMemoryFact, ...]:
        self._validate_query(query)
        visible = tuple(
            item
            for item in self._events
            if item.conversation_id == query.conversation_id
            and item.branch_id in query.branch_lineage
            and (
                query.visible_source_message_ids is None
                or item.source_message_id in query.visible_source_message_ids
            )
            and self._is_known_by(item, query.speaker_character_id)
            and self._is_usable(
                item,
                self._decision_states.get(item.id, item.approval),
                query.current_source_message_id,
            )
        )
        superseded_ids = transitive_superseded_event_ids(visible, self._events)
        facts = tuple(
            CanonicalMemoryFact(
                event_id=item.id,
                subject_id=item.subject_id,
                kind=item.kind,
                slot=item.slot,
                value=item.value,
                state=(
                    MemoryFactState.HISTORICAL
                    if item.id in superseded_ids
                    else MemoryFactState.ACTIVE
                ),
                source_message_id=item.source_message_id,
                effective_at=item.effective_at,
            )
            for item in visible
            if query.include_historical or item.id not in superseded_ids
        )
        return tuple(sorted(facts, key=self._fact_sort_key))

    def review(
        self,
        conversation_id: str,
        branch_lineage: tuple[str, ...],
        visible_source_message_ids: frozenset[str],
    ) -> tuple[CanonicalMemoryReviewItem, ...]:
        if not conversation_id.strip() or not branch_lineage:
            raise ValueError("memory review conversation and branch lineage are required")
        visible = tuple(
            item
            for item in self._events
            if item.conversation_id == conversation_id
            and item.branch_id in branch_lineage
            and item.source_message_id in visible_source_message_ids
            and self._decision_states.get(item.id, item.approval)
            not in {MemoryApprovalState.REJECTED, MemoryApprovalState.UNDONE}
        )
        superseded_ids = transitive_superseded_event_ids(visible, self._events)
        items = tuple(
            CanonicalMemoryReviewItem(
                event_id=item.id,
                subject_id=item.subject_id,
                kind=item.kind,
                slot=item.slot,
                value=item.value,
                approval=self._decision_states.get(item.id, item.approval),
                source_message_id=item.source_message_id,
                known_by_character_ids=item.known_by_character_ids,
                effective_at=item.effective_at,
                is_active=item.id not in superseded_ids,
            )
            for item in visible
        )
        return tuple(
            sorted(
                items,
                key=lambda item: (
                    0 if item.kind is MemoryKind.SAFETY_CONSTRAINT else 1,
                    -item.effective_at.timestamp(),
                    item.event_id,
                ),
            )
        )

    def _validate_events(self) -> None:
        by_id: dict[str, CanonicalMemoryEvent] = {}
        for item in self._events:
            self._validate_required_fields(item)
            if item.approval in {
                MemoryApprovalState.REJECTED,
                MemoryApprovalState.UNDONE,
            }:
                raise ValueError(
                    "rejection and undo must be append-only approval decisions"
                )
            if item.id in by_id:
                raise ValueError(f"duplicate canonical memory event id: {item.id}")
            by_id[item.id] = item

        for item in self._events:
            if item.supersedes_event_id is None:
                continue
            target = by_id.get(item.supersedes_event_id)
            if target is None:
                raise ValueError("superseded canonical memory event does not exist")
            comparable = (
                target.conversation_id,
                target.subject_id,
                target.kind,
                target.slot,
                target.cardinality,
            )
            replacement = (
                item.conversation_id,
                item.subject_id,
                item.kind,
                item.slot,
                item.cardinality,
            )
            if replacement != comparable:
                raise ValueError(
                    "replacement must keep the same subject, kind, slot, and cardinality"
                )
            if item.recorded_at < target.recorded_at:
                raise ValueError("replacement cannot be recorded before its target")
            if item.id == target.id:
                raise ValueError("canonical memory event cannot supersede itself")

    def _validate_and_reduce_decisions(self) -> dict[str, MemoryApprovalState]:
        events_by_id = {item.id: item for item in self._events}
        known_ids = set(events_by_id)
        states = {item.id: item.approval for item in self._events}
        for decision in self._decisions:
            if not decision.id.strip() or not decision.source_message_id.strip():
                raise ValueError("memory approval decision fields cannot be blank")
            if decision.recorded_at.tzinfo is None:
                raise ValueError("memory approval decision timestamp must be timezone-aware")
            if decision.id in known_ids:
                raise ValueError(f"duplicate canonical memory ledger id: {decision.id}")
            known_ids.add(decision.id)
            target = events_by_id.get(decision.target_event_id)
            if target is None:
                raise ValueError("memory approval decision target does not exist")
            if decision.recorded_at < target.recorded_at:
                raise ValueError("memory approval decision cannot precede its target")
            current = states[target.id]
            allowed = {
                MemoryApprovalState.PENDING_CONFIRMATION: {
                    MemoryApprovalState.CONFIRMED,
                    MemoryApprovalState.REJECTED,
                },
                MemoryApprovalState.AUTO_SAVED: {MemoryApprovalState.UNDONE},
                MemoryApprovalState.CONFIRMED: {MemoryApprovalState.UNDONE},
            }.get(current, set())
            if decision.state not in allowed:
                raise ValueError(
                    f"invalid memory approval transition: {current} -> {decision.state}"
                )
            states[target.id] = decision.state
        return states

    @staticmethod
    def _validate_required_fields(item: CanonicalMemoryEvent) -> None:
        values = (
            item.id,
            item.conversation_id,
            item.branch_id,
            item.subject_id,
            item.slot,
            item.value,
            item.source_message_id,
        )
        if any(not value.strip() for value in values):
            raise ValueError("canonical memory event fields cannot be blank")
        if item.effective_at.tzinfo is None or item.recorded_at.tzinfo is None:
            raise ValueError("canonical memory event timestamps must be timezone-aware")
        if any(not character_id.strip() for character_id in item.known_by_character_ids):
            raise ValueError("knowledge scope character ids cannot be blank")

    @staticmethod
    def _validate_query(query: MemoryProjectionQuery) -> None:
        if not query.conversation_id.strip() or not query.speaker_character_id.strip():
            raise ValueError("memory projection conversation and speaker are required")
        if not query.branch_lineage or any(
            not branch_id.strip() for branch_id in query.branch_lineage
        ):
            raise ValueError("memory projection requires a non-empty branch lineage")
        if len(query.branch_lineage) != len(set(query.branch_lineage)):
            raise ValueError("memory projection branch lineage cannot contain duplicates")
        if query.visible_source_message_ids is not None and any(
            not message_id.strip() for message_id in query.visible_source_message_ids
        ):
            raise ValueError("memory projection source message ids cannot be blank")

    @staticmethod
    def _is_known_by(item: CanonicalMemoryEvent, speaker_id: str) -> bool:
        return (
            not item.known_by_character_ids
            or speaker_id in item.known_by_character_ids
        )

    @staticmethod
    def _is_usable(
        item: CanonicalMemoryEvent,
        approval: MemoryApprovalState,
        current_source_message_id: str | None,
    ) -> bool:
        if approval in _PERSISTED_APPROVAL_STATES:
            return True
        return (
            approval is MemoryApprovalState.PENDING_CONFIRMATION
            and current_source_message_id is not None
            and item.source_message_id == current_source_message_id
        )

    @staticmethod
    def _fact_sort_key(fact: CanonicalMemoryFact) -> tuple[int, int, float, str]:
        kind_priority = 0 if fact.kind is MemoryKind.SAFETY_CONSTRAINT else 1
        state_priority = 0 if fact.state is MemoryFactState.ACTIVE else 1
        return (kind_priority, state_priority, -fact.effective_at.timestamp(), fact.event_id)
