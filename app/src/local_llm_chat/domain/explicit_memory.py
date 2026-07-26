from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


MAX_EXPLICIT_MEMORY_VALUE_CHARACTERS = 200


@dataclass(frozen=True, slots=True)
class ExplicitMemoryEvent:
    id: str
    request_id: str
    conversation_id: str
    branch_id: str
    source_message_id: str
    character_id: str
    value: str
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class ExplicitMemoryDecision:
    id: str
    conversation_id: str
    target_event_id: str
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class ExplicitMemoryReviewItem:
    event_id: str
    source_message_id: str
    character_id: str
    value: str
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class ExplicitMemoryProjectionQuery:
    conversation_id: str
    branch_lineage: tuple[str, ...]
    character_id: str
    visible_source_message_ids: frozenset[str]


class ExplicitMemoryLedger:
    """Projects user-confirmed memories without crossing branch or character scope."""

    def __init__(
        self,
        events: tuple[ExplicitMemoryEvent, ...],
        decisions: tuple[ExplicitMemoryDecision, ...] = (),
    ) -> None:
        self._events = tuple(sorted(events, key=lambda item: (item.recorded_at, item.id)))
        self._decisions = decisions
        self._validate()

    def project(
        self, query: ExplicitMemoryProjectionQuery
    ) -> tuple[ExplicitMemoryReviewItem, ...]:
        if (
            not query.conversation_id.strip()
            or not query.branch_lineage
            or not query.character_id.strip()
        ):
            raise ValueError("explicit memory projection scope is required")
        undone_ids = frozenset(item.target_event_id for item in self._decisions)
        projected = (
            ExplicitMemoryReviewItem(
                event_id=item.id,
                source_message_id=item.source_message_id,
                character_id=item.character_id,
                value=item.value,
                recorded_at=item.recorded_at,
            )
            for item in self._events
            if item.id not in undone_ids
            and item.conversation_id == query.conversation_id
            and item.branch_id in query.branch_lineage
            and item.source_message_id in query.visible_source_message_ids
            and item.character_id == query.character_id
        )
        return tuple(
            sorted(projected, key=lambda item: (-item.recorded_at.timestamp(), item.event_id))
        )

    def _validate(self) -> None:
        event_ids: set[str] = set()
        request_ids: set[str] = set()
        for event in self._events:
            required = (
                event.id,
                event.request_id,
                event.conversation_id,
                event.branch_id,
                event.source_message_id,
                event.character_id,
                event.value,
            )
            if any(not value.strip() for value in required):
                raise ValueError("explicit memory event fields cannot be blank")
            if len(event.value) > MAX_EXPLICIT_MEMORY_VALUE_CHARACTERS:
                raise ValueError("explicit memory value is too long")
            if event.recorded_at.tzinfo is None:
                raise ValueError("explicit memory timestamp must be timezone-aware")
            if event.id in event_ids or event.request_id in request_ids:
                raise ValueError("explicit memory event identity must be unique")
            event_ids.add(event.id)
            request_ids.add(event.request_id)

        decision_ids: set[str] = set()
        target_ids: set[str] = set()
        for decision in self._decisions:
            if (
                not decision.id.strip()
                or not decision.conversation_id.strip()
                or not decision.target_event_id.strip()
            ):
                raise ValueError("explicit memory decision fields cannot be blank")
            if decision.recorded_at.tzinfo is None:
                raise ValueError("explicit memory decision timestamp must be timezone-aware")
            if (
                decision.id in decision_ids
                or decision.target_event_id in target_ids
            ):
                raise ValueError("explicit memory can be undone only once")
            if decision.target_event_id not in event_ids:
                raise ValueError("explicit memory decision target does not exist")
            decision_ids.add(decision.id)
            target_ids.add(decision.target_event_id)
