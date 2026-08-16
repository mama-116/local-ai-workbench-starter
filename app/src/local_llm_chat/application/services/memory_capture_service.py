from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid5

from local_llm_chat.application.services.memory_candidate_service import (
    MemoryCandidateService,
)
from local_llm_chat.domain.canonical_memory import CanonicalMemoryEvent
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.memory_candidates import (
    MemoryCandidate,
    MemoryCandidateRequest,
)
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.states import (
    MemoryApprovalState,
    MemoryCandidateDisposition,
    MemoryCardinality,
    MemoryKind,
)


_MEMORY_EVENT_NAMESPACE = UUID("8a37fbc1-ca67-4ecf-a54a-f54cc0f31475")
_PROGRESSIVE_MEMORY_NAMESPACE = UUID("c19f1db5-dd4e-4324-b61e-1cb928ca61cb")
_FOOD_DETAIL_PATTERN = re.compile(
    r"^\s*(?:私は|自分は)\s*(?P<value>[^。！？!?\r\n]{1,180}?(?:アイス|味))"
    r"\s*[。！？!?]?\s*$"
)
_FOOD_LATEST_CORRECTION_PATTERN = re.compile(
    r"^\s*(?:でも)?\s*今は\s*(?P<value>[^。！？!?\r\n]{1,180}?(?:アイス|味))"
    r"\s*が一番好き\s*[。！？!?]?\s*$"
)
_FOOD_CONDITION_STATEMENT_PATTERN = re.compile(
    r"^\s*(?P<value>(?:ちょっと|少し|やや|かなり|完全に)?\s*"
    r"(?:溶け(?:かけ|た|ている)|冷え(?:かけ|た|ている)|温め(?:た|ている)"
    r"|焼き(?:たて|かけ)|凍(?:った|らせた)|熱々|あつあつ|ひえひえ"
    r"|冷たい|温かい|ぬるい|常温)(?:状態|もの|方|の)?)"
    r"\s*が好き\s*[。！？!?]?\s*$"
)
_MULTIPLE_FOOD_OPTIONS_PATTERN = re.compile(
    r"(?P<first>[^。！？!?\r\n、]{1,80}味)\s*と\s*"
    r"(?P<second>[^。！？!?\r\n、]{1,80}味)"
)


@dataclass(frozen=True, slots=True)
class MemoryCaptureRequest:
    conversation_id: str
    branch_id: str
    source_message_id: str
    model_name: str
    author_subject_id: str
    allowed_subject_ids: frozenset[str]
    allowed_knowledge_character_ids: frozenset[str]
    known_by_character_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class MemoryCaptureResult:
    candidates: tuple[MemoryCandidate, ...]
    persisted_event_ids: tuple[str, ...]
    extractor_unavailable: bool = False


class MemoryCaptureState(StrEnum):
    PROCESSING = "processing"
    SAVED = "saved"
    NO_CANDIDATES = "no_candidates"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MemoryCaptureUpdate:
    conversation_id: str
    branch_id: str
    source_message_id: str
    state: MemoryCaptureState
    persisted_event_ids: tuple[str, ...] = ()


MemoryCaptureSubscriber = Callable[[MemoryCaptureUpdate], Awaitable[None]]


class MemoryCaptureRunner(Protocol):
    async def capture(self, request: MemoryCaptureRequest) -> MemoryCaptureResult: ...


class MemoryCaptureScheduler(Protocol):
    async def request_capture(
        self, request: MemoryCaptureRequest, run_id: str
    ) -> None: ...


class MemoryCaptureService:
    """Captures source-backed candidates from one persisted user message."""

    def __init__(
        self, repository: AppRepository, candidate_service: MemoryCandidateService
    ) -> None:
        self._repository = repository
        self._candidate_service = candidate_service

    async def capture(self, request: MemoryCaptureRequest) -> MemoryCaptureResult:
        if not request.known_by_character_ids:
            raise ValidationError(
                "automatic memory requires at least one listener character"
            )
        if not request.known_by_character_ids.issubset(
            request.allowed_knowledge_character_ids
        ):
            raise ValidationError(
                "automatic memory listener is outside the allowed knowledge scope"
            )
        source = await self._repository.get_memory_source_message(
            request.conversation_id,
            request.branch_id,
            request.source_message_id,
        )
        progressive_event = await self._progressive_preference_event(
            request, source.content, source.created_at
        )
        if progressive_event is not None:
            await self._repository.append_canonical_memory_event(progressive_event)
            return MemoryCaptureResult((), (progressive_event.id,))
        generation = await self._candidate_service.generate_with_status(
            MemoryCandidateRequest(
                conversation_id=request.conversation_id,
                branch_id=request.branch_id,
                source_message_id=request.source_message_id,
                model_name=request.model_name,
                content=source.content,
                author_subject_id=request.author_subject_id,
                allowed_subject_ids=request.allowed_subject_ids,
                allowed_knowledge_character_ids=(
                    request.allowed_knowledge_character_ids
                ),
                known_by_character_ids=request.known_by_character_ids,
            )
        )
        candidates = generation.candidates
        events_by_id: dict[str, CanonicalMemoryEvent] = {}
        for candidate in candidates:
            event = self._to_event(candidate, source.created_at)
            if event is not None:
                events_by_id.setdefault(event.id, event)
        events = tuple(events_by_id.values())
        persisted_event_ids = await self._repository.append_captured_memory_events(
            events
        )
        return MemoryCaptureResult(
            candidates, persisted_event_ids, generation.extractor_unavailable
        )

    async def _progressive_preference_event(
        self,
        request: MemoryCaptureRequest,
        content: str,
        source_created_at: datetime,
    ) -> CanonicalMemoryEvent | None:
        detail_match = _FOOD_DETAIL_PATTERN.fullmatch(content)
        correction_match = _FOOD_LATEST_CORRECTION_PATTERN.fullmatch(content)
        condition_match = _FOOD_CONDITION_STATEMENT_PATTERN.fullmatch(content)
        if (
            detail_match is None
            and correction_match is None
            and condition_match is None
        ):
            return None

        review_items = await self._repository.list_canonical_memory_review_items(
            request.conversation_id, request.branch_id
        )
        active_foods = tuple(
            item
            for item in review_items
            if item.is_active
            and item.subject_id == request.author_subject_id
            and item.kind is MemoryKind.PREFERENCE
            and item.slot == "liked_food"
            and item.approval
            in {MemoryApprovalState.AUTO_SAVED, MemoryApprovalState.CONFIRMED}
        )
        if any(
            item.known_by_character_ids != request.known_by_character_ids
            for item in active_foods
        ):
            # A replacement is global, while knowledge is per character.
            # Merging different listener snapshots would leak or hide updates.
            return None
        if not active_foods:
            if condition_match is None:
                return None
            recent = await self._repository.list_memory_source_messages(
                request.conversation_id,
                request.branch_id,
                request.source_message_id,
                8,
            )
            ambiguous_sources = tuple(
                message.id
                for message in recent[:-1]
                if _MULTIPLE_FOOD_OPTIONS_PATTERN.search(message.content)
            )
            if not ambiguous_sources:
                return None
            condition = condition_match.group("value").strip()
            source_ids = (*ambiguous_sources, request.source_message_id)
            identity = json.dumps(
                {
                    "conversation_id": request.conversation_id,
                    "branch_id": request.branch_id,
                    "sources": source_ids,
                    "subject_id": request.author_subject_id,
                    "slot": "liked_food_condition",
                    "value": condition,
                    "approval": MemoryApprovalState.PENDING_CONFIRMATION.value,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            return CanonicalMemoryEvent(
                id=str(uuid5(_PROGRESSIVE_MEMORY_NAMESPACE, identity)),
                conversation_id=request.conversation_id,
                branch_id=request.branch_id,
                subject_id=request.author_subject_id,
                kind=MemoryKind.PREFERENCE,
                slot="liked_food_condition",
                value=condition,
                cardinality=MemoryCardinality.MULTIPLE,
                approval=MemoryApprovalState.PENDING_CONFIRMATION,
                source_message_id=request.source_message_id,
                known_by_character_ids=request.known_by_character_ids,
                supersedes_event_id=None,
                effective_at=source_created_at,
                recorded_at=source_created_at,
                source_message_ids=source_ids,
            )

        sources: tuple[str, ...]
        if correction_match is not None:
            if len(active_foods) != 1:
                return None
            target = active_foods[0]
            value = correction_match.group("value").strip()
            if value == target.value:
                return None
            approval = MemoryApprovalState.AUTO_SAVED
            slot = target.slot
            supersedes_event_id = target.event_id
            # An explicit latest preference is independently grounded in the
            # correction turn. The superseded event retains its own evidence.
            sources = (request.source_message_id,)
            known_by = target.known_by_character_ids
        elif detail_match is not None:
            if len(active_foods) != 1:
                return None
            target = active_foods[0]
            value = detail_match.group("value").strip()
            if value == target.value:
                return None
            approval = MemoryApprovalState.AUTO_SAVED
            slot = target.slot
            supersedes_event_id = target.event_id
            sources = (*target.source_message_ids, request.source_message_id)
            known_by = target.known_by_character_ids
        else:
            assert condition_match is not None
            condition = condition_match.group("value").strip()
            if len(active_foods) == 1:
                target = active_foods[0]
                value = f"{target.value}（{condition}）"
                approval = MemoryApprovalState.AUTO_SAVED
                slot = target.slot
                supersedes_event_id = target.event_id
                sources = (*target.source_message_ids, request.source_message_id)
                known_by = target.known_by_character_ids
            else:
                value = condition
                approval = MemoryApprovalState.PENDING_CONFIRMATION
                slot = "liked_food_condition"
                supersedes_event_id = None
                prior_sources = tuple(
                    source_id
                    for item in active_foods
                    for source_id in item.source_message_ids
                )
                sources = (*prior_sources[-7:], request.source_message_id)
                known_by = request.known_by_character_ids

        source_ids = tuple(dict.fromkeys(sources))
        identity = json.dumps(
            {
                "conversation_id": request.conversation_id,
                "branch_id": request.branch_id,
                "sources": source_ids,
                "subject_id": request.author_subject_id,
                "slot": slot,
                "value": value,
                "approval": approval.value,
                "known_by": sorted(known_by),
                "supersedes": supersedes_event_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return CanonicalMemoryEvent(
            id=str(uuid5(_PROGRESSIVE_MEMORY_NAMESPACE, identity)),
            conversation_id=request.conversation_id,
            branch_id=request.branch_id,
            subject_id=request.author_subject_id,
            kind=MemoryKind.PREFERENCE,
            slot=slot,
            value=value,
            cardinality=MemoryCardinality.MULTIPLE,
            approval=approval,
            source_message_id=request.source_message_id,
            known_by_character_ids=known_by,
            supersedes_event_id=supersedes_event_id,
            effective_at=source_created_at,
            recorded_at=source_created_at,
            source_message_ids=source_ids,
        )

    @staticmethod
    def _to_event(
        candidate: MemoryCandidate, source_created_at: datetime
    ) -> CanonicalMemoryEvent | None:
        if candidate.disposition is MemoryCandidateDisposition.BLOCK:
            return None
        if (
            candidate.subject_id is None
            or candidate.cardinality is None
            or candidate.initial_approval is None
        ):
            raise ValidationError(
                "persistable memory candidate is missing required event fields"
            )
        identity = json.dumps(
            {
                "conversation_id": candidate.conversation_id,
                "branch_id": candidate.branch_id,
                "source_message_id": candidate.source_message_id,
                "subject_id": candidate.subject_id,
                "kind": candidate.kind.value,
                "slot": candidate.slot,
                "value": candidate.value,
                "cardinality": candidate.cardinality.value,
                "approval": candidate.initial_approval.value,
                "known_by": sorted(candidate.known_by_character_ids),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        event_id = str(uuid5(_MEMORY_EVENT_NAMESPACE, identity))
        # Capture only appends new facts. Resolving SINGLE-slot conflicts and
        # selecting supersedes_event_id belongs to the conflict-resolution slice.
        return CanonicalMemoryEvent(
            id=event_id,
            conversation_id=candidate.conversation_id,
            branch_id=candidate.branch_id,
            subject_id=candidate.subject_id,
            kind=candidate.kind,
            slot=candidate.slot,
            value=candidate.value,
            cardinality=candidate.cardinality,
            approval=candidate.initial_approval,
            source_message_id=candidate.source_message_id,
            known_by_character_ids=candidate.known_by_character_ids,
            supersedes_event_id=None,
            effective_at=source_created_at,
            recorded_at=source_created_at,
        )


class QueuedMemoryCaptureScheduler:
    """Runs capture off the response path; failures never fail the chat response."""

    def __init__(
        self,
        runner: MemoryCaptureRunner,
        repository: AppRepository,
        queue_size: int = 16,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._runner = runner
        self._repository = repository
        self._queue: asyncio.Queue[tuple[MemoryCaptureRequest, str]] = asyncio.Queue(
            maxsize=queue_size
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._subscribers: list[MemoryCaptureSubscriber] = []
        self._closing = False

    def subscribe(self, subscriber: MemoryCaptureSubscriber) -> None:
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)

    async def request_capture(
        self, request: MemoryCaptureRequest, run_id: str
    ) -> None:
        if self._closing:
            return
        self._ensure_worker()
        await self._notify(
            MemoryCaptureUpdate(
                request.conversation_id,
                request.branch_id,
                request.source_message_id,
                MemoryCaptureState.PROCESSING,
                (),
            ),
            run_id,
        )
        try:
            self._queue.put_nowait((request, run_id))
        except asyncio.QueueFull:
            await self._log_best_effort(
                "warning", "memory_capture_queue_full", {}, run_id
            )
            await self._notify(
                MemoryCaptureUpdate(
                    request.conversation_id,
                    request.branch_id,
                    request.source_message_id,
                    MemoryCaptureState.FAILED,
                    (),
                ),
                run_id,
            )

    async def wait_until_idle(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        self._closing = True
        task = self._worker_task
        self._worker_task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()

    def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._run_worker(), name="local-memory-capture-worker"
            )

    async def _run_worker(self) -> None:
        while True:
            request, run_id = await self._queue.get()
            try:
                result = await self._runner.capture(request)
                if result.extractor_unavailable:
                    await self._log_best_effort(
                        "warning",
                        "memory_capture_fallback_used",
                        {"error_type": "OllamaUnavailable"},
                        run_id,
                    )
                state = (
                    MemoryCaptureState.SAVED
                    if result.persisted_event_ids
                    else MemoryCaptureState.NO_CANDIDATES
                )
                await self._notify(
                    MemoryCaptureUpdate(
                        request.conversation_id,
                        request.branch_id,
                        request.source_message_id,
                        state,
                        result.persisted_event_ids,
                    ),
                    run_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                await self._log_best_effort(
                    "error",
                    "memory_capture_failed",
                    {"error_type": type(error).__name__},
                    run_id,
                )
                await self._notify(
                    MemoryCaptureUpdate(
                        request.conversation_id,
                        request.branch_id,
                        request.source_message_id,
                        MemoryCaptureState.FAILED,
                        (),
                    ),
                    run_id,
                )
            finally:
                self._queue.task_done()

    async def _notify(self, update: MemoryCaptureUpdate, run_id: str) -> None:
        for subscriber in tuple(self._subscribers):
            try:
                await subscriber(update)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                await self._log_best_effort(
                    "error",
                    "memory_capture_subscriber_failed",
                    {"error_type": type(error).__name__},
                    run_id,
                )

    async def _log_best_effort(
        self,
        level: str,
        event_type: str,
        details: dict[str, object],
        run_id: str,
    ) -> None:
        with suppress(Exception):
            await self._repository.log_event(level, event_type, details, run_id)
