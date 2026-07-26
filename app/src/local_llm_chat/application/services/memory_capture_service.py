from __future__ import annotations

import asyncio
import json
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
from local_llm_chat.application.services.relationship_profile_service import (
    RelationshipProfileService,
)
from local_llm_chat.domain.canonical_memory import CanonicalMemoryEvent
from local_llm_chat.domain.errors import AppError, ValidationError
from local_llm_chat.domain.memory_candidates import (
    MemoryCandidate,
    MemoryCandidateRequest,
)
from local_llm_chat.domain.ports.repositories import AppRepository
from local_llm_chat.domain.ports.relationship_candidate_extractor import (
    RelationshipCandidateExtractor,
)
from local_llm_chat.domain.relationship_profile import (
    ProfileApproval,
    ProfileOrigin,
    RelationshipCandidateRequest,
)
from local_llm_chat.domain.states import MemoryCandidateDisposition


_MEMORY_EVENT_NAMESPACE = UUID("8a37fbc1-ca67-4ecf-a54a-f54cc0f31475")


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
    profile_event_ids: tuple[str, ...] = ()
    relationship_event_ids: tuple[str, ...] = ()
    profile_capture_failed: bool = False
    relationship_capture_failed: bool = False


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
    profile_event_ids: tuple[str, ...] = ()
    relationship_event_ids: tuple[str, ...] = ()


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
        self,
        repository: AppRepository,
        candidate_service: MemoryCandidateService,
        relationship_profiles: RelationshipProfileService | None = None,
        relationship_extractor: RelationshipCandidateExtractor | None = None,
    ) -> None:
        self._repository = repository
        self._candidate_service = candidate_service
        self._relationship_profiles = relationship_profiles
        self._relationship_extractor = relationship_extractor

    async def capture(self, request: MemoryCaptureRequest) -> MemoryCaptureResult:
        source = await self._repository.get_memory_source_message(
            request.conversation_id,
            request.branch_id,
            request.source_message_id,
        )
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
        profile_event_ids: list[str] = []
        profile_capture_failed = False
        if self._relationship_profiles is not None:
            existing_items = await self._relationship_profiles.list_profile_items(
                request.conversation_id
            )
            for candidate in candidates:
                if (
                    candidate.disposition is MemoryCandidateDisposition.BLOCK
                    or candidate.subject_id != "user"
                ):
                    continue
                matching = [
                    item
                    for item in existing_items
                    if item.item_kind == candidate.kind.value
                    and item.item_name == candidate.slot
                ]
                current = matching[-1] if matching else None
                if current is not None and current.value == candidate.value:
                    continue
                if current is not None and (
                    current.origin is ProfileOrigin.USER_ASSERTED
                    or current.approval is ProfileApproval.PENDING_CONFIRMATION
                ):
                    continue
                try:
                    profile_event = (
                        await self._relationship_profiles.propose_ai_profile_item(
                            conversation_id=request.conversation_id,
                            branch_id=request.branch_id,
                            source_message_id=request.source_message_id,
                            item_kind=candidate.kind.value,
                            item_name=candidate.slot,
                            value=candidate.value,
                            low_risk_explicit=(
                                current is None
                                and candidate.disposition
                                is MemoryCandidateDisposition.AUTO_SAVE
                            ),
                            recorded_at=source.created_at,
                            supersedes_event_id=(
                                current.event_id if current is not None else None
                            ),
                        )
                    )
                except AppError:
                    profile_capture_failed = True
                    continue
                if profile_event.id not in profile_event_ids:
                    profile_event_ids.append(profile_event.id)
                    existing_items = (
                        await self._relationship_profiles.list_profile_items(
                            request.conversation_id
                        )
                    )
        relationship_event_ids: tuple[str, ...] = ()
        relationship_capture_failed = False
        if (
            self._relationship_profiles is not None
            and self._relationship_extractor is not None
        ):
            try:
                relationship_event_ids = (
                    await self._relationship_profiles.capture_relationship_candidates(
                        request=RelationshipCandidateRequest(
                            conversation_id=request.conversation_id,
                            branch_id=request.branch_id,
                            source_message_id=request.source_message_id,
                            model_name=request.model_name,
                            content=source.content,
                            allowed_character_ids=(
                                request.allowed_knowledge_character_ids
                            ),
                        ),
                        extractor=self._relationship_extractor,
                        recorded_at=source.created_at,
                    )
                )
            except asyncio.CancelledError:
                raise
            except AppError:
                relationship_capture_failed = True
        return MemoryCaptureResult(
            candidates,
            persisted_event_ids,
            generation.extractor_unavailable,
            tuple(profile_event_ids),
            relationship_event_ids,
            profile_capture_failed,
            relationship_capture_failed,
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
                if result.relationship_capture_failed:
                    await self._log_best_effort(
                        "warning",
                        "relationship_capture_failed",
                        {"error_type": "RelationshipCaptureError"},
                        run_id,
                    )
                if result.profile_capture_failed:
                    await self._log_best_effort(
                        "warning",
                        "profile_capture_skipped",
                        {"error_type": "ProfileCapturePolicyError"},
                        run_id,
                    )
                state = (
                    MemoryCaptureState.SAVED
                    if (
                        result.persisted_event_ids
                        or result.profile_event_ids
                        or result.relationship_event_ids
                    )
                    else MemoryCaptureState.NO_CANDIDATES
                )
                await self._notify(
                    MemoryCaptureUpdate(
                        request.conversation_id,
                        request.branch_id,
                        request.source_message_id,
                        state,
                        result.persisted_event_ids,
                        result.profile_event_ids,
                        result.relationship_event_ids,
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
