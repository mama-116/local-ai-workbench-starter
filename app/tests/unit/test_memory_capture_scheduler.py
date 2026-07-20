from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from local_llm_chat.application.services.memory_capture_service import (
    MemoryCaptureUpdate,
    MemoryCaptureRequest,
    MemoryCaptureResult,
    QueuedMemoryCaptureScheduler,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


class RecordingCaptureRunner:
    def __init__(
        self,
        error: Exception | None = None,
        persisted_event_ids: tuple[str, ...] = (),
    ) -> None:
        self.error = error
        self.persisted_event_ids = persisted_event_ids
        self.requests: list[MemoryCaptureRequest] = []

    async def capture(self, request: MemoryCaptureRequest) -> MemoryCaptureResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return MemoryCaptureResult((), self.persisted_event_ids)


def capture_request(
    conversation_id: str = "conversation-1",
    branch_id: str = "branch-1",
    source_message_id: str = "message-1",
) -> MemoryCaptureRequest:
    return MemoryCaptureRequest(
        conversation_id=conversation_id,
        branch_id=branch_id,
        source_message_id=source_message_id,
        model_name="conversation-model",
        author_subject_id="user",
        allowed_subject_ids=frozenset({"user", "character-1"}),
        allowed_knowledge_character_ids=frozenset({"character-1"}),
    )


async def make_capture_context(
    repository: SQLiteAppRepository,
) -> tuple[MemoryCaptureRequest, str]:
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation(
        "conversation", character.id, profile.id
    )
    session = await repository.start_send(conversation.id, "source")
    return (
        capture_request(
            conversation.id, session.branch_id, session.user_message.id
        ),
        session.run.id,
    )


@pytest.mark.asyncio
async def test_scheduler_runs_capture_outside_request_path(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    request, run_id = await make_capture_context(repository)
    runner = RecordingCaptureRunner()
    scheduler = QueuedMemoryCaptureScheduler(runner, repository)

    await scheduler.request_capture(request, run_id)
    assert runner.requests == []

    await scheduler.wait_until_idle()
    assert runner.requests == [request]
    await scheduler.close()


@pytest.mark.asyncio
async def test_capture_failure_is_logged_without_private_error_text(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    request, run_id = await make_capture_context(repository)
    runner = RecordingCaptureRunner(RuntimeError("PRIVATE USER CONTENT"))
    scheduler = QueuedMemoryCaptureScheduler(runner, repository)

    await scheduler.request_capture(request, run_id)
    await scheduler.wait_until_idle()
    await scheduler.close()

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT event_type, run_id, details_json FROM app_events "
            "WHERE event_type = 'memory_capture_failed'"
        ).fetchone()
    assert row == (
        "memory_capture_failed",
        run_id,
        '{"error_type": "RuntimeError"}',
    )


@pytest.mark.asyncio
async def test_full_queue_skips_capture_and_logs_warning(tmp_path: Path) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    request, run_id = await make_capture_context(repository)
    runner = RecordingCaptureRunner()
    scheduler = QueuedMemoryCaptureScheduler(runner, repository, queue_size=1)

    await scheduler.request_capture(request, run_id)
    await scheduler.request_capture(request, run_id)
    await scheduler.wait_until_idle()
    await scheduler.close()

    assert runner.requests == [request]
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT event_type, run_id FROM app_events "
            "WHERE event_type = 'memory_capture_queue_full'"
        ).fetchone()
    assert row == ("memory_capture_queue_full", run_id)


def test_scheduler_rejects_non_positive_queue_size(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")

    with pytest.raises(ValueError, match="queue_size must be positive"):
        QueuedMemoryCaptureScheduler(RecordingCaptureRunner(), repository, 0)


@pytest.mark.asyncio
async def test_scheduler_notifies_with_ids_after_persistence_and_isolates_subscribers(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    request, run_id = await make_capture_context(repository)
    scheduler = QueuedMemoryCaptureScheduler(
        RecordingCaptureRunner(persisted_event_ids=("memory-1",)), repository
    )
    updates: list[MemoryCaptureUpdate] = []

    async def failing(_: MemoryCaptureUpdate) -> None:
        raise RuntimeError("PRIVATE CALLBACK DETAIL")

    async def record(update: MemoryCaptureUpdate) -> None:
        updates.append(update)

    scheduler.subscribe(failing)
    scheduler.subscribe(record)
    await scheduler.request_capture(request, run_id)
    await scheduler.wait_until_idle()
    await scheduler.close()

    assert updates == [
        MemoryCaptureUpdate(
            request.conversation_id,
            request.branch_id,
            ("memory-1",),
        )
    ]


@pytest.mark.asyncio
async def test_scheduler_ignores_capture_requested_after_close(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    request, run_id = await make_capture_context(repository)
    runner = RecordingCaptureRunner()
    scheduler = QueuedMemoryCaptureScheduler(runner, repository)

    await scheduler.close()
    await scheduler.request_capture(request, run_id)
    await scheduler.wait_until_idle()

    assert runner.requests == []
