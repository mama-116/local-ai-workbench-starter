from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_llm_chat.application.services.scheduler_service import SchedulerService
from local_llm_chat.domain.models import JobRun, ScheduledJob
from local_llm_chat.domain.states import JobRunState
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


class PausingClaimRepository(SQLiteAppRepository):
    def __init__(self, database_path: Path) -> None:
        super().__init__(database_path)
        self.claim_started = asyncio.Event()
        self.allow_claim = asyncio.Event()

    async def claim_due_job(
        self, now: datetime
    ) -> tuple[ScheduledJob, JobRun] | None:
        self.claim_started.set()
        await self.allow_claim.wait()
        return await super().claim_due_job(now)


@pytest.mark.asyncio
async def test_two_schedulers_claim_one_due_run_only_once(tmp_path: Path) -> None:
    database = tmp_path / "chat.sqlite3"
    first_repository = SQLiteAppRepository(database)
    second_repository = SQLiteAppRepository(database)
    await first_repository.initialize()
    await second_repository.initialize()
    job = await first_repository.create_scheduled_job(
        "hourly-local", "record", 3600, NOW, {"value": "safe"}
    )
    calls: list[str] = []

    async def record(_job: object, _run: object) -> None:
        calls.append("called")

    first = SchedulerService(first_repository, {"record": record})
    second = SchedulerService(second_repository, {"record": record})

    await asyncio.gather(first.run_due_once(NOW), second.run_due_once(NOW))

    runs = await first_repository.list_job_runs(job.id)
    assert calls == ["called"]
    assert len(runs) == 1
    assert runs[0].state is JobRunState.COMPLETED


@pytest.mark.asyncio
async def test_failure_and_manual_retry_are_separate_attempts(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    job = await repository.create_scheduled_job(
        "retry-local", "sometimes", 3600, NOW, {}
    )
    attempts = 0

    async def sometimes(_job: object, _run: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("broken")

    scheduler = SchedulerService(repository, {"sometimes": sometimes})
    await scheduler.run_due_once(NOW)
    first = (await repository.list_job_runs(job.id))[0]

    await scheduler.retry_failed(first.id, NOW)

    runs = await repository.list_job_runs(job.id)
    assert [(run.attempt, run.state) for run in runs] == [
        (1, JobRunState.FAILED),
        (2, JobRunState.COMPLETED),
    ]
    assert runs[1].retry_of_run_id == runs[0].id


@pytest.mark.asyncio
async def test_restart_marks_interrupted_run_failed_without_auto_retry(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    job = await repository.create_scheduled_job(
        "recover-local", "noop", 3600, NOW, {}
    )
    claimed = await repository.claim_due_job(NOW)
    assert claimed is not None

    await repository.recover_interrupted_runs()

    runs = await repository.list_job_runs(job.id)
    assert len(runs) == 1
    assert runs[0].state is JobRunState.FAILED
    assert runs[0].failure_reason == "previous_session_interrupted"


@pytest.mark.asyncio
async def test_close_cancels_only_active_handler_and_records_stop(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    job = await repository.create_scheduled_job(
        "stop-local", "blocking", 3600, NOW, {}
    )
    started = asyncio.Event()

    async def blocking(_job: object, _run: object) -> None:
        started.set()
        await asyncio.Event().wait()

    scheduler = SchedulerService(repository, {"blocking": blocking})
    execution = asyncio.create_task(scheduler.run_due_once(NOW))
    await started.wait()

    await scheduler.close()
    run = (await repository.list_job_runs(job.id))[0]
    assert run.state is JobRunState.FAILED
    assert run.failure_reason == "application_stopped"
    await execution


@pytest.mark.asyncio
async def test_background_driver_polls_due_jobs_until_closed(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    job = await repository.create_scheduled_job(
        "background-local", "notify", 3600, NOW, {}
    )
    completed = asyncio.Event()

    async def notify(_job: object, _run: object) -> None:
        completed.set()

    scheduler = SchedulerService(repository, {"notify": notify})
    scheduler.start(0.01, lambda: NOW)
    await asyncio.wait_for(completed.wait(), timeout=1)
    for _ in range(100):
        runs = await repository.list_job_runs(job.id)
        if runs and runs[0].state is JobRunState.COMPLETED:
            break
        await asyncio.sleep(0.01)
    await scheduler.close()

    run = (await repository.list_job_runs(job.id))[0]
    assert run.state is JobRunState.COMPLETED


@pytest.mark.asyncio
async def test_close_waits_for_claim_registration_before_cancelling(
    tmp_path: Path,
) -> None:
    repository = PausingClaimRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    job = await repository.create_scheduled_job(
        "claim-race-local", "blocking", 3600, NOW, {}
    )

    async def blocking(_job: object, _run: object) -> None:
        await asyncio.Event().wait()

    scheduler = SchedulerService(repository, {"blocking": blocking})
    execution = asyncio.create_task(scheduler.run_due_once(NOW))
    await repository.claim_started.wait()
    closing = asyncio.create_task(scheduler.close())
    await asyncio.sleep(0)
    repository.allow_claim.set()

    await closing
    await execution

    run = (await repository.list_job_runs(job.id))[0]
    assert run.state is JobRunState.FAILED
    assert run.failure_reason == "application_stopped"
