from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from local_llm_chat.domain.models import JobRun, ScheduledJob
from local_llm_chat.domain.states import JobRunState

ScheduledJobHandler = Callable[[ScheduledJob, JobRun], Awaitable[None]]


class SchedulerRepository(Protocol):
    async def claim_due_job(
        self, now: datetime
    ) -> tuple[ScheduledJob, JobRun] | None: ...

    async def create_job_retry(
        self, run_id: str, now: datetime
    ) -> tuple[ScheduledJob, JobRun]: ...

    async def finish_job_run(
        self,
        run_id: str,
        state: JobRunState,
        failure_reason: str | None = None,
    ) -> JobRun: ...

    async def log_event(
        self,
        level: str,
        event_type: str,
        details: dict[str, object],
        run_id: str | None = None,
    ) -> None: ...


class SchedulerService:
    def __init__(
        self,
        repository: SchedulerRepository,
        handlers: Mapping[str, ScheduledJobHandler],
        max_runs_per_tick: int = 10,
    ) -> None:
        if max_runs_per_tick < 1:
            raise ValueError("max_runs_per_tick must be positive")
        self._repository = repository
        self._handlers = dict(handlers)
        self._max_runs_per_tick = max_runs_per_tick
        self._active: dict[str, asyncio.Task[JobRun]] = {}
        self._claim_lock = asyncio.Lock()
        self._worker: asyncio.Task[None] | None = None
        self._closing = False

    def start(
        self,
        poll_interval_seconds: float = 1.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self._closing:
            raise RuntimeError("scheduler is closed")
        if self._worker is not None and not self._worker.done():
            return
        current_time = clock or (lambda: datetime.now(UTC))
        self._worker = asyncio.create_task(
            self._run_loop(poll_interval_seconds, current_time),
            name="local-scheduler-worker",
        )

    async def run_due_once(self, now: datetime) -> int:
        executed = 0
        for _ in range(self._max_runs_per_tick):
            async with self._claim_lock:
                if self._closing:
                    break
                claimed = await self._repository.claim_due_job(now)
                if claimed is not None:
                    job, run = claimed
                    task = await self._begin_execution(job, run)
            if claimed is None:
                break
            await self._await_execution(run.id, task)
            executed += 1
        return executed

    async def retry_failed(self, run_id: str, now: datetime) -> JobRun:
        async with self._claim_lock:
            if self._closing:
                raise RuntimeError("scheduler is closed")
            job, run = await self._repository.create_job_retry(run_id, now)
            task = await self._begin_execution(job, run)
        return await self._await_execution(run.id, task)

    async def close(self) -> None:
        self._closing = True
        worker = self._worker
        self._worker = None
        async with self._claim_lock:
            pass
        active: list[asyncio.Task[Any]] = list(self._active.values())
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    async def _run_loop(
        self,
        poll_interval_seconds: float,
        clock: Callable[[], datetime],
    ) -> None:
        while not self._closing:
            try:
                await self.run_due_once(clock())
            except asyncio.CancelledError:
                raise
            except Exception as error:
                await self._repository.log_event(
                    "error",
                    "scheduler_tick_failed",
                    {"error_type": type(error).__name__},
                )
            await asyncio.sleep(poll_interval_seconds)

    async def _begin_execution(
        self, job: ScheduledJob, run: JobRun
    ) -> asyncio.Task[JobRun]:
        handler = self._handlers.get(job.handler_name)
        async def execute_registered_handler() -> JobRun:
            if handler is None:
                return await self._repository.finish_job_run(
                    run.id, JobRunState.FAILED, "handler_not_registered"
                )
            try:
                await handler(job, run)
            except asyncio.CancelledError:
                stopped = await self._repository.finish_job_run(
                    run.id,
                    JobRunState.FAILED,
                    "application_stopped" if self._closing else "cancelled",
                )
                if not self._closing:
                    raise
                return stopped
            except Exception as error:
                return await self._repository.finish_job_run(
                    run.id, JobRunState.FAILED, type(error).__name__
                )
            return await self._repository.finish_job_run(
                run.id, JobRunState.COMPLETED
            )

        task = asyncio.create_task(
            execute_registered_handler(),
            name=f"scheduled-job-{job.id}-{run.id}",
        )
        self._active[run.id] = task
        # Let the wrapper enter its cancellation guard before close() can claim
        # the lock and cancel it.
        await asyncio.sleep(0)
        return task

    async def _await_execution(
        self, run_id: str, task: asyncio.Task[JobRun]
    ) -> JobRun:
        try:
            return await task
        finally:
            if self._active.get(run_id) is task:
                self._active.pop(run_id, None)
