from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from local_llm_chat.domain.models import LatestTelemetry, TelemetryMetric
from local_llm_chat.domain.ports.telemetry import TelemetryCollector


class TelemetryRepository(Protocol):
    async def save_telemetry_metrics(
        self, run_id: str, metrics: tuple[TelemetryMetric, ...]
    ) -> None: ...

    async def get_latest_telemetry(
        self, conversation_id: str | None = None
    ) -> LatestTelemetry | None: ...


class TelemetryService:
    def __init__(
        self,
        repository: TelemetryRepository,
        collectors: list[TelemetryCollector],
    ) -> None:
        self._repository = repository
        self._collectors = tuple(collectors)
        self._tasks: set[asyncio.Task[None]] = set()
        self._subscribers: list[Callable[[str], Awaitable[None]]] = []
        self._closing = False

    def subscribe(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self._subscribers.append(callback)

    def request_capture(self, run_id: str) -> None:
        if self._closing:
            return
        task = asyncio.create_task(self._capture_and_notify(run_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _capture_and_notify(self, run_id: str) -> None:
        try:
            await self.capture(run_id)
        except Exception:
            return
        for subscriber in tuple(self._subscribers):
            try:
                await subscriber(run_id)
            except Exception:
                continue

    async def capture(self, run_id: str) -> tuple[TelemetryMetric, ...]:
        gathered: list[TelemetryMetric] = []
        for collector in self._collectors:
            try:
                gathered.extend(await collector.collect())
            except Exception:
                gathered.append(
                    TelemetryMetric(
                        f"{collector.name}_status",
                        None,
                        "",
                        collector.name,
                        "収集に失敗しました",
                    )
                )
        metrics = tuple(gathered)
        await self._repository.save_telemetry_metrics(run_id, metrics)
        return metrics

    async def latest(
        self, conversation_id: str | None = None
    ) -> LatestTelemetry | None:
        return await self._repository.get_latest_telemetry(conversation_id)

    async def close(self) -> None:
        self._closing = True
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
