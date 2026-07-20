import asyncio

import pytest

from local_llm_chat.application.services.telemetry_service import TelemetryService
from local_llm_chat.domain.models import TelemetryMetric


class FakeRepository:
    def __init__(self) -> None:
        self.saved: tuple[TelemetryMetric, ...] = ()
        self.saved_event = asyncio.Event()

    async def save_telemetry_metrics(
        self, run_id: str, metrics: tuple[TelemetryMetric, ...]
    ) -> None:
        assert run_id == "run-1"
        self.saved = metrics
        self.saved_event.set()

    async def get_latest_telemetry(self, conversation_id: str | None = None) -> None:
        return None


class FailingCollector:
    name = "broken-sensor"

    async def collect(self) -> tuple[TelemetryMetric, ...]:
        raise RuntimeError("sensor details must not leak")


class PartialCollector:
    name = "test-host"

    async def collect(self) -> tuple[TelemetryMetric, ...]:
        return (
            TelemetryMetric("cpu_percent", 42.5, "%", "test-host"),
            TelemetryMetric(
                "ram_percent", None, "%", "test-host", "取得に対応していません"
            ),
        )


class WaitingCollector:
    name = "waiting"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def collect(self) -> tuple[TelemetryMetric, ...]:
        self.started.set()
        await self.release.wait()
        return (TelemetryMetric("cpu_percent", 1.0, "%", self.name),)


@pytest.mark.asyncio
async def test_capture_keeps_partial_values_and_turns_collector_failure_into_missing(
) -> None:
    repository = FakeRepository()
    service = TelemetryService(repository, [PartialCollector(), FailingCollector()])

    metrics = await service.capture("run-1")

    assert metrics[0].value == 42.5
    assert metrics[1].value is None
    assert metrics[1].unavailable_reason == "取得に対応していません"
    assert metrics[2].value is None
    assert metrics[2].source == "broken-sensor"
    assert metrics[2].unavailable_reason == "収集に失敗しました"
    assert repository.saved == metrics


@pytest.mark.asyncio
async def test_requested_capture_runs_in_background_without_blocking_chat() -> None:
    repository = FakeRepository()
    collector = WaitingCollector()
    service = TelemetryService(repository, [collector])

    service.request_capture("run-1")
    await asyncio.wait_for(collector.started.wait(), timeout=0.2)
    assert repository.saved == ()

    collector.release.set()
    await asyncio.wait_for(repository.saved_event.wait(), timeout=0.2)
    await service.close()
    assert next(iter(repository.saved)).value == 1.0


@pytest.mark.asyncio
async def test_close_cancels_running_capture_instead_of_waiting_for_collector() -> None:
    repository = FakeRepository()
    collector = WaitingCollector()
    service = TelemetryService(repository, [collector])
    service.request_capture("run-1")
    await asyncio.wait_for(collector.started.wait(), timeout=0.2)

    await asyncio.wait_for(service.close(), timeout=0.2)

    assert repository.saved == ()


@pytest.mark.asyncio
async def test_capture_requested_after_close_is_ignored() -> None:
    repository = FakeRepository()
    collector = WaitingCollector()
    service = TelemetryService(repository, [collector])

    await service.close()
    service.request_capture("run-1")
    await asyncio.sleep(0)

    assert not collector.started.is_set()
    assert repository.saved == ()
