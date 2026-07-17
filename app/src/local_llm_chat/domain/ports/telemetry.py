from __future__ import annotations

from typing import Protocol

from local_llm_chat.domain.models import TelemetryMetric


class TelemetryCollector(Protocol):
    @property
    def name(self) -> str: ...

    async def collect(self) -> tuple[TelemetryMetric, ...]: ...
