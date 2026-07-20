from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from local_llm_chat.domain.group_turns import (
    TurnBatchDraft,
    TurnBatchGenerationRequest,
)
from local_llm_chat.domain.ports.llm_provider import LLMProvider


class TurnBatchProviderRegistry(Protocol):
    def get(self, provider_name: str) -> LLMProvider: ...

    def cloud_is_disabled(self, provider_name: str) -> bool: ...


TurnBatchStreamCallback = Callable[[str], Awaitable[None]]


async def no_turn_batch_stream_update(_: str) -> None:
    return None


class TurnBatchGenerator(Protocol):
    async def generate(
        self,
        request: TurnBatchGenerationRequest,
        on_update: TurnBatchStreamCallback = no_turn_batch_stream_update,
    ) -> TurnBatchDraft: ...
