from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from local_llm_chat.domain.models import (
    ChatChunk,
    ChatRequest,
    ModelInfo,
    ProviderConnection,
    ProviderMetadata,
)


class LLMProvider(Protocol):
    @property
    def metadata(self) -> ProviderMetadata: ...

    async def health(self) -> bool: ...

    async def list_models(self) -> list[ModelInfo]: ...

    async def inspect_model(self, model_name: str) -> ModelInfo: ...

    def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatChunk]: ...

    async def close(self) -> None: ...


class LLMProviderRegistry(Protocol):
    @property
    def default_provider_name(self) -> str: ...

    def list_connections(self) -> list[ProviderConnection]: ...

    def get(self, provider_name: str) -> LLMProvider: ...

    def cloud_is_disabled(self, provider_name: str) -> bool: ...

    async def save_connection(
        self,
        connection_id: str | None,
        display_name: str,
        endpoint: str,
        cloud_disabled_confirmed: bool,
    ) -> ProviderConnection: ...

    async def close(self) -> None: ...
