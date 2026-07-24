from __future__ import annotations

from typing import Protocol

from local_llm_chat.domain.models import (
    ModelInfo,
    ProviderConnection,
    ProviderMetadata,
)


class EmbeddingProvider(Protocol):
    @property
    def metadata(self) -> ProviderMetadata: ...

    async def list_embedding_models(self) -> list[ModelInfo]: ...

    async def inspect_embedding_model(self, model_name: str) -> ModelInfo: ...

    async def embed(
        self, model_name: str, inputs: tuple[str, ...]
    ) -> tuple[tuple[float, ...], ...]: ...


class EmbeddingProviderRegistry(Protocol):
    def list_connections(self) -> list[ProviderConnection]: ...

    def get_embedding(self, provider_name: str) -> EmbeddingProvider: ...

    def cloud_is_disabled(self, provider_name: str) -> bool: ...
