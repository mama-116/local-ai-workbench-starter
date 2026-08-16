from __future__ import annotations

from typing import Any, cast

from local_llm_chat.domain.errors import FreeOperationBlocked
from local_llm_chat.domain.models import CharacterVersion, ModelInfo, ProviderConnection
from local_llm_chat.domain.relationship_behavior import RelationshipStyle
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.ports.llm_provider import LLMProviderRegistry
from local_llm_chat.domain.ports.repositories import AppRepository


class ProfileService:
    def __init__(
        self,
        repository: AppRepository,
        providers: LLMProviderRegistry,
        free_policy: FreeOperationPolicy,
    ) -> None:
        self._repository = repository
        self._providers = providers
        self._free_policy = free_policy

    async def list_characters(self) -> list[CharacterVersion]:
        return await self._repository.list_character_versions()

    async def save_character(
        self,
        display_name: str,
        system_prompt: str,
        character_id: str | None = None,
        relationship_style: RelationshipStyle | None = None,
    ) -> CharacterVersion:
        return await self._repository.create_character_version(
            display_name, system_prompt, character_id, relationship_style
        )

    def list_connections(self) -> list[ProviderConnection]:
        return self._providers.list_connections()

    @property
    def default_provider_name(self) -> str:
        return self._providers.default_provider_name

    async def list_visible_models(self, provider_name: str) -> list[ModelInfo]:
        provider = self._providers.get(provider_name)
        self._free_policy.require_provider(provider.metadata)
        models = await provider.list_models()
        allowed: list[ModelInfo] = []
        for model in models:
            try:
                self._free_policy.require_model(model)
            except FreeOperationBlocked:
                continue
            allowed.append(model)
        return allowed

    async def list_allowed_models(self, provider_name: str) -> list[ModelInfo]:
        self._free_policy.require_cloud_disabled(
            self._providers.cloud_is_disabled(provider_name)
        )
        return await self.list_visible_models(provider_name)

    def cloud_is_disabled(self, provider_name: str) -> bool:
        return self._providers.cloud_is_disabled(provider_name)

    async def ollama_is_available(self, provider_name: str) -> bool:
        return await self._providers.get(provider_name).health()

    async def save_connection(
        self,
        connection_id: str | None,
        display_name: str,
        endpoint: str,
        cloud_disabled_confirmed: bool,
        relationship_behavior_allowed: bool = False,
    ) -> ProviderConnection:
        registry = cast(Any, self._providers)
        return cast(
            ProviderConnection,
            await registry.save_connection(
                connection_id,
                display_name,
                endpoint,
                cloud_disabled_confirmed,
                relationship_behavior_allowed,
            ),
        )
