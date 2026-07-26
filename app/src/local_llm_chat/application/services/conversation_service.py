from __future__ import annotations

from typing import Any

from local_llm_chat.domain.models import (
    BranchInfo,
    Conversation,
    ConversationSelection,
    Message,
)
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.ports.llm_provider import LLMProvider, LLMProviderRegistry
from local_llm_chat.domain.ports.repositories import AppRepository

_DEFAULT_CHAT_PARAMETERS: dict[str, Any] = {
    "num_ctx": 4096,
    "temperature": 0.3,
}


class ConversationService:
    def __init__(
        self,
        repository: AppRepository,
        providers: LLMProviderRegistry,
        free_policy: FreeOperationPolicy,
    ) -> None:
        self._repository = repository
        self._providers = providers
        self._free_policy = free_policy

    async def list_conversations(self) -> list[Conversation]:
        return await self._repository.list_conversations()

    async def list_archived_conversations(self) -> list[Conversation]:
        return await self._repository.list_archived_conversations()

    async def create_conversation(
        self,
        title: str,
        character_version_id: str,
        provider_name: str,
        model_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> Conversation:
        provider = await self._require_model(provider_name, model_name)
        profile = await self._repository.ensure_model_profile(
            provider.metadata.name,
            model_name,
            parameters or dict(_DEFAULT_CHAT_PARAMETERS),
        )
        return await self._repository.create_conversation(
            title, character_version_id, profile.id
        )

    async def update_selection(
        self,
        conversation_id: str,
        character_version_id: str,
        provider_name: str,
        model_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        provider = await self._require_model(provider_name, model_name)
        profile = await self._repository.ensure_model_profile(
            provider.metadata.name,
            model_name,
            parameters or dict(_DEFAULT_CHAT_PARAMETERS),
        )
        await self._repository.update_conversation_selection(
            conversation_id, character_version_id, profile.id
        )

    async def archive(self, conversation_id: str) -> None:
        await self._repository.archive_conversation(conversation_id)

    async def restore(self, conversation_id: str) -> None:
        await self._repository.restore_conversation(conversation_id)

    async def empty_trash(self, conversation_ids: tuple[str, ...]) -> int:
        return await self._repository.delete_archived_conversations(
            conversation_ids
        )

    async def set_auto_translate(self, conversation_id: str, enabled: bool) -> None:
        await self._repository.set_conversation_auto_translate(
            conversation_id, enabled
        )

    async def messages(self, conversation_id: str) -> list[Message]:
        return await self._repository.list_active_messages(conversation_id)

    async def selection(self, conversation_id: str) -> ConversationSelection:
        conversation = await self._repository.get_conversation(conversation_id)
        character = await self._repository.get_character_version(
            conversation.character_version_id
        )
        profile = await self._repository.get_model_profile(
            conversation.model_profile_id
        )
        return ConversationSelection(character=character, model_profile=profile)

    async def branches(self, conversation_id: str) -> list[BranchInfo]:
        return await self._repository.list_branches(conversation_id)

    async def all_branches(self, conversation_id: str) -> list[BranchInfo]:
        return await self._repository.list_all_branches(conversation_id)

    async def activate_branch(self, conversation_id: str, branch_id: str) -> None:
        await self._repository.activate_branch(conversation_id, branch_id)

    async def hide_branch(self, conversation_id: str, branch_id: str) -> None:
        await self._repository.hide_branch(conversation_id, branch_id)

    async def restore_branch(self, conversation_id: str, branch_id: str) -> None:
        await self._repository.restore_branch(conversation_id, branch_id)

    async def _require_model(
        self, provider_name: str, model_name: str
    ) -> LLMProvider:
        provider = self._providers.get(provider_name)
        self._free_policy.require_cloud_disabled(
            self._providers.cloud_is_disabled(provider_name)
        )
        self._free_policy.require_provider(provider.metadata)
        model = await provider.inspect_model(model_name)
        self._free_policy.require_model(model)
        return provider
