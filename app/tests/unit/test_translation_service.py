from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from local_llm_chat.application.services.translation_service import TranslationService
from local_llm_chat.domain.models import (
    ChatChunk,
    ChatRequest,
    ModelInfo,
    ProviderConnection,
    ProviderMetadata,
)
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.states import (
    CostClass,
    Locality,
    MessageState,
    TranslationState,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


class TranslationProvider:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            "ollama-local",
            Locality.LOCAL,
            CostClass.NO_CHARGE,
            "http://127.0.0.1:11434",
        )

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return [await self.inspect_model("gemma4:12b")]

    async def inspect_model(self, model_name: str) -> ModelInfo:
        return ModelInfo(model_name, 1, "gguf", "gemma", "12B", "Q4")

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.requests.append(request)
        yield ChatChunk(content="こんにちは")
        yield ChatChunk(content="、世界。", done=True)

    async def close(self) -> None:
        return None


class FakeRegistry:
    def __init__(self, provider: TranslationProvider) -> None:
        self.provider = provider

    @property
    def default_provider_name(self) -> str:
        return "ollama-local"

    def list_connections(self) -> list[ProviderConnection]:
        return []

    def get(self, provider_name: str) -> TranslationProvider:
        assert provider_name == "ollama-local"
        return self.provider

    def cloud_is_disabled(self, provider_name: str) -> bool:
        assert provider_name == "ollama-local"
        return True

    async def save_connection(
        self,
        connection_id: str | None,
        display_name: str,
        endpoint: str,
        cloud_disabled_confirmed: bool,
    ) -> ProviderConnection:
        raise AssertionError("not used")

    async def close(self) -> None:
        return None


class EmptyTranslationProvider(TranslationProvider):
    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.requests.append(request)
        yield ChatChunk(done=True)


class WaitingTranslationProvider(TranslationProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        yield ChatChunk(content="translated", done=True)


async def make_response(
    repository: SQLiteAppRepository,
    content: str,
) -> str:
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "gemma4:12b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "翻訳試験", character.id, profile.id
    )
    session = await repository.start_send(conversation.id, "質問")
    response = await repository.finish_response(
        session, content, MessageState.COMPLETED
    )
    return response.id


@pytest.mark.asyncio
async def test_translates_foreign_response_without_changing_original(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    message_id = await make_response(repository, "Hello, world.")
    provider = TranslationProvider()
    service = TranslationService(
        repository, provider, FakeRegistry(provider), FreeOperationPolicy()
    )

    try:
        queued = await service.request_translation(message_id)
        await service.wait_until_idle()

        assert queued is not None
        translation = await repository.get_current_translation(message_id)
        assert translation is not None
        assert translation.content == "こんにちは、世界。"
        assert translation.model == "llama3.1:latest"
        assert translation.state is TranslationState.COMPLETED
        assert (await repository.get_message(message_id)).content == "Hello, world."
        assert provider.requests[0].model == "llama3.1:latest"
        assert provider.requests[0].options["num_ctx"] == 4096
        assert provider.requests[0].options["num_predict"] == 1200
        assert "Hello, world." in provider.requests[0].messages[-1].content
        assert "Do not leave Chinese words" in provider.requests[0].messages[-1].content
        assert "FINAL JAPANESE TRANSLATION" in provider.requests[0].messages[-1].content
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_reuses_translation_and_force_creates_a_new_attempt(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    first_message_id = await make_response(repository, "Hello, world.")
    second_message_id = await make_response(repository, "Hello, world.")
    provider = TranslationProvider()
    service = TranslationService(
        repository, provider, FakeRegistry(provider), FreeOperationPolicy()
    )

    try:
        await service.request_translation(first_message_id)
        await service.wait_until_idle()
        reused = await service.request_translation(second_message_id)
        assert reused is not None
        assert reused.reused_from_id is not None
        assert len(provider.requests) == 1

        retried = await service.request_translation(second_message_id, force=True)
        await service.wait_until_idle()
        assert retried is not None
        assert retried.id != reused.id
        assert len(provider.requests) == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_skips_automatic_translation_for_japanese_response(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    message_id = await make_response(repository, "日本語の回答です。")
    provider = TranslationProvider()
    service = TranslationService(
        repository, provider, FakeRegistry(provider), FreeOperationPolicy()
    )

    try:
        assert await service.request_translation(message_id) is None
        assert await repository.get_current_translation(message_id) is None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_translation_failure_keeps_original_and_can_be_retried(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    message_id = await make_response(repository, "Hello, world.")
    provider = EmptyTranslationProvider()
    service = TranslationService(
        repository, provider, FakeRegistry(provider), FreeOperationPolicy()
    )

    try:
        await service.request_translation(message_id)
        await service.wait_until_idle()

        failed = await repository.get_current_translation(message_id)
        assert failed is not None
        assert failed.state is TranslationState.FAILED
        assert (await repository.get_message(message_id)).content == "Hello, world."
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_translation_requested_after_close_is_ignored(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    message_id = await make_response(repository, "Hello, world.")
    provider = TranslationProvider()
    service = TranslationService(
        repository, provider, FakeRegistry(provider), FreeOperationPolicy()
    )

    await service.close()
    result = await service.request_translation(message_id)

    assert result is None
    assert await repository.get_current_translation(message_id) is None
    assert provider.requests == []


@pytest.mark.asyncio
async def test_close_fails_running_and_queued_translations(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    running_message_id = await make_response(repository, "First response.")
    queued_message_id = await make_response(repository, "Second response.")
    provider = WaitingTranslationProvider()
    service = TranslationService(
        repository, provider, FakeRegistry(provider), FreeOperationPolicy()
    )

    await service.request_translation(running_message_id)
    await asyncio.wait_for(provider.started.wait(), timeout=0.2)
    await service.request_translation(queued_message_id)
    await service.close()
    await service.wait_until_idle()

    running = await repository.get_current_translation(running_message_id)
    queued = await repository.get_current_translation(queued_message_id)
    assert running is not None
    assert running.state is TranslationState.FAILED
    assert running.error_code == "application_closed"
    assert queued is not None
    assert queued.state is TranslationState.FAILED
    assert queued.error_code == "application_closed"
