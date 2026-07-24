from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import json

import pytest

from local_llm_chat.application.services.memory_extraction_settings_service import (
    MemoryExtractionSettingsService,
)
from local_llm_chat.domain.errors import (
    FreeOperationBlocked,
    ModelUnavailable,
    OllamaUnavailable,
    ValidationError,
)
from local_llm_chat.domain.models import (
    ChatChunk,
    ChatRequest,
    ModelInfo,
    ModelRoleSetting,
    ProviderConnection,
    ProviderMetadata,
)
from local_llm_chat.domain.memory_candidates import MemoryCandidateRequest
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.states import CostClass, Locality, ModelRole
from local_llm_chat.infrastructure.llm.configured_memory_candidate_extractor import (
    ConfiguredMemoryCandidateExtractor,
)


def model(
    name: str,
    *,
    digest: str = "digest-1",
    capabilities: tuple[str, ...] = ("completion",),
    size_bytes: int = 1_000,
) -> ModelInfo:
    return ModelInfo(
        name=name,
        size_bytes=size_bytes,
        format="gguf",
        family="test",
        parameter_size="1B",
        quantization="Q4",
        digest=digest,
        capabilities=capabilities,
    )


@dataclass
class FakeProvider:
    provider_name: str
    endpoint: str
    models: dict[str, ModelInfo]
    locality: Locality = Locality.LOCAL
    missing_on_inspect: set[str] = field(default_factory=set)
    list_calls: int = 0
    inspect_calls: list[str] = field(default_factory=list)
    chat_requests: list[ChatRequest] = field(default_factory=list)
    response_content: str = field(
        default_factory=lambda: json.dumps(
            {"candidates": []}, ensure_ascii=False
        )
    )
    response_completed: bool = True

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            self.provider_name,
            self.locality,
            CostClass.NO_CHARGE,
            self.endpoint,
        )

    async def list_models(self) -> list[ModelInfo]:
        self.list_calls += 1
        return list(self.models.values())

    async def inspect_model(self, model_name: str) -> ModelInfo:
        self.inspect_calls.append(model_name)
        if model_name in self.missing_on_inspect:
            raise ModelUnavailable("model disappeared")
        try:
            return self.models[model_name]
        except KeyError as error:
            raise ValidationError("model missing") from error

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def stream_chat(
        self, request: ChatRequest
    ) -> AsyncIterator[ChatChunk]:
        self.chat_requests.append(request)
        yield ChatChunk(
            self.response_content,
            done=self.response_completed,
        )


@dataclass
class FakeRegistry:
    connections: list[ProviderConnection]
    providers: dict[str, FakeProvider]
    cloud_disabled: dict[str, bool]
    default_provider_name: str = "ollama-local"

    def list_connections(self) -> list[ProviderConnection]:
        return list(self.connections)

    def get(self, provider_name: str) -> FakeProvider:
        try:
            return self.providers[provider_name]
        except KeyError as error:
            raise ValidationError("provider missing") from error

    def cloud_is_disabled(self, provider_name: str) -> bool:
        return self.cloud_disabled.get(provider_name, False)

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


@dataclass
class FakeRepository:
    setting: ModelRoleSetting | None = None
    save_calls: int = 0

    async def get_model_role_setting(
        self, role: ModelRole
    ) -> ModelRoleSetting | None:
        assert role is ModelRole.MEMORY_EXTRACTION
        return self.setting

    async def save_model_role_setting(
        self, setting: ModelRoleSetting
    ) -> ModelRoleSetting:
        self.save_calls += 1
        self.setting = setting
        return setting


def registry() -> FakeRegistry:
    local = ProviderConnection(
        "local",
        "ollama-local",
        "このPC",
        "http://127.0.0.1:11434",
        True,
        True,
    )
    dgx = ProviderConnection(
        "dgx",
        "ollama-dgx",
        "DGX Spark",
        "http://192.168.1.50:11434",
        True,
    )
    return FakeRegistry(
        [local, dgx],
        {
            "ollama-local": FakeProvider(
                "ollama-local",
                local.endpoint,
                {
                    "chat:latest": model("chat:latest"),
                    "embed:latest": model(
                        "embed:latest", capabilities=("embedding",)
                    ),
                    "cloud:latest": model("cloud:latest"),
                },
            ),
            "ollama-dgx": FakeProvider(
                "ollama-dgx",
                dgx.endpoint,
                {"gemma:31b": model("gemma:31b", digest="dgx-digest")},
            ),
        },
        {"ollama-local": True, "ollama-dgx": True},
    )


@pytest.mark.asyncio
async def test_lists_only_safe_completion_models_for_each_connection() -> None:
    providers = registry()
    service = MemoryExtractionSettingsService(
        FakeRepository(), providers, FreeOperationPolicy()
    )

    local = await service.list_models("ollama-local")
    dgx = await service.list_models("ollama-dgx")

    assert [item.name for item in local] == ["chat:latest"]
    assert [item.name for item in dgx] == ["gemma:31b"]


@pytest.mark.asyncio
async def test_model_disappearing_during_listing_is_omitted() -> None:
    providers = registry()
    provider = providers.providers["ollama-local"]
    provider.missing_on_inspect.add("chat:latest")
    service = MemoryExtractionSettingsService(
        FakeRepository(), providers, FreeOperationPolicy()
    )

    models = await service.list_models("ollama-local")

    assert models == []


@pytest.mark.asyncio
async def test_configure_saves_connection_model_digest_and_endpoint_identity() -> None:
    repository = FakeRepository()
    providers = registry()
    service = MemoryExtractionSettingsService(
        repository, providers, FreeOperationPolicy()
    )

    setting = await service.configure("ollama-dgx", "gemma:31b")

    assert setting.role is ModelRole.MEMORY_EXTRACTION
    assert setting.connection_id == "dgx"
    assert setting.provider_name == "ollama-dgx"
    assert setting.model_name == "gemma:31b"
    assert setting.model_digest == "dgx-digest"
    assert len(setting.endpoint_fingerprint) == 64
    assert repository.save_calls == 1


@pytest.mark.asyncio
async def test_rejects_unconfirmed_cloud_before_model_requests() -> None:
    providers = registry()
    providers.cloud_disabled["ollama-dgx"] = False
    provider = providers.providers["ollama-dgx"]
    service = MemoryExtractionSettingsService(
        FakeRepository(), providers, FreeOperationPolicy()
    )

    with pytest.raises(FreeOperationBlocked):
        await service.configure("ollama-dgx", "gemma:31b")

    assert provider.inspect_calls == []


@pytest.mark.asyncio
async def test_rejects_remote_provider_before_model_requests() -> None:
    providers = registry()
    provider = providers.providers["ollama-dgx"]
    provider.locality = Locality.REMOTE
    service = MemoryExtractionSettingsService(
        FakeRepository(), providers, FreeOperationPolicy()
    )

    with pytest.raises(FreeOperationBlocked):
        await service.list_models("ollama-dgx")

    assert provider.list_calls == 0
    assert provider.inspect_calls == []


@pytest.mark.asyncio
async def test_rejects_non_completion_or_digestless_model() -> None:
    providers = registry()
    repository = FakeRepository()
    service = MemoryExtractionSettingsService(
        repository, providers, FreeOperationPolicy()
    )

    with pytest.raises(ValidationError, match="文章生成"):
        await service.configure("ollama-local", "embed:latest")

    providers.providers["ollama-local"].models["chat:latest"] = model(
        "chat:latest", digest=""
    )
    with pytest.raises(ValidationError, match="digest"):
        await service.configure("ollama-local", "chat:latest")

    assert repository.save_calls == 0


@pytest.mark.asyncio
async def test_execution_revalidates_endpoint_model_and_digest() -> None:
    repository = FakeRepository()
    providers = registry()
    service = MemoryExtractionSettingsService(
        repository, providers, FreeOperationPolicy()
    )
    saved = await service.configure("ollama-dgx", "gemma:31b")

    assert await service.validated_configuration() == saved

    providers.providers["ollama-dgx"].models["gemma:31b"] = model(
        "gemma:31b", digest="changed"
    )
    with pytest.raises(ValidationError, match="更新"):
        await service.validated_configuration()

    providers.providers["ollama-dgx"].models["gemma:31b"] = model(
        "gemma:31b", digest="dgx-digest"
    )
    providers.providers["ollama-dgx"].endpoint = (
        "http://192.168.1.51:11434"
    )
    with pytest.raises(ValidationError, match="接続先"):
        await service.validated_configuration()


@pytest.mark.asyncio
async def test_missing_setting_or_registered_connection_is_rejected() -> None:
    providers = registry()
    service = MemoryExtractionSettingsService(
        FakeRepository(), providers, FreeOperationPolicy()
    )

    with pytest.raises(ValidationError, match="設定されていません"):
        await service.validated_configuration()

    repository = FakeRepository()
    configured = MemoryExtractionSettingsService(
        repository, providers, FreeOperationPolicy()
    )
    await configured.configure("ollama-dgx", "gemma:31b")
    providers.connections = [
        item for item in providers.connections if item.id != "dgx"
    ]
    with pytest.raises(ValidationError, match="登録されていない"):
        await configured.validated_configuration()


@pytest.mark.asyncio
async def test_configured_extractor_uses_selected_dgx_model_not_chat_model() -> None:
    repository = FakeRepository()
    providers = registry()
    settings = MemoryExtractionSettingsService(
        repository, providers, FreeOperationPolicy()
    )
    await settings.configure("ollama-dgx", "gemma:31b")
    extractor = ConfiguredMemoryCandidateExtractor(settings, providers)
    request = MemoryCandidateRequest(
        conversation_id="conversation-1",
        branch_id="branch-1",
        source_message_id="message-1",
        model_name="conversation-model-on-another-provider",
        content="アイスが好き",
        author_subject_id="user",
        allowed_subject_ids=frozenset({"user"}),
        allowed_knowledge_character_ids=frozenset({"character-1"}),
    )

    assert await extractor.extract(request) == ()

    dgx = providers.providers["ollama-dgx"]
    assert len(dgx.chat_requests) == 1
    assert dgx.chat_requests[0].model == "gemma:31b"
    assert dgx.chat_requests[0].response_format is not None
    assert providers.providers["ollama-local"].chat_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_content", "response_completed"),
    (
        ('{"candidates": []}', False),
        ("not-json", True),
    ),
)
async def test_configured_extractor_rejects_incomplete_or_invalid_output(
    response_content: str, response_completed: bool
) -> None:
    repository = FakeRepository()
    providers = registry()
    settings = MemoryExtractionSettingsService(
        repository, providers, FreeOperationPolicy()
    )
    await settings.configure("ollama-dgx", "gemma:31b")
    dgx = providers.providers["ollama-dgx"]
    dgx.response_content = response_content
    dgx.response_completed = response_completed
    extractor = ConfiguredMemoryCandidateExtractor(settings, providers)
    request = MemoryCandidateRequest(
        "conversation-1",
        "branch-1",
        "message-1",
        "chat-model",
        "アイスが好き",
        "user",
        frozenset({"user"}),
        frozenset({"character-1"}),
    )

    with pytest.raises(OllamaUnavailable, match="記憶抽出"):
        await extractor.extract(request)
