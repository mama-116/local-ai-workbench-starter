from __future__ import annotations

from local_llm_chat.application.services.memory_extraction_settings_service import (
    MemoryExtractionSettingsService,
)
from local_llm_chat.domain.errors import (
    AppError,
    OllamaUnavailable,
    ValidationError,
)
from local_llm_chat.domain.memory_candidates import (
    MemoryCandidateDraft,
    MemoryCandidateRequest,
)
from local_llm_chat.domain.models import ProviderMetadata
from local_llm_chat.domain.ports.llm_provider import LLMProviderRegistry
from local_llm_chat.infrastructure.llm.ollama_memory_candidate_extractor import (
    MAX_EXTRACTOR_RESPONSE_CHARACTERS,
    OllamaMemoryCandidateExtractor,
    memory_extraction_chat_request,
)


class ConfiguredMemoryCandidateExtractor:
    def __init__(
        self,
        settings: MemoryExtractionSettingsService,
        providers: LLMProviderRegistry,
    ) -> None:
        self._settings = settings
        self._providers = providers

    @property
    def metadata(self) -> ProviderMetadata:
        setting = self._settings.cached_configuration
        provider_name = (
            setting.provider_name
            if setting is not None
            else self._providers.default_provider_name
        )
        return self._providers.get(provider_name).metadata

    @property
    def cloud_is_disabled(self) -> bool:
        setting = self._settings.cached_configuration
        provider_name = (
            setting.provider_name
            if setting is not None
            else self._providers.default_provider_name
        )
        return self._providers.cloud_is_disabled(provider_name)

    async def close(self) -> None:
        return None

    async def extract(
        self, request: MemoryCandidateRequest
    ) -> tuple[MemoryCandidateDraft, ...]:
        try:
            setting = await self._settings.validated_configuration()
        except AppError as error:
            raise OllamaUnavailable(
                "記憶抽出設定を利用できないため、安全な簡易抽出へ切り替えます。"
            ) from error
        provider = self._providers.get(setting.provider_name)
        chat_request = memory_extraction_chat_request(
            request, setting.model_name
        )
        parts: list[str] = []
        characters = 0
        completed = False
        try:
            async for chunk in provider.stream_chat(chat_request):
                if chunk.content:
                    characters += len(chunk.content)
                    if characters > MAX_EXTRACTOR_RESPONSE_CHARACTERS:
                        raise OllamaUnavailable("記憶抽出の応答が長すぎます。")
                    parts.append(chunk.content)
                if chunk.done:
                    completed = True
        except AppError as error:
            raise OllamaUnavailable(
                "記憶抽出用Ollamaの応答を取得できません。"
            ) from error
        if not completed:
            raise OllamaUnavailable("記憶抽出の応答が完了前に終了しました。")
        try:
            return OllamaMemoryCandidateExtractor.parse_content(
                "".join(parts), request.content
            )
        except ValidationError as error:
            raise OllamaUnavailable("記憶抽出の応答が不正です。") from error
