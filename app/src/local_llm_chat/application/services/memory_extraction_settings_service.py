from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Protocol

from local_llm_chat.domain.errors import (
    FreeOperationBlocked,
    ModelUnavailable,
    ValidationError,
)
from local_llm_chat.domain.models import (
    ModelInfo,
    ModelRoleSetting,
    ProviderConnection,
)
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.ports.llm_provider import LLMProviderRegistry
from local_llm_chat.domain.states import ModelRole


class ModelRoleSettingsRepository(Protocol):
    async def get_model_role_setting(
        self, role: ModelRole
    ) -> ModelRoleSetting | None: ...

    async def save_model_role_setting(
        self, setting: ModelRoleSetting
    ) -> ModelRoleSetting: ...


class MemoryExtractionSettingsService:
    def __init__(
        self,
        repository: ModelRoleSettingsRepository,
        providers: LLMProviderRegistry,
        free_policy: FreeOperationPolicy,
    ) -> None:
        self._repository = repository
        self._providers = providers
        self._free_policy = free_policy

    async def configuration(self) -> ModelRoleSetting | None:
        return await self._repository.get_model_role_setting(
            ModelRole.MEMORY_EXTRACTION
        )

    async def list_models(self, provider_name: str) -> list[ModelInfo]:
        provider = self._providers.get(provider_name)
        self._require_safe_provider(provider_name)
        allowed: list[ModelInfo] = []
        for listed in await provider.list_models():
            try:
                inspected = await provider.inspect_model(listed.name)
            except ModelUnavailable:
                continue
            if "completion" not in inspected.capabilities:
                continue
            try:
                self._free_policy.require_model(inspected)
            except FreeOperationBlocked:
                continue
            allowed.append(inspected)
        return sorted(allowed, key=lambda item: item.name.casefold())

    async def configure(
        self, provider_name: str, model_name: str
    ) -> ModelRoleSetting:
        connection = self._find_connection(provider_name)
        provider = self._providers.get(provider_name)
        self._require_safe_provider(provider_name)
        model = await provider.inspect_model(model_name)
        self._validate_model(model)
        setting = ModelRoleSetting(
            role=ModelRole.MEMORY_EXTRACTION,
            connection_id=connection.id,
            provider_name=provider_name,
            endpoint_fingerprint=self._fingerprint(provider.metadata.endpoint),
            model_name=model.name,
            model_digest=model.digest,
            updated_at=datetime.now(UTC),
        )
        return await self._repository.save_model_role_setting(setting)

    async def validated_configuration(self) -> ModelRoleSetting:
        setting = await self.configuration()
        if setting is None:
            raise ValidationError("記憶抽出モデルが設定されていません。")
        connection = self._find_connection(setting.provider_name)
        if connection.id != setting.connection_id:
            raise ValidationError(
                "記憶抽出の接続先が変更されています。設定を保存し直してください。"
            )
        provider = self._providers.get(setting.provider_name)
        self._require_safe_provider(setting.provider_name)
        if (
            self._fingerprint(provider.metadata.endpoint)
            != setting.endpoint_fingerprint
        ):
            raise ValidationError(
                "記憶抽出の接続先が変更されています。設定を保存し直してください。"
            )
        model = await provider.inspect_model(setting.model_name)
        self._validate_model(model)
        if model.digest != setting.model_digest:
            raise ValidationError(
                "記憶抽出モデルが更新されています。設定を保存し直してください。"
            )
        return setting

    def _require_safe_provider(self, provider_name: str) -> None:
        provider = self._providers.get(provider_name)
        self._free_policy.require_provider(provider.metadata)
        self._free_policy.require_cloud_disabled(
            self._providers.cloud_is_disabled(provider_name)
        )

    def _find_connection(self, provider_name: str) -> ProviderConnection:
        connection = next(
            (
                item
                for item in self._providers.list_connections()
                if item.provider_name == provider_name
            ),
            None,
        )
        if connection is None:
            raise ValidationError("登録されていない記憶抽出接続先です。")
        return connection

    def _validate_model(self, model: ModelInfo) -> None:
        self._free_policy.require_model(model)
        if "completion" not in model.capabilities:
            raise ValidationError(
                "選択したモデルは記憶抽出に必要な文章生成へ対応していません。"
            )
        if not model.digest:
            raise ValidationError("記憶抽出モデルのdigestを確認できません。")

    @staticmethod
    def _fingerprint(endpoint: str) -> str:
        return hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
