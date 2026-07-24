from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from uuid import uuid4

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import ProviderConnection
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.ports.llm_provider import LLMProvider
from local_llm_chat.domain.ports.embedding_provider import EmbeddingProvider
from local_llm_chat.infrastructure.llm.ollama_provider import OllamaProvider
from local_llm_chat.infrastructure.settings import is_ollama_cloud_disabled

LOCAL_PROVIDER_NAME = "ollama-local"
LOCAL_ENDPOINT = "http://127.0.0.1:11434"

_DEFAULT_CONNECTIONS = (
    ProviderConnection(
        id="local",
        provider_name=LOCAL_PROVIDER_NAME,
        display_name="このPC",
        endpoint=LOCAL_ENDPOINT,
        cloud_disabled_confirmed=False,
        is_builtin=True,
    ),
    ProviderConnection(
        id="lan-192-168-1-17",
        provider_name="ollama-lan-192-168-1-17",
        display_name="LAN 192.168.1.17",
        endpoint="http://192.168.1.17:11434",
        cloud_disabled_confirmed=False,
    ),
)


class OllamaProviderRegistry:
    def __init__(
        self,
        connections_path: Path,
        local_server_config_path: Path,
        free_policy: FreeOperationPolicy,
    ) -> None:
        self._connections_path = connections_path
        self._local_server_config_path = local_server_config_path
        self._free_policy = free_policy
        self._connections = self._load_connections()
        self._providers = {
            connection.provider_name: OllamaProvider(
                connection.endpoint, name=connection.provider_name
            )
            for connection in self._connections
        }

    @property
    def default_provider_name(self) -> str:
        return LOCAL_PROVIDER_NAME

    def list_connections(self) -> list[ProviderConnection]:
        return list(self._connections)

    def get(self, provider_name: str) -> LLMProvider:
        try:
            return self._providers[provider_name]
        except KeyError as error:
            raise ValidationError("登録されていないOllama接続先です。") from error

    def get_embedding(self, provider_name: str) -> EmbeddingProvider:
        try:
            return self._providers[provider_name]
        except KeyError as error:
            raise ValidationError("登録されていないOllama接続先です。") from error

    def cloud_is_disabled(self, provider_name: str) -> bool:
        connection = self._connection(provider_name)
        if connection.provider_name == LOCAL_PROVIDER_NAME:
            return is_ollama_cloud_disabled(self._local_server_config_path)
        return connection.cloud_disabled_confirmed

    async def save_connection(
        self,
        connection_id: str | None,
        display_name: str,
        endpoint: str,
        cloud_disabled_confirmed: bool,
    ) -> ProviderConnection:
        name = display_name.strip()
        normalized_endpoint = endpoint.strip().rstrip("/")
        if not name:
            raise ValidationError("接続先の表示名を入力してください。")
        self._free_policy.require_endpoint(normalized_endpoint)

        existing = next(
            (item for item in self._connections if item.id == connection_id), None
        )
        if existing is not None and existing.is_builtin:
            raise ValidationError("このPCの接続設定は変更できません。")
        if any(
            item.endpoint == normalized_endpoint and item.id != connection_id
            for item in self._connections
        ):
            raise ValidationError("同じOllama接続先がすでに登録されています。")

        target_id = existing.id if existing else f"lan-{uuid4()}"
        provider_name = existing.provider_name if existing else f"ollama-{target_id}"
        saved = ProviderConnection(
            id=target_id,
            provider_name=provider_name,
            display_name=name,
            endpoint=normalized_endpoint,
            cloud_disabled_confirmed=cloud_disabled_confirmed,
        )
        self._connections = [
            saved if item.id == target_id else item for item in self._connections
        ]
        if existing is None:
            self._connections.append(saved)
        old_provider = self._providers.get(provider_name)
        self._providers[provider_name] = OllamaProvider(
            saved.endpoint, name=provider_name
        )
        if old_provider is not None:
            await old_provider.close()
        await asyncio.to_thread(self._write_connections)
        return saved

    async def close(self) -> None:
        await asyncio.gather(
            *(provider.close() for provider in self._providers.values())
        )

    def _connection(self, provider_name: str) -> ProviderConnection:
        connection = next(
            (item for item in self._connections if item.provider_name == provider_name),
            None,
        )
        if connection is None:
            raise ValidationError("登録されていないOllama接続先です。")
        return connection

    def _load_connections(self) -> list[ProviderConnection]:
        loaded: list[ProviderConnection] = []
        try:
            raw = json.loads(self._connections_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            raw = []
        if isinstance(raw, list):
            for item in raw:
                connection = self._parse_connection(item)
                if connection is not None:
                    loaded.append(connection)

        by_id = {item.id: item for item in loaded}
        for default in _DEFAULT_CONNECTIONS:
            by_id.setdefault(default.id, default)
        connections = list(by_id.values())
        connections.sort(key=lambda item: (not item.is_builtin, item.display_name.lower()))
        return connections

    def _parse_connection(self, raw: object) -> ProviderConnection | None:
        if not isinstance(raw, dict):
            return None
        try:
            connection = ProviderConnection(
                id=str(raw["id"]),
                provider_name=str(raw["provider_name"]),
                display_name=str(raw["display_name"]),
                endpoint=str(raw["endpoint"]).rstrip("/"),
                cloud_disabled_confirmed=bool(raw["cloud_disabled_confirmed"]),
                is_builtin=bool(raw.get("is_builtin", False)),
            )
            self._free_policy.require_endpoint(connection.endpoint)
        except (KeyError, TypeError, ValidationError):
            return None
        except Exception:
            return None
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", connection.provider_name):
            return None
        return connection

    def _write_connections(self) -> None:
        self._connections_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "id": item.id,
                "provider_name": item.provider_name,
                "display_name": item.display_name,
                "endpoint": item.endpoint,
                "cloud_disabled_confirmed": item.cloud_disabled_confirmed,
                "is_builtin": item.is_builtin,
            }
            for item in self._connections
        ]
        temporary = self._connections_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self._connections_path)
