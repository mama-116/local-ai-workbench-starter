from pathlib import Path

import pytest

from local_llm_chat.domain.errors import FreeOperationBlocked
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.states import ProviderKind
from local_llm_chat.infrastructure.llm.ollama_registry import OllamaProviderRegistry


@pytest.mark.asyncio
async def test_includes_local_and_requested_lan_connection(tmp_path: Path) -> None:
    registry = OllamaProviderRegistry(
        tmp_path / "connections.json",
        tmp_path / "server.json",
        FreeOperationPolicy(),
    )
    try:
        endpoints = {item.endpoint for item in registry.list_connections()}
        assert "http://127.0.0.1:11434" in endpoints
        assert "http://192.168.1.17:11434" in endpoints
        assert "http://192.168.1.17:18080" in endpoints
        vllm = next(
            item
            for item in registry.list_connections()
            if item.endpoint == "http://192.168.1.17:18080"
        )
        assert vllm.provider_kind is ProviderKind.VLLM
        assert registry.cloud_is_disabled("ollama-lan-192-168-1-17") is False
        assert registry.cloud_is_disabled("vllm-lan-192-168-1-17") is True
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_saves_lan_confirmation_and_reloads_it(tmp_path: Path) -> None:
    connections_path = tmp_path / "connections.json"
    registry = OllamaProviderRegistry(
        connections_path,
        tmp_path / "server.json",
        FreeOperationPolicy(),
    )
    try:
        saved = await registry.save_connection(
            "lan-192-168-1-17",
            "推論PC",
            "http://192.168.1.17:11434",
            True,
        )
        assert registry.cloud_is_disabled(saved.provider_name) is True
    finally:
        await registry.close()

    reloaded = OllamaProviderRegistry(
        connections_path,
        tmp_path / "server.json",
        FreeOperationPolicy(),
    )
    try:
        restored = next(
            item
            for item in reloaded.list_connections()
            if item.id == "lan-192-168-1-17"
        )
        assert restored.display_name == "推論PC"
        assert restored.cloud_disabled_confirmed is True
    finally:
        await reloaded.close()


@pytest.mark.asyncio
async def test_rejects_public_connection(tmp_path: Path) -> None:
    registry = OllamaProviderRegistry(
        tmp_path / "connections.json",
        tmp_path / "server.json",
        FreeOperationPolicy(),
    )
    try:
        with pytest.raises(FreeOperationBlocked):
            await registry.save_connection(
                None,
                "外部",
                "http://8.8.8.8:11434",
                True,
            )
    finally:
        await registry.close()
