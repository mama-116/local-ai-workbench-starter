from __future__ import annotations

import asyncio
import sqlite3
import struct
from pathlib import Path

import pytest

from local_llm_chat.application.services.embedding_index_service import (
    EmbeddingIndexService,
)
from local_llm_chat.application.services.rag_service import RagService
from local_llm_chat.domain.errors import FreeOperationBlocked, OllamaUnavailable
from local_llm_chat.domain.models import (
    ModelInfo,
    ProviderConnection,
    ProviderMetadata,
)
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.states import (
    CostClass,
    EmbeddingIndexState,
    Locality,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


class FakeEmbeddingProvider:
    def __init__(self, endpoint: str) -> None:
        self._metadata = ProviderMetadata(
            "fake", Locality.LOCAL, CostClass.NO_CHARGE, endpoint
        )
        self.fail_index_for: set[str] = set()
        self.fail_inputs_containing: set[str] = set()
        self.block_index_for: dict[str, asyncio.Event] = {}
        self.digest_overrides: dict[str, str] = {}

    @property
    def metadata(self) -> ProviderMetadata:
        return self._metadata

    async def list_embedding_models(self) -> list[ModelInfo]:
        return [
            self._model("embed-good"),
            self._model("embed-fails-during-index"),
        ]

    async def inspect_embedding_model(self, model_name: str) -> ModelInfo:
        return self._model(model_name)

    async def embed(
        self, model_name: str, inputs: tuple[str, ...]
    ) -> tuple[tuple[float, ...], ...]:
        if len(inputs) == 60:
            documents = tuple(_unit(index) for index in range(10))
            positives = tuple(
                _unit(index)
                for index in range(10)
                for _ in range(3)
            )
            negatives = tuple((0.0,) * 10 for _ in range(20))
            return documents + positives + negatives
        if any(
            marker in text
            for marker in self.fail_inputs_containing
            for text in inputs
        ):
            raise OllamaUnavailable("増分索引中に接続が切れました。")
        blocker = self.block_index_for.get(model_name)
        if blocker is not None:
            await blocker.wait()
        if model_name in self.fail_index_for:
            raise OllamaUnavailable("索引中に接続が切れました。")
        return tuple(_unit(0) for _ in inputs)

    def _model(self, name: str) -> ModelInfo:
        return ModelInfo(
            name=name,
            size_bytes=1024,
            format="gguf",
            family="test",
            parameter_size="small",
            quantization="F16",
            digest=self.digest_overrides.get(name, f"digest-{name}"),
            capabilities=("embedding",),
        )


class FakeEmbeddingRegistry:
    def __init__(self) -> None:
        self.local = FakeEmbeddingProvider("http://127.0.0.1:11434")
        self.dgx = FakeEmbeddingProvider("http://192.168.1.17:11434")
        self._connections = [
            ProviderConnection(
                "local",
                "ollama-local",
                "このPC",
                self.local.metadata.endpoint,
                False,
                True,
            ),
            ProviderConnection(
                "dgx",
                "ollama-dgx",
                "DGX Spark",
                self.dgx.metadata.endpoint,
                True,
            ),
        ]
        self.disabled_providers: set[str] = set()

    def list_connections(self) -> list[ProviderConnection]:
        return list(self._connections)

    def get_embedding(self, provider_name: str) -> FakeEmbeddingProvider:
        return self.local if provider_name == "ollama-local" else self.dgx

    def cloud_is_disabled(self, provider_name: str) -> bool:
        return provider_name not in self.disabled_providers


async def _wait_for_state(
    service: EmbeddingIndexService,
    state: EmbeddingIndexState,
) -> None:
    for _ in range(200):
        configuration = await service.configuration()
        if configuration.desired and configuration.desired.state is state:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"embedding state did not become {state}")


async def test_switch_failure_keeps_previous_index_and_lexical_fallback(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    await RagService(repository).register_document(
        "cats.md", "猫は狭い箱に入ることを好みます。".encode()
    )
    registry = FakeEmbeddingRegistry()
    service = EmbeddingIndexService(
        repository, registry, FreeOperationPolicy()
    )
    try:
        assert [model.name for model in await service.list_models("ollama-local")] == [
            "embed-fails-during-index",
            "embed-good",
        ]
        assert [model.name for model in await service.list_models("ollama-dgx")] == [
            "embed-fails-during-index",
            "embed-good",
        ]

        await service.configure("ollama-local", "embed-good")
        await _wait_for_state(service, EmbeddingIndexState.READY)
        first = await service.configuration()
        assert first.active is not None
        assert first.active.provider_name == "ollama-local"
        assert await service.search("猫の好み", 5)

        registry.local.digest_overrides["embed-good"] = "changed-digest"
        hybrid = RagService(repository, service)
        fallback = await hybrid.search("猫")
        assert [result.citation.document_title for result in fallback] == ["cats.md"]
        registry.local.digest_overrides.clear()

        registry.local.fail_inputs_containing.add("新資料")
        await hybrid.register_document(
            "new.md", "新資料は索引中の追加を再現します。".encode()
        )
        await _wait_for_state(service, EmbeddingIndexState.FAILED)
        incremental_failure = await service.configuration()
        assert incremental_failure.active is not None
        assert incremental_failure.active.id == first.active.id
        assert await service.search("猫の好み", 5)

        registry.dgx.fail_index_for.add("embed-fails-during-index")
        await service.configure("ollama-dgx", "embed-fails-during-index")
        await _wait_for_state(service, EmbeddingIndexState.FAILED)
        failed = await service.configuration()

        assert failed.desired is not None
        assert failed.desired.provider_name == "ollama-dgx"
        assert failed.active is not None
        assert failed.active.id == first.active.id
        assert "接続が切れました" in (failed.desired.last_error or "")
    finally:
        await service.close()


async def test_restart_repairs_corrupt_vector_and_lan_confirmation_is_required(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    await RagService(repository).register_document(
        "restart.md", "再起動後も索引を復元します。".encode()
    )
    registry = FakeEmbeddingRegistry()
    first_service = EmbeddingIndexService(
        repository, registry, FreeOperationPolicy()
    )
    await first_service.configure("ollama-local", "embed-good")
    await _wait_for_state(first_service, EmbeddingIndexState.READY)
    configured = await first_service.configuration()
    assert configured.active is not None
    profile_id = configured.active.id
    await first_service.close()

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE chunk_embeddings SET vector_blob = ? WHERE profile_id = ?",
            (
                struct.pack("<10f", float("nan"), *([0.0] * 9)),
                profile_id,
            ),
        )
        connection.commit()
    assert await repository.list_chunks_needing_embeddings(profile_id)

    resumed = EmbeddingIndexService(repository, registry, FreeOperationPolicy())
    try:
        await resumed.resume()
        for _ in range(200):
            configuration = await resumed.configuration()
            if (
                configuration.active is not None
                and configuration.active.state is EmbeddingIndexState.READY
                and not await repository.list_chunks_needing_embeddings(profile_id)
            ):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("corrupt embedding was not rebuilt")
    finally:
        await resumed.close()

    registry.disabled_providers.add("ollama-dgx")
    guarded = EmbeddingIndexService(repository, registry, FreeOperationPolicy())
    try:
        with pytest.raises(FreeOperationBlocked, match="Cloud"):
            await guarded.list_models("ollama-dgx")
    finally:
        await guarded.close()


async def test_document_added_during_build_is_included_before_activation(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    await RagService(repository).register_document(
        "first.md", "最初の資料です。".encode()
    )
    registry = FakeEmbeddingRegistry()
    blocker = asyncio.Event()
    registry.local.block_index_for["embed-good"] = blocker
    service = EmbeddingIndexService(repository, registry, FreeOperationPolicy())
    rag = RagService(repository, service)
    try:
        await service.configure("ollama-local", "embed-good")
        await _wait_for_state(service, EmbeddingIndexState.BUILDING)
        await rag.register_document("second.md", "構築中に追加した資料です。".encode())
        blocker.set()
        await _wait_for_state(service, EmbeddingIndexState.READY)

        configuration = await service.configuration()
        assert configuration.active is not None
        assert configuration.active.total_chunks == 2
        assert configuration.active.embedded_chunks == 2
    finally:
        blocker.set()
        await service.close()


async def test_completed_superseded_profile_is_cached_without_reactivation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    await RagService(repository).register_document(
        "switch.md", "切替競合を確認する資料です。".encode()
    )
    registry = FakeEmbeddingRegistry()
    blocker = asyncio.Event()
    registry.local.block_index_for["embed-good"] = blocker
    service = EmbeddingIndexService(repository, registry, FreeOperationPolicy())
    try:
        await service.configure("ollama-local", "embed-good")
        await _wait_for_state(service, EmbeddingIndexState.BUILDING)
        first = await service.configuration()
        assert first.desired is not None
        superseded_id = first.desired.id

        await service.configure("ollama-dgx", "embed-fails-during-index")
        await _wait_for_state(service, EmbeddingIndexState.READY)
        selected = await service.configuration()
        assert selected.active is not None
        selected_id = selected.active.id
        assert selected_id != superseded_id

        blocker.set()
        for _ in range(200):
            with sqlite3.connect(database_path) as connection:
                row = connection.execute(
                    "SELECT state FROM embedding_profiles WHERE id = ?",
                    (superseded_id,),
                ).fetchone()
            if row is not None and row[0] == EmbeddingIndexState.READY.value:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("superseded profile was not cached as ready")

        final = await service.configuration()
        assert final.active is not None
        assert final.active.id == selected_id
    finally:
        blocker.set()
        await service.close()


def _unit(index: int) -> tuple[float, ...]:
    return tuple(1.0 if position == index else 0.0 for position in range(10))
