from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Protocol

from local_llm_chat.domain.errors import FreeOperationBlocked, ValidationError
from local_llm_chat.domain.models import (
    DocumentChunk,
    EmbeddingConfiguration,
    EmbeddingProfile,
    ModelInfo,
    RagSearchResult,
)
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.ports.embedding_provider import (
    EmbeddingProvider,
    EmbeddingProviderRegistry,
)
from local_llm_chat.domain.states import EmbeddingIndexState


_BATCH_SIZE = 16
_EVALUATION_CASES = (
    ("東京は日本の首都で、政治と経済の中心です。", ("日本の首都はどこ", "国内政治の中心都市", "東京の位置づけ")),
    ("富士山は日本で最も標高が高い山です。", ("日本一高い山", "国内最高峰", "富士山の高さの特徴")),
    ("ブロッコリーにはビタミンCが含まれます。", ("緑の野菜のビタミン", "ブロッコリーの栄養", "ビタミンCを含む野菜")),
    ("犬は毎日の散歩と適度な運動を必要とします。", ("犬の運動習慣", "ペットを歩かせる頻度", "犬の健康に必要な活動")),
    ("水は標準気圧で摂氏100度になると沸騰します。", ("水が沸く温度", "標準気圧での沸点", "100度になる液体")),
    ("春には桜が咲き、多くの人が花見を楽しみます。", ("春の代表的な花", "花見の季節", "桜を眺める行事")),
    ("地震への備えとして水と非常食を保管します。", ("災害用の備蓄", "地震前に用意する物", "非常時の食料と飲料")),
    ("Pythonは読みやすさを重視したプログラミング言語です。", ("初心者にも読みやすい言語", "Pythonの設計上の特徴", "可読性を重視する開発言語")),
    ("電車は時刻表に沿って駅と駅の間を運行します。", ("鉄道の運行予定", "駅を結ぶ交通機関", "電車が従う予定表")),
    ("睡眠は心身の回復と記憶の整理に重要です。", ("休息が記憶に与える効果", "心身を回復させる習慣", "睡眠の役割")),
)
_NEGATIVE_QUERIES = (
    "量子もつれの数式",
    "火星探査機の軌道",
    "中世ヨーロッパの甲冑",
    "深海魚の発光器官",
    "為替デリバティブの評価",
    "古代エジプトの象形文字",
    "半導体露光装置",
    "オペラの声楽技法",
    "宇宙背景放射",
    "暗号資産の合意形成",
    "陶磁器の焼成温度",
    "野球の守備位置",
    "サンゴ礁の生態系",
    "航空機の揚力",
    "ワインの発酵",
    "裁判員制度",
    "海底ケーブル",
    "素粒子加速器",
    "古典音楽の和声",
    "ロボットアームの制御",
)


class EmbeddingRepository(Protocol):
    async def save_embedding_profile(
        self, profile: EmbeddingProfile
    ) -> EmbeddingProfile: ...

    async def set_desired_embedding_profile(self, profile_id: str) -> None: ...

    async def get_embedding_configuration(self) -> EmbeddingConfiguration: ...

    async def update_embedding_profile(
        self,
        profile_id: str,
        state: EmbeddingIndexState,
        total_chunks: int,
        embedded_chunks: int,
        last_error: str | None = None,
    ) -> EmbeddingProfile: ...

    async def complete_and_activate_embedding_profile(
        self, profile_id: str, total_chunks: int
    ) -> None: ...

    async def list_chunks_needing_embeddings(
        self, profile_id: str
    ) -> list[DocumentChunk]: ...

    async def save_chunk_embeddings(
        self,
        profile_id: str,
        embeddings: tuple[tuple[str, str, tuple[float, ...]], ...],
    ) -> None: ...

    async def count_profile_embeddings(self, profile_id: str) -> tuple[int, int]: ...

    async def search_semantic_chunks(
        self,
        profile_id: str,
        query_vector: tuple[float, ...],
        min_similarity: float,
        top_k: int,
        document_ids: tuple[str, ...] | None = None,
    ) -> list[RagSearchResult]: ...


UpdateCallback = Callable[[], Awaitable[None]]


class EmbeddingIndexService:
    def __init__(
        self,
        repository: EmbeddingRepository,
        providers: EmbeddingProviderRegistry,
        free_policy: FreeOperationPolicy,
    ) -> None:
        self._repository = repository
        self._providers = providers
        self._free_policy = free_policy
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._subscribers: list[UpdateCallback] = []
        self._closed = False

    def subscribe(self, callback: UpdateCallback) -> None:
        self._subscribers.append(callback)

    async def configuration(self) -> EmbeddingConfiguration:
        return await self._repository.get_embedding_configuration()

    async def list_models(self, provider_name: str) -> list[ModelInfo]:
        provider = self._providers.get_embedding(provider_name)
        self._free_policy.require_provider(provider.metadata)
        self._free_policy.require_cloud_disabled(
            self._providers.cloud_is_disabled(provider_name)
        )
        allowed: list[ModelInfo] = []
        for model in await provider.list_embedding_models():
            try:
                self._free_policy.require_model(model)
            except FreeOperationBlocked:
                continue
            allowed.append(model)
        return sorted(allowed, key=lambda item: item.name.casefold())

    async def configure(
        self, provider_name: str, model_name: str
    ) -> EmbeddingConfiguration:
        connection = next(
            (
                item
                for item in self._providers.list_connections()
                if item.provider_name == provider_name
            ),
            None,
        )
        if connection is None:
            raise ValidationError("登録されていない埋め込み接続先です。")
        provider = self._providers.get_embedding(provider_name)
        self._free_policy.require_provider(provider.metadata)
        self._free_policy.require_cloud_disabled(
            self._providers.cloud_is_disabled(provider_name)
        )
        model = await provider.inspect_embedding_model(model_name)
        self._free_policy.require_model(model)
        if not model.digest:
            raise ValidationError("埋め込みモデルのdigestを確認できません。")

        threshold, dimensions = await self._calibrate(provider_name, model_name)
        endpoint_fingerprint = hashlib.sha256(
            provider.metadata.endpoint.encode("utf-8")
        ).hexdigest()
        profile_key = "\0".join(
            (
                connection.id,
                endpoint_fingerprint,
                model_name,
                model.digest,
                str(dimensions),
            )
        )
        profile_id = hashlib.sha256(profile_key.encode("utf-8")).hexdigest()
        now = datetime.now(UTC)
        profile = await self._repository.save_embedding_profile(
            EmbeddingProfile(
                id=profile_id,
                connection_id=connection.id,
                provider_name=provider_name,
                endpoint_fingerprint=endpoint_fingerprint,
                model_name=model_name,
                model_digest=model.digest,
                vector_dimensions=dimensions,
                min_similarity=threshold,
                state=EmbeddingIndexState.PENDING,
                total_chunks=0,
                embedded_chunks=0,
                last_error=None,
                created_at=now,
                updated_at=now,
            )
        )
        await self._repository.set_desired_embedding_profile(profile.id)
        self._schedule(profile)
        await self._notify()
        return await self.configuration()

    async def resume(self) -> None:
        configuration = await self.configuration()
        if configuration.desired is not None:
            self._schedule(configuration.desired)

    async def documents_changed(self) -> None:
        await self._schedule_desired()

    async def search(
        self,
        query: str,
        top_k: int,
        document_ids: tuple[str, ...] | None = None,
    ) -> list[RagSearchResult]:
        configuration = await self.configuration()
        profile = configuration.active
        if profile is None:
            return []
        provider = await self._validated_provider(profile)
        vectors = await provider.embed(profile.model_name, (query,))
        if not vectors or len(vectors[0]) != profile.vector_dimensions:
            return []
        return await self._repository.search_semantic_chunks(
            profile.id,
            vectors[0],
            profile.min_similarity,
            top_k,
            document_ids,
        )

    async def close(self) -> None:
        self._closed = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _schedule_desired(self) -> None:
        configuration = await self.configuration()
        if configuration.desired is not None:
            self._schedule(configuration.desired)

    def _schedule(self, profile: EmbeddingProfile) -> None:
        if self._closed:
            return
        running = self._tasks.get(profile.id)
        if running is not None and not running.done():
            return
        task = asyncio.create_task(self._build(profile))
        self._tasks[profile.id] = task
        task.add_done_callback(lambda _: self._tasks.pop(profile.id, None))

    async def _build(self, profile: EmbeddingProfile) -> None:
        try:
            provider = await self._validated_provider(profile)
            while True:
                missing = await self._repository.list_chunks_needing_embeddings(
                    profile.id
                )
                total, _ = await self._repository.count_profile_embeddings(profile.id)
                completed = max(0, total - len(missing))
                await self._repository.update_embedding_profile(
                    profile.id,
                    EmbeddingIndexState.BUILDING,
                    total,
                    completed,
                )
                await self._notify()
                if not missing:
                    break
                for start in range(0, len(missing), _BATCH_SIZE):
                    batch = missing[start : start + _BATCH_SIZE]
                    vectors = await provider.embed(
                        profile.model_name, tuple(chunk.content for chunk in batch)
                    )
                    if len(vectors) != len(batch):
                        raise ValidationError("埋め込み応答件数が一致しません。")
                    records = tuple(
                        (
                            chunk.id,
                            hashlib.sha256(
                                chunk.content.encode("utf-8")
                            ).hexdigest(),
                            vector,
                        )
                        for chunk, vector in zip(batch, vectors, strict=True)
                    )
                    await self._repository.save_chunk_embeddings(
                        profile.id, records
                    )
                    completed += len(batch)
                    await self._repository.update_embedding_profile(
                        profile.id,
                        EmbeddingIndexState.BUILDING,
                        total,
                        completed,
                    )
                    await self._notify()
            total, embedded = await self._repository.count_profile_embeddings(profile.id)
            if embedded < total:
                raise ValidationError("埋め込み索引の件数が一致しません。")
            await self._repository.complete_and_activate_embedding_profile(
                profile.id, total
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            try:
                total, embedded = await self._repository.count_profile_embeddings(
                    profile.id
                )
                await self._repository.update_embedding_profile(
                    profile.id,
                    EmbeddingIndexState.FAILED,
                    total,
                    embedded,
                    str(error),
                )
            except Exception:
                pass
        finally:
            await self._notify()

    async def _calibrate(
        self, provider_name: str, model_name: str
    ) -> tuple[float, int]:
        provider = self._providers.get_embedding(provider_name)
        documents = tuple(item[0] for item in _EVALUATION_CASES)
        positives = tuple(
            (document_index, query)
            for document_index, (_, queries) in enumerate(_EVALUATION_CASES)
            for query in queries
        )
        inputs = (
            documents
            + tuple(query for _, query in positives)
            + _NEGATIVE_QUERIES
        )
        vectors = await provider.embed(model_name, inputs)
        if len(vectors) != len(inputs) or not vectors:
            raise ValidationError("埋め込み品質評価を完了できませんでした。")
        dimensions = len(vectors[0])
        if dimensions <= 0 or any(len(vector) != dimensions for vector in vectors):
            raise ValidationError("埋め込みベクトルの次元が一致しません。")
        document_vectors = vectors[: len(documents)]
        positive_vectors = vectors[
            len(documents) : len(documents) + len(positives)
        ]
        negative_vectors = vectors[len(documents) + len(positives) :]
        positive_scores = [
            [
                _cosine(query_vector, document_vector)
                for document_vector in document_vectors
            ]
            for query_vector in positive_vectors
        ]
        negative_scores = [
            [
                _cosine(query_vector, document_vector)
                for document_vector in document_vectors
            ]
            for query_vector in negative_vectors
        ]
        candidates = sorted(
            {
                -1.0,
                1.0,
                *(
                    max(-1.0, min(1.0, score + 1e-6))
                    for scores in positive_scores + negative_scores
                    for score in scores
                ),
            }
        )
        passing: list[tuple[float, float, float, float]] = []
        for threshold in candidates:
            ranks: list[int | None] = []
            for (expected, _), scores in zip(positives, positive_scores, strict=True):
                ranked = [
                    index
                    for index, score in sorted(
                        enumerate(scores), key=lambda item: item[1], reverse=True
                    )
                    if score >= threshold
                ][:5]
                ranks.append(
                    ranked.index(expected) + 1 if expected in ranked else None
                )
            recall = sum(rank is not None for rank in ranks) / len(ranks)
            mrr = sum(1 / rank for rank in ranks if rank is not None) / len(ranks)
            negative_no_hit = sum(
                all(score < threshold for score in scores)
                for scores in negative_scores
            ) / len(negative_scores)
            if recall >= 0.90 and mrr >= 0.80 and negative_no_hit >= 0.85:
                passing.append((mrr, negative_no_hit, recall, threshold))
        if not passing:
            raise ValidationError(
                "この埋め込みモデルは日本語RAGの品質基準を満たしませんでした。"
            )
        _, _, _, threshold = max(passing)
        return threshold, dimensions

    async def _validated_provider(
        self, profile: EmbeddingProfile
    ) -> EmbeddingProvider:
        provider = self._providers.get_embedding(profile.provider_name)
        self._free_policy.require_provider(provider.metadata)
        self._free_policy.require_cloud_disabled(
            self._providers.cloud_is_disabled(profile.provider_name)
        )
        current_fingerprint = hashlib.sha256(
            provider.metadata.endpoint.encode("utf-8")
        ).hexdigest()
        if current_fingerprint != profile.endpoint_fingerprint:
            raise ValidationError(
                "接続先が変更されています。用途別モデル設定を保存し直してください。"
            )
        model = await provider.inspect_embedding_model(profile.model_name)
        self._free_policy.require_model(model)
        if model.digest != profile.model_digest:
            raise ValidationError(
                "埋め込みモデルのdigestが変更されています。"
                "用途別モデル設定を保存し直してください。"
            )
        return provider

    async def _notify(self) -> None:
        if self._subscribers:
            await asyncio.gather(
                *(callback() for callback in self._subscribers),
                return_exceptions=True,
            )


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return sum(
        first * second for first, second in zip(left, right, strict=True)
    ) / (left_norm * right_norm)
