from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_llm_chat.application.services.relationship_profile_service import (
    RelationshipProfileService,
)
from local_llm_chat.application.services.relationship_turn_reception_service import (
    RelationshipTurnReceptionService,
)
from local_llm_chat.domain.models import ProviderMetadata
from local_llm_chat.domain.relationship_profile import (
    EvidenceContext,
    RelationshipCandidateDraft,
    RelationshipCandidateRequest,
    RelationshipMeaning,
    RelationshipApproval,
    RelationshipSeverity,
)
from local_llm_chat.domain.states import CostClass, Locality
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


NOW = datetime(2026, 7, 27, tzinfo=UTC)


class Extractor:
    def __init__(self, draft: RelationshipCandidateDraft) -> None:
        self._draft = draft

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            "test", Locality.LOCAL, CostClass.NO_CHARGE, "http://127.0.0.1:11434"
        )

    @property
    def cloud_is_disabled(self) -> bool:
        return True

    async def extract(
        self, request: RelationshipCandidateRequest
    ) -> tuple[RelationshipCandidateDraft, ...]:
        return (self._draft,)

    async def close(self) -> None:
        return None


async def _setup(
    database: Path, content: str
) -> tuple[
    SQLiteAppRepository,
    RelationshipProfileService,
    str,
    str,
    str,
    str,
]:
    repository = SQLiteAppRepository(database)
    await repository.initialize()
    character = await repository.ensure_default_character()
    model = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation(
        "turn reception", character.id, model.id
    )
    session = await repository.start_send(conversation.id, content)
    return (
        repository,
        RelationshipProfileService(repository),
        conversation.id,
        conversation.active_branch_id,
        session.user_message.id,
        character.character_id,
    )


@pytest.mark.asyncio
async def test_pending_conflict_is_available_to_same_turn_context_without_metric_change(
    tmp_path: Path,
) -> None:
    content = "さっきの言い方は嫌だった。説明してほしい"
    repository, profiles, conversation_id, branch_id, message_id, character_id = (
        await _setup(tmp_path / "conflict.sqlite3", content)
    )
    service = RelationshipTurnReceptionService(
        repository,
        profiles,
        Extractor(
            RelationshipCandidateDraft(
                character_id,
                RelationshipMeaning.CONFLICT,
                RelationshipSeverity.LOW,
                EvidenceContext.DIRECT,
                0,
                len(content),
            )
        ),
    )

    first = await service.capture(
        conversation_id=conversation_id,
        branch_id=branch_id,
        source_message_id=message_id,
        character_ids=(character_id,),
    )
    second = await service.capture(
        conversation_id=conversation_id,
        branch_id=branch_id,
        source_message_id=message_id,
        character_ids=(character_id,),
    )
    context = await profiles.render_generation_context(
        conversation_id=conversation_id,
        character_ids=(character_id,),
        provider_endpoint="http://127.0.0.1:11434",
        current_source_message_id=message_id,
    )
    snapshot = await profiles.relationship_snapshot(conversation_id, character_id)

    assert first.event_ids == second.event_ids
    assert first.meanings == (RelationshipMeaning.CONFLICT,)
    assert '"turn_reception":"conflict"' in context
    assert snapshot.metrics.affinity == 50
    assert len(snapshot.pending_events) == 1


@pytest.mark.asyncio
async def test_safe_positive_reception_auto_applies_once_and_lan_envelope_is_anonymous(
    tmp_path: Path,
) -> None:
    content = "約束どおり戻ったよ"
    repository, profiles, conversation_id, branch_id, message_id, character_id = (
        await _setup(tmp_path / "positive.sqlite3", content)
    )
    service = RelationshipTurnReceptionService(
        repository,
        profiles,
        Extractor(
            RelationshipCandidateDraft(
                character_id,
                RelationshipMeaning.KEPT_COMMITMENT,
                RelationshipSeverity.LOW,
                EvidenceContext.DIRECT,
                0,
                len(content),
            )
        ),
    )

    await service.capture(
        conversation_id=conversation_id,
        branch_id=branch_id,
        source_message_id=message_id,
        character_ids=(character_id,),
    )
    await service.capture(
        conversation_id=conversation_id,
        branch_id=branch_id,
        source_message_id=message_id,
        character_ids=(character_id,),
    )
    snapshot = await profiles.relationship_snapshot(conversation_id, character_id)
    envelope = await profiles.render_generation_context(
        conversation_id=conversation_id,
        character_ids=(character_id,),
        provider_endpoint="http://192.168.1.17:11434",
        allow_private_lan_behavior=True,
        current_source_message_id=message_id,
    )

    assert snapshot.metrics.affinity == 51
    assert snapshot.metrics.trust == 32
    assert len(snapshot.metrics.applied_event_ids) == 1
    assert '"turn_reception":"kept_commitment"' in envelope
    assert character_id not in envelope
    assert message_id not in envelope
    assert content not in envelope


@pytest.mark.asyncio
async def test_repeated_safe_positive_receptions_apply_five_of_ten_without_duplicate_runs(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "cadence.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    model = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation(
        "cadence", character.id, model.id
    )
    profiles = RelationshipProfileService(repository)

    for index in range(10):
        content = f"ありがとう。助かったよ {index}"
        session = await repository.start_send(conversation.id, content)
        service = RelationshipTurnReceptionService(
            repository,
            profiles,
            Extractor(
                RelationshipCandidateDraft(
                    character.character_id,
                    RelationshipMeaning.POSITIVE_INTERACTION,
                    RelationshipSeverity.LOW,
                    EvidenceContext.DIRECT,
                    0,
                    len(content),
                )
            ),
        )
        await service.capture(
            conversation_id=conversation.id,
            branch_id=conversation.active_branch_id,
            source_message_id=session.user_message.id,
            character_ids=(character.character_id,),
        )
        await service.capture(
            conversation_id=conversation.id,
            branch_id=conversation.active_branch_id,
            source_message_id=session.user_message.id,
            character_ids=(character.character_id,),
        )

    continuity = await repository.get_continuity_for_conversation(conversation.id)
    events = await repository.list_relationship_events(
        continuity.id,
        continuity.user_profile_id,
        character.character_id,
        include_unapplied=True,
    )

    assert len(events) == 10
    assert (
        sum(event.approval is RelationshipApproval.AUTO_APPLIED for event in events)
        == 5
    )
