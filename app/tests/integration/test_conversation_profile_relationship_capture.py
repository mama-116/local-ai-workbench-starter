from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_llm_chat.application.services.memory_candidate_service import (
    DEFAULT_MEMORY_TEMPLATES,
    MemoryCandidateService,
)
from local_llm_chat.application.services.memory_capture_service import (
    MemoryCaptureRequest,
    MemoryCaptureService,
)
from local_llm_chat.application.services.relationship_profile_service import (
    RelationshipProfileService,
)
from local_llm_chat.domain.memory_candidates import (
    MemoryCandidateDraft,
    MemoryCandidateRequest,
)
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import ProviderMetadata
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.relationship_profile import (
    EvidenceContext,
    RelationshipApproval,
    RelationshipCandidateDraft,
    RelationshipCandidateRequest,
    RelationshipMeaning,
    RelationshipSeverity,
)
from local_llm_chat.domain.states import (
    CostClass,
    Locality,
    MemoryEvidenceMode,
    MemoryKind,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


@dataclass
class MemoryExtractor:
    draft: MemoryCandidateDraft

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            "memory", Locality.LOCAL, CostClass.NO_CHARGE, "http://127.0.0.1:11434"
        )

    @property
    def cloud_is_disabled(self) -> bool:
        return True

    async def extract(
        self, request: MemoryCandidateRequest
    ) -> tuple[MemoryCandidateDraft, ...]:
        return (self.draft,)


@dataclass
class RelationshipExtractor:
    draft: RelationshipCandidateDraft

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            "relationship",
            Locality.LOCAL,
            CostClass.NO_CHARGE,
            "http://127.0.0.1:11434",
        )

    @property
    def cloud_is_disabled(self) -> bool:
        return True

    async def extract(
        self, request: RelationshipCandidateRequest
    ) -> tuple[RelationshipCandidateDraft, ...]:
        return (self.draft,)


@pytest.mark.asyncio
async def test_chat_capture_creates_profile_and_confirmable_relationship(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    model = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation(
        "capture", character.id, model.id
    )
    content = "一番好きなのはチョコ。ありがとう"
    session = await repository.start_send(conversation.id, content)
    favorite = "一番好きなのはチョコ"
    thanks = "ありがとう"
    relationship_profiles = RelationshipProfileService(repository)
    memory = MemoryCandidateService(
        MemoryExtractor(
            MemoryCandidateDraft(
                "user",
                MemoryKind.PREFERENCE,
                "favorite_food",
                "チョコ",
                MemoryEvidenceMode.EXPLICIT,
                0,
                len(favorite),
            )
        ),
        FreeOperationPolicy(),
        DEFAULT_MEMORY_TEMPLATES,
    )
    capture = MemoryCaptureService(
        repository,
        memory,
        relationship_profiles,
        RelationshipExtractor(
            RelationshipCandidateDraft(
                character.character_id,
                RelationshipMeaning.POSITIVE_INTERACTION,
                RelationshipSeverity.LOW,
                EvidenceContext.DIRECT,
                content.index(thanks),
                len(content),
            )
        ),
    )

    result = await capture.capture(
        MemoryCaptureRequest(
            conversation.id,
            conversation.active_branch_id,
            session.user_message.id,
            "model",
            "user",
            frozenset({"user", character.character_id}),
            frozenset({character.character_id}),
        )
    )

    assert len(result.profile_event_ids) == 1
    assert len(result.relationship_event_ids) == 1
    profile_items = await relationship_profiles.list_profile_items(conversation.id)
    assert [(item.item_name, item.value) for item in profile_items] == [
        ("favorite_food", "チョコ")
    ]
    snapshot = await relationship_profiles.relationship_snapshot(
        conversation.id, character.character_id
    )
    assert snapshot.metrics.affinity == 50
    assert [event.approval for event in snapshot.pending_events] == [
        RelationshipApproval.PENDING_CONFIRMATION
    ]
    retry = await capture.capture(
        MemoryCaptureRequest(
            conversation.id,
            conversation.active_branch_id,
            session.user_message.id,
            "model",
            "user",
            frozenset({"user", character.character_id}),
            frozenset({character.character_id}),
        )
    )
    assert retry.profile_event_ids == ()
    assert len(
        await relationship_profiles.list_profile_items(conversation.id)
    ) == 1
    assert len(
        (
            await relationship_profiles.relationship_snapshot(
                conversation.id, character.character_id
            )
        ).pending_events
    ) == 1

    event_id = result.relationship_event_ids[0]
    await relationship_profiles.decide_relationship_candidate(
        conversation_id=conversation.id,
        event_id=event_id,
        state="confirmed",
        operation_id="confirm",
        recorded_at=NOW,
    )
    confirmed = await relationship_profiles.relationship_snapshot(
        conversation.id, character.character_id
    )
    assert confirmed.metrics.affinity == 51

    await relationship_profiles.decide_relationship_candidate(
        conversation_id=conversation.id,
        event_id=event_id,
        state="undone",
        operation_id="undo",
        recorded_at=NOW,
    )
    undone = await relationship_profiles.relationship_snapshot(
        conversation.id, character.character_id
    )
    assert undone.metrics.affinity == 50

    other = await repository.create_conversation("other", character.id, model.id)
    with pytest.raises(ValidationError, match="現在のキャスト"):
        await relationship_profiles.decide_relationship_candidate(
            conversation_id=other.id,
            event_id=event_id,
            state="confirmed",
            operation_id="wrong-worldline",
            recorded_at=NOW,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context",
    (
        EvidenceContext.QUOTED,
        EvidenceContext.HYPOTHETICAL,
        EvidenceContext.NARRATIVE,
        EvidenceContext.THIRD_PARTY,
        EvidenceContext.ROLEPLAY,
        EvidenceContext.UNKNOWN,
    ),
)
async def test_non_direct_context_never_creates_relationship_candidate(
    tmp_path: Path, context: EvidenceContext
) -> None:
    repository = SQLiteAppRepository(tmp_path / f"{context.value}.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    model = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation(
        "context", character.id, model.id
    )
    content = "最低だ"
    session = await repository.start_send(conversation.id, content)
    service = RelationshipProfileService(repository)
    ids = await service.capture_relationship_candidates(
        request=RelationshipCandidateRequest(
            conversation.id,
            conversation.active_branch_id,
            session.user_message.id,
            "model",
            content,
            frozenset({character.character_id}),
        ),
        extractor=RelationshipExtractor(
            RelationshipCandidateDraft(
                character.character_id,
                RelationshipMeaning.BOUNDARY_VIOLATION,
                RelationshipSeverity.HIGH,
                context,
                0,
                len(content),
            )
        ),
        recorded_at=NOW,
    )
    assert ids == ()
