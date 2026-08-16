import sqlite3
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

import pytest

from local_llm_chat.application.services.memory_candidate_service import (
    MemoryCandidateService,
)
from local_llm_chat.application.services.memory_capture_service import (
    MemoryCaptureRequest,
    MemoryCaptureService,
)
from local_llm_chat.domain.canonical_memory import (
    CanonicalMemoryEvent,
    MemoryApprovalDecision,
)
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.memory_candidates import (
    MemoryCandidateDraft,
    MemoryCandidateRequest,
    MemoryTemplate,
)
from local_llm_chat.domain.models import ProviderMetadata
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.states import (
    CostClass,
    Locality,
    MemoryApprovalState,
    MemoryCandidateDisposition,
    MemoryCardinality,
    MemoryEvidenceMode,
    MemoryFactState,
    MemoryKind,
    MessageState,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


@dataclass
class FakeExtractor:
    drafts: tuple[MemoryCandidateDraft, ...]
    calls: int = 0

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            "memory-local",
            Locality.LOCAL,
            CostClass.NO_CHARGE,
            "http://127.0.0.1:11434",
        )

    @property
    def cloud_is_disabled(self) -> bool:
        return True

    async def extract(
        self, request: MemoryCandidateRequest
    ) -> tuple[MemoryCandidateDraft, ...]:
        self.calls += 1
        return self.drafts


def _span(content: str, evidence: str) -> tuple[int, int]:
    start = content.index(evidence)
    return start, start + len(evidence)


@pytest.mark.asyncio
async def test_capture_loads_trusted_message_classifies_and_atomically_persists(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile(
        "ollama-local", "test-local", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "記憶取得", character.id, profile.id
    )
    content = "一番好きなのはチョコアイス。苺アレルギーです。田中は猫が好きそう。"
    session = await repository.start_send(conversation.id, content)

    favorite = "一番好きなのはチョコアイス"
    allergy = "苺アレルギーです"
    inference = "田中は猫が好きそう"
    extractor = FakeExtractor(
        (
            MemoryCandidateDraft(
                "user",
                MemoryKind.PREFERENCE,
                "favorite_food",
                "チョコアイス",
                MemoryEvidenceMode.EXPLICIT,
                *_span(content, favorite),
            ),
            MemoryCandidateDraft(
                "user",
                MemoryKind.SAFETY_CONSTRAINT,
                "food_allergy",
                "苺",
                MemoryEvidenceMode.EXPLICIT,
                *_span(content, allergy),
            ),
            MemoryCandidateDraft(
                "character-tanaka",
                MemoryKind.PREFERENCE,
                "liked_food",
                "猫",
                MemoryEvidenceMode.INFERRED,
                *_span(content, inference),
            ),
        )
    )
    classifier = MemoryCandidateService(
        extractor,
        FreeOperationPolicy(),
        (
            MemoryTemplate(
                MemoryKind.PREFERENCE,
                "favorite_food",
                MemoryCardinality.SINGLE,
                False,
                (r"一番好きなのは\s*{value}",),
            ),
            MemoryTemplate(
                MemoryKind.SAFETY_CONSTRAINT,
                "food_allergy",
                MemoryCardinality.MULTIPLE,
                True,
                (r"{value}\s*アレルギー",),
            ),
            MemoryTemplate(
                MemoryKind.PREFERENCE,
                "liked_food",
                MemoryCardinality.MULTIPLE,
                False,
                (r"{value}\s*が好き",),
            ),
        ),
    )
    capture = MemoryCaptureService(repository, classifier)
    request = MemoryCaptureRequest(
        conversation_id=conversation.id,
        branch_id=conversation.active_branch_id,
        source_message_id=session.user_message.id,
        model_name="test-local",
        author_subject_id="user",
        allowed_subject_ids=frozenset({"user", "character-tanaka"}),
        allowed_knowledge_character_ids=frozenset({character.character_id}),
        known_by_character_ids=frozenset({character.character_id}),
    )

    result = await capture.capture(request)

    assert [item.disposition for item in result.candidates] == [
        MemoryCandidateDisposition.AUTO_SAVE,
        MemoryCandidateDisposition.REQUIRE_CONFIRMATION,
        MemoryCandidateDisposition.BLOCK,
    ]
    assert len(result.persisted_event_ids) == 2
    current = await repository.project_canonical_memory(
        conversation.id,
        conversation.active_branch_id,
        character.character_id,
        current_source_message_id=session.user_message.id,
    )
    future = await repository.project_canonical_memory(
        conversation.id,
        conversation.active_branch_id,
        character.character_id,
    )
    assert {fact.value for fact in current} == {"チョコアイス", "苺"}
    assert [fact.value for fact in future] == ["チョコアイス"]

    retry = await capture.capture(request)
    assert retry.persisted_event_ids == result.persisted_event_ids
    assert extractor.calls == 2

    reopened = SQLiteAppRepository(database_path)
    await reopened.initialize()
    assert await reopened.project_canonical_memory(
        conversation.id,
        conversation.active_branch_id,
        character.character_id,
        current_source_message_id=session.user_message.id,
    ) == current

    def extra_event(identifier: str, known_by: frozenset[str]) -> CanonicalMemoryEvent:
        return CanonicalMemoryEvent(
            id=identifier,
            conversation_id=conversation.id,
            branch_id=conversation.active_branch_id,
            subject_id="user",
            kind=MemoryKind.PREFERENCE,
            slot="liked_food",
            value=identifier,
            cardinality=MemoryCardinality.MULTIPLE,
            approval=MemoryApprovalState.AUTO_SAVED,
            source_message_id=session.user_message.id,
            known_by_character_ids=known_by,
            supersedes_event_id=None,
            effective_at=session.user_message.created_at,
            recorded_at=session.user_message.created_at,
        )

    with pytest.raises(ValidationError):
        await reopened.append_canonical_memory_events(
            (
                extra_event("would-partially-save", frozenset()),
                extra_event("invalid-secret", frozenset({"missing-character"})),
            )
        )
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM canonical_memory_events"
        ).fetchone() == (2,)


@pytest.mark.asyncio
async def test_capture_rejects_untrusted_source_before_extraction(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile("ollama-local", "test-local", {})
    conversation = await repository.create_conversation("記憶取得", character.id, profile.id)
    session = await repository.start_send(conversation.id, "一番好きなのはチョコアイス")
    extractor = FakeExtractor(())
    classifier = MemoryCandidateService(extractor, FreeOperationPolicy(), ())
    capture = MemoryCaptureService(repository, classifier)

    with pytest.raises(ValidationError):
        await capture.capture(
            MemoryCaptureRequest(
                conversation_id=conversation.id,
                branch_id="not-this-branch",
                source_message_id=session.user_message.id,
                model_name="test-local",
                author_subject_id="user",
                allowed_subject_ids=frozenset({"user"}),
                allowed_knowledge_character_ids=frozenset({character.character_id}),
                known_by_character_ids=frozenset({character.character_id}),
            )
        )

    assert extractor.calls == 0


@pytest.mark.asyncio
async def test_capture_single_value_supersedes_old_fact_without_deleting_history(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile("ollama-local", "test-local", {})
    conversation = await repository.create_conversation("好みの変更", character.id, profile.id)
    template = MemoryTemplate(
        MemoryKind.PREFERENCE,
        "favorite_food",
        MemoryCardinality.SINGLE,
        False,
        (r"一番好きなのは\s*{value}",),
    )

    async def capture_favorite(value: str) -> tuple[str, ...]:
        content = f"一番好きなのは{value}"
        session = await repository.start_send(conversation.id, content)
        extractor = FakeExtractor(
            (
                MemoryCandidateDraft(
                    "user",
                    MemoryKind.PREFERENCE,
                    "favorite_food",
                    value,
                    MemoryEvidenceMode.EXPLICIT,
                    0,
                    len(content),
                ),
            )
        )
        service = MemoryCaptureService(
            repository,
            MemoryCandidateService(extractor, FreeOperationPolicy(), (template,)),
        )
        result = await service.capture(
            MemoryCaptureRequest(
                conversation.id,
                conversation.active_branch_id,
                session.user_message.id,
                "test-local",
                "user",
                frozenset({"user"}),
                frozenset({character.character_id}),
                frozenset({character.character_id}),
            )
        )
        await repository.finish_response(session, "覚えた", MessageState.COMPLETED)
        return result.persisted_event_ids

    first_ids = await capture_favorite("ストロベリーアイス")
    second_ids = await capture_favorite("チョコアイス")
    repeated_ids = await capture_favorite("チョコアイス")

    active = await repository.project_canonical_memory(
        conversation.id, conversation.active_branch_id, character.character_id
    )
    history = await repository.project_canonical_memory(
        conversation.id,
        conversation.active_branch_id,
        character.character_id,
        include_historical=True,
    )
    assert len(first_ids) == 1
    assert len(second_ids) == 1
    assert repeated_ids == ()
    assert [(fact.value, fact.state) for fact in active] == [
        ("チョコアイス", MemoryFactState.ACTIVE)
    ]
    assert {(fact.value, fact.state) for fact in history} == {
        ("ストロベリーアイス", MemoryFactState.HISTORICAL),
        ("チョコアイス", MemoryFactState.ACTIVE),
    }
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM canonical_memory_events"
        ).fetchone() == (2,)


@pytest.mark.asyncio
async def test_delayed_older_capture_cannot_replace_newer_single_value(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile("ollama-local", "test-local", {})
    conversation = await repository.create_conversation("遅延した記憶", character.id, profile.id)
    older = await repository.start_send(
        conversation.id, "一番好きなのはストロベリーアイス"
    )
    await repository.finish_response(older, "覚えた", MessageState.COMPLETED)
    newer = await repository.start_send(conversation.id, "一番好きなのはチョコアイス")
    await repository.finish_response(newer, "覚えた", MessageState.COMPLETED)
    template = MemoryTemplate(
        MemoryKind.PREFERENCE,
        "favorite_food",
        MemoryCardinality.SINGLE,
        False,
        (r"一番好きなのは\s*{value}",),
    )

    async def capture_session(
        source_message_id: str, content: str, value: str
    ) -> tuple[str, ...]:
        extractor = FakeExtractor(
            (
                MemoryCandidateDraft(
                    "user",
                    MemoryKind.PREFERENCE,
                    "favorite_food",
                    value,
                    MemoryEvidenceMode.EXPLICIT,
                    0,
                    len(content),
                ),
            )
        )
        service = MemoryCaptureService(
            repository,
            MemoryCandidateService(extractor, FreeOperationPolicy(), (template,)),
        )
        result = await service.capture(
            MemoryCaptureRequest(
                conversation.id,
                conversation.active_branch_id,
                source_message_id,
                "test-local",
                "user",
                frozenset({"user"}),
                frozenset({character.character_id}),
                frozenset({character.character_id}),
            )
        )
        return result.persisted_event_ids

    newer_ids = await capture_session(
        newer.user_message.id, newer.user_message.content, "チョコアイス"
    )
    older_ids = await capture_session(
        older.user_message.id,
        older.user_message.content,
        "ストロベリーアイス",
    )

    assert len(newer_ids) == 1
    assert older_ids == ()
    active = await repository.project_canonical_memory(
        conversation.id, conversation.active_branch_id, character.character_id
    )
    assert [fact.value for fact in active] == ["チョコアイス"]
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM canonical_memory_events"
        ).fetchone() == (1,)


@pytest.mark.asyncio
async def test_single_slot_waits_for_pending_confirmation_before_next_change(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile("ollama-local", "test-local", {})
    conversation = await repository.create_conversation("確認待ち", character.id, profile.id)
    template = MemoryTemplate(
        MemoryKind.GOAL,
        "current_goal",
        MemoryCardinality.SINGLE,
        True,
        (r"今の目標は\s*{value}",),
    )

    async def capture_goal(value: str) -> tuple[str, ...]:
        content = f"今の目標は{value}"
        session = await repository.start_send(conversation.id, content)
        extractor = FakeExtractor(
            (
                MemoryCandidateDraft(
                    "user",
                    MemoryKind.GOAL,
                    "current_goal",
                    value,
                    MemoryEvidenceMode.EXPLICIT,
                    0,
                    len(content),
                ),
            )
        )
        service = MemoryCaptureService(
            repository,
            MemoryCandidateService(extractor, FreeOperationPolicy(), (template,)),
        )
        try:
            result = await service.capture(
                MemoryCaptureRequest(
                    conversation.id,
                    conversation.active_branch_id,
                    session.user_message.id,
                    "test-local",
                    "user",
                    frozenset({"user"}),
                    frozenset({character.character_id}),
                    frozenset({character.character_id}),
                )
            )
            return result.persisted_event_ids
        finally:
            await repository.finish_response(session, "確認する", MessageState.COMPLETED)

    first_ids = await capture_goal("小説家になること")
    with pytest.raises(ValidationError, match="unresolved confirmation"):
        await capture_goal("漫画家になること")

    assert len(first_ids) == 1
    assert await repository.project_canonical_memory(
        conversation.id, conversation.active_branch_id, character.character_id
    ) == ()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM canonical_memory_events"
        ).fetchone() == (1,)


@pytest.mark.asyncio
async def test_single_conflict_uses_transitive_history_and_message_order(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile("ollama-local", "test-local", {})
    conversation = await repository.create_conversation("履歴順", character.id, profile.id)
    sessions = []
    for value in ("苺", "チョコ", "バニラ", "抹茶"):
        session = await repository.start_send(conversation.id, value)
        await repository.finish_response(session, "覚えた", MessageState.COMPLETED)
        sessions.append(session)

    def captured_event(index: int, recorded_offset: timedelta = timedelta()) -> CanonicalMemoryEvent:
        source = sessions[index].user_message
        return CanonicalMemoryEvent(
            id=f"preference-{index}",
            conversation_id=conversation.id,
            branch_id=conversation.active_branch_id,
            subject_id="user",
            kind=MemoryKind.PREFERENCE,
            slot="favorite_food",
            value=source.content,
            cardinality=MemoryCardinality.SINGLE,
            approval=MemoryApprovalState.AUTO_SAVED,
            source_message_id=source.id,
            known_by_character_ids=frozenset(),
            supersedes_event_id=None,
            effective_at=source.created_at + recorded_offset,
            recorded_at=source.created_at + recorded_offset,
        )

    for index in range(3):
        assert await repository.append_captured_memory_events(
            (captured_event(index),)
        ) == (f"preference-{index}",)
    await repository.append_memory_approval_decision(
        MemoryApprovalDecision(
            id="undo-middle",
            target_event_id="preference-1",
            state=MemoryApprovalState.UNDONE,
            source_message_id=sessions[2].user_message.id,
            recorded_at=sessions[2].user_message.created_at,
        )
    )

    # Simulate a wall-clock rollback: the fourth message is later in the branch,
    # but its clock timestamp is earlier than the current fact.
    clock_rollback = sessions[0].user_message.created_at - sessions[3].user_message.created_at
    persisted = await repository.append_captured_memory_events(
        (captured_event(3, clock_rollback - timedelta(seconds=1)),)
    )

    assert persisted == ("preference-3",)
    active = await repository.project_canonical_memory(
        conversation.id, conversation.active_branch_id, character.character_id
    )
    assert [fact.value for fact in active] == ["抹茶"]


@pytest.mark.asyncio
async def test_single_conflict_excludes_parent_facts_after_branch_fork(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile("ollama-local", "test-local", {})
    conversation = await repository.create_conversation("分岐記憶", character.id, profile.id)
    first = await repository.start_send(conversation.id, "苺")
    await repository.finish_response(first, "覚えた", MessageState.COMPLETED)
    second = await repository.start_send(conversation.id, "チョコ")
    await repository.finish_response(second, "覚えた", MessageState.COMPLETED)

    def event_for(
        identifier: str, branch_id: str, source_message_id: str, value: str
    ) -> CanonicalMemoryEvent:
        source = first.user_message if source_message_id == first.user_message.id else second.user_message
        return CanonicalMemoryEvent(
            id=identifier,
            conversation_id=conversation.id,
            branch_id=branch_id,
            subject_id="user",
            kind=MemoryKind.PREFERENCE,
            slot="favorite_food",
            value=value,
            cardinality=MemoryCardinality.SINGLE,
            approval=MemoryApprovalState.AUTO_SAVED,
            source_message_id=source_message_id,
            known_by_character_ids=frozenset(),
            supersedes_event_id=None,
            effective_at=source.created_at,
            recorded_at=source.created_at,
        )

    root_branch_id = conversation.active_branch_id
    await repository.append_captured_memory_events(
        (event_for("root-苺", root_branch_id, first.user_message.id, "苺"),)
    )
    await repository.append_captured_memory_events(
        (event_for("root-チョコ", root_branch_id, second.user_message.id, "チョコ"),)
    )
    child = await repository.start_rewrite(
        conversation.id, second.user_message.id, "バニラ"
    )
    await repository.finish_response(child, "覚えた", MessageState.COMPLETED)
    child_event = CanonicalMemoryEvent(
        id="child-バニラ",
        conversation_id=conversation.id,
        branch_id=child.branch_id,
        subject_id="user",
        kind=MemoryKind.PREFERENCE,
        slot="favorite_food",
        value="バニラ",
        cardinality=MemoryCardinality.SINGLE,
        approval=MemoryApprovalState.AUTO_SAVED,
        source_message_id=child.user_message.id,
        known_by_character_ids=frozenset(),
        supersedes_event_id=None,
        effective_at=child.user_message.created_at,
        recorded_at=child.user_message.created_at,
    )

    with pytest.raises(ValidationError, match="outside the selected branch path"):
        await repository.append_canonical_memory_event(
            replace(
                child_event,
                id="invalid-cross-fork-replacement",
                supersedes_event_id="root-チョコ",
            )
        )
    assert await repository.append_captured_memory_events((child_event,)) == (
        child_event.id,
    )
    root = await repository.project_canonical_memory(
        conversation.id, root_branch_id, character.character_id
    )
    branch = await repository.project_canonical_memory(
        conversation.id, child.branch_id, character.character_id
    )
    assert [fact.value for fact in root] == ["チョコ"]
    assert [fact.value for fact in branch] == ["バニラ"]
