from pathlib import Path
from dataclasses import dataclass, replace

import pytest

from local_llm_chat.application.services.explicit_memory_service import (
    ExplicitMemorySaveRequest,
    ExplicitMemoryService,
)
from local_llm_chat.application.services.memory_candidate_service import (
    DEFAULT_MEMORY_TEMPLATES,
    MemoryCandidateService,
)
from local_llm_chat.application.services.memory_capture_service import (
    MemoryCaptureRequest,
    MemoryCaptureService,
)
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.memory_candidates import (
    MemoryCandidateDraft,
    MemoryCandidateRequest,
)
from local_llm_chat.domain.models import (
    CharacterVersion,
    Conversation,
    Message,
    ProviderMetadata,
)
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.domain.states import (
    CostClass,
    Locality,
    MemoryApprovalState,
    MemoryKind,
    MessageState,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


@dataclass
class EmptyExtractor:
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
        return ()


async def _conversation(
    repository: SQLiteAppRepository,
) -> tuple[CharacterVersion, Conversation, Message, Message]:
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile(
        "ollama-local", "test-local", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "明示記憶", character.id, profile.id
    )
    first = await repository.start_send(
        conversation.id, "スーパーカップのチョコチップ味が好き！"
    )
    await repository.finish_response(first, "いいね", MessageState.COMPLETED)
    second = await repository.start_send(
        conversation.id, "ちょっと溶けかけが好き！"
    )
    await repository.finish_response(second, "覚えた", MessageState.COMPLETED)
    return character, conversation, first.user_message, second.user_message


@pytest.mark.asyncio
async def test_explicit_memory_preserves_multiple_sources_and_is_idempotent(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character, conversation, first, second = await _conversation(repository)
    service = ExplicitMemoryService(
        repository,
        MemoryCandidateService(
            EmptyExtractor(), FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
        ),
    )

    draft = await service.prepare(
        conversation.id, conversation.active_branch_id, second.id
    )
    assert [message.id for message in draft.source_messages] == [first.id, second.id]
    assert draft.selected_source_message_ids == (second.id,)
    assert draft.auto_known_by_character_ids == frozenset(
        {character.character_id}
    )
    suggestion = await service.suggest(
        conversation.id,
        conversation.active_branch_id,
        (first.id, second.id),
        MemoryKind.PREFERENCE,
        character.character_id,
    )
    assert suggestion.item == "スーパーカップのチョコチップ味"
    assert suggestion.condition == "ちょっと溶けかけ"
    assert (
        suggestion.summary
        == "スーパーカップのチョコチップ味は、ちょっと溶けかけが好き"
    )

    request = ExplicitMemorySaveRequest(
        request_id="request-1",
        conversation_id=conversation.id,
        branch_id=conversation.active_branch_id,
        source_message_ids=(first.id, second.id),
        item=suggestion.item,
        condition=suggestion.condition,
        kind=MemoryKind.PREFERENCE,
        known_by_character_ids=frozenset({character.character_id}),
    )
    first_id = await service.save(request)
    assert await service.save(request) == first_id
    with pytest.raises(ValidationError, match="already exists"):
        await service.save(
            ExplicitMemorySaveRequest(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                branch_id=request.branch_id,
                source_message_ids=request.source_message_ids,
                item="スーパーカップのチョコチップ味",
                condition=None,
                kind=request.kind,
                known_by_character_ids=request.known_by_character_ids,
            )
        )

    items = await repository.list_canonical_memory_review_items(
        conversation.id, conversation.active_branch_id
    )
    assert len(items) == 1
    assert items[0].approval is MemoryApprovalState.CONFIRMED
    assert items[0].known_by_character_ids == frozenset({character.character_id})
    assert items[0].source_message_ids == (first.id, second.id)
    assert items[0].value == suggestion.summary
    assert tuple(
        (attribute.key, attribute.value, attribute.source_message_id)
        for attribute in items[0].attributes
    ) == (
        ("condition", "ちょっと溶けかけ", second.id),
        ("item", "スーパーカップのチョコチップ味", first.id),
    )


@pytest.mark.asyncio
async def test_explicit_memory_requires_an_explicit_registered_knowledge_scope(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character, conversation, first, _ = await _conversation(repository)
    other = await repository.create_character_version("別の人物", "別の人物として話す。")
    service = ExplicitMemoryService(repository)

    draft = await service.prepare(
        conversation.id, conversation.active_branch_id, first.id
    )
    assert {
        option.character_id for option in draft.knowledge_character_options
    } == {character.character_id, other.character_id}

    base = ExplicitMemorySaveRequest(
        request_id="knowledge-scope",
        conversation_id=conversation.id,
        branch_id=conversation.active_branch_id,
        source_message_ids=(first.id,),
        item="スーパーカップのチョコチップ味",
        condition=None,
        kind=MemoryKind.PREFERENCE,
        known_by_character_ids=frozenset(),
    )
    with pytest.raises(ValidationError, match="at least one character"):
        await service.save(
            base
        )
    with pytest.raises(ValidationError, match="registered character"):
        await service.save(
            replace(base, known_by_character_ids=frozenset({"missing"}))
        )

    await service.save(
        replace(
            base,
            known_by_character_ids=frozenset(
                {character.character_id, other.character_id}
            ),
        )
    )
    items = await repository.list_canonical_memory_review_items(
        conversation.id, conversation.active_branch_id
    )
    assert items[0].known_by_character_ids == frozenset(
        {character.character_id, other.character_id}
    )


@pytest.mark.asyncio
async def test_explicit_memory_rejects_invented_text_and_non_user_evidence(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character, conversation, first, second = await _conversation(repository)
    service = ExplicitMemoryService(repository)

    with pytest.raises(ValidationError, match="原文"):
        await service.save(
            ExplicitMemorySaveRequest(
                request_id="invented",
                conversation_id=conversation.id,
                branch_id=conversation.active_branch_id,
                source_message_ids=(first.id, second.id),
                item="抹茶味",
                condition=None,
                kind=MemoryKind.PREFERENCE,
                known_by_character_ids=frozenset({character.character_id}),
            )
        )
    with pytest.raises(ValidationError, match="原文"):
        await service.save(
            ExplicitMemorySaveRequest(
                request_id="invented-condition",
                conversation_id=conversation.id,
                branch_id=conversation.active_branch_id,
                source_message_ids=(first.id, second.id),
                item="スーパーカップのチョコチップ味",
                condition="熱々",
                kind=MemoryKind.PREFERENCE,
                known_by_character_ids=frozenset({character.character_id}),
            )
        )

    messages = await repository.list_active_messages(conversation.id)
    assistant = next(
        message
        for message in messages
        if message.role.value == "assistant" and message.state is MessageState.COMPLETED
    )
    with pytest.raises(ValidationError, match="ユーザー発言"):
        await repository.get_memory_source_messages(
            conversation.id,
            conversation.active_branch_id,
            (assistant.id,),
        )


@pytest.mark.asyncio
async def test_explicit_memory_uses_safe_short_fallback_without_extractor(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile(
        "ollama-local", "test-local", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "安全な短文化", character.id, profile.id
    )
    send = await repository.start_send(
        conversation.id,
        "先輩、私実はスーパーカップのチョコチップ味が好きなんだ！",
    )
    await repository.finish_response(send, "応答", MessageState.COMPLETED)
    service = ExplicitMemoryService(repository)

    suggestion = await service.suggest(
        conversation.id,
        conversation.active_branch_id,
        (send.user_message.id,),
        MemoryKind.PREFERENCE,
        character.character_id,
    )

    assert suggestion.item == "スーパーカップのチョコチップ味"
    assert suggestion.condition is None
    assert suggestion.summary == "スーパーカップのチョコチップ味が好き"
    assert suggestion.extractor_unavailable


@pytest.mark.asyncio
async def test_automatic_memory_progressively_refines_food_with_all_sources(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile(
        "ollama-local", "test-local", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "段階記憶", character.id, profile.id
    )
    candidate_service = MemoryCandidateService(
        EmptyExtractor(), FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
    )
    capture = MemoryCaptureService(repository, candidate_service)
    source_ids: list[str] = []

    async def turn(content: str) -> None:
        session = await repository.start_send(conversation.id, content)
        await repository.finish_response(session, "応答", MessageState.COMPLETED)
        source_ids.append(session.user_message.id)
        await capture.capture(
            MemoryCaptureRequest(
                conversation_id=conversation.id,
                branch_id=session.branch_id,
                source_message_id=session.user_message.id,
                model_name="test-local",
                author_subject_id="user",
                allowed_subject_ids=frozenset({"user", character.character_id}),
                    allowed_knowledge_character_ids=frozenset(
                        {character.character_id}
                    ),
                    known_by_character_ids=frozenset(
                        {character.character_id}
                    ),
            )
        )

    await turn("アイスが好き。チョコ味！")
    await turn("先輩はどのアイスメーカーのやつが好き？")
    await turn("私はスーパーカップのチョコチップ味！")
    await turn("ちょっと溶けかけが好き！")

    items = await repository.list_canonical_memory_review_items(
        conversation.id, conversation.active_branch_id
    )
    active = [item for item in items if item.is_active]
    assert len(active) == 1
    assert active[0].value == "スーパーカップのチョコチップ味（ちょっと溶けかけ）"
    assert active[0].source_message_ids == (
        source_ids[0],
        source_ids[2],
        source_ids[3],
    )


@pytest.mark.asyncio
async def test_ambiguous_cross_turn_condition_waits_for_confirmation(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile(
        "ollama-local", "test-local", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "曖昧な段階記憶", character.id, profile.id
    )
    capture = MemoryCaptureService(
        repository,
        MemoryCandidateService(
            EmptyExtractor(), FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
        ),
    )
    first = await repository.start_send(
        conversation.id, "チョコ味とバニラ味を買った"
    )
    await repository.finish_response(first, "応答", MessageState.COMPLETED)
    second = await repository.start_send(
        conversation.id, "少し溶けた方が好き"
    )
    await repository.finish_response(second, "応答", MessageState.COMPLETED)

    result = await capture.capture(
        MemoryCaptureRequest(
            conversation_id=conversation.id,
            branch_id=second.branch_id,
            source_message_id=second.user_message.id,
            model_name="test-local",
            author_subject_id="user",
                allowed_subject_ids=frozenset({"user", character.character_id}),
                allowed_knowledge_character_ids=frozenset({character.character_id}),
                known_by_character_ids=frozenset({character.character_id}),
        )
    )

    assert len(result.persisted_event_ids) == 1
    items = await repository.list_canonical_memory_review_items(
        conversation.id, conversation.active_branch_id
    )
    assert items[0].approval is MemoryApprovalState.PENDING_CONFIRMATION
    assert items[0].source_message_ids == (
        first.user_message.id,
        second.user_message.id,
    )


@pytest.mark.asyncio
async def test_latest_food_preference_replaces_prior_memory_from_new_evidence(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.create_character_version("先輩", "先輩として話す")
    profile = await repository.ensure_model_profile(
        "ollama-local", "test-local", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "好みの訂正", character.id, profile.id
    )
    capture = MemoryCaptureService(
        repository,
        MemoryCandidateService(
            EmptyExtractor(), FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
        ),
    )

    async def turn(content: str) -> Message:
        session = await repository.start_send(conversation.id, content)
        await repository.finish_response(session, "応答", MessageState.COMPLETED)
        await capture.capture(
            MemoryCaptureRequest(
                conversation_id=conversation.id,
                branch_id=session.branch_id,
                source_message_id=session.user_message.id,
                model_name="test-local",
                author_subject_id="user",
                allowed_subject_ids=frozenset({"user", character.character_id}),
                    allowed_knowledge_character_ids=frozenset(
                        {character.character_id}
                    ),
                    known_by_character_ids=frozenset(
                        {character.character_id}
                    ),
            )
        )
        return session.user_message

    first = await turn("チョコアイスが好き")
    correction = await turn("でも今はバニラアイスが一番好き")

    items = await repository.list_canonical_memory_review_items(
        conversation.id, conversation.active_branch_id
    )
    active = [item for item in items if item.is_active]
    assert len(active) == 1
    assert active[0].value == "バニラアイス"
    assert active[0].source_message_ids == (correction.id,)
    assert next(item for item in items if item.event_id != active[0].event_id).source_message_ids == (
        first.id,
    )


@pytest.mark.asyncio
async def test_progressive_memory_does_not_merge_across_listener_changes(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    first_character = await repository.create_character_version(
        "先輩", "先輩として話す"
    )
    second_character = await repository.create_character_version(
        "後輩", "後輩として話す"
    )
    profile = await repository.ensure_model_profile(
        "ollama-local", "test-local", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "参加者変更", first_character.id, profile.id
    )
    capture = MemoryCaptureService(
        repository,
        MemoryCandidateService(
            EmptyExtractor(), FreeOperationPolicy(), DEFAULT_MEMORY_TEMPLATES
        ),
    )

    async def capture_for(content: str, character_id: str) -> None:
        session = await repository.start_send(conversation.id, content)
        await repository.finish_response(
            session, "応答", MessageState.COMPLETED
        )
        await capture.capture(
            MemoryCaptureRequest(
                conversation_id=conversation.id,
                branch_id=session.branch_id,
                source_message_id=session.user_message.id,
                model_name="test-local",
                author_subject_id="user",
                allowed_subject_ids=frozenset({"user", character_id}),
                allowed_knowledge_character_ids=frozenset({character_id}),
                known_by_character_ids=frozenset({character_id}),
            )
        )

    await capture_for("チョコアイスが好き", first_character.character_id)
    await capture_for(
        "ちょっと溶けかけが好き", second_character.character_id
    )

    items = await repository.list_canonical_memory_review_items(
        conversation.id, conversation.active_branch_id
    )
    active = [item for item in items if item.is_active]
    assert len(active) == 1
    assert active[0].value == "チョコアイス"
    assert active[0].known_by_character_ids == frozenset(
        {first_character.character_id}
    )
