from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_llm_chat.application.services.chat_coordinator import ChatCoordinator
from local_llm_chat.application.services.conversation_group_settings_service import (
    ConversationGroupSettingsService,
)
from local_llm_chat.application.services.memory_capture_service import (
    MemoryCaptureRequest,
)
from local_llm_chat.application.services.turn_batch_generation_service import (
    TurnBatchGenerationService,
)
from local_llm_chat.application.services.turn_batch_service import TurnBatchService
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.group_turns import (
    TurnBatchDraft,
    TurnBatchGenerationRequest,
    TurnSegmentDraft,
)
from local_llm_chat.domain.models import Conversation, Message
from local_llm_chat.domain.states import (
    MessageState,
    MessageRole,
    RunState,
    TurnBatchState,
    TurnMode,
    TurnRepairState,
    TurnSpeakerKind,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


async def no_update(_: str) -> None:
    return None


async def no_notice(_: str, __: bool) -> None:
    return None


class SingleChatStub:
    def __init__(self, response: Message) -> None:
        self.response = response
        self.send_count = 0
        self.regenerate_count = 0

    async def send_message(
        self,
        conversation_id: str,
        content: str,
        on_update: Callable[[str], Awaitable[None]] = no_update,
        on_notice: Callable[[str, bool], Awaitable[None]] = no_notice,
    ) -> Message:
        self.send_count += 1
        return self.response

    async def rewrite_message(
        self,
        conversation_id: str,
        source_message_id: str,
        content: str,
        on_update: Callable[[str], Awaitable[None]] = no_update,
        on_notice: Callable[[str, bool], Awaitable[None]] = no_notice,
    ) -> Message:
        return self.response

    async def regenerate_message(
        self,
        conversation_id: str,
        source_message_id: str,
        on_update: Callable[[str], Awaitable[None]] = no_update,
        on_notice: Callable[[str, bool], Awaitable[None]] = no_notice,
    ) -> Message:
        self.regenerate_count += 1
        return self.response


class RecordingGenerator:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.requests: list[TurnBatchGenerationRequest] = []

    async def generate(
        self,
        request: TurnBatchGenerationRequest,
        on_update: Callable[[str], Awaitable[None]] = no_update,
    ) -> TurnBatchDraft:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        first = request.formal_characters[0]
        await on_update(f"{first.display_name}: streaming")
        return TurnBatchDraft(
            mode=request.mode,
            formal_character_ids=tuple(
                character.character_id for character in request.formal_characters
            ),
            guest_ids=(),
            prompt_version=request.prompt_version,
            state=TurnBatchState.COMPLETED,
            repair_state=TurnRepairState.NOT_NEEDED,
            segments=(
                TurnSegmentDraft(
                    TurnSpeakerKind.CHARACTER,
                    first.character_id,
                    first.display_name,
                    "グループ応答",
                ),
            ),
            spotlight_character_id=request.spotlight_character_id,
            prompt_tokens=80,
            output_tokens=20,
            total_duration_ns=2_000_000_000,
            generation_duration_ns=1_000_000_000,
        )


class SettingsChangingGenerator(RecordingGenerator):
    def __init__(
        self, repository: SQLiteAppRepository, conversation_id: str
    ) -> None:
        super().__init__()
        self._repository = repository
        self._conversation_id = conversation_id

    async def generate(
        self,
        request: TurnBatchGenerationRequest,
        on_update: Callable[[str], Awaitable[None]] = no_update,
    ) -> TurnBatchDraft:
        draft = await super().generate(request, on_update)
        await ConversationGroupSettingsService(self._repository).configure(
            self._conversation_id,
            enabled=True,
            mode=TurnMode.ROUND_TABLE,
        )
        return draft


class RecordingMemoryScheduler:
    def __init__(self) -> None:
        self.requests: list[tuple[MemoryCaptureRequest, str]] = []

    async def request_capture(
        self, request: MemoryCaptureRequest, run_id: str
    ) -> None:
        self.requests.append((request, run_id))


async def make_conversation(repository: SQLiteAppRepository) -> Conversation:
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "group-model", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "group", character.id, profile.id
    )
    return conversation


def coordinator(
    repository: SQLiteAppRepository,
    single: SingleChatStub,
    generator: RecordingGenerator,
    memory: RecordingMemoryScheduler | None = None,
) -> ChatCoordinator:
    return ChatCoordinator(
        repository,
        single,
        TurnBatchGenerationService(repository, generator),
        TurnBatchService(repository),
        memory_capture_scheduler=memory,
    )


@pytest.mark.asyncio
async def test_disabled_conversation_uses_existing_single_chat_path(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation = await make_conversation(repository)
    placeholder = Message(
        "single-response",
        conversation.id,
        None,
        None,
        MessageRole.ASSISTANT,
        "単独応答",
        MessageState.COMPLETED,
        datetime.now(UTC),
        datetime.now(UTC),
    )
    single = SingleChatStub(placeholder)
    generator = RecordingGenerator()

    response = await coordinator(repository, single, generator).send_message(
        conversation.id, "質問"
    )

    assert response is placeholder
    assert single.send_count == 1
    assert generator.requests == []


@pytest.mark.asyncio
async def test_enabled_conversation_generates_and_finishes_one_turn_batch(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation = await make_conversation(repository)
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.STORY
    )
    placeholder = Message(
        "unused",
        conversation.id,
        None,
        None,
        MessageRole.ASSISTANT,
        "",
        MessageState.COMPLETED,
        datetime.now(UTC),
        datetime.now(UTC),
    )
    single = SingleChatStub(placeholder)
    generator = RecordingGenerator()
    updates: list[str] = []

    async def on_update(content: str) -> None:
        updates.append(content)

    response = await coordinator(repository, single, generator).send_message(
        conversation.id, "みんなはどう思う？", on_update
    )

    assert single.send_count == 0
    assert len(generator.requests) == 1
    assert response.state is MessageState.COMPLETED
    assert response.content.endswith("グループ応答")
    assert updates == [
        f"{generator.requests[0].formal_characters[0].display_name}: streaming",
        response.content,
    ]
    batch = await repository.get_turn_batch_for_response(response.id)
    assert len(batch.segments) == 1
    latest = await repository.get_latest_telemetry(conversation.id)
    assert latest is not None
    assert latest.run.prompt_tokens == 80
    assert latest.run.output_tokens == 20
    assert latest.run.total_duration_ns == 2_000_000_000
    assert latest.run.generation_duration_ns == 1_000_000_000
    assert latest.run.response_duration_ms is not None
    assert latest.run.response_duration_ms >= 0
    assert latest.tokens_per_second == pytest.approx(20.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_message_state", "expected_run_state"),
    [
        (ValidationError("broken output"), MessageState.FAILED, RunState.FAILED),
        (asyncio.CancelledError(), MessageState.CANCELLED, RunState.CANCELLED),
    ],
)
async def test_group_failure_always_terminates_started_response(
    tmp_path: Path,
    error: BaseException,
    expected_message_state: MessageState,
    expected_run_state: RunState,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation = await make_conversation(repository)
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.STORY
    )
    placeholder = Message(
        "unused",
        conversation.id,
        None,
        None,
        MessageRole.ASSISTANT,
        "",
        MessageState.COMPLETED,
        datetime.now(UTC),
        datetime.now(UTC),
    )

    with pytest.raises(type(error)):
        await coordinator(
            repository, SingleChatStub(placeholder), RecordingGenerator(error)
        ).send_message(conversation.id, "失敗する質問")

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        response = connection.execute(
            "SELECT state FROM messages WHERE conversation_id = ? AND role = 'assistant' ORDER BY created_at DESC LIMIT 1",
            (conversation.id,),
        ).fetchone()
        run = connection.execute(
            "SELECT state FROM runs WHERE conversation_id = ? ORDER BY rowid DESC LIMIT 1",
            (conversation.id,),
        ).fetchone()
    assert response is not None and response["state"] == expected_message_state.value
    assert run is not None and run["state"] == expected_run_state.value


@pytest.mark.asyncio
async def test_settings_changed_during_generation_rejects_batch_and_terminates_run(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation = await make_conversation(repository)
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.STORY
    )
    placeholder = Message(
        "unused",
        conversation.id,
        None,
        None,
        MessageRole.ASSISTANT,
        "",
        MessageState.COMPLETED,
        datetime.now(UTC),
        datetime.now(UTC),
    )

    with pytest.raises(ValidationError):
        await coordinator(
            repository,
            SingleChatStub(placeholder),
            SettingsChangingGenerator(repository, conversation.id),
        ).send_message(conversation.id, "生成中にモード変更")

    with sqlite3.connect(database_path) as connection:
        response_state = connection.execute(
            "SELECT state FROM messages WHERE conversation_id = ? AND role = 'assistant' ORDER BY created_at DESC LIMIT 1",
            (conversation.id,),
        ).fetchone()
        run_state = connection.execute(
            "SELECT state FROM runs WHERE conversation_id = ? ORDER BY rowid DESC LIMIT 1",
            (conversation.id,),
        ).fetchone()
        batch_count = connection.execute("SELECT COUNT(*) FROM turn_batches").fetchone()
    assert response_state == (MessageState.FAILED.value,)
    assert run_state == (RunState.FAILED.value,)
    assert batch_count == (0,)


async def make_group_response(
    repository: SQLiteAppRepository,
    conversation: Conversation,
    service: ChatCoordinator,
) -> Message:
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.STORY
    )
    return await service.send_message(conversation.id, "original group prompt")


def row_counts(database_path: Path) -> tuple[int, int, int, int]:
    with sqlite3.connect(database_path) as connection:
        return tuple(
            int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("branches", "messages", "runs", "turn_batches")
        )  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_turn_batch_regeneration_forks_from_source_batch_and_reuses_user_message(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation = await make_conversation(repository)
    placeholder = Message(
        "unused",
        conversation.id,
        None,
        None,
        MessageRole.ASSISTANT,
        "",
        MessageState.COMPLETED,
        datetime.now(UTC),
        datetime.now(UTC),
    )
    single = SingleChatStub(placeholder)
    generator = RecordingGenerator()
    memory = RecordingMemoryScheduler()
    service = coordinator(repository, single, generator, memory)
    source_response = await make_group_response(repository, conversation, service)
    source_batch = await repository.get_turn_batch_for_response(source_response.id)

    alternate = await repository.start_rewrite(
        conversation.id, source_batch.source_message_id, "alternate prompt"
    )
    await repository.finish_response(
        alternate, "alternate response", MessageState.COMPLETED
    )
    counts_before = row_counts(database_path)

    regenerated = await service.regenerate_turn_batch(
        conversation.id,
        source_response.id,
        expected_active_branch_id=alternate.branch_id,
    )

    regenerated_batch = await repository.get_turn_batch_for_response(regenerated.id)
    branches = await repository.list_branches(conversation.id)
    regenerated_branch = next(
        branch for branch in branches if branch.id == regenerated_batch.branch_id
    )
    assert regenerated_branch.parent_branch_id == source_batch.branch_id
    assert regenerated_branch.forked_from_message_id == source_response.id
    assert regenerated_batch.source_message_id == source_batch.source_message_id
    assert regenerated.source_message_id == source_response.id
    assert await repository.get_turn_batch_for_response(source_response.id) == source_batch
    assert row_counts(database_path) == (
        counts_before[0] + 1,
        counts_before[1] + 1,
        counts_before[2] + 1,
        counts_before[3] + 1,
    )
    assert single.regenerate_count == 0
    assert len(generator.requests) == 2
    assert len(memory.requests) == 1
    regenerated_context = tuple(
        message.content for message in generator.requests[1].messages
    )
    assert "original group prompt" in regenerated_context
    assert "alternate prompt" not in regenerated_context
    assert "alternate response" not in regenerated_context


@pytest.mark.asyncio
async def test_turn_batch_regeneration_rejects_stale_active_branch_without_writes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation = await make_conversation(repository)
    placeholder = Message(
        "unused",
        conversation.id,
        None,
        None,
        MessageRole.ASSISTANT,
        "",
        MessageState.COMPLETED,
        datetime.now(UTC),
        datetime.now(UTC),
    )
    service = coordinator(
        repository, SingleChatStub(placeholder), RecordingGenerator()
    )
    source_response = await make_group_response(repository, conversation, service)
    counts_before = row_counts(database_path)

    with pytest.raises(ValidationError, match="active branch"):
        await service.regenerate_turn_batch(
            conversation.id,
            source_response.id,
            expected_active_branch_id="stale-branch",
        )

    assert row_counts(database_path) == counts_before


@pytest.mark.asyncio
async def test_turn_batch_regeneration_rejects_second_click_with_same_branch_token(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation = await make_conversation(repository)
    placeholder = Message(
        "unused",
        conversation.id,
        None,
        None,
        MessageRole.ASSISTANT,
        "",
        MessageState.COMPLETED,
        datetime.now(UTC),
        datetime.now(UTC),
    )
    service = coordinator(
        repository, SingleChatStub(placeholder), RecordingGenerator()
    )
    source_response = await make_group_response(repository, conversation, service)
    original_active_branch_id = (
        await repository.get_conversation(conversation.id)
    ).active_branch_id

    await service.regenerate_turn_batch(
        conversation.id,
        source_response.id,
        expected_active_branch_id=original_active_branch_id,
    )
    counts_after_first = row_counts(database_path)
    with pytest.raises(ValidationError, match="active branch"):
        await service.regenerate_turn_batch(
            conversation.id,
            source_response.id,
            expected_active_branch_id=original_active_branch_id,
        )

    assert row_counts(database_path) == counts_after_first


@pytest.mark.asyncio
async def test_turn_batch_regeneration_rejects_disabled_group_before_writes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation = await make_conversation(repository)
    placeholder = Message(
        "unused",
        conversation.id,
        None,
        None,
        MessageRole.ASSISTANT,
        "",
        MessageState.COMPLETED,
        datetime.now(UTC),
        datetime.now(UTC),
    )
    service = coordinator(
        repository, SingleChatStub(placeholder), RecordingGenerator()
    )
    source_response = await make_group_response(repository, conversation, service)
    active_branch_id = (await repository.get_conversation(conversation.id)).active_branch_id
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=False, mode=TurnMode.STORY
    )
    counts_before = row_counts(database_path)

    with pytest.raises(ValidationError, match="disabled"):
        await service.regenerate_turn_batch(
            conversation.id,
            source_response.id,
            expected_active_branch_id=active_branch_id,
        )

    assert row_counts(database_path) == counts_before


@pytest.mark.asyncio
async def test_turn_batch_regeneration_rejects_non_batch_response_without_writes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation = await make_conversation(repository)
    await ConversationGroupSettingsService(repository).configure(
        conversation.id, enabled=True, mode=TurnMode.STORY
    )
    legacy = await repository.start_send(conversation.id, "legacy prompt")
    legacy_response = await repository.finish_response(
        legacy, "legacy response", MessageState.COMPLETED
    )
    placeholder = Message(
        "unused",
        conversation.id,
        None,
        None,
        MessageRole.ASSISTANT,
        "",
        MessageState.COMPLETED,
        datetime.now(UTC),
        datetime.now(UTC),
    )
    service = coordinator(
        repository, SingleChatStub(placeholder), RecordingGenerator()
    )
    counts_before = row_counts(database_path)

    with pytest.raises(ValidationError, match="turn batch was not found"):
        await service.regenerate_turn_batch(
            conversation.id,
            legacy_response.id,
            expected_active_branch_id=legacy.branch_id,
        )

    assert row_counts(database_path) == counts_before


@pytest.mark.asyncio
async def test_turn_batch_regeneration_failure_terminates_only_new_branch(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation = await make_conversation(repository)
    placeholder = Message(
        "unused",
        conversation.id,
        None,
        None,
        MessageRole.ASSISTANT,
        "",
        MessageState.COMPLETED,
        datetime.now(UTC),
        datetime.now(UTC),
    )
    initial_service = coordinator(
        repository, SingleChatStub(placeholder), RecordingGenerator()
    )
    source_response = await make_group_response(
        repository, conversation, initial_service
    )
    source_batch = await repository.get_turn_batch_for_response(source_response.id)
    active_branch_id = (await repository.get_conversation(conversation.id)).active_branch_id
    failing_service = coordinator(
        repository,
        SingleChatStub(placeholder),
        RecordingGenerator(ValidationError("broken regenerated output")),
    )

    with pytest.raises(ValidationError, match="broken regenerated output"):
        await failing_service.regenerate_turn_batch(
            conversation.id,
            source_response.id,
            expected_active_branch_id=active_branch_id,
        )

    assert await repository.get_turn_batch_for_response(source_response.id) == source_batch
    with sqlite3.connect(database_path) as connection:
        latest_message = connection.execute(
            "SELECT state FROM messages WHERE role = 'assistant' ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        latest_run = connection.execute(
            "SELECT state FROM runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        batch_count = connection.execute("SELECT COUNT(*) FROM turn_batches").fetchone()
    assert latest_message == (MessageState.FAILED.value,)
    assert latest_run == (RunState.FAILED.value,)
    assert batch_count == (1,)


@pytest.mark.asyncio
async def test_turn_batch_regeneration_cancellation_terminates_only_new_branch(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation = await make_conversation(repository)
    placeholder = Message(
        "unused",
        conversation.id,
        None,
        None,
        MessageRole.ASSISTANT,
        "",
        MessageState.COMPLETED,
        datetime.now(UTC),
        datetime.now(UTC),
    )
    initial_service = coordinator(
        repository, SingleChatStub(placeholder), RecordingGenerator()
    )
    source_response = await make_group_response(
        repository, conversation, initial_service
    )
    source_batch = await repository.get_turn_batch_for_response(source_response.id)
    active_branch_id = (await repository.get_conversation(conversation.id)).active_branch_id
    cancelled_service = coordinator(
        repository,
        SingleChatStub(placeholder),
        RecordingGenerator(asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await cancelled_service.regenerate_turn_batch(
            conversation.id,
            source_response.id,
            expected_active_branch_id=active_branch_id,
        )

    assert await repository.get_turn_batch_for_response(source_response.id) == source_batch
    with sqlite3.connect(database_path) as connection:
        latest_message = connection.execute(
            "SELECT state FROM messages WHERE role = 'assistant' ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        latest_run = connection.execute(
            "SELECT state FROM runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        batch_count = connection.execute("SELECT COUNT(*) FROM turn_batches").fetchone()
    assert latest_message == (MessageState.CANCELLED.value,)
    assert latest_run == (RunState.CANCELLED.value,)
    assert batch_count == (1,)


@pytest.mark.asyncio
async def test_turn_batch_regeneration_rejects_batch_from_another_conversation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    source_conversation = await make_conversation(repository)
    target_conversation = await make_conversation(repository)
    placeholder = Message(
        "unused",
        source_conversation.id,
        None,
        None,
        MessageRole.ASSISTANT,
        "",
        MessageState.COMPLETED,
        datetime.now(UTC),
        datetime.now(UTC),
    )
    service = coordinator(
        repository, SingleChatStub(placeholder), RecordingGenerator()
    )
    source_response = await make_group_response(
        repository, source_conversation, service
    )
    await ConversationGroupSettingsService(repository).configure(
        target_conversation.id, enabled=True, mode=TurnMode.STORY
    )
    target_active_branch_id = (
        await repository.get_conversation(target_conversation.id)
    ).active_branch_id
    counts_before = row_counts(database_path)

    with pytest.raises(ValidationError, match="another conversation"):
        await service.regenerate_turn_batch(
            target_conversation.id,
            source_response.id,
            expected_active_branch_id=target_active_branch_id,
        )

    assert row_counts(database_path) == counts_before
