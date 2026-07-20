from __future__ import annotations

from pathlib import Path

import pytest

from local_llm_chat.application.services.conversation_group_settings_service import (
    ConversationGroupSettingsService,
)
from local_llm_chat.application.services.conversation_timeline_service import (
    ConversationTimelineService,
)
from local_llm_chat.application.services.turn_batch_service import TurnBatchService
from local_llm_chat.domain.group_turns import TurnBatchDraft, TurnSegmentDraft
from local_llm_chat.domain.states import (
    MessageRole,
    MessageState,
    TurnBatchState,
    TurnMode,
    TurnRepairState,
    TurnSpeakerKind,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


async def make_repository(
    tmp_path: Path,
) -> tuple[SQLiteAppRepository, str, str]:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "group-model", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "timeline", character.id, profile.id
    )
    return repository, conversation.id, character.character_id


@pytest.mark.asyncio
async def test_legacy_messages_remain_one_timeline_item_each(tmp_path: Path) -> None:
    repository, conversation_id, _ = await make_repository(tmp_path)
    session = await repository.start_send(conversation_id, "通常の質問")
    response = await repository.finish_response(
        session, "通常の回答", MessageState.COMPLETED
    )

    items = await ConversationTimelineService(repository).list_items(conversation_id)

    assert [item.message.id for item in items] == [
        session.user_message.id,
        response.id,
    ]
    assert all(item.segment is None for item in items)


@pytest.mark.asyncio
async def test_turn_batch_replaces_compatibility_message_with_ordered_segments(
    tmp_path: Path,
) -> None:
    repository, conversation_id, character_id = await make_repository(tmp_path)
    await ConversationGroupSettingsService(repository).configure(
        conversation_id, enabled=True, mode=TurnMode.STORY
    )
    session = await repository.start_send(conversation_id, "二人で答えて")
    draft = TurnBatchDraft(
        mode=TurnMode.STORY,
        formal_character_ids=(character_id,),
        guest_ids=(),
        prompt_version="group-turn-v1",
        state=TurnBatchState.COMPLETED,
        repair_state=TurnRepairState.NOT_NEEDED,
        segments=(
            TurnSegmentDraft(
                TurnSpeakerKind.NARRATOR, None, "ナレーター", "場面が動いた。"
            ),
            TurnSegmentDraft(
                TurnSpeakerKind.CHARACTER,
                character_id,
                "先輩",
                "俺はこう思う。",
            ),
        ),
    )
    batch = await TurnBatchService(repository).finish(session, draft)

    items = await ConversationTimelineService(repository).list_items(conversation_id)

    assert len(items) == 3
    assert items[0].message.role is MessageRole.USER
    assert [item.segment.content for item in items[1:] if item.segment] == [
        "場面が動いた。",
        "俺はこう思う。",
    ]
    assert all(item.message.id == batch.response_message_id for item in items[1:])
    assert [item.is_batch_end for item in items[1:]] == [False, True]
    assert all(item.turn_batch == batch for item in items[1:])
    assert not any(
        item.segment is None and item.message.id == batch.response_message_id
        for item in items
    )


@pytest.mark.asyncio
async def test_partial_batch_exposes_only_persisted_segments_and_partial_state(
    tmp_path: Path,
) -> None:
    repository, conversation_id, character_id = await make_repository(tmp_path)
    await ConversationGroupSettingsService(repository).configure(
        conversation_id, enabled=True, mode=TurnMode.STORY
    )
    session = await repository.start_send(conversation_id, "途中で切れる")
    await TurnBatchService(repository).finish(
        session,
        TurnBatchDraft(
            mode=TurnMode.STORY,
            formal_character_ids=(character_id,),
            guest_ids=(),
            prompt_version="group-turn-v1",
            state=TurnBatchState.PARTIAL,
            repair_state=TurnRepairState.FAILED,
            segments=(
                TurnSegmentDraft(
                    TurnSpeakerKind.CHARACTER,
                    character_id,
                    "先輩",
                    "ここまでは確実。",
                ),
            ),
            error_code="structured_output_truncated",
        ),
    )

    items = await ConversationTimelineService(repository).list_items(conversation_id)

    group_item = items[-1]
    assert group_item.segment is not None
    assert group_item.segment.content == "ここまでは確実。"
    assert group_item.turn_batch is not None
    assert group_item.turn_batch.state is TurnBatchState.PARTIAL
    assert group_item.is_batch_end
