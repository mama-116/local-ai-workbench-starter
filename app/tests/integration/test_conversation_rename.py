from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from local_llm_chat.application.services.conversation_service import (
    ConversationService,
)
from local_llm_chat.domain.errors import ConversationNotFound, ValidationError
from local_llm_chat.domain.models import Conversation
from local_llm_chat.domain.policies.free_operation import FreeOperationPolicy
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


async def _make_conversation(repository: SQLiteAppRepository) -> Conversation:
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "rename-model", {"num_ctx": 4096}
    )
    return await repository.create_conversation(
        "変更前", character.id, profile.id
    )


def _service(repository: SQLiteAppRepository) -> ConversationService:
    return ConversationService(
        repository,
        cast(Any, None),
        FreeOperationPolicy(),
    )


@pytest.mark.asyncio
async def test_rename_rejects_blank_and_overlong_titles_without_changing_data(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation = await _make_conversation(repository)
    service = _service(repository)

    with pytest.raises(ValidationError, match="会話名を入力"):
        await service.rename(conversation.id, " \t ")
    with pytest.raises(ValidationError, match="120文字以内"):
        await service.rename(conversation.id, "名" * 121)

    assert (await repository.get_conversation(conversation.id)).title == "変更前"


@pytest.mark.asyncio
async def test_rename_trims_and_persists_after_repository_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation = await _make_conversation(repository)

    renamed = await _service(repository).rename(
        conversation.id, "  夏の旅行計画  "
    )

    assert renamed.title == "夏の旅行計画"
    reopened = SQLiteAppRepository(database_path)
    await reopened.initialize()
    assert (await reopened.get_conversation(conversation.id)).title == "夏の旅行計画"


@pytest.mark.asyncio
async def test_rename_rejects_conversation_archived_after_menu_was_opened(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation = await _make_conversation(repository)
    await repository.archive_conversation(conversation.id)

    with pytest.raises(ConversationNotFound):
        await _service(repository).rename(conversation.id, "変更後")
