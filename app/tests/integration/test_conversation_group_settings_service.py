from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from local_llm_chat.application.services.conversation_cast_service import (
    ConversationCastService,
)
from local_llm_chat.application.services.conversation_group_settings_service import (
    ConversationGroupSettingsService,
)
from local_llm_chat.domain.errors import PersistenceError, ValidationError
from local_llm_chat.domain.states import TurnMode
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


async def make_conversation(
    repository: SQLiteAppRepository,
) -> tuple[str, str, str]:
    await repository.initialize()
    first = await repository.ensure_default_character()
    second = await repository.create_character_version("Second", "second prompt")
    profile = await repository.ensure_model_profile(
        "ollama-local", "group-model", {"num_ctx": 8192}
    )
    conversation = await repository.create_conversation(
        "group settings", first.id, profile.id
    )
    await ConversationCastService(repository).set_cast(
        conversation.id, (first.id, second.id)
    )
    return conversation.id, first.character_id, second.character_id


@pytest.mark.asyncio
async def test_new_conversation_defaults_to_disabled_story_and_restarts(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation_id, _, _ = await make_conversation(repository)

    saved = await ConversationGroupSettingsService(repository).get(conversation_id)

    assert saved.enabled is False
    assert saved.mode is TurnMode.STORY
    assert saved.spotlight_character_id is None
    restarted = SQLiteAppRepository(database_path)
    assert await ConversationGroupSettingsService(restarted).get(
        conversation_id
    ) == saved


@pytest.mark.asyncio
async def test_spotlight_requires_a_registered_character_and_restarts(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation_id, _, second_character_id = await make_conversation(repository)
    service = ConversationGroupSettingsService(repository)

    with pytest.raises(ValidationError):
        await service.configure(
            conversation_id,
            enabled=True,
            mode=TurnMode.SPOTLIGHT,
            spotlight_character_id="outside-cast",
        )

    saved = await service.configure(
        conversation_id,
        enabled=True,
        mode=TurnMode.SPOTLIGHT,
        spotlight_character_id=second_character_id,
    )
    restarted = SQLiteAppRepository(database_path)
    assert await ConversationGroupSettingsService(restarted).get(
        conversation_id
    ) == saved


@pytest.mark.asyncio
async def test_removing_spotlight_character_falls_back_to_story_atomically(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id, first_character_id, second_character_id = (
        await make_conversation(repository)
    )
    settings = ConversationGroupSettingsService(repository)
    await settings.configure(
        conversation_id,
        enabled=True,
        mode=TurnMode.SPOTLIGHT,
        spotlight_character_id=second_character_id,
    )
    cast = await ConversationCastService(repository).get(conversation_id)
    first_version_id = next(
        member.character_version_id
        for member in cast.members
        if member.character_id == first_character_id
    )

    await ConversationCastService(repository).set_cast(
        conversation_id, (first_version_id,)
    )

    restored = await settings.get(conversation_id)
    assert restored.enabled is True
    assert restored.mode is TurnMode.STORY
    assert restored.spotlight_character_id is None


@pytest.mark.asyncio
async def test_failed_settings_update_rolls_back_the_previous_mode(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation_id, _, _ = await make_conversation(repository)
    service = ConversationGroupSettingsService(repository)
    before = await service.configure(
        conversation_id, enabled=True, mode=TurnMode.ROUND_TABLE
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_group_settings_update BEFORE UPDATE "
            "ON conversation_group_settings "
            "BEGIN SELECT RAISE(ABORT, 'forced'); END"
        )

    with pytest.raises(PersistenceError):
        await service.configure(
            conversation_id, enabled=True, mode=TurnMode.STORY
        )

    assert await service.get(conversation_id) == before
