from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from local_llm_chat.application.services.conversation_group_configuration_service import (
    ConversationGroupConfigurationService,
)
from local_llm_chat.domain.errors import PersistenceError, ValidationError
from local_llm_chat.domain.states import TurnMode
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


async def make_conversation(
    repository: SQLiteAppRepository,
) -> tuple[str, str, str, str]:
    await repository.initialize()
    first = await repository.ensure_default_character()
    second = await repository.create_character_version("Second", "second prompt")
    third = await repository.create_character_version("Third", "third prompt")
    profile = await repository.ensure_model_profile(
        "ollama-local", "group-model", {"num_ctx": 8192}
    )
    conversation = await repository.create_conversation(
        "configuration", first.id, profile.id
    )
    return conversation.id, first.id, second.id, third.id


@pytest.mark.asyncio
async def test_saves_cast_mode_and_spotlight_as_one_configuration(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation_id, first_id, second_id, _ = await make_conversation(repository)
    second = await repository.get_character_version(second_id)
    service = ConversationGroupConfigurationService(repository)

    saved = await service.save(
        conversation_id,
        character_version_ids=(second_id, first_id),
        enabled=True,
        mode=TurnMode.SPOTLIGHT,
        spotlight_character_id=second.character_id,
    )

    assert tuple(member.character_version_id for member in saved.cast.members) == (
        second_id,
        first_id,
    )
    assert saved.settings.enabled
    assert saved.settings.mode is TurnMode.SPOTLIGHT
    assert saved.settings.spotlight_character_id == second.character_id
    restarted = ConversationGroupConfigurationService(
        SQLiteAppRepository(database_path)
    )
    assert await restarted.get(conversation_id) == saved


@pytest.mark.asyncio
async def test_invalid_spotlight_leaves_cast_and_settings_unchanged(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id, first_id, second_id, _ = await make_conversation(repository)
    service = ConversationGroupConfigurationService(repository)
    before = await service.get(conversation_id)

    with pytest.raises(ValidationError):
        await service.save(
            conversation_id,
            character_version_ids=(second_id, first_id),
            enabled=True,
            mode=TurnMode.SPOTLIGHT,
            spotlight_character_id="outside-cast",
        )

    assert await service.get(conversation_id) == before


@pytest.mark.asyncio
async def test_settings_write_failure_rolls_back_cast_and_primary_character(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation_id, first_id, second_id, _ = await make_conversation(repository)
    service = ConversationGroupConfigurationService(repository)
    before = await service.get(conversation_id)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_group_configuration BEFORE UPDATE "
            "ON conversation_group_settings "
            "BEGIN SELECT RAISE(ABORT, 'forced'); END"
        )

    with pytest.raises(PersistenceError):
        await service.save(
            conversation_id,
            character_version_ids=(second_id, first_id),
            enabled=True,
            mode=TurnMode.ROUND_TABLE,
        )

    assert await service.get(conversation_id) == before
    assert (await repository.get_conversation(conversation_id)).character_version_id == (
        first_id
    )


@pytest.mark.asyncio
async def test_disabled_configuration_normalizes_mode_and_spotlight(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id, first_id, second_id, _ = await make_conversation(repository)
    service = ConversationGroupConfigurationService(repository)

    with pytest.raises(ValidationError):
        await service.save(
            conversation_id,
            character_version_ids=(first_id,),
            enabled=False,
            mode=TurnMode.SPOTLIGHT,
            spotlight_character_id="character",
        )
    with pytest.raises(ValidationError):
        await service.save(
            conversation_id,
            character_version_ids=(first_id, second_id),
            enabled=False,
            mode=TurnMode.STORY,
        )

    saved = await service.save(
        conversation_id,
        character_version_ids=(first_id,),
        enabled=False,
        mode=TurnMode.STORY,
    )
    assert saved.settings.mode is TurnMode.STORY
    assert saved.settings.spotlight_character_id is None
