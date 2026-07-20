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


async def make_conversation(repository: SQLiteAppRepository) -> tuple[str, str]:
    await repository.initialize()
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "cast-model", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "cast", character.id, profile.id
    )
    return conversation.id, character.id


@pytest.mark.asyncio
async def test_migration_backfills_existing_conversation_as_one_person_cast(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    migrations = (
        Path(__file__).parents[2]
        / "src"
        / "local_llm_chat"
        / "infrastructure"
        / "persistence"
        / "migrations"
    )
    now = "2026-07-19T00:00:00+00:00"
    with sqlite3.connect(database_path) as connection:
        for migration in sorted(migrations.glob("*.sql")):
            version = int(migration.stem.split("_", maxsplit=1)[0])
            if version >= 13:
                continue
            connection.executescript(migration.read_text(encoding="utf-8"))
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                (version, now),
            )
        connection.execute(
            "INSERT INTO characters(id, display_name, created_at) VALUES('c1', 'one', ?)",
            (now,),
        )
        connection.execute(
            "INSERT INTO character_versions(id, character_id, version, system_prompt, created_at) "
            "VALUES('v1', 'c1', 1, 'prompt', ?)",
            (now,),
        )
        connection.execute(
            "INSERT INTO model_profiles(id, provider, model_name, parameters_json, created_at, updated_at) "
            "VALUES('m1', 'ollama-local', 'model', '{}', ?, ?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO conversations(id, title, character_version_id, model_profile_id, created_at, updated_at) "
            "VALUES('conversation-1', 'old', 'v1', 'm1', ?, ?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO branches(id, conversation_id, created_at) "
            "VALUES('branch-1', 'conversation-1', ?)",
            (now,),
        )
        connection.execute(
            "UPDATE conversations SET active_branch_id = 'branch-1' WHERE id = 'conversation-1'"
        )

    repository = SQLiteAppRepository(database_path)
    await repository.initialize()

    cast = await ConversationCastService(repository).get("conversation-1")
    assert tuple(member.character_id for member in cast.members) == ("c1",)
    assert tuple(member.character_version_id for member in cast.members) == ("v1",)
    settings = await ConversationGroupSettingsService(repository).get(
        "conversation-1"
    )
    assert settings.enabled is False
    assert settings.mode is TurnMode.STORY
    assert settings.spotlight_character_id is None


@pytest.mark.asyncio
async def test_new_single_conversation_has_a_persistent_one_person_cast(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation_id, version_id = await make_conversation(repository)

    saved = await ConversationCastService(repository).get(conversation_id)

    assert len(saved.members) == 1
    assert saved.members[0].character_version_id == version_id
    assert saved.members[0].position == 0
    restarted = SQLiteAppRepository(database_path)
    assert await ConversationCastService(restarted).get(conversation_id) == saved


@pytest.mark.asyncio
async def test_cast_accepts_five_unique_characters_and_preserves_order(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation_id, first_version_id = await make_conversation(repository)
    versions = [await repository.get_character_version(first_version_id)]
    for index in range(2, 6):
        versions.append(
            await repository.create_character_version(
                f"character-{index}", f"prompt-{index}"
            )
        )
    ordered_version_ids = tuple(version.id for version in reversed(versions))

    saved = await ConversationCastService(repository).set_cast(
        conversation_id, ordered_version_ids
    )

    assert tuple(member.character_version_id for member in saved.members) == (
        ordered_version_ids
    )
    assert tuple(member.position for member in saved.members) == tuple(range(5))
    assert (await repository.get_conversation(conversation_id)).character_version_id == (
        ordered_version_ids[0]
    )
    restarted = SQLiteAppRepository(database_path)
    assert await ConversationCastService(restarted).get(conversation_id) == saved


@pytest.mark.asyncio
async def test_cast_rejects_more_than_five_and_two_versions_of_same_character(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id, first_version_id = await make_conversation(repository)
    first = await repository.get_character_version(first_version_id)
    newer = await repository.create_character_version(
        "renamed", "new prompt", character_id=first.character_id
    )

    with pytest.raises(ValidationError):
        await ConversationCastService(repository).set_cast(
            conversation_id, tuple(f"version-{index}" for index in range(6))
        )
    with pytest.raises(ValidationError):
        await ConversationCastService(repository).set_cast(
            conversation_id, (first.id, newer.id)
        )


@pytest.mark.asyncio
async def test_cast_replacement_rolls_back_roster_and_primary_on_insert_failure(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    conversation_id, first_version_id = await make_conversation(repository)
    second = await repository.create_character_version("second", "second prompt")
    before = await ConversationCastService(repository).get(conversation_id)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_second_cast_member BEFORE INSERT "
            "ON conversation_cast_members WHEN NEW.position = 1 "
            "BEGIN SELECT RAISE(ABORT, 'forced'); END"
        )

    with pytest.raises(PersistenceError):
        await ConversationCastService(repository).set_cast(
            conversation_id, (second.id, first_version_id)
        )

    assert await ConversationCastService(repository).get(conversation_id) == before
    assert (await repository.get_conversation(conversation_id)).character_version_id == (
        first_version_id
    )


@pytest.mark.asyncio
async def test_legacy_character_selection_cannot_bypass_a_multi_person_cast(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    conversation_id, first_version_id = await make_conversation(repository)
    second = await repository.create_character_version("second", "second prompt")
    outside = await repository.create_character_version("outside", "outside prompt")
    service = ConversationCastService(repository)
    before = await service.set_cast(conversation_id, (first_version_id, second.id))
    conversation = await repository.get_conversation(conversation_id)

    with pytest.raises(ValidationError):
        await repository.update_conversation_selection(
            conversation_id, outside.id, conversation.model_profile_id
        )

    assert await service.get(conversation_id) == before
    assert (await repository.get_conversation(conversation_id)).character_version_id == (
        first_version_id
    )
