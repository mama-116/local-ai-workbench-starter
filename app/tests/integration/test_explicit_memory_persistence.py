from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from local_llm_chat.application.services.explicit_memory_service import (
    ExplicitMemoryService,
    RememberExplicitMemoryRequest,
    UndoExplicitMemoryRequest,
)
from local_llm_chat.application.services.conversation_group_settings_service import (
    ConversationGroupSettingsService,
)
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.states import MessageState, TurnMode
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


@pytest.mark.asyncio
async def test_legacy_explicit_memory_version_17_is_reconciled(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-explicit.sqlite3"
    migrations = (
        Path(__file__).parents[2]
        / "src"
        / "local_llm_chat"
        / "infrastructure"
        / "persistence"
        / "migrations"
    )
    with sqlite3.connect(database_path) as connection:
        for migration in sorted(migrations.glob("*.sql")):
            version = int(migration.stem.split("_", maxsplit=1)[0])
            if version > 16:
                continue
            connection.executescript(migration.read_text(encoding="utf-8"))
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                (version, "2026-07-24T00:00:00+00:00"),
            )
        connection.executescript(
            """
            CREATE TABLE explicit_memory_events (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                branch_id TEXT NOT NULL,
                source_message_id TEXT NOT NULL,
                character_id TEXT NOT NULL REFERENCES characters(id),
                value TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                UNIQUE(id, conversation_id),
                FOREIGN KEY(branch_id, conversation_id)
                    REFERENCES branches(id, conversation_id),
                FOREIGN KEY(source_message_id, conversation_id)
                    REFERENCES messages(id, conversation_id)
            );
            CREATE TABLE explicit_memory_decisions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                target_event_id TEXT NOT NULL UNIQUE,
                state TEXT NOT NULL CHECK(state = 'undone'),
                recorded_at TEXT NOT NULL,
                FOREIGN KEY(target_event_id, conversation_id)
                    REFERENCES explicit_memory_events(id, conversation_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES(17, ?)",
            ("2026-07-24T00:00:00+00:00",),
        )

    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    await repository.initialize()

    with sqlite3.connect(database_path) as check:
        assert check.execute(
            "SELECT version FROM schema_migrations WHERE version IN (17, 22, 23) "
            "ORDER BY version"
        ).fetchall() == [(17,), (22,), (23,)]
        guard_columns = {
            row[1]
            for row in check.execute(
                "PRAGMA table_info(conversation_deletion_guards)"
            ).fetchall()
        }
        assert guard_columns == {"conversation_id", "authorized_at"}
        assert check.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.asyncio
async def test_explicit_memory_is_character_scoped_and_survives_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    alice = await repository.create_character_version("アリス", "アリスとして話す")
    bob = await repository.create_character_version("ボブ", "ボブとして話す")
    profile = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation("明示記憶", alice.id, profile.id)
    session = await repository.start_send(conversation.id, "毎週日曜は散歩する")
    service = ExplicitMemoryService(repository)
    request = RememberExplicitMemoryRequest(
        "dialog-1",
        conversation.id,
        session.branch_id,
        session.user_message.id,
        alice.character_id,
        "毎週日曜は散歩する",
    )

    saved = await service.remember(request)
    duplicate = await service.remember(request)

    assert duplicate.id == saved.id
    restarted = SQLiteAppRepository(database_path)
    await restarted.initialize()
    assert [
        item.value
        for item in await restarted.project_explicit_memory(
            conversation.id, session.branch_id, alice.character_id
        )
    ] == ["毎週日曜は散歩する"]
    assert (
        await restarted.project_explicit_memory(
            conversation.id, session.branch_id, bob.character_id
        )
        == ()
    )


@pytest.mark.asyncio
async def test_explicit_memory_follows_character_identity_and_undo_is_append_only(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    alice = await repository.create_character_version("アリス", "初版")
    alice_v2 = await repository.create_character_version(
        "アリス", "改訂版", character_id=alice.character_id
    )
    profile = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation("記憶", alice.id, profile.id)
    session = await repository.start_send(conversation.id, "猫の名前はミケ")
    service = ExplicitMemoryService(repository)
    saved = await service.remember(
        RememberExplicitMemoryRequest(
            "dialog-1",
            conversation.id,
            session.branch_id,
            session.user_message.id,
            alice.character_id,
            "猫の名前はミケ",
        )
    )

    await repository.update_conversation_selection(
        conversation.id, alice_v2.id, profile.id
    )
    assert [
        item.event_id
        for item in await service.list_items(
            conversation.id, session.branch_id, alice.character_id
        )
    ] == [saved.id]

    await service.undo(
        UndoExplicitMemoryRequest(
            conversation.id,
            session.branch_id,
            saved.id,
            alice.character_id,
        )
    )
    assert (
        await service.list_items(
            conversation.id, session.branch_id, alice.character_id
        )
        == ()
    )

    restored = await service.remember(
        RememberExplicitMemoryRequest(
            "dialog-2",
            conversation.id,
            session.branch_id,
            session.user_message.id,
            alice.character_id,
            "猫の名前はミケ",
        )
    )
    assert restored.id != saved.id


@pytest.mark.asyncio
async def test_empty_trash_deletes_explicit_memory_with_its_conversation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    character = await repository.create_character_version("アリス", "初版")
    profile = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation(
        "削除する会話", character.id, profile.id
    )
    session = await repository.start_send(conversation.id, "猫の名前はミケ")
    service = ExplicitMemoryService(repository)
    saved = await service.remember(
        RememberExplicitMemoryRequest(
            "dialog-delete",
            conversation.id,
            session.branch_id,
            session.user_message.id,
            character.character_id,
            "猫の名前はミケ",
        )
    )
    await service.undo(
        UndoExplicitMemoryRequest(
            conversation.id,
            session.branch_id,
            saved.id,
            character.character_id,
        )
    )

    await repository.archive_conversation(conversation.id)
    assert await repository.delete_archived_conversations((conversation.id,)) == 1

    with sqlite3.connect(database_path) as check:
        assert check.execute(
            "SELECT COUNT(*) FROM explicit_memory_events"
        ).fetchone() == (0,)
        assert check.execute(
            "SELECT COUNT(*) FROM explicit_memory_decisions"
        ).fetchone() == (0,)
        assert check.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.asyncio
async def test_explicit_memory_rejects_invalid_content_and_changed_character(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    alice = await repository.create_character_version("アリス", "アリス")
    bob = await repository.create_character_version("ボブ", "ボブ")
    profile = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation("記憶", alice.id, profile.id)
    session = await repository.start_send(conversation.id, "覚えて")
    service = ExplicitMemoryService(repository)

    for request_id, value in (("empty", ""), ("long", "長" * 201)):
        with pytest.raises(ValidationError):
            await service.remember(
                RememberExplicitMemoryRequest(
                    request_id,
                    conversation.id,
                    session.branch_id,
                    session.user_message.id,
                    alice.character_id,
                    value,
                )
            )

    await repository.update_conversation_selection(conversation.id, bob.id, profile.id)
    with pytest.raises(ValidationError, match="保存条件"):
        await service.remember(
            RememberExplicitMemoryRequest(
                "stale-dialog",
                conversation.id,
                session.branch_id,
                session.user_message.id,
                alice.character_id,
                "現在のキャラクターではない",
            )
        )


@pytest.mark.asyncio
async def test_explicit_memory_rejects_assistant_messages_and_group_chat(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.create_character_version("アリス", "アリス")
    profile = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation(
        "境界", character.id, profile.id
    )
    session = await repository.start_send(conversation.id, "利用者発言")
    service = ExplicitMemoryService(repository)

    with pytest.raises(ValidationError):
        await service.remember(
            RememberExplicitMemoryRequest(
                "assistant",
                conversation.id,
                session.branch_id,
                session.assistant_message.id,
                character.character_id,
                "AI発言",
            )
        )

    await ConversationGroupSettingsService(repository).configure(
        conversation.id,
        enabled=True,
        mode=TurnMode.STORY,
    )
    with pytest.raises(ValidationError, match="保存条件"):
        await service.remember(
            RememberExplicitMemoryRequest(
                "group",
                conversation.id,
                session.branch_id,
                session.user_message.id,
                character.character_id,
                "グループでは保存しない",
            )
        )


@pytest.mark.asyncio
async def test_explicit_memory_never_crosses_branch_or_conversation(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    character = await repository.create_character_version("アリス", "アリス")
    profile = await repository.ensure_model_profile("ollama-local", "model", {})
    conversation = await repository.create_conversation(
        "分岐", character.id, profile.id
    )
    root = await repository.start_send(conversation.id, "root")
    await repository.finish_response(root, "root response", MessageState.COMPLETED)
    child = await repository.start_rewrite(
        conversation.id, root.user_message.id, "child"
    )
    service = ExplicitMemoryService(repository)
    saved = await service.remember(
        RememberExplicitMemoryRequest(
            "child-dialog",
            conversation.id,
            child.branch_id,
            child.user_message.id,
            character.character_id,
            "child only",
        )
    )

    assert [
        item.event_id
        for item in await service.list_items(
            conversation.id, child.branch_id, character.character_id
        )
    ] == [saved.id]
    await repository.activate_branch(conversation.id, root.branch_id)
    assert (
        await service.list_items(
            conversation.id, root.branch_id, character.character_id
        )
        == ()
    )
    with pytest.raises(ValidationError, match="保存条件"):
        await service.remember(
            RememberExplicitMemoryRequest(
                "wrong-branch",
                conversation.id,
                root.branch_id,
                child.user_message.id,
                character.character_id,
                "not visible",
            )
        )

    other = await repository.create_conversation("別会話", character.id, profile.id)
    with pytest.raises(ValidationError):
        await service.list_items(
            other.id, root.branch_id, character.character_id
        )
