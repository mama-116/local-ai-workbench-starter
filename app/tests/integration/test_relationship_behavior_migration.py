from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from local_llm_chat.domain.relationship_behavior import (
    DEFAULT_RELATIONSHIP_STYLE,
    RelationshipConflictResponse,
    RelationshipExpressiveness,
    RelationshipPace,
    RelationshipPriority,
    RelationshipStyle,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


def _apply_migrations_before_24(database: Path) -> None:
    migration_dir = (
        Path(__file__).parents[2]
        / "src"
        / "local_llm_chat"
        / "infrastructure"
        / "persistence"
        / "migrations"
    )
    connection = sqlite3.connect(database)
    try:
        for migration in sorted(migration_dir.glob("*.sql")):
            version = int(migration.stem.split("_", 1)[0])
            if version >= 22:
                continue
            connection.executescript(migration.read_text(encoding="utf-8"))
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                VALUES(?, '2026-07-27T00:00:00+00:00')
                """,
                (version,),
            )
        connection.execute(
            """
            INSERT INTO characters(id, display_name, created_at)
            VALUES('character-existing', '既存人物', '2026-07-27T00:00:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO character_versions(
                id, character_id, version, system_prompt, created_at
            ) VALUES(
                'version-existing', 'character-existing', 1, '既存指示',
                '2026-07-27T00:00:00+00:00'
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_existing_database_migrates_to_default_style_and_is_idempotent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "existing.sqlite3"
    _apply_migrations_before_24(database)
    repository = SQLiteAppRepository(database)

    await repository.initialize()
    await repository.initialize()

    existing = await repository.get_character_version("version-existing")
    assert existing.relationship_style == DEFAULT_RELATIONSHIP_STYLE
    connection = sqlite3.connect(database)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = 24"
        ).fetchone() == (1,)
        event_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(relationship_events)")
        }
        assert {
            "affinity_delta",
            "trust_delta",
            "tension_delta",
        }.issubset(event_columns)
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_character_style_is_versioned_and_persisted(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "new.sqlite3")
    await repository.initialize()
    original = await repository.ensure_default_character()
    style = RelationshipStyle(
        attachment_pace=RelationshipPace.QUICK,
        expressiveness=RelationshipExpressiveness.EXPRESSIVE,
        priority=RelationshipPriority.COMMITMENTS,
        conflict_response=RelationshipConflictResponse.REPAIR_SEEKING,
        recovery_pace=RelationshipPace.SLOW,
    )

    updated = await repository.create_character_version(
        original.display_name,
        original.system_prompt,
        original.character_id,
        style,
    )
    restored = await repository.get_character_version(updated.id)

    assert updated.version == original.version + 1
    assert restored.relationship_style == style
    assert (await repository.get_character_version(original.id)).relationship_style == (
        DEFAULT_RELATIONSHIP_STYLE
    )
