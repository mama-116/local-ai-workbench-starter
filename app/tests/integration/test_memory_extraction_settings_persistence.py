from datetime import UTC, datetime
from pathlib import Path
import sqlite3

import pytest

from local_llm_chat.domain.models import ModelRoleSetting
from local_llm_chat.domain.states import ModelRole
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


@pytest.mark.asyncio
async def test_memory_extraction_setting_survives_repository_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    setting = ModelRoleSetting(
        role=ModelRole.MEMORY_EXTRACTION,
        connection_id="dgx",
        provider_name="ollama-dgx",
        endpoint_fingerprint="a" * 64,
        model_name="gemma:31b",
        model_digest="digest-1",
        updated_at=datetime(2026, 7, 24, 1, 2, 3, tzinfo=UTC),
    )

    saved = await repository.save_model_role_setting(setting)

    restarted = SQLiteAppRepository(database_path)
    await restarted.initialize()
    loaded = await restarted.get_model_role_setting(
        ModelRole.MEMORY_EXTRACTION
    )

    assert saved == setting
    assert loaded == setting


@pytest.mark.asyncio
async def test_embedding_role_remains_compatible_with_extended_table(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()

    configuration = await repository.get_embedding_configuration()

    assert configuration.desired is None
    assert configuration.active is None


@pytest.mark.asyncio
async def test_migration_preserves_existing_active_embedding_profile(
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
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    for migration in sorted(migrations.glob("*.sql")):
        version = int(migration.stem.split("_", maxsplit=1)[0])
        if version >= 18:
            continue
        connection.executescript(migration.read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
            (version, datetime.now(UTC).isoformat()),
        )
    now = datetime.now(UTC).isoformat()
    connection.execute(
        """
        INSERT INTO embedding_profiles(
            id, connection_id, provider_name, endpoint_fingerprint,
            model_name, model_digest, vector_dimensions, min_similarity,
            state, total_chunks, embedded_chunks, last_error,
            created_at, updated_at
        ) VALUES(
            'profile-1', 'local', 'ollama-local', 'fingerprint',
            'embed:latest', 'digest', 3, 0.5,
            'ready', 0, 0, NULL, ?, ?
        )
        """,
        (now, now),
    )
    connection.execute(
        """
        INSERT INTO model_role_settings(
            role, desired_profile_id, active_profile_id, updated_at
        ) VALUES('embedding', 'profile-1', 'profile-1', ?)
        """,
        (now,),
    )
    connection.commit()
    connection.close()

    repository = SQLiteAppRepository(database_path)
    await repository.initialize()
    configuration = await repository.get_embedding_configuration()

    assert configuration.desired is not None
    assert configuration.desired.id == "profile-1"
    assert configuration.active is not None
    assert configuration.active.id == "profile-1"
