from __future__ import annotations

from pathlib import Path

import pytest

from local_llm_chat.application.services.builtin_tool_provider import (
    BuiltInToolProvider,
)
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


@pytest.mark.asyncio
async def test_provider_exposes_common_contract_and_executes_search_and_read(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "notes.md").write_text("needle details", encoding="utf-8")
    await repository.set_tool_folder_grant("conversation", allowed)
    provider = BuiltInToolProvider(repository)

    search = await provider.execute(
        "conversation", provider.SEARCH_NAME, {"query": "needle"}
    )
    read = await provider.execute(
        "conversation", provider.READ_NAME, {"path": "notes.md"}
    )

    assert {tool.name for tool in provider.list_tools()} == {
        provider.SEARCH_NAME,
        provider.READ_NAME,
    }
    assert search.item_count == 1
    assert read.item_count == 1
    assert "needle details" in read.content


@pytest.mark.asyncio
async def test_provider_denies_after_folder_grant_is_revoked(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    await repository.set_tool_folder_grant("conversation", allowed)
    await repository.revoke_tool_folder_grant("conversation")

    with pytest.raises(ValidationError, match="許可されていません"):
        await BuiltInToolProvider(repository).execute(
            "conversation", BuiltInToolProvider.SEARCH_NAME, {"query": "x"}
        )
