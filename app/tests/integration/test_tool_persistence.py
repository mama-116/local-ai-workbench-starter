from __future__ import annotations

from pathlib import Path

import pytest

from local_llm_chat.domain.models import ToolCallRequest
from local_llm_chat.domain.states import ToolCallState
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


@pytest.mark.asyncio
async def test_folder_grant_can_be_saved_restored_and_revoked(tmp_path: Path) -> None:
    database = tmp_path / "chat.sqlite3"
    repository = SQLiteAppRepository(database)
    await repository.initialize()
    folder = tmp_path / "allowed"
    folder.mkdir()

    saved = await repository.set_tool_folder_grant("conversation", folder)
    reopened = SQLiteAppRepository(database)
    restored = await reopened.get_tool_folder_grant("conversation")

    assert restored == saved
    await reopened.revoke_tool_folder_grant("conversation")
    assert await reopened.get_tool_folder_grant("conversation") is None


@pytest.mark.asyncio
async def test_tool_audit_state_transitions_do_not_expose_result_body(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    call = await repository.create_tool_call(
        "conversation", "run", "builtin", ToolCallRequest("call", "read_text", {"path": "a.md"})
    )
    await repository.mark_tool_call_running(call.id)
    await repository.finish_tool_call(
        call.id, ToolCallState.COMPLETED, "private body", 1
    )

    [audit] = await repository.list_tool_calls("conversation")
    assert audit.state is ToolCallState.COMPLETED
    assert audit.result_size_bytes == len("private body".encode())
    assert audit.result_sha256 is not None
    assert audit.result_content is None
    assert audit.started_at is not None
    assert audit.completed_at is not None
