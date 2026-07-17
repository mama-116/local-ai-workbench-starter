from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcp import types
from mcp.server import NotificationOptions

from local_llm_chat.application.services.tool_coordinator import ToolCoordinator
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import ToolCallRequest
from local_llm_chat.domain.states import ToolCallState
from local_llm_chat.infrastructure.mcp.profile import local_notes_profile
from local_llm_chat.infrastructure.mcp.local_notes_server import server
from local_llm_chat.infrastructure.mcp.tool_provider import (
    McpCallResult,
    SdkMcpSessionRunner,
    TrustedMcpToolProvider,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


@pytest.mark.asyncio
async def test_bundled_mcp_searches_and_reads_only_the_current_conversation_root(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "notes.md").write_text("first-only needle", encoding="utf-8")
    (second / "notes.md").write_text("second-only needle", encoding="utf-8")
    await repository.set_tool_folder_grant("first-conversation", first)
    await repository.set_tool_folder_grant("second-conversation", second)
    provider = TrustedMcpToolProvider(repository, local_notes_profile())

    first_search = await provider.execute(
        "first-conversation", provider.SEARCH_NAME, {"query": "needle"}
    )
    second_read = await provider.execute(
        "second-conversation", provider.READ_NAME, {"path": "notes.md"}
    )

    assert "first-only" in first_search.content
    assert "second-only" not in first_search.content
    assert "second-only" in second_read.content
    assert "first-only" not in second_read.content


@pytest.mark.asyncio
async def test_bundled_mcp_denies_invalid_path_and_revoked_grant(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "notes.md").write_text("safe", encoding="utf-8")
    await repository.set_tool_folder_grant("conversation", allowed)
    provider = TrustedMcpToolProvider(repository, local_notes_profile())

    with pytest.raises(ValidationError, match="許可されていない"):
        await provider.execute(
            "conversation", provider.READ_NAME, {"path": "../notes.md"}
        )

    await repository.revoke_tool_folder_grant("conversation")
    with pytest.raises(ValidationError, match="許可されていません"):
        await provider.execute(
            "conversation", provider.SEARCH_NAME, {"query": "safe"}
        )


class BrokenRunner:
    async def call(
        self,
        profile: object,
        allowed_root: Path,
        tool_name: str,
        arguments: dict[str, object],
    ) -> McpCallResult:
        del profile, allowed_root, tool_name, arguments
        raise EOFError("server stopped")


class SlowRunner:
    async def call(
        self,
        profile: object,
        allowed_root: Path,
        tool_name: str,
        arguments: dict[str, object],
    ) -> McpCallResult:
        del profile, allowed_root, tool_name, arguments
        await asyncio.sleep(1)
        return McpCallResult("late", 1)


class MalformedProtocolRunner:
    async def call(
        self,
        profile: object,
        allowed_root: Path,
        tool_name: str,
        arguments: dict[str, object],
    ) -> McpCallResult:
        del profile, allowed_root, tool_name, arguments
        raise ValueError("malformed JSON-RPC")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner", [BrokenRunner(), SlowRunner(), MalformedProtocolRunner()]
)
async def test_mcp_stop_and_timeout_are_contained_and_audited(
    tmp_path: Path,
    runner: BrokenRunner | SlowRunner | MalformedProtocolRunner,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    await repository.set_tool_folder_grant("conversation", allowed)
    provider = TrustedMcpToolProvider(repository, local_notes_profile(), runner)
    coordinator = ToolCoordinator(repository, (provider,), timeout_seconds=0.01)

    results = await coordinator.execute_calls(
        "conversation",
        "run",
        (ToolCallRequest("call", provider.SEARCH_NAME, {"query": "x"}),),
    )

    assert results[0].is_error
    audits = await repository.list_tool_calls("conversation")
    assert audits[0].state is ToolCallState.FAILED


@pytest.mark.parametrize(
    "capabilities",
    [
        types.ServerCapabilities(
            tools=types.ToolsCapability(), resources=types.ResourcesCapability()
        ),
        types.ServerCapabilities(
            tools=types.ToolsCapability(), prompts=types.PromptsCapability()
        ),
        types.ServerCapabilities(
            tools=types.ToolsCapability(), tasks=types.ServerTasksCapability()
        ),
        types.ServerCapabilities(
            tools=types.ToolsCapability(), experimental={"forbidden": {}}
        ),
    ],
)
def test_rejects_resources_prompts_tasks_and_experimental_capabilities(
    capabilities: types.ServerCapabilities,
) -> None:
    forbidden = types.InitializeResult(
        protocolVersion="2025-11-25",
        serverInfo=types.Implementation(name="local-notes", version="1"),
        capabilities=capabilities,
    )

    with pytest.raises(RuntimeError, match="forbidden capability"):
        SdkMcpSessionRunner._require_capabilities(forbidden)


def test_bundled_server_exposes_only_tool_capability() -> None:
    capabilities = server.get_capabilities(
        notification_options=NotificationOptions(),
        experimental_capabilities={},
    )

    assert capabilities.tools is not None
    assert capabilities.resources is None
    assert capabilities.prompts is None
    assert capabilities.tasks is None
    assert not capabilities.experimental
