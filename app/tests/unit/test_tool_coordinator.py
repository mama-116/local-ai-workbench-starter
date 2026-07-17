from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_llm_chat.application.services.tool_coordinator import ToolCoordinator
from local_llm_chat.domain.models import (
    ToolCallRequest,
    ToolDefinition,
    ToolProviderResult,
)
from local_llm_chat.domain.states import ToolCallState
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


class FakeToolProvider:
    @property
    def name(self) -> str:
        return "fake"

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        return (ToolDefinition("echo", "echo", {"type": "object"}),)

    async def execute(
        self, conversation_id: str, tool_name: str, arguments: dict[str, object]
    ) -> ToolProviderResult:
        assert conversation_id
        if arguments.get("delay"):
            await asyncio.sleep(0.05)
        return ToolProviderResult(str(arguments.get("text", "ok")), 1)


@pytest.mark.asyncio
async def test_executes_sequentially_and_audits_completed_calls(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    coordinator = ToolCoordinator(repository, (FakeToolProvider(),))

    results = await coordinator.execute_calls(
        "conversation", "run", (
            ToolCallRequest("one", "echo", {"text": "first"}),
            ToolCallRequest("two", "echo", {"text": "second"}),
        )
    )

    assert [result.content for result in results] == ["first", "second"]
    audits = await repository.list_tool_calls("conversation")
    assert [audit.state for audit in audits] == [
        ToolCallState.COMPLETED,
        ToolCallState.COMPLETED,
    ]
    assert audits[0].result_sha256 is not None
    assert audits[0].result_content is None


@pytest.mark.asyncio
async def test_rejects_calls_over_per_turn_limit_without_executing(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    coordinator = ToolCoordinator(repository, (FakeToolProvider(),))
    calls = tuple(ToolCallRequest(str(i), "echo", {}) for i in range(4))

    results = await coordinator.execute_calls("conversation", "run", calls)

    assert len(results) == 4
    assert all(result.is_error for result in results)
    audits = await repository.list_tool_calls("conversation")
    assert len(audits) == 4
    assert all(audit.state is ToolCallState.DENIED for audit in audits)


@pytest.mark.asyncio
async def test_carries_call_and_result_budgets_across_rounds(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    coordinator = ToolCoordinator(repository, (FakeToolProvider(),))
    first = await coordinator.execute_calls(
        "conversation",
        "run",
        (ToolCallRequest("one", "echo", {"text": "a" * 40_000}),),
    )
    first_size = len(first[0].content.encode("utf-8"))

    second = await coordinator.execute_calls(
        "conversation",
        "run",
        (ToolCallRequest("two", "echo", {"text": "b" * 40_000}),),
        calls_used=1,
        result_bytes_used=first_size,
    )
    denied = await coordinator.execute_calls(
        "conversation",
        "run",
        (ToolCallRequest("four", "echo", {}),),
        calls_used=3,
        result_bytes_used=first_size + len(second[0].content.encode("utf-8")),
    )

    assert first_size + len(second[0].content.encode("utf-8")) == 64 * 1024
    assert denied[0].is_error
    audits = await repository.list_tool_calls("conversation")
    assert audits[-1].state is ToolCallState.DENIED


@pytest.mark.asyncio
async def test_unknown_tool_and_timeout_are_contained_and_audited(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    coordinator = ToolCoordinator(repository, (FakeToolProvider(),), timeout_seconds=0.01)

    results = await coordinator.execute_calls(
        "conversation", "run", (
            ToolCallRequest("unknown", "missing", {}),
            ToolCallRequest("slow", "echo", {"delay": True}),
        )
    )

    assert len(results) == 2
    assert all(result.is_error for result in results)
    audits = await repository.list_tool_calls("conversation")
    assert audits[0].state is ToolCallState.DENIED
    assert audits[1].state is ToolCallState.FAILED
