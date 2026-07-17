from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import (
    AgentExecutionLimits,
    AgentRun,
    AgentStep,
)
from local_llm_chat.domain.states import (
    AgentRunState,
    AgentStepState,
    AgentToolEffect,
    DataClassification,
    CostClass,
    Locality,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


async def create_conversation(repository: SQLiteAppRepository) -> str:
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "qwen3.5:9b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "Agent監査", character.id, profile.id
    )
    return conversation.id


def running_agent_run(identifier: str, conversation_id: str) -> AgentRun:
    return AgentRun(
        id=identifier,
        conversation_id=conversation_id,
        objective="日本語と絵文字を監査する 🧪",
        allowed_tools=("local_read",),
        limits=AgentExecutionLimits(),
        state=AgentRunState.RUNNING,
        failure_reason=None,
        created_at=NOW,
        started_at=NOW,
    )


def proposed_step(run_id: str) -> AgentStep:
    return AgentStep(
        id="step-1",
        run_id=run_id,
        ordinal=1,
        tool_name="local_read",
        arguments={"query": "日本語 🧪"},
        action_hash="a" * 64,
        data_classification=DataClassification.PRIVATE,
        effect=AgentToolEffect.READ,
        destination=Locality.LOCAL,
        cost_class=CostClass.NO_CHARGE,
        cost_units=1,
        state=AgentStepState.PROPOSED,
        result_size_bytes=None,
        result_sha256=None,
        restore_token=None,
        failure_reason=None,
        created_at=NOW,
    )


@pytest.mark.asyncio
async def test_agent_run_and_private_step_audit_survive_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "chat.sqlite3"
    first = SQLiteAppRepository(database)
    await first.initialize()
    conversation_id = await create_conversation(first)
    run = await first.create_agent_run(running_agent_run("run-1", conversation_id))
    await first.create_agent_step(proposed_step(run.id))

    reopened = SQLiteAppRepository(database)
    await reopened.initialize()

    stored_run = await reopened.get_agent_run(run.id)
    stored_steps = await reopened.list_agent_steps(run.id)
    assert stored_run.objective == "日本語と絵文字を監査する 🧪"
    assert stored_run.allowed_tools == ("local_read",)
    assert stored_steps[0].arguments == {"query": "日本語 🧪"}
    assert stored_steps[0].data_classification is DataClassification.PRIVATE
    assert stored_steps[0].destination is Locality.LOCAL
    assert stored_steps[0].cost_class is CostClass.NO_CHARGE


@pytest.mark.asyncio
async def test_restart_fails_interrupted_agent_run_and_never_resumes_it(
    tmp_path: Path,
) -> None:
    database = tmp_path / "chat.sqlite3"
    first = SQLiteAppRepository(database)
    await first.initialize()
    conversation_id = await create_conversation(first)
    run = await first.create_agent_run(running_agent_run("run-1", conversation_id))
    await first.create_agent_step(proposed_step(run.id))

    restarted = SQLiteAppRepository(database)
    await restarted.initialize()
    await restarted.recover_interrupted_runs()

    recovered_run = await restarted.get_agent_run(run.id)
    recovered_steps = await restarted.list_agent_steps(run.id)
    assert recovered_run.state is AgentRunState.FAILED
    assert recovered_run.failure_reason == "previous_session_interrupted"
    assert recovered_steps[0].state is AgentStepState.FAILED
    assert recovered_steps[0].failure_reason == "previous_session_interrupted"


@pytest.mark.asyncio
async def test_database_allows_only_one_running_agent_across_instances(
    tmp_path: Path,
) -> None:
    database = tmp_path / "chat.sqlite3"
    first = SQLiteAppRepository(database)
    second = SQLiteAppRepository(database)
    await first.initialize()
    await second.initialize()
    conversation_id = await create_conversation(first)

    results = await asyncio.gather(
        first.create_agent_run(running_agent_run("run-1", conversation_id)),
        second.create_agent_run(running_agent_run("run-2", conversation_id)),
        return_exceptions=True,
    )

    assert sum(isinstance(result, AgentRun) for result in results) == 1
    assert sum(isinstance(result, ValidationError) for result in results) == 1
