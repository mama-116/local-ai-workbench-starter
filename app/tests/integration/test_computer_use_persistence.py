from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from local_llm_chat.application.services.computer_use_service import (
    computer_plan_hash,
)
from local_llm_chat.domain.models import (
    ComputerActionAudit,
    ComputerActionRequest,
    ComputerPlan,
    ComputerPlanApproval,
    ComputerUseLimits,
    ComputerUseRun,
)
from local_llm_chat.domain.states import (
    ComputerActionState,
    ComputerActionType,
    ComputerUseRunState,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)
from local_llm_chat.domain.errors import ValidationError


NOW = datetime(2026, 7, 18, 1, 0, tzinfo=UTC)


async def create_conversation(repository: SQLiteAppRepository) -> str:
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile(
        "ollama-local", "qwen3.5:9b", {"num_ctx": 4096}
    )
    conversation = await repository.create_conversation(
        "Computer Use監査", character.id, profile.id
    )
    return conversation.id


def plan(conversation_id: str) -> ComputerPlan:
    return ComputerPlan(
        conversation_id,
        "メモ帳へ日本語と絵文字を入力して保存しない",
        "observation-1",
        (
            ComputerActionRequest(
                "launch",
                ComputerActionType.LAUNCH_ALLOWED_APP,
                "windows_notepad",
            ),
            ComputerActionRequest(
                "click",
                ComputerActionType.CLICK_UIA_ELEMENT,
                "notepad_edit",
            ),
            ComputerActionRequest(
                "type",
                ComputerActionType.TYPE_PLAIN_TEXT,
                "notepad_edit",
                "安全確認🙂",
            ),
        ),
    )


def awaiting_run(identifier: str, value: ComputerPlan) -> ComputerUseRun:
    return ComputerUseRun(
        identifier,
        value.conversation_id,
        value.objective,
        value.observation_id,
        computer_plan_hash(value),
        ComputerUseLimits(),
        len(value.actions),
        ComputerUseRunState.AWAITING_APPROVAL,
        None,
        NOW,
    )


async def add_all_actions(
    repository: SQLiteAppRepository, run: ComputerUseRun, value: ComputerPlan
) -> list[ComputerActionAudit]:
    stored: list[ComputerActionAudit] = []
    for ordinal, request in enumerate(value.actions, start=1):
        stored.append(
            await repository.create_computer_action(
                ComputerActionAudit(
                    f"action-{ordinal}",
                    run.id,
                    ordinal,
                    request,
                    ComputerActionState.PROPOSED,
                    None,
                    NOW,
                )
            )
        )
    return stored


@pytest.mark.asyncio
async def test_private_action_audit_survives_reopen(tmp_path: Path) -> None:
    database = tmp_path / "chat.sqlite3"
    first = SQLiteAppRepository(database)
    await first.initialize()
    conversation_id = await create_conversation(first)
    value = plan(conversation_id)
    run = await first.create_computer_use_run(awaiting_run("run-1", value))
    await first.create_computer_action(
        ComputerActionAudit(
            "action-1",
            run.id,
            1,
            value.actions[2],
            ComputerActionState.PROPOSED,
            None,
            NOW,
        )
    )

    reopened = SQLiteAppRepository(database)
    await reopened.initialize()

    stored_run = await reopened.get_computer_use_run(run.id)
    stored_actions = await reopened.list_computer_actions(run.id)
    assert stored_run.objective == "メモ帳へ日本語と絵文字を入力して保存しない"
    assert stored_actions[0].request.text == "安全確認🙂"
    assert stored_actions[0].request.action_type is ComputerActionType.TYPE_PLAIN_TEXT


@pytest.mark.asyncio
async def test_approval_can_be_consumed_only_once_across_instances(
    tmp_path: Path,
) -> None:
    database = tmp_path / "chat.sqlite3"
    first = SQLiteAppRepository(database)
    second = SQLiteAppRepository(database)
    await first.initialize()
    await second.initialize()
    conversation_id = await create_conversation(first)
    value = plan(conversation_id)
    run = await first.create_computer_use_run(awaiting_run("run-1", value))
    await add_all_actions(first, run, value)
    approval = ComputerPlanApproval("approval-1", run.plan_hash, NOW)
    await first.create_computer_plan_approval(run.id, approval)

    results = await asyncio.gather(
        first.verify_and_consume(approval, run.plan_hash, NOW),
        second.verify_and_consume(approval, run.plan_hash, NOW),
    )

    assert sorted(results) == [False, True]
    assert (await first.get_computer_use_run(run.id)).state is ComputerUseRunState.RUNNING
    assert await second.verify_and_consume(approval, run.plan_hash, NOW) is False


@pytest.mark.asyncio
async def test_repository_rejects_approval_after_plan_deadline(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    conversation_id = await create_conversation(repository)
    value = plan(conversation_id)
    run = await repository.create_computer_use_run(awaiting_run("run-1", value))
    await add_all_actions(repository, run, value)
    approval = ComputerPlanApproval("approval-1", run.plan_hash, NOW)
    await repository.create_computer_plan_approval(run.id, approval)

    assert not await repository.verify_and_consume(
        approval,
        run.plan_hash,
        NOW + timedelta(seconds=run.limits.approval_timeout_seconds),
    )
    assert (
        await repository.get_computer_use_run(run.id)
    ).state is ComputerUseRunState.AWAITING_APPROVAL


@pytest.mark.asyncio
async def test_restart_fails_unfinished_run_and_actions(tmp_path: Path) -> None:
    database = tmp_path / "chat.sqlite3"
    first = SQLiteAppRepository(database)
    await first.initialize()
    conversation_id = await create_conversation(first)
    value = plan(conversation_id)
    run = await first.create_computer_use_run(awaiting_run("run-1", value))
    await first.create_computer_action(
        ComputerActionAudit(
            "action-1",
            run.id,
            1,
            value.actions[0],
            ComputerActionState.PROPOSED,
            None,
            NOW,
        )
    )

    restarted = SQLiteAppRepository(database)
    await restarted.initialize()
    await restarted.recover_interrupted_runs()

    stored_run = await restarted.get_computer_use_run(run.id)
    stored_actions = await restarted.list_computer_actions(run.id)
    assert stored_run.state is ComputerUseRunState.FAILED
    assert stored_run.failure_reason == "previous_session_interrupted"
    assert stored_actions[0].state is ComputerActionState.FAILED
    assert stored_actions[0].failure_reason == "previous_session_interrupted"


@pytest.mark.asyncio
async def test_terminal_audit_cannot_be_rewritten(tmp_path: Path) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    conversation_id = await create_conversation(repository)
    value = plan(conversation_id)
    run = await repository.create_computer_use_run(awaiting_run("run-1", value))
    actions = await add_all_actions(repository, run, value)
    approval = ComputerPlanApproval("approval-1", run.plan_hash, NOW)
    await repository.create_computer_plan_approval(run.id, approval)
    assert await repository.verify_and_consume(approval, run.plan_hash, NOW)
    completed_actions: list[ComputerActionAudit] = []
    for action in actions:
        action = await repository.update_computer_action(
            replace(action, state=ComputerActionState.APPROVED)
        )
        action = await repository.update_computer_action(
            replace(action, state=ComputerActionState.RUNNING, started_at=NOW)
        )
        completed_actions.append(
            await repository.update_computer_action(
                replace(
                    action,
                    state=ComputerActionState.COMPLETED,
                    completed_at=NOW,
                )
            )
        )
    running = await repository.get_computer_use_run(run.id)
    completed = await repository.finish_computer_use_run(
        replace(running, state=ComputerUseRunState.COMPLETED, completed_at=NOW)
    )

    with pytest.raises(ValidationError):
        await repository.update_computer_action(
            replace(completed_actions[0], state=ComputerActionState.FAILED)
        )
    with pytest.raises(ValidationError):
        await repository.finish_computer_use_run(
            replace(completed, state=ComputerUseRunState.FAILED)
        )


@pytest.mark.asyncio
async def test_incomplete_action_audit_blocks_approval_and_completion(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    conversation_id = await create_conversation(repository)
    value = plan(conversation_id)
    run = await repository.create_computer_use_run(awaiting_run("run-1", value))
    await repository.create_computer_action(
        ComputerActionAudit(
            "action-1",
            run.id,
            1,
            value.actions[0],
            ComputerActionState.PROPOSED,
            None,
            NOW,
        )
    )
    approval = ComputerPlanApproval("approval-1", run.plan_hash, NOW)
    await repository.create_computer_plan_approval(run.id, approval)

    assert await repository.verify_and_consume(approval, run.plan_hash, NOW) is False
    stored = await repository.get_computer_use_run(run.id)
    assert stored.state is ComputerUseRunState.AWAITING_APPROVAL
    with pytest.raises(ValidationError):
        await repository.finish_computer_use_run(
            replace(stored, state=ComputerUseRunState.COMPLETED)
        )


@pytest.mark.asyncio
async def test_action_cannot_be_added_after_approval_is_consumed(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    conversation_id = await create_conversation(repository)
    value = plan(conversation_id)
    run = await repository.create_computer_use_run(awaiting_run("run-1", value))
    await add_all_actions(repository, run, value)
    approval = ComputerPlanApproval("approval-1", run.plan_hash, NOW)
    await repository.create_computer_plan_approval(run.id, approval)
    assert await repository.verify_and_consume(approval, run.plan_hash, NOW)

    with pytest.raises(ValidationError):
        await repository.create_computer_action(
            ComputerActionAudit(
                "action-extra",
                run.id,
                4,
                value.actions[2],
                ComputerActionState.PROPOSED,
                None,
                NOW,
            )
        )
