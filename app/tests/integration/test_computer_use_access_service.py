from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from local_llm_chat.application.services.computer_use_access_service import (
    ComputerUseAccessService,
)
from local_llm_chat.application.services.computer_use_service import (
    ComputerUseCoordinator,
)
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import ComputerPlanApproval, DesktopSecurityContext
from local_llm_chat.domain.states import (
    ComputerActionState,
    ComputerUseRunState,
    DesktopIntegrityLevel,
)
from local_llm_chat.infrastructure.computer_use.fake_action_broker import (
    FakeDesktopActionBroker,
)
from local_llm_chat.infrastructure.persistence.sqlite_repositories import (
    SQLiteAppRepository,
)


NOW = datetime(2026, 7, 18, 2, 0, tzinfo=UTC)


def safe_context() -> DesktopSecurityContext:
    return DesktopSecurityContext(
        "same-user",
        "same-user",
        1,
        1,
        DesktopIntegrityLevel.MEDIUM,
        DesktopIntegrityLevel.MEDIUM,
        False,
        False,
        False,
        False,
    )


async def conversation_id(repository: SQLiteAppRepository) -> str:
    character = await repository.ensure_default_character()
    profile = await repository.ensure_model_profile("local", "fake", {})
    conversation = await repository.create_conversation(
        "Fake Computer Use", character.id, profile.id
    )
    return conversation.id


@pytest.mark.asyncio
async def test_fake_review_requires_approval_then_persists_completed_audit(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    broker = FakeDesktopActionBroker(repository, clock=lambda: NOW)
    service = ComputerUseAccessService(
        repository,
        ComputerUseCoordinator(broker, clock=lambda: NOW),
        clock=lambda: NOW,
    )
    plan, run = await service.stage_fake_notepad_plan(
        await conversation_id(repository), "安全確認🙂"
    )

    assert broker.calls == []
    assert run.state is ComputerUseRunState.AWAITING_APPROVAL
    assert all(
        action.state is ComputerActionState.PROPOSED
        for action in await repository.list_computer_actions(run.id)
    )

    completed = await service.approve_and_execute_fake(
        run.id, plan, safe_context()
    )

    assert completed.state is ComputerUseRunState.COMPLETED
    assert len(broker.calls) == 3
    assert all(
        action.state is ComputerActionState.COMPLETED
        for action in await repository.list_computer_actions(run.id)
    )
    reopened = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await reopened.initialize()
    history = await reopened.list_computer_use_runs(plan.conversation_id)
    assert history[0].id == run.id
    assert history[0].state is ComputerUseRunState.COMPLETED


@pytest.mark.asyncio
async def test_cancelling_review_persists_cancelled_without_fake_calls(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    broker = FakeDesktopActionBroker(repository, clock=lambda: NOW)
    service = ComputerUseAccessService(
        repository,
        ComputerUseCoordinator(broker, clock=lambda: NOW),
        clock=lambda: NOW,
    )
    _plan, run = await service.stage_fake_notepad_plan(
        await conversation_id(repository), "取り消す🙂"
    )

    cancelled = await service.cancel(run.id)

    assert cancelled.state is ComputerUseRunState.CANCELLED
    assert broker.calls == []
    assert all(
        action.state is ComputerActionState.CANCELLED
        for action in await repository.list_computer_actions(run.id)
    )


@pytest.mark.asyncio
async def test_approval_expires_from_plan_display_time_before_fake_broker(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    current = NOW
    broker = FakeDesktopActionBroker(repository, clock=lambda: current)
    service = ComputerUseAccessService(
        repository,
        ComputerUseCoordinator(broker, clock=lambda: current),
        clock=lambda: current,
    )
    plan, run = await service.stage_fake_notepad_plan(
        await conversation_id(repository), "期限切れ🙂"
    )
    current = NOW + timedelta(seconds=30)

    assert await service.approval_remaining_seconds(run.id) == 0
    with pytest.raises(ValidationError, match="承認期限が切れました"):
        await service.approve_and_execute_fake(run.id, plan, safe_context())

    expired = await repository.get_computer_use_run(run.id)
    assert expired.state is ComputerUseRunState.CANCELLED
    assert expired.failure_reason == "approval_expired"
    assert broker.calls == []
    assert all(
        action.state is ComputerActionState.CANCELLED
        and action.failure_reason == "approval_expired"
        for action in await repository.list_computer_actions(run.id)
    )


@pytest.mark.asyncio
async def test_broker_independently_rejects_approval_after_plan_deadline(
    tmp_path: Path,
) -> None:
    repository = SQLiteAppRepository(tmp_path / "chat.sqlite3")
    await repository.initialize()
    current = NOW
    broker = FakeDesktopActionBroker(repository, clock=lambda: current)
    coordinator = ComputerUseCoordinator(broker, clock=lambda: current)
    service = ComputerUseAccessService(
        repository, coordinator, clock=lambda: current
    )
    plan, run = await service.stage_fake_notepad_plan(
        await conversation_id(repository), "独立検査🙂"
    )
    current = NOW + timedelta(seconds=30)
    approval = ComputerPlanApproval("late-approval", run.plan_hash, current)
    await repository.create_computer_plan_approval(run.id, approval)

    result = await coordinator.execute(plan, safe_context(), approval)

    assert result.state is ComputerUseRunState.DENIED
    assert result.failure_reason == "plan_approval_required"
    assert broker.calls == []
    assert (
        await repository.get_computer_use_run(run.id)
    ).state is ComputerUseRunState.AWAITING_APPROVAL
