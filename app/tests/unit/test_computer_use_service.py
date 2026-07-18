from __future__ import annotations

from dataclasses import replace
import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from local_llm_chat.application.services.computer_use_service import (
    ComputerUseCoordinator,
)
from local_llm_chat.domain.errors import ComputerActionDenied
from local_llm_chat.domain.models import (
    ComputerActionExecution,
    ComputerActionRequest,
    ComputerPlan,
    ComputerPlanApproval,
    DesktopSecurityContext,
)
from local_llm_chat.domain.states import (
    ComputerActionType,
    ComputerUseRunState,
    DesktopIntegrityLevel,
)
from local_llm_chat.infrastructure.computer_use.fake_action_broker import (
    FakeDesktopActionBroker,
)


NOW = datetime(2026, 7, 18, 0, 0, tzinfo=UTC)


class OneTimeApprovalVerifier:
    def __init__(self, valid_ids: set[str]) -> None:
        self.valid_ids = valid_ids
        self.consumed: set[str] = set()

    async def verify_and_consume(
        self, approval: ComputerPlanApproval, plan_hash: str
    ) -> bool:
        if (
            approval.id not in self.valid_ids
            or approval.id in self.consumed
            or approval.plan_hash != plan_hash
        ):
            return False
        self.consumed.add(approval.id)
        return True


def safe_context() -> DesktopSecurityContext:
    return DesktopSecurityContext(
        controller_user_sid="S-1-5-21-test",
        target_user_sid="S-1-5-21-test",
        controller_session_id=1,
        target_session_id=1,
        controller_integrity=DesktopIntegrityLevel.MEDIUM,
        target_integrity=DesktopIntegrityLevel.MEDIUM,
        controller_elevated=False,
        target_elevated=False,
        ui_access=False,
        secure_desktop=False,
    )


def fixed_plan() -> ComputerPlan:
    return ComputerPlan(
        conversation_id="conversation-1",
        objective="メモ帳へ試験文を入力して保存しない",
        observation_id="observation-1",
        actions=(
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
                text="安全確認🙂",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_unapproved_text_plan_never_reaches_fake_provider() -> None:
    broker = FakeDesktopActionBroker(
        OneTimeApprovalVerifier(set()), clock=lambda: NOW
    )
    coordinator = ComputerUseCoordinator(
        broker,
        clock=lambda: NOW,
    )

    result = await coordinator.execute(fixed_plan(), safe_context())

    assert result.state is ComputerUseRunState.DENIED
    assert result.failure_reason == "plan_approval_required"
    assert broker.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("context", "reason"),
    [
        (replace(safe_context(), controller_elevated=True), "controller_elevated"),
        (
            replace(safe_context(), target_user_sid="S-1-5-21-other"),
            "user_sid_mismatch",
        ),
        (replace(safe_context(), target_session_id=2), "session_mismatch"),
        (replace(safe_context(), ui_access=True), "ui_access_denied"),
    ],
)
async def test_privilege_boundary_denies_before_provider(
    context: DesktopSecurityContext, reason: str
) -> None:
    broker = FakeDesktopActionBroker(
        OneTimeApprovalVerifier({"approved"}), clock=lambda: NOW
    )
    coordinator = ComputerUseCoordinator(
        broker,
        clock=lambda: NOW,
    )
    plan = fixed_plan()
    approval = ComputerPlanApproval(
        "approved", coordinator.approval_hash(plan), NOW
    )
    result = await coordinator.execute(plan, context, approval)

    assert result.state is ComputerUseRunState.DENIED
    assert result.failure_reason == reason
    assert broker.calls == []


@pytest.mark.asyncio
async def test_unknown_target_is_denied_even_with_matching_approval() -> None:
    broker = FakeDesktopActionBroker(
        OneTimeApprovalVerifier({"approved"}), clock=lambda: NOW
    )
    coordinator = ComputerUseCoordinator(
        broker,
        clock=lambda: NOW,
    )
    plan = ComputerPlan(
        conversation_id="conversation-1",
        objective="AIが最適と判断した操作",
        observation_id="observation-1",
        actions=(
            ComputerActionRequest(
                "type",
                ComputerActionType.TYPE_PLAIN_TEXT,
                "windows_settings",
                text="delete everything",
            ),
        ),
    )
    approval = ComputerPlanApproval(
        "approved", coordinator.approval_hash(plan), NOW
    )

    result = await coordinator.execute(plan, safe_context(), approval)

    assert result.state is ComputerUseRunState.DENIED
    assert result.failure_reason == "fixed_action_sequence_required"
    assert broker.calls == []


@pytest.mark.asyncio
async def test_matching_approval_executes_only_the_three_fixed_fake_actions() -> None:
    broker = FakeDesktopActionBroker(
        OneTimeApprovalVerifier({"approved"}), clock=lambda: NOW
    )
    coordinator = ComputerUseCoordinator(
        broker,
        clock=lambda: NOW,
    )
    plan = fixed_plan()
    approval = ComputerPlanApproval(
        "approved", coordinator.approval_hash(plan), NOW
    )

    result = await coordinator.execute(plan, safe_context(), approval)

    assert result.state is ComputerUseRunState.COMPLETED
    assert [call.action_type for call in broker.calls] == [
        ComputerActionType.LAUNCH_ALLOWED_APP,
        ComputerActionType.CLICK_UIA_ELEMENT,
        ComputerActionType.TYPE_PLAIN_TEXT,
    ]


@pytest.mark.asyncio
async def test_direct_broker_call_still_requires_matching_approval() -> None:
    broker = FakeDesktopActionBroker(
        OneTimeApprovalVerifier(set()), clock=lambda: NOW
    )

    with pytest.raises(ComputerActionDenied, match="plan_approval_required"):
        await broker.execute_plan(fixed_plan(), safe_context(), None)

    assert broker.calls == []


@pytest.mark.asyncio
async def test_broker_consumes_approval_only_once() -> None:
    broker = FakeDesktopActionBroker(
        OneTimeApprovalVerifier({"approved"}), clock=lambda: NOW
    )
    coordinator = ComputerUseCoordinator(broker, clock=lambda: NOW)
    plan = fixed_plan()
    approval = ComputerPlanApproval(
        "approved", coordinator.approval_hash(plan), NOW
    )

    first = await coordinator.execute(plan, safe_context(), approval)
    second = await coordinator.execute(plan, safe_context(), approval)

    assert first.state is ComputerUseRunState.COMPLETED
    assert second.state is ComputerUseRunState.DENIED
    assert second.failure_reason == "plan_approval_required"
    assert len(broker.calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "approved_at",
    [NOW - timedelta(seconds=31), NOW + timedelta(seconds=1), datetime(2026, 7, 18)],
)
async def test_stale_future_or_naive_approval_never_reaches_broker(
    approved_at: datetime,
) -> None:
    broker = FakeDesktopActionBroker(
        OneTimeApprovalVerifier({"approved"}), clock=lambda: NOW
    )
    coordinator = ComputerUseCoordinator(broker, clock=lambda: NOW)
    plan = fixed_plan()
    approval = ComputerPlanApproval(
        "approved", coordinator.approval_hash(plan), approved_at
    )

    result = await coordinator.execute(plan, safe_context(), approval)

    assert result.state is ComputerUseRunState.DENIED
    assert result.failure_reason == "plan_approval_required"
    assert broker.calls == []


@pytest.mark.asyncio
async def test_plan_change_invalidates_existing_approval() -> None:
    broker = FakeDesktopActionBroker(
        OneTimeApprovalVerifier({"approved"}), clock=lambda: NOW
    )
    coordinator = ComputerUseCoordinator(broker, clock=lambda: NOW)
    original = fixed_plan()
    approval = ComputerPlanApproval(
        "approved", coordinator.approval_hash(original), NOW
    )
    changed = replace(
        original,
        actions=(
            original.actions[0],
            original.actions[1],
            replace(original.actions[2], text="変更された入力"),
        ),
    )

    result = await coordinator.execute(changed, safe_context(), approval)

    assert result.state is ComputerUseRunState.DENIED
    assert result.failure_reason == "plan_approval_required"
    assert broker.calls == []


class BlockingApprovalVerifier:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def verify_and_consume(
        self, _approval: ComputerPlanApproval, _plan_hash: str
    ) -> bool:
        self.entered.set()
        await self.release.wait()
        return True


@pytest.mark.asyncio
async def test_concurrent_run_is_denied_and_cancel_requests_stop() -> None:
    verifier = BlockingApprovalVerifier()
    broker = FakeDesktopActionBroker(verifier, clock=lambda: NOW)
    coordinator = ComputerUseCoordinator(broker, clock=lambda: NOW)
    plan = fixed_plan()
    approval = ComputerPlanApproval(
        "approved", coordinator.approval_hash(plan), NOW
    )
    running = asyncio.create_task(
        coordinator.execute(plan, safe_context(), approval)
    )
    await verifier.entered.wait()

    concurrent = await coordinator.execute(plan, safe_context(), approval)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert concurrent.state is ComputerUseRunState.DENIED
    assert concurrent.failure_reason == "computer_use_busy"
    assert broker.stop_calls == 1
    assert broker.calls == []


class MisleadingFailure(RuntimeError):
    reason = "looks_like_a_denial"


class PartiallyFailingBroker:
    def __init__(self, *, fail_stop: bool = False) -> None:
        self.stop_calls = 0
        self.fail_stop = fail_stop

    async def execute_plan(
        self,
        _plan: ComputerPlan,
        _context: DesktopSecurityContext,
        _approval: ComputerPlanApproval | None,
    ) -> tuple[ComputerActionExecution, ...]:
        raise MisleadingFailure("partial failure")

    async def stop(self) -> None:
        self.stop_calls += 1
        if self.fail_stop:
            raise RuntimeError("stop failed")


@pytest.mark.asyncio
async def test_arbitrary_reason_attribute_cannot_bypass_failure_stop() -> None:
    broker = PartiallyFailingBroker()
    coordinator = ComputerUseCoordinator(broker, clock=lambda: NOW)
    plan = fixed_plan()
    approval = ComputerPlanApproval(
        "approved", coordinator.approval_hash(plan), NOW
    )

    result = await coordinator.execute(plan, safe_context(), approval)

    assert result.state is ComputerUseRunState.FAILED
    assert result.failure_reason == "provider_failed:MisleadingFailure"
    assert broker.stop_calls == 1


@pytest.mark.asyncio
async def test_stop_failure_is_not_hidden() -> None:
    broker = PartiallyFailingBroker(fail_stop=True)
    coordinator = ComputerUseCoordinator(broker, clock=lambda: NOW)
    plan = fixed_plan()
    approval = ComputerPlanApproval(
        "approved", coordinator.approval_hash(plan), NOW
    )

    result = await coordinator.execute(plan, safe_context(), approval)

    assert result.state is ComputerUseRunState.FAILED
    assert result.failure_reason == (
        "provider_failed:MisleadingFailure;stop_failed"
    )
    assert broker.stop_calls == 1
