from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from local_llm_chat.application.services.computer_use_service import (
    ComputerUseCoordinator,
    computer_plan_hash,
)
from local_llm_chat.domain.errors import ValidationError
from local_llm_chat.domain.models import (
    ComputerActionAudit,
    ComputerActionRequest,
    ComputerPlan,
    ComputerPlanApproval,
    ComputerUseRun,
    DesktopSecurityContext,
)
from local_llm_chat.domain.ports.computer_use import ComputerUseRepository
from local_llm_chat.domain.states import (
    ComputerActionState,
    ComputerActionType,
    ComputerUseRunState,
)


class ComputerUseAccessService:
    """Connects focused review, one-time approval, fake execution, and audit."""

    def __init__(
        self,
        repository: ComputerUseRepository,
        coordinator: ComputerUseCoordinator,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._coordinator = coordinator
        self._clock = clock or (lambda: datetime.now(UTC))

    async def stage_fake_notepad_plan(
        self, conversation_id: str, text: str
    ) -> tuple[ComputerPlan, ComputerUseRun]:
        plan = ComputerPlan(
            conversation_id,
            "Fake Brokerでメモ帳計画を検査する（OS入力なし）",
            f"fake-observation-{uuid4()}",
            (
                ComputerActionRequest(
                    f"fake-launch-{uuid4()}",
                    ComputerActionType.LAUNCH_ALLOWED_APP,
                    "windows_notepad",
                ),
                ComputerActionRequest(
                    f"fake-click-{uuid4()}",
                    ComputerActionType.CLICK_UIA_ELEMENT,
                    "notepad_edit",
                ),
                ComputerActionRequest(
                    f"fake-type-{uuid4()}",
                    ComputerActionType.TYPE_PLAIN_TEXT,
                    "notepad_edit",
                    text,
                ),
            ),
        )
        now = self._aware_now()
        run = await self._repository.create_computer_use_run(
            ComputerUseRun(
                str(uuid4()),
                conversation_id,
                plan.objective,
                plan.observation_id,
                computer_plan_hash(plan),
                plan.limits,
                len(plan.actions),
                ComputerUseRunState.AWAITING_APPROVAL,
                None,
                now,
            )
        )
        for ordinal, request in enumerate(plan.actions, start=1):
            await self._repository.create_computer_action(
                ComputerActionAudit(
                    str(uuid4()),
                    run.id,
                    ordinal,
                    request,
                    ComputerActionState.PROPOSED,
                    None,
                    now,
                )
            )
        return plan, run

    async def approve_and_execute_fake(
        self,
        run_id: str,
        plan: ComputerPlan,
        context: DesktopSecurityContext,
    ) -> ComputerUseRun:
        run = await self._repository.get_computer_use_run(run_id)
        plan_hash = computer_plan_hash(plan)
        if (
            run.state is not ComputerUseRunState.AWAITING_APPROVAL
            or run.plan_hash != plan_hash
            or run.conversation_id != plan.conversation_id
        ):
            raise ValidationError("承認対象のFake計画が一致しません。")
        now = self._aware_now()
        if self._remaining_seconds(run, now) <= 0:
            await self._cancel_with_reason(run, "approval_expired", now)
            raise ValidationError("承認期限が切れました。")
        approval = ComputerPlanApproval(str(uuid4()), plan_hash, now)
        await self._repository.create_computer_plan_approval(run.id, approval)
        execution = await self._coordinator.execute(plan, context, approval)
        now = self._aware_now()
        if execution.state is ComputerUseRunState.COMPLETED:
            completed = set(execution.completed_action_ids)
            actions = await self._repository.list_computer_actions(run.id)
            if completed != {action.request.id for action in actions}:
                execution = replace(
                    execution,
                    state=ComputerUseRunState.FAILED,
                    failure_reason="fake_action_audit_mismatch",
                )
            else:
                for action in actions:
                    approved = await self._repository.update_computer_action(
                        replace(action, state=ComputerActionState.APPROVED)
                    )
                    running = await self._repository.update_computer_action(
                        replace(
                            approved,
                            state=ComputerActionState.RUNNING,
                            started_at=now,
                        )
                    )
                    await self._repository.update_computer_action(
                        replace(
                            running,
                            state=ComputerActionState.COMPLETED,
                            completed_at=now,
                        )
                    )
                current = await self._repository.get_computer_use_run(run.id)
                return await self._repository.finish_computer_use_run(
                    replace(
                        current,
                        state=ComputerUseRunState.COMPLETED,
                        completed_at=now,
                    )
                )

        terminal_state = (
            ComputerActionState.DENIED
            if execution.state is ComputerUseRunState.DENIED
            else ComputerActionState.FAILED
        )
        await self._finish_actions(
            run.id, terminal_state, execution.failure_reason, now
        )
        current = await self._repository.get_computer_use_run(run.id)
        return await self._repository.finish_computer_use_run(
            replace(
                current,
                state=execution.state,
                failure_reason=execution.failure_reason,
                completed_at=now,
            )
        )

    async def cancel(self, run_id: str) -> ComputerUseRun:
        run = await self._repository.get_computer_use_run(run_id)
        if run.state is not ComputerUseRunState.AWAITING_APPROVAL:
            raise ValidationError("承認待ちのFake計画だけを取り消せます。")
        return await self._cancel_with_reason(
            run, "user_cancelled", self._aware_now()
        )

    async def expire(self, run_id: str) -> ComputerUseRun:
        run = await self._repository.get_computer_use_run(run_id)
        if run.state is not ComputerUseRunState.AWAITING_APPROVAL:
            return run
        now = self._aware_now()
        if self._remaining_seconds(run, now) > 0:
            raise ValidationError("承認期限はまだ切れていません。")
        return await self._cancel_with_reason(run, "approval_expired", now)

    async def approval_remaining_seconds(self, run_id: str) -> int:
        run = await self._repository.get_computer_use_run(run_id)
        if run.state is not ComputerUseRunState.AWAITING_APPROVAL:
            return 0
        return self._remaining_seconds(run, self._aware_now())

    async def _cancel_with_reason(
        self, run: ComputerUseRun, reason: str, now: datetime
    ) -> ComputerUseRun:
        await self._finish_actions(
            run.id, ComputerActionState.CANCELLED, reason, now
        )
        return await self._repository.finish_computer_use_run(
            replace(
                run,
                state=ComputerUseRunState.CANCELLED,
                failure_reason=reason,
                completed_at=now,
            )
        )

    async def history(
        self, conversation_id: str
    ) -> list[ComputerUseRun]:
        return await self._repository.list_computer_use_runs(conversation_id)

    async def actions(self, run_id: str) -> list[ComputerActionAudit]:
        return await self._repository.list_computer_actions(run_id)

    async def _finish_actions(
        self,
        run_id: str,
        state: ComputerActionState,
        reason: str | None,
        now: datetime,
    ) -> None:
        for action in await self._repository.list_computer_actions(run_id):
            if action.state in {
                ComputerActionState.PROPOSED,
                ComputerActionState.APPROVED,
                ComputerActionState.RUNNING,
            }:
                await self._repository.update_computer_action(
                    replace(
                        action,
                        state=state,
                        failure_reason=reason,
                        completed_at=now,
                    )
                )

    def _aware_now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now

    @staticmethod
    def _remaining_seconds(run: ComputerUseRun, now: datetime) -> int:
        age = (now.astimezone(UTC) - run.created_at.astimezone(UTC)).total_seconds()
        if age < 0:
            return 0
        remaining = run.limits.approval_timeout_seconds - age
        return max(0, math.ceil(remaining))
