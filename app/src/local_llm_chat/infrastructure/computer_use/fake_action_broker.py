from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

from local_llm_chat.application.services.computer_use_service import (
    computer_plan_hash,
)
from local_llm_chat.domain.errors import ComputerActionDenied
from local_llm_chat.domain.models import (
    ComputerActionExecution,
    ComputerActionRequest,
    ComputerPlan,
    ComputerPlanApproval,
    DesktopSecurityContext,
)
from local_llm_chat.domain.policies.computer_use import ComputerUsePolicy
from local_llm_chat.domain.ports.computer_use import ComputerPlanApprovalVerifier
from local_llm_chat.domain.states import ComputerPolicyDecision


class FakeDesktopActionBroker:
    """Revalidates complete plans while producing zero operating-system input."""

    def __init__(
        self,
        approval_verifier: ComputerPlanApprovalVerifier,
        *,
        policy: ComputerUsePolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._approval_verifier = approval_verifier
        self._policy = policy or ComputerUsePolicy()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._run_lock = asyncio.Lock()
        self.calls: list[ComputerActionRequest] = []
        self.stop_calls = 0

    async def execute_plan(
        self,
        plan: ComputerPlan,
        context: DesktopSecurityContext,
        approval: ComputerPlanApproval | None,
    ) -> tuple[ComputerActionExecution, ...]:
        if self._run_lock.locked():
            raise ComputerActionDenied("computer_use_busy")
        async with self._run_lock:
            result = self._policy.evaluate_plan(plan, context)
            if result.decision is ComputerPolicyDecision.DENY:
                raise ComputerActionDenied(result.reason or "plan_denied")
            plan_hash = computer_plan_hash(plan)
            verified_at = self._clock()
            if (
                approval is None
                or not approval.id
                or approval.plan_hash != plan_hash
                or not self._approval_is_fresh(plan, approval, verified_at)
            ):
                raise ComputerActionDenied("plan_approval_required")
            try:
                approved = await self._approval_verifier.verify_and_consume(
                    approval, plan_hash, verified_at
                )
            except Exception:
                approved = False
            if not approved:
                raise ComputerActionDenied("plan_approval_required")

            executions: list[ComputerActionExecution] = []
            for action in plan.actions:
                action_result = self._policy.evaluate_action(action, context)
                if action_result.decision is not ComputerPolicyDecision.ALLOW:
                    raise ComputerActionDenied(
                        action_result.reason or "action_denied"
                    )
                self.calls.append(action)
                executions.append(
                    ComputerActionExecution(action.id, "fake_completed")
                )
            return tuple(executions)

    async def stop(self) -> None:
        self.stop_calls += 1

    def _approval_is_fresh(
        self,
        plan: ComputerPlan,
        approval: ComputerPlanApproval,
        now: datetime,
    ) -> bool:
        approved_at = approval.approved_at
        if approved_at.tzinfo is None or approved_at.utcoffset() is None:
            return False
        if now.tzinfo is None or now.utcoffset() is None:
            return False
        age_seconds = (
            now.astimezone(UTC) - approved_at.astimezone(UTC)
        ).total_seconds()
        return 0 <= age_seconds <= plan.limits.approval_timeout_seconds
