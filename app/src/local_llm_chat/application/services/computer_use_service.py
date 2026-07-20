from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime

from local_llm_chat.domain.errors import ComputerActionDenied
from local_llm_chat.domain.models import (
    ComputerPlan,
    ComputerPlanApproval,
    ComputerUseExecution,
    DesktopSecurityContext,
)
from local_llm_chat.domain.policies.computer_use import ComputerUsePolicy
from local_llm_chat.domain.ports.computer_use import (
    DesktopActionBroker,
)
from local_llm_chat.domain.states import (
    ComputerPolicyDecision,
    ComputerUseRunState,
)


def computer_plan_hash(plan: ComputerPlan) -> str:
    payload = json.dumps(
        {
            "conversation_id": plan.conversation_id,
            "objective": plan.objective,
            "observation_id": plan.observation_id,
            "actions": [
                {
                    "id": action.id,
                    "action_type": action.action_type.value,
                    "target_profile_id": action.target_profile_id,
                    "text": action.text,
                }
                for action in plan.actions
            ],
            "limits": {
                "max_actions": plan.limits.max_actions,
                "max_duration_seconds": plan.limits.max_duration_seconds,
                "approval_timeout_seconds": plan.limits.approval_timeout_seconds,
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ComputerUseCoordinator:
    def __init__(
        self,
        provider: DesktopActionBroker,
        *,
        policy: ComputerUsePolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._provider = provider
        self._policy = policy or ComputerUsePolicy()
        self._clock = clock or (lambda: datetime.now(UTC))

    @staticmethod
    def approval_hash(plan: ComputerPlan) -> str:
        return computer_plan_hash(plan)

    async def execute(
        self,
        plan: ComputerPlan,
        context: DesktopSecurityContext,
        approval: ComputerPlanApproval | None = None,
    ) -> ComputerUseExecution:
        policy_result = self._policy.evaluate_plan(plan, context)
        if policy_result.decision is ComputerPolicyDecision.DENY:
            return ComputerUseExecution(
                ComputerUseRunState.DENIED, policy_result.reason
            )
        plan_hash = computer_plan_hash(plan)
        if (
            approval is None
            or not approval.id
            or approval.plan_hash != plan_hash
            or not self._approval_is_fresh(plan, approval)
        ):
            return ComputerUseExecution(
                ComputerUseRunState.DENIED, "plan_approval_required"
            )

        try:
            async with asyncio.timeout(plan.limits.max_duration_seconds):
                executions = await self._provider.execute_plan(
                    plan, context, approval
                )
        except TimeoutError:
            stop_failed = await self._safe_stop()
            return ComputerUseExecution(
                ComputerUseRunState.FAILED,
                (
                    "duration_exceeded;stop_failed"
                    if stop_failed
                    else "duration_exceeded"
                ),
            )
        except asyncio.CancelledError:
            await self._safe_stop()
            raise
        except ComputerActionDenied as error:
            return ComputerUseExecution(
                ComputerUseRunState.DENIED, error.reason
            )
        except Exception as error:
            stop_failed = await self._safe_stop()
            failure_reason = f"provider_failed:{type(error).__name__}"
            if stop_failed:
                failure_reason += ";stop_failed"
            return ComputerUseExecution(
                ComputerUseRunState.FAILED,
                failure_reason,
            )
        return ComputerUseExecution(
            ComputerUseRunState.COMPLETED,
            None,
            tuple(execution.action_id for execution in executions),
        )

    async def _safe_stop(self) -> bool:
        try:
            await self._provider.stop()
        except Exception:
            return True
        return False

    def _approval_is_fresh(
        self, plan: ComputerPlan, approval: ComputerPlanApproval
    ) -> bool:
        approved_at = approval.approved_at
        if approved_at.tzinfo is None or approved_at.utcoffset() is None:
            return False
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        age_seconds = (
            now.astimezone(UTC) - approved_at.astimezone(UTC)
        ).total_seconds()
        return 0 <= age_seconds <= plan.limits.approval_timeout_seconds
