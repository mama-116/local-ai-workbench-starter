from __future__ import annotations

from typing import Protocol

from local_llm_chat.domain.models import (
    ComputerActionExecution,
    ComputerPlan,
    ComputerPlanApproval,
    DesktopSecurityContext,
)


class ComputerPlanApprovalVerifier(Protocol):
    async def verify_and_consume(
        self, approval: ComputerPlanApproval, plan_hash: str
    ) -> bool: ...


class DesktopActionBroker(Protocol):
    async def execute_plan(
        self,
        plan: ComputerPlan,
        context: DesktopSecurityContext,
        approval: ComputerPlanApproval | None,
    ) -> tuple[ComputerActionExecution, ...]: ...

    async def stop(self) -> None: ...
