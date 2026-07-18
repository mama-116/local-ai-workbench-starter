from __future__ import annotations

from datetime import datetime
from typing import Protocol

from local_llm_chat.domain.models import (
    ComputerActionAudit,
    ComputerActionExecution,
    ComputerPlan,
    ComputerPlanApproval,
    ComputerUseRun,
    DesktopSecurityContext,
)


class ComputerPlanApprovalVerifier(Protocol):
    async def verify_and_consume(
        self,
        approval: ComputerPlanApproval,
        plan_hash: str,
        verified_at: datetime,
    ) -> bool: ...


class DesktopActionBroker(Protocol):
    async def execute_plan(
        self,
        plan: ComputerPlan,
        context: DesktopSecurityContext,
        approval: ComputerPlanApproval | None,
    ) -> tuple[ComputerActionExecution, ...]: ...

    async def stop(self) -> None: ...


class ComputerUseRepository(ComputerPlanApprovalVerifier, Protocol):
    async def create_computer_use_run(
        self, run: ComputerUseRun
    ) -> ComputerUseRun: ...

    async def get_computer_use_run(self, run_id: str) -> ComputerUseRun: ...

    async def list_computer_use_runs(
        self, conversation_id: str, limit: int = 50
    ) -> list[ComputerUseRun]: ...

    async def create_computer_action(
        self, action: ComputerActionAudit
    ) -> ComputerActionAudit: ...

    async def list_computer_actions(
        self, run_id: str
    ) -> list[ComputerActionAudit]: ...

    async def update_computer_action(
        self, action: ComputerActionAudit
    ) -> ComputerActionAudit: ...

    async def finish_computer_use_run(
        self, run: ComputerUseRun
    ) -> ComputerUseRun: ...

    async def create_computer_plan_approval(
        self, run_id: str, approval: ComputerPlanApproval
    ) -> None: ...
