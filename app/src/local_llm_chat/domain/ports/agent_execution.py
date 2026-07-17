from __future__ import annotations

from datetime import datetime
from typing import Protocol

from local_llm_chat.domain.models import (
    AgentActionApproval,
    AgentRun,
    AgentStep,
    AgentStepRequest,
    AgentToolDescriptor,
    AgentToolExecution,
)
from local_llm_chat.domain.states import CostClass, DataClassification, Locality


class AgentToolProvider(Protocol):
    @property
    def descriptor(self) -> AgentToolDescriptor: ...

    def classify(self, arguments: dict[str, object]) -> DataClassification: ...

    async def execute(self, arguments: dict[str, object]) -> AgentToolExecution: ...

    async def restore(self, restore_token: str) -> None: ...


class AgentApprovalVerifier(Protocol):
    async def verify_and_consume(
        self, approval: AgentActionApproval, action_hash: str
    ) -> bool: ...


class AgentExecutionRepository(Protocol):
    async def create_agent_run(self, run: AgentRun) -> AgentRun: ...

    async def get_agent_run(self, run_id: str) -> AgentRun: ...

    async def create_agent_step(self, step: AgentStep) -> AgentStep: ...

    async def finish_agent_step(self, step: AgentStep) -> AgentStep: ...

    async def finish_agent_run(self, run: AgentRun) -> AgentRun: ...

    async def list_agent_steps(self, run_id: str) -> list[AgentStep]: ...

    async def recover_interrupted_agent_runs(self, now: datetime) -> int: ...


class AgentRequestSource(Protocol):
    @property
    def locality(self) -> Locality: ...

    @property
    def cost_class(self) -> CostClass: ...

    async def next_step(
        self,
        run: AgentRun,
        previous_result: AgentToolExecution | None,
    ) -> AgentStepRequest | None: ...
