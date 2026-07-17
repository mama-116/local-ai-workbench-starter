from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from local_llm_chat.application.services.agent_execution_service import (
    AgentExecutionCoordinator,
)
from local_llm_chat.domain.errors import AgentToolFailure
from local_llm_chat.domain.models import (
    AgentActionApproval,
    AgentRun,
    AgentStep,
    AgentStepRequest,
    AgentToolDescriptor,
    AgentToolExecution,
)
from local_llm_chat.domain.states import (
    AgentRunState,
    AgentStepState,
    AgentToolEffect,
    CostClass,
    DataClassification,
    Locality,
)


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def free_descriptor(
    name: str,
    effect: AgentToolEffect,
    destination: Locality,
    cost_units: int = 1,
) -> AgentToolDescriptor:
    return AgentToolDescriptor(
        name,
        effect,
        destination,
        cost_units=cost_units,
        cost_class=CostClass.NO_CHARGE,
    )


class MemoryAgentRepository:
    def __init__(self) -> None:
        self.runs: dict[str, AgentRun] = {}
        self.steps: dict[str, AgentStep] = {}

    async def create_agent_run(self, run: AgentRun) -> AgentRun:
        self.runs[run.id] = run
        return run

    async def get_agent_run(self, run_id: str) -> AgentRun:
        return self.runs[run_id]

    async def create_agent_step(self, step: AgentStep) -> AgentStep:
        self.steps[step.id] = step
        return step

    async def finish_agent_step(self, step: AgentStep) -> AgentStep:
        self.steps[step.id] = step
        return step

    async def finish_agent_run(self, run: AgentRun) -> AgentRun:
        self.runs[run.id] = run
        return run

    async def list_agent_steps(self, run_id: str) -> list[AgentStep]:
        return sorted(
            (step for step in self.steps.values() if step.run_id == run_id),
            key=lambda step: step.ordinal,
        )

    async def recover_interrupted_agent_runs(
        self, now: datetime
    ) -> int:
        recovered = 0
        for run_id, run in tuple(self.runs.items()):
            if run.state is AgentRunState.RUNNING:
                self.runs[run_id] = replace(
                    run,
                    state=AgentRunState.FAILED,
                    failure_reason="previous_session_interrupted",
                    completed_at=now,
                )
                recovered += 1
        return recovered


class ListRequestSource:
    def __init__(
        self,
        requests: list[AgentStepRequest],
        *,
        locality: Locality = Locality.LOCAL,
        cost_class: CostClass = CostClass.NO_CHARGE,
    ) -> None:
        self.requests = list(requests)
        self._locality = locality
        self._cost_class = cost_class

    @property
    def locality(self) -> Locality:
        return self._locality

    @property
    def cost_class(self) -> CostClass:
        return self._cost_class

    async def next_step(
        self,
        _run: AgentRun,
        _previous_result: AgentToolExecution | None,
    ) -> AgentStepRequest | None:
        return self.requests.pop(0) if self.requests else None


class AdaptiveRequestSource:
    def __init__(self) -> None:
        self.observations: list[AgentToolExecution | None] = []

    @property
    def locality(self) -> Locality:
        return Locality.LOCAL

    @property
    def cost_class(self) -> CostClass:
        return CostClass.NO_CHARGE

    async def next_step(
        self,
        _run: AgentRun,
        previous_result: AgentToolExecution | None,
    ) -> AgentStepRequest | None:
        self.observations.append(previous_result)
        if previous_result is None:
            return request("adaptive-1")
        assert previous_result.result_content == "ok"
        return None


class FakeAgentTool:
    def __init__(
        self,
        descriptor: AgentToolDescriptor,
        classification: DataClassification = DataClassification.PUBLIC_RESULT,
        *,
        fail_on_call: int | None = None,
        fail_with_restore_token_on_call: int | None = None,
        fail_restore: bool = False,
    ) -> None:
        self._descriptor = (
            replace(descriptor, cost_class=CostClass.NO_CHARGE)
            if descriptor.cost_class is CostClass.UNKNOWN
            else descriptor
        )
        self._classification = classification
        self.fail_on_call = fail_on_call
        self.fail_with_restore_token_on_call = fail_with_restore_token_on_call
        self.fail_restore = fail_restore
        self.calls: list[dict[str, object]] = []
        self.restored: list[str] = []

    @property
    def descriptor(self) -> AgentToolDescriptor:
        return self._descriptor

    def classify(self, _arguments: dict[str, object]) -> DataClassification:
        return self._classification

    async def execute(self, arguments: dict[str, object]) -> AgentToolExecution:
        self.calls.append(arguments)
        if self.fail_with_restore_token_on_call == len(self.calls):
            raise AgentToolFailure(
                "planned partial failure", f"restore-{len(self.calls)}"
            )
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("planned failure")
        token = (
            f"restore-{len(self.calls)}"
            if self.descriptor.effect is AgentToolEffect.REVERSIBLE_WRITE
            else None
        )
        return AgentToolExecution("ok", token)

    async def restore(self, restore_token: str) -> None:
        self.restored.append(restore_token)
        if self.fail_restore:
            raise RuntimeError("planned restore failure")

    def replace_descriptor(self, descriptor: AgentToolDescriptor) -> None:
        self._descriptor = (
            replace(descriptor, cost_class=CostClass.NO_CHARGE)
            if descriptor.cost_class is CostClass.UNKNOWN
            else descriptor
        )


class OneTimeApprovalVerifier:
    def __init__(self, valid_ids: set[str]) -> None:
        self.valid_ids = valid_ids
        self.consumed: set[str] = set()

    async def verify_and_consume(
        self, approval: AgentActionApproval, action_hash: str
    ) -> bool:
        if (
            approval.id not in self.valid_ids
            or approval.id in self.consumed
            or approval.action_hash != action_hash
        ):
            return False
        self.consumed.add(approval.id)
        return True


class AlwaysApproveVerifier:
    async def verify_and_consume(
        self, _approval: AgentActionApproval, _action_hash: str
    ) -> bool:
        return True


def request(identifier: str, tool_name: str = "local_read") -> AgentStepRequest:
    return AgentStepRequest(identifier, tool_name, {"value": identifier})


@pytest.mark.asyncio
async def test_sixth_requested_step_is_audited_but_never_executed() -> None:
    repository = MemoryAgentRepository()
    provider = FakeAgentTool(
        AgentToolDescriptor("local_read", AgentToolEffect.READ, Locality.LOCAL)
    )
    coordinator = AgentExecutionCoordinator(
        repository,
        (provider,),
        OneTimeApprovalVerifier(set()),
        clock=lambda: NOW,
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="bounded read",
        allowed_tools=("local_read",),
        source=ListRequestSource([request(str(index)) for index in range(6)]),
    )

    assert run.state is AgentRunState.DENIED
    assert run.failure_reason == "max_steps_exceeded"
    assert len(provider.calls) == 5
    steps = await repository.list_agent_steps(run.id)
    assert len(steps) == 6
    assert steps[-1].state is AgentStepState.DENIED
    assert steps[-1].failure_reason == "max_steps_exceeded"


@pytest.mark.asyncio
async def test_private_external_send_is_denied_before_provider_call() -> None:
    repository = MemoryAgentRepository()
    provider = FakeAgentTool(
        AgentToolDescriptor(
            "post", AgentToolEffect.EXTERNAL_POST, Locality.LAN
        ),
        DataClassification.PRIVATE,
    )
    coordinator = AgentExecutionCoordinator(
        repository,
        (provider,),
        OneTimeApprovalVerifier({"approved"}),
        clock=lambda: NOW,
    )
    proposed = request("post-1", "post")
    approval = AgentActionApproval(
        "approved",
        coordinator.approval_hash(
            proposed, "conversation-1", "must stay local", ("post",)
        ),
        NOW,
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="must stay local",
        allowed_tools=("post",),
        source=ListRequestSource([proposed]),
        approvals={proposed.id: approval},
    )

    assert run.state is AgentRunState.DENIED
    assert run.failure_reason == "private_external_send_denied"
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("locality", "cost_class"),
    [
        (Locality.REMOTE, CostClass.NO_CHARGE),
        (Locality.LOCAL, CostClass.PAID),
    ],
)
async def test_non_local_or_costly_request_source_is_never_called(
    locality: Locality,
    cost_class: CostClass,
) -> None:
    repository = MemoryAgentRepository()
    provider = FakeAgentTool(
        AgentToolDescriptor("local_read", AgentToolEffect.READ, Locality.LOCAL)
    )
    source = ListRequestSource(
        [request("never")], locality=locality, cost_class=cost_class
    )
    coordinator = AgentExecutionCoordinator(
        repository,
        (provider,),
        OneTimeApprovalVerifier(set()),
        clock=lambda: NOW,
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="must use local source",
        allowed_tools=("local_read",),
        source=source,
    )

    assert run.state is AgentRunState.DENIED
    assert run.failure_reason == "request_source_not_local_no_charge"
    assert len(source.requests) == 1
    assert provider.calls == []


@pytest.mark.asyncio
async def test_external_actions_each_need_a_fresh_matching_approval() -> None:
    repository = MemoryAgentRepository()
    provider = FakeAgentTool(
        AgentToolDescriptor(
            "post", AgentToolEffect.EXTERNAL_POST, Locality.REMOTE
        )
    )
    verifier = AlwaysApproveVerifier()
    coordinator = AgentExecutionCoordinator(
        repository, (provider,), verifier, clock=lambda: NOW
    )
    first = AgentStepRequest("first", "post", {"value": "same"})
    second = AgentStepRequest("second", "post", {"value": "same"})
    reused = AgentActionApproval(
        "only-once",
        coordinator.approval_hash(
            first, "conversation-1", "two posts", ("post",)
        ),
        NOW,
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="two posts",
        allowed_tools=("post",),
        source=ListRequestSource([first, second]),
        approvals={first.id: reused, second.id: reused},
    )

    assert len(provider.calls) == 1
    assert run.state is AgentRunState.DENIED
    assert run.failure_reason == "approval_required"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "approved_at",
    [NOW - timedelta(seconds=31), NOW + timedelta(seconds=1), datetime(2026, 7, 17)],
)
async def test_stale_future_or_naive_approval_is_rejected(
    approved_at: datetime,
) -> None:
    repository = MemoryAgentRepository()
    provider = FakeAgentTool(
        AgentToolDescriptor(
            "post", AgentToolEffect.EXTERNAL_POST, Locality.REMOTE
        )
    )
    coordinator = AgentExecutionCoordinator(
        repository,
        (provider,),
        OneTimeApprovalVerifier({"approved"}),
        clock=lambda: NOW,
    )
    proposed = request("post-1", "post")
    approval = AgentActionApproval(
        "approved",
        coordinator.approval_hash(
            proposed, "conversation-1", "expiry test", ("post",)
        ),
        approved_at,
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="expiry test",
        allowed_tools=("post",),
        source=ListRequestSource([proposed]),
        approvals={proposed.id: approval},
    )

    assert run.state is AgentRunState.DENIED
    assert run.failure_reason == "approval_required"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_approval_for_another_conversation_is_rejected() -> None:
    repository = MemoryAgentRepository()
    provider = FakeAgentTool(
        AgentToolDescriptor(
            "post", AgentToolEffect.EXTERNAL_POST, Locality.REMOTE
        )
    )
    coordinator = AgentExecutionCoordinator(
        repository,
        (provider,),
        OneTimeApprovalVerifier({"approved"}),
        clock=lambda: NOW,
    )
    proposed = request("post-1", "post")
    approval = AgentActionApproval(
        "approved",
        coordinator.approval_hash(
            proposed, "conversation-2", "same objective", ("post",)
        ),
        NOW,
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="same objective",
        allowed_tools=("post",),
        source=ListRequestSource([proposed]),
        approvals={proposed.id: approval},
    )

    assert run.state is AgentRunState.DENIED
    assert run.failure_reason == "approval_required"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_provider_definition_change_invalidates_existing_approval() -> None:
    repository = MemoryAgentRepository()
    provider = FakeAgentTool(
        AgentToolDescriptor(
            "post", AgentToolEffect.EXTERNAL_POST, Locality.REMOTE
        )
    )
    coordinator = AgentExecutionCoordinator(
        repository,
        (provider,),
        OneTimeApprovalVerifier({"approved"}),
        clock=lambda: NOW,
    )
    proposed = request("post-1", "post")
    approval = AgentActionApproval(
        "approved",
        coordinator.approval_hash(
            proposed, "conversation-1", "changed definition", ("post",)
        ),
        NOW,
    )
    provider.replace_descriptor(
        AgentToolDescriptor(
            "post",
            AgentToolEffect.EXTERNAL_DELETE,
            Locality.REMOTE,
            cost_units=2,
        )
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="changed definition",
        allowed_tools=("post",),
        source=ListRequestSource([proposed]),
        approvals={proposed.id: approval},
    )

    assert run.state is AgentRunState.DENIED
    assert run.failure_reason == "approval_required"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_later_failure_restores_completed_reversible_steps_in_reverse() -> None:
    repository = MemoryAgentRepository()
    writer = FakeAgentTool(
        AgentToolDescriptor(
            "write", AgentToolEffect.REVERSIBLE_WRITE, Locality.LOCAL
        )
    )
    failing = FakeAgentTool(
        AgentToolDescriptor("read", AgentToolEffect.READ, Locality.LOCAL),
        fail_on_call=1,
    )
    first = request("write-1", "write")
    second = request("write-2", "write")
    coordinator = AgentExecutionCoordinator(
        repository,
        (writer, failing),
        OneTimeApprovalVerifier({"write-approved-1", "write-approved-2"}),
        clock=lambda: NOW,
    )
    allowed_tools = ("write", "read")
    first_approval = AgentActionApproval(
        "write-approved-1",
        coordinator.approval_hash(
            first, "conversation-1", "restore on failure", allowed_tools
        ),
        NOW,
    )
    second_approval = AgentActionApproval(
        "write-approved-2",
        coordinator.approval_hash(
            second, "conversation-1", "restore on failure", allowed_tools
        ),
        NOW,
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="restore on failure",
        allowed_tools=allowed_tools,
        source=ListRequestSource(
            [first, second, request("read-1", "read")]
        ),
        approvals={first.id: first_approval, second.id: second_approval},
    )

    assert run.state is AgentRunState.FAILED
    assert writer.restored == ["restore-2", "restore-1"]
    steps = await repository.list_agent_steps(run.id)
    assert steps[0].state is AgentStepState.RESTORED
    assert steps[1].state is AgentStepState.RESTORED
    assert steps[2].state is AgentStepState.FAILED


@pytest.mark.asyncio
async def test_partial_write_failure_with_journal_token_is_restored() -> None:
    repository = MemoryAgentRepository()
    writer = FakeAgentTool(
        AgentToolDescriptor(
            "write", AgentToolEffect.REVERSIBLE_WRITE, Locality.LOCAL
        ),
        fail_with_restore_token_on_call=1,
    )
    proposed = request("write-1", "write")
    coordinator = AgentExecutionCoordinator(
        repository,
        (writer,),
        OneTimeApprovalVerifier({"approved"}),
        clock=lambda: NOW,
    )
    approval = AgentActionApproval(
        "approved",
        coordinator.approval_hash(
            proposed, "conversation-1", "partial write", ("write",)
        ),
        NOW,
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="partial write",
        allowed_tools=("write",),
        source=ListRequestSource([proposed]),
        approvals={proposed.id: approval},
    )

    assert run.state is AgentRunState.FAILED
    assert writer.restored == ["restore-1"]
    steps = await repository.list_agent_steps(run.id)
    assert steps[0].state is AgentStepState.RESTORED


@pytest.mark.asyncio
async def test_unjournaled_write_failure_is_marked_restore_failed() -> None:
    repository = MemoryAgentRepository()
    writer = FakeAgentTool(
        AgentToolDescriptor(
            "write", AgentToolEffect.REVERSIBLE_WRITE, Locality.LOCAL
        ),
        fail_on_call=1,
    )
    proposed = request("write-1", "write")
    coordinator = AgentExecutionCoordinator(
        repository,
        (writer,),
        OneTimeApprovalVerifier({"approved"}),
        clock=lambda: NOW,
    )
    approval = AgentActionApproval(
        "approved",
        coordinator.approval_hash(
            proposed, "conversation-1", "unjournaled write", ("write",)
        ),
        NOW,
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="unjournaled write",
        allowed_tools=("write",),
        source=ListRequestSource([proposed]),
        approvals={proposed.id: approval},
    )

    assert run.state is AgentRunState.FAILED
    assert run.failure_reason is not None
    assert run.failure_reason.endswith(";restore_failed")
    steps = await repository.list_agent_steps(run.id)
    assert steps[0].state is AgentStepState.RESTORE_FAILED


@pytest.mark.asyncio
async def test_cost_budget_stops_before_the_next_provider_call() -> None:
    repository = MemoryAgentRepository()
    provider = FakeAgentTool(
        AgentToolDescriptor(
            "costly", AgentToolEffect.READ, Locality.LOCAL, cost_units=2
        )
    )
    coordinator = AgentExecutionCoordinator(
        repository,
        (provider,),
        OneTimeApprovalVerifier(set()),
        clock=lambda: NOW,
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="bounded cost",
        allowed_tools=("costly",),
        source=ListRequestSource(
            [request("1", "costly"), request("2", "costly"), request("3", "costly")]
        ),
    )

    assert run.state is AgentRunState.DENIED
    assert run.failure_reason == "cost_budget_exceeded"
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_duration_boundary_stops_before_request_or_provider() -> None:
    repository = MemoryAgentRepository()
    provider = FakeAgentTool(
        AgentToolDescriptor("local_read", AgentToolEffect.READ, Locality.LOCAL)
    )
    elapsed_values = iter((0.0, 60.0))
    source = ListRequestSource([request("never")])
    coordinator = AgentExecutionCoordinator(
        repository,
        (provider,),
        OneTimeApprovalVerifier(set()),
        clock=lambda: NOW,
        monotonic=lambda: next(elapsed_values),
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="bounded duration",
        allowed_tools=("local_read",),
        source=source,
    )

    assert run.state is AgentRunState.FAILED
    assert run.failure_reason == "duration_exceeded"
    assert provider.calls == []
    assert len(source.requests) == 1


@pytest.mark.asyncio
async def test_source_returning_no_step_after_deadline_cannot_complete_run() -> None:
    repository = MemoryAgentRepository()
    provider = FakeAgentTool(
        AgentToolDescriptor("local_read", AgentToolEffect.READ, Locality.LOCAL)
    )
    elapsed_values = iter((0.0, 0.0, 60.0))
    coordinator = AgentExecutionCoordinator(
        repository,
        (provider,),
        OneTimeApprovalVerifier(set()),
        clock=lambda: NOW,
        monotonic=lambda: next(elapsed_values),
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="late completion",
        allowed_tools=("local_read",),
        source=ListRequestSource([]),
    )

    assert run.state is AgentRunState.FAILED
    assert run.failure_reason == "duration_exceeded"


@pytest.mark.asyncio
async def test_late_provider_result_cannot_complete_run_after_deadline() -> None:
    repository = MemoryAgentRepository()
    provider = FakeAgentTool(
        AgentToolDescriptor("local_read", AgentToolEffect.READ, Locality.LOCAL)
    )
    elapsed_values = iter((0.0, 0.0, 0.0, 0.0, 0.0, 60.0))
    coordinator = AgentExecutionCoordinator(
        repository,
        (provider,),
        OneTimeApprovalVerifier(set()),
        clock=lambda: NOW,
        monotonic=lambda: next(elapsed_values),
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="late result",
        allowed_tools=("local_read",),
        source=ListRequestSource([request("late")]),
    )

    assert run.state is AgentRunState.FAILED
    assert run.failure_reason == "duration_exceeded"
    assert len(provider.calls) == 1
    steps = await repository.list_agent_steps(run.id)
    assert steps[0].state is AgentStepState.FAILED


@pytest.mark.asyncio
async def test_previous_local_result_is_available_to_next_step_source() -> None:
    repository = MemoryAgentRepository()
    provider = FakeAgentTool(
        AgentToolDescriptor("local_read", AgentToolEffect.READ, Locality.LOCAL)
    )
    source = AdaptiveRequestSource()
    coordinator = AgentExecutionCoordinator(
        repository,
        (provider,),
        OneTimeApprovalVerifier(set()),
        clock=lambda: NOW,
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="adaptive local read",
        allowed_tools=("local_read",),
        source=source,
    )

    assert run.state is AgentRunState.COMPLETED
    assert source.observations[0] is None
    assert source.observations[1] == AgentToolExecution("ok")


@pytest.mark.asyncio
async def test_restore_failure_is_audited_and_keeps_run_failed() -> None:
    repository = MemoryAgentRepository()
    writer = FakeAgentTool(
        AgentToolDescriptor(
            "write", AgentToolEffect.REVERSIBLE_WRITE, Locality.LOCAL
        ),
        fail_restore=True,
    )
    failing = FakeAgentTool(
        AgentToolDescriptor("read", AgentToolEffect.READ, Locality.LOCAL),
        fail_on_call=1,
    )
    write = request("write-1", "write")
    coordinator = AgentExecutionCoordinator(
        repository,
        (writer, failing),
        OneTimeApprovalVerifier({"approved"}),
        clock=lambda: NOW,
    )
    allowed_tools = ("write", "read")
    approval = AgentActionApproval(
        "approved",
        coordinator.approval_hash(
            write, "conversation-1", "audit failed restore", allowed_tools
        ),
        NOW,
    )

    run = await coordinator.execute(
        conversation_id="conversation-1",
        objective="audit failed restore",
        allowed_tools=allowed_tools,
        source=ListRequestSource([write, request("read", "read")]),
        approvals={write.id: approval},
    )

    assert run.state is AgentRunState.FAILED
    assert run.failure_reason is not None
    assert run.failure_reason.endswith(";restore_failed")
    steps = await repository.list_agent_steps(run.id)
    assert steps[0].state is AgentStepState.RESTORE_FAILED
    assert steps[0].failure_reason == "restore_failed:RuntimeError"
