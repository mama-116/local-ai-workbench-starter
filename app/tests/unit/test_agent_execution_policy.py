from local_llm_chat.domain.models import (
    AgentExecutionLimits,
    AgentPolicyResult,
    AgentToolDescriptor,
)
from local_llm_chat.domain.policies.agent_execution import AgentExecutionPolicy
from local_llm_chat.domain.states import (
    AgentPolicyDecision,
    AgentToolEffect,
    CostClass,
    DataClassification,
    Locality,
)
import pytest


def free_descriptor(
    *,
    name: str,
    effect: AgentToolEffect,
    destination: Locality,
    cost_units: int = 1,
    cost_class: CostClass = CostClass.NO_CHARGE,
) -> AgentToolDescriptor:
    return AgentToolDescriptor(
        name,
        effect,
        destination,
        cost_units=cost_units,
        cost_class=cost_class,
    )


def test_sixth_step_is_denied_before_provider_execution() -> None:
    result = AgentExecutionPolicy().evaluate(
        descriptor=free_descriptor(
            name="local_read",
            effect=AgentToolEffect.READ,
            destination=Locality.LOCAL,
        ),
        classification=DataClassification.PRIVATE,
        limits=AgentExecutionLimits(),
        allowed_tools=("local_read",),
        steps_used=5,
        cost_units_used=4,
        elapsed_seconds=1.0,
    )

    assert result.decision is AgentPolicyDecision.DENY
    assert result.reason == "max_steps_exceeded"


@pytest.mark.parametrize(
    ("steps_used", "cost_units_used", "elapsed_seconds", "reason"),
    [
        (0, 5, 0.0, "cost_budget_exceeded"),
        (0, 0, 60.0, "duration_exceeded"),
    ],
)
def test_each_execution_limit_stops_at_its_boundary(
    steps_used: int,
    cost_units_used: int,
    elapsed_seconds: float,
    reason: str,
) -> None:
    result = AgentExecutionPolicy().evaluate(
        descriptor=free_descriptor(
            name="local_read",
            effect=AgentToolEffect.READ,
            destination=Locality.LOCAL,
        ),
        classification=DataClassification.LOCAL_OPERATIONAL,
        limits=AgentExecutionLimits(),
        allowed_tools=("local_read",),
        steps_used=steps_used,
        cost_units_used=cost_units_used,
        elapsed_seconds=elapsed_seconds,
    )

    assert result.decision is AgentPolicyDecision.DENY
    assert result.reason == reason


def test_tool_outside_run_allow_list_is_denied() -> None:
    result = AgentExecutionPolicy().evaluate(
        descriptor=free_descriptor(
            name="unexpected",
            effect=AgentToolEffect.READ,
            destination=Locality.LOCAL,
        ),
        classification=DataClassification.PUBLIC_RESULT,
        limits=AgentExecutionLimits(),
        allowed_tools=("local_read",),
        steps_used=0,
        cost_units_used=0,
        elapsed_seconds=0.0,
    )

    assert result == AgentPolicyResult(
        AgentPolicyDecision.DENY, "tool_not_allowed"
    )


@pytest.mark.parametrize("destination", [Locality.LAN, Locality.REMOTE])
def test_private_data_cannot_leave_the_pc_even_with_a_known_destination(
    destination: Locality,
) -> None:
    result = AgentExecutionPolicy().evaluate(
        descriptor=free_descriptor(
            name="external",
            effect=AgentToolEffect.EXTERNAL_POST,
            destination=destination,
        ),
        classification=DataClassification.PRIVATE,
        limits=AgentExecutionLimits(),
        allowed_tools=("external",),
        steps_used=0,
        cost_units_used=0,
        elapsed_seconds=0.0,
    )

    assert result == AgentPolicyResult(
        AgentPolicyDecision.DENY, "private_external_send_denied"
    )


def test_unknown_destination_is_denied() -> None:
    result = AgentExecutionPolicy().evaluate(
        descriptor=free_descriptor(
            name="unknown",
            effect=AgentToolEffect.READ,
            destination=Locality.UNKNOWN,
        ),
        classification=DataClassification.PUBLIC_RESULT,
        limits=AgentExecutionLimits(),
        allowed_tools=("unknown",),
        steps_used=0,
        cost_units_used=0,
        elapsed_seconds=0.0,
    )

    assert result == AgentPolicyResult(
        AgentPolicyDecision.DENY, "destination_unknown"
    )


@pytest.mark.parametrize(
    "effect",
    [
        AgentToolEffect.REVERSIBLE_WRITE,
        AgentToolEffect.EXTERNAL_POST,
        AgentToolEffect.EXTERNAL_DELETE,
    ],
)
def test_every_effectful_action_requires_approval(effect: AgentToolEffect) -> None:
    result = AgentExecutionPolicy().evaluate(
        descriptor=free_descriptor(
            name="effectful",
            effect=effect,
            destination=(
                Locality.LOCAL
                if effect is AgentToolEffect.REVERSIBLE_WRITE
                else Locality.REMOTE
            ),
        ),
        classification=DataClassification.PUBLIC_RESULT,
        limits=AgentExecutionLimits(),
        allowed_tools=("effectful",),
        steps_used=0,
        cost_units_used=0,
        elapsed_seconds=0.0,
    )

    assert result == AgentPolicyResult(AgentPolicyDecision.REQUIRE_APPROVAL)


def test_billing_is_denied_even_if_an_approval_could_be_supplied() -> None:
    result = AgentExecutionPolicy().evaluate(
        descriptor=free_descriptor(
            name="billing",
            effect=AgentToolEffect.BILLING,
            destination=Locality.REMOTE,
        ),
        classification=DataClassification.PUBLIC_RESULT,
        limits=AgentExecutionLimits(),
        allowed_tools=("billing",),
        steps_used=0,
        cost_units_used=0,
        elapsed_seconds=0.0,
    )

    assert result == AgentPolicyResult(
        AgentPolicyDecision.DENY, "billing_denied_in_strict_free"
    )


@pytest.mark.parametrize(
    "cost_class",
    [CostClass.FREE_TIER, CostClass.PAID, CostClass.UNKNOWN],
)
def test_any_cost_class_other_than_no_charge_is_denied(
    cost_class: CostClass,
) -> None:
    result = AgentExecutionPolicy().evaluate(
        descriptor=free_descriptor(
            name="costly_read",
            effect=AgentToolEffect.READ,
            destination=Locality.LOCAL,
            cost_class=cost_class,
        ),
        classification=DataClassification.PUBLIC_RESULT,
        limits=AgentExecutionLimits(),
        allowed_tools=("costly_read",),
        steps_used=0,
        cost_units_used=0,
        elapsed_seconds=0.0,
    )

    assert result == AgentPolicyResult(
        AgentPolicyDecision.DENY, "cost_class_not_allowed"
    )


def test_local_read_within_all_limits_is_allowed() -> None:
    result = AgentExecutionPolicy().evaluate(
        descriptor=free_descriptor(
            name="local_read",
            effect=AgentToolEffect.READ,
            destination=Locality.LOCAL,
        ),
        classification=DataClassification.PRIVATE,
        limits=AgentExecutionLimits(),
        allowed_tools=("local_read",),
        steps_used=4,
        cost_units_used=4,
        elapsed_seconds=59.999,
    )

    assert result == AgentPolicyResult(AgentPolicyDecision.ALLOW)


def test_execution_limits_reject_non_positive_values() -> None:
    with pytest.raises(ValueError):
        AgentExecutionLimits(max_steps=0)
    with pytest.raises(ValueError):
        AgentExecutionLimits(max_cost_units=0)
    with pytest.raises(ValueError):
        AgentExecutionLimits(max_duration_seconds=0.0)
    with pytest.raises(ValueError):
        AgentExecutionLimits(max_steps=6)
    with pytest.raises(ValueError):
        AgentExecutionLimits(max_cost_units=6)
    with pytest.raises(ValueError):
        AgentExecutionLimits(max_duration_seconds=60.001)
    with pytest.raises(ValueError):
        AgentExecutionLimits(max_duration_seconds=float("nan"))
