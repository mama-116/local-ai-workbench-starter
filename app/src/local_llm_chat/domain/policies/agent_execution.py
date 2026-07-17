from __future__ import annotations

from local_llm_chat.domain.models import (
    AgentExecutionLimits,
    AgentPolicyResult,
    AgentToolDescriptor,
)
from local_llm_chat.domain.states import (
    AgentPolicyDecision,
    AgentToolEffect,
    CostClass,
    DataClassification,
    Locality,
)


class AgentExecutionPolicy:
    def evaluate(
        self,
        *,
        descriptor: AgentToolDescriptor,
        classification: DataClassification,
        limits: AgentExecutionLimits,
        allowed_tools: tuple[str, ...],
        steps_used: int,
        cost_units_used: int,
        elapsed_seconds: float,
    ) -> AgentPolicyResult:
        if descriptor.name not in allowed_tools:
            return AgentPolicyResult(
                AgentPolicyDecision.DENY, "tool_not_allowed"
            )
        if descriptor.cost_class is not CostClass.NO_CHARGE:
            return AgentPolicyResult(
                AgentPolicyDecision.DENY, "cost_class_not_allowed"
            )
        if descriptor.effect is AgentToolEffect.BILLING:
            return AgentPolicyResult(
                AgentPolicyDecision.DENY, "billing_denied_in_strict_free"
            )
        if descriptor.destination is Locality.UNKNOWN:
            return AgentPolicyResult(
                AgentPolicyDecision.DENY, "destination_unknown"
            )
        if (
            classification is DataClassification.PRIVATE
            and descriptor.destination is not Locality.LOCAL
        ):
            return AgentPolicyResult(
                AgentPolicyDecision.DENY, "private_external_send_denied"
            )
        if steps_used >= limits.max_steps:
            return AgentPolicyResult(
                AgentPolicyDecision.DENY, "max_steps_exceeded"
            )
        if cost_units_used + descriptor.cost_units > limits.max_cost_units:
            return AgentPolicyResult(
                AgentPolicyDecision.DENY, "cost_budget_exceeded"
            )
        if elapsed_seconds >= limits.max_duration_seconds:
            return AgentPolicyResult(
                AgentPolicyDecision.DENY, "duration_exceeded"
            )
        if descriptor.effect in {
            AgentToolEffect.REVERSIBLE_WRITE,
            AgentToolEffect.EXTERNAL_POST,
            AgentToolEffect.EXTERNAL_DELETE,
        }:
            return AgentPolicyResult(AgentPolicyDecision.REQUIRE_APPROVAL)
        return AgentPolicyResult(AgentPolicyDecision.ALLOW)
