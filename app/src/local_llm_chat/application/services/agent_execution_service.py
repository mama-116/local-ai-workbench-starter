from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from local_llm_chat.domain.errors import AgentToolFailure
from local_llm_chat.domain.models import (
    AgentActionApproval,
    AgentExecutionLimits,
    AgentRun,
    AgentStep,
    AgentStepRequest,
    AgentToolDescriptor,
    AgentToolExecution,
)
from local_llm_chat.domain.policies.agent_execution import AgentExecutionPolicy
from local_llm_chat.domain.ports.agent_execution import (
    AgentApprovalVerifier,
    AgentExecutionRepository,
    AgentRequestSource,
    AgentToolProvider,
)
from local_llm_chat.domain.states import (
    AgentPolicyDecision,
    AgentRunState,
    AgentStepState,
    AgentToolEffect,
    CostClass,
    DataClassification,
    Locality,
)


def agent_action_hash(
    request: AgentStepRequest,
    *,
    descriptor: AgentToolDescriptor,
    classification: DataClassification,
    limits: AgentExecutionLimits,
    allowed_tools: tuple[str, ...],
    conversation_id: str,
    objective: str,
) -> str:
    payload = json.dumps(
        {
            "tool_name": request.tool_name,
            "arguments": request.arguments,
            "effect": descriptor.effect.value,
            "destination": descriptor.destination.value,
            "cost_units": descriptor.cost_units,
            "cost_class": descriptor.cost_class.value,
            "data_classification": classification.value,
            "limits": {
                "max_cost_units": limits.max_cost_units,
                "max_steps": limits.max_steps,
                "max_duration_seconds": limits.max_duration_seconds,
            },
            "allowed_tools": sorted(set(allowed_tools)),
            "conversation_id": conversation_id,
            "objective": objective,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class AgentExecutionCoordinator:
    def __init__(
        self,
        repository: AgentExecutionRepository,
        providers: tuple[AgentToolProvider, ...],
        approval_verifier: AgentApprovalVerifier,
        *,
        limits: AgentExecutionLimits | None = None,
        policy: AgentExecutionPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        provider_map = {provider.descriptor.name: provider for provider in providers}
        if len(provider_map) != len(providers):
            raise ValueError("agent tool names must be unique")
        self._repository = repository
        self._providers = provider_map
        self._approval_verifier = approval_verifier
        self._limits = limits or AgentExecutionLimits()
        self._policy = policy or AgentExecutionPolicy()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._run_lock = asyncio.Lock()

    def approval_hash(
        self,
        request: AgentStepRequest,
        conversation_id: str,
        objective: str,
        allowed_tools: tuple[str, ...],
    ) -> str:
        provider = self._providers.get(request.tool_name)
        if provider is None:
            raise ValueError("tool is not registered")
        classification = provider.classify(request.arguments)
        return agent_action_hash(
            request,
            descriptor=provider.descriptor,
            classification=classification,
            limits=self._limits,
            allowed_tools=tuple(dict.fromkeys(allowed_tools)),
            conversation_id=conversation_id,
            objective=objective,
        )

    async def execute(
        self,
        *,
        conversation_id: str,
        objective: str,
        allowed_tools: tuple[str, ...],
        source: AgentRequestSource,
        approvals: Mapping[str, AgentActionApproval] | None = None,
    ) -> AgentRun:
        async with self._run_lock:
            return await self._execute_locked(
                conversation_id=conversation_id,
                objective=objective,
                allowed_tools=allowed_tools,
                source=source,
                approvals=approvals or {},
            )

    async def _execute_locked(
        self,
        *,
        conversation_id: str,
        objective: str,
        allowed_tools: tuple[str, ...],
        source: AgentRequestSource,
        approvals: Mapping[str, AgentActionApproval],
    ) -> AgentRun:
        now = self._aware_now()
        run = AgentRun(
            id=str(uuid4()),
            conversation_id=conversation_id,
            objective=objective,
            allowed_tools=tuple(dict.fromkeys(allowed_tools)),
            limits=self._limits,
            state=AgentRunState.RUNNING,
            failure_reason=None,
            created_at=now,
            started_at=now,
        )
        run = await self._repository.create_agent_run(run)
        try:
            source_is_local_and_free = (
                source.locality is Locality.LOCAL
                and source.cost_class is CostClass.NO_CHARGE
            )
        except Exception:
            source_is_local_and_free = False
        if not source_is_local_and_free:
            return await self._finish_run(
                run,
                AgentRunState.DENIED,
                "request_source_not_local_no_charge",
            )
        started = self._monotonic()
        accepted_steps: list[AgentStep] = []
        seen_request_ids: set[str] = set()
        cost_units_used = 0
        previous_result: AgentToolExecution | None = None
        consumed_approval_ids: set[str] = set()

        try:
            while True:
                elapsed = max(0.0, self._monotonic() - started)
                remaining = self._limits.max_duration_seconds - elapsed
                if remaining <= 0:
                    return await self._fail_run(
                        run,
                        AgentRunState.FAILED,
                        "duration_exceeded",
                        accepted_steps,
                    )
                try:
                    request = await asyncio.wait_for(
                        source.next_step(run, previous_result), timeout=remaining
                    )
                except TimeoutError:
                    return await self._fail_run(
                        run,
                        AgentRunState.FAILED,
                        "duration_exceeded",
                        accepted_steps,
                    )
                if (
                    max(0.0, self._monotonic() - started)
                    >= self._limits.max_duration_seconds
                ):
                    return await self._fail_run(
                        run,
                        AgentRunState.FAILED,
                        "duration_exceeded",
                        accepted_steps,
                    )
                if request is None:
                    return await self._finish_run(run, AgentRunState.COMPLETED)
                if not request.id or request.id in seen_request_ids:
                    return await self._deny_unknown_request(
                        run,
                        request,
                        len(accepted_steps) + 1,
                        "duplicate_or_empty_request_id",
                        accepted_steps,
                    )
                seen_request_ids.add(request.id)
                try:
                    json.dumps(request.arguments, sort_keys=True)
                except (TypeError, ValueError):
                    return await self._deny_unknown_request(
                        run,
                        replace(request, arguments={}),
                        len(accepted_steps) + 1,
                        "arguments_not_json",
                        accepted_steps,
                        action_hash="",
                    )

                provider = self._providers.get(request.tool_name)
                if provider is None:
                    return await self._deny_unknown_request(
                        run,
                        request,
                        len(accepted_steps) + 1,
                        "tool_not_registered",
                        accepted_steps,
                        action_hash="",
                    )
                descriptor = provider.descriptor
                try:
                    classification = provider.classify(request.arguments)
                except Exception:
                    return await self._deny_known_request(
                        run,
                        request,
                        descriptor,
                        None,
                        len(accepted_steps) + 1,
                        "",
                        "classification_failed",
                        accepted_steps,
                    )
                action_hash = agent_action_hash(
                    request,
                    descriptor=descriptor,
                    classification=classification,
                    limits=self._limits,
                    allowed_tools=run.allowed_tools,
                    conversation_id=run.conversation_id,
                    objective=run.objective,
                )

                elapsed = max(0.0, self._monotonic() - started)
                policy_result = self._policy.evaluate(
                    descriptor=descriptor,
                    classification=classification,
                    limits=self._limits,
                    allowed_tools=run.allowed_tools,
                    steps_used=len(accepted_steps),
                    cost_units_used=cost_units_used,
                    elapsed_seconds=elapsed,
                )
                if policy_result.decision is AgentPolicyDecision.DENY:
                    return await self._deny_known_request(
                        run,
                        request,
                        descriptor,
                        classification,
                        len(accepted_steps) + 1,
                        action_hash,
                        policy_result.reason or "policy_denied",
                        accepted_steps,
                    )
                if policy_result.decision is AgentPolicyDecision.REQUIRE_APPROVAL:
                    approval = approvals.get(request.id)
                    try:
                        approved = (
                            approval is not None
                            and bool(approval.id)
                            and approval.id not in consumed_approval_ids
                            and self._approval_is_fresh(approval)
                            and await self._approval_verifier.verify_and_consume(
                                approval, action_hash
                            )
                        )
                    except Exception:
                        approved = False
                    if not approved:
                        return await self._deny_known_request(
                            run,
                            request,
                            descriptor,
                            classification,
                            len(accepted_steps) + 1,
                            action_hash,
                            "approval_required",
                            accepted_steps,
                        )
                    if approval is not None:  # narrowed after the fail-closed branch
                        consumed_approval_ids.add(approval.id)

                step = AgentStep(
                    id=str(uuid4()),
                    run_id=run.id,
                    ordinal=len(accepted_steps) + 1,
                    tool_name=request.tool_name,
                    arguments=dict(request.arguments),
                    action_hash=action_hash,
                    data_classification=classification,
                    effect=descriptor.effect,
                    destination=descriptor.destination,
                    cost_class=descriptor.cost_class,
                    cost_units=descriptor.cost_units,
                    state=AgentStepState.PROPOSED,
                    result_size_bytes=None,
                    result_sha256=None,
                    restore_token=None,
                    failure_reason=None,
                    created_at=self._aware_now(),
                )
                step = await self._repository.create_agent_step(step)
                step = await self._repository.finish_agent_step(
                    replace(
                        step,
                        state=AgentStepState.RUNNING,
                        started_at=self._aware_now(),
                    )
                )
                elapsed = max(0.0, self._monotonic() - started)
                remaining = self._limits.max_duration_seconds - elapsed
                if remaining <= 0:
                    failed = await self._repository.finish_agent_step(
                        replace(
                            step,
                            state=AgentStepState.FAILED,
                            failure_reason="duration_exceeded",
                            completed_at=self._aware_now(),
                        )
                    )
                    accepted_steps.append(failed)
                    return await self._fail_run(
                        run,
                        AgentRunState.FAILED,
                        "duration_exceeded",
                        accepted_steps,
                    )
                try:
                    execution = await asyncio.wait_for(
                        provider.execute(request.arguments), timeout=remaining
                    )
                except TimeoutError:
                    failed = await self._repository.finish_agent_step(
                        replace(
                            step,
                            state=AgentStepState.FAILED,
                            failure_reason="duration_exceeded",
                            completed_at=self._aware_now(),
                        )
                    )
                    accepted_steps.append(failed)
                    return await self._fail_run(
                        run,
                        AgentRunState.FAILED,
                        "duration_exceeded",
                        accepted_steps,
                    )
                except AgentToolFailure as error:
                    failure_state = (
                        AgentStepState.FAILED
                        if error.restore_token is not None
                        or descriptor.effect is not AgentToolEffect.REVERSIBLE_WRITE
                        else AgentStepState.RESTORE_FAILED
                    )
                    failed = await self._repository.finish_agent_step(
                        replace(
                            step,
                            state=failure_state,
                            restore_token=error.restore_token,
                            failure_reason="provider_failed:AgentToolFailure",
                            completed_at=self._aware_now(),
                        )
                    )
                    accepted_steps.append(failed)
                    return await self._fail_run(
                        run,
                        AgentRunState.FAILED,
                        failed.failure_reason or "provider_failed",
                        accepted_steps,
                    )
                except Exception as error:
                    failure_state = (
                        AgentStepState.RESTORE_FAILED
                        if descriptor.effect is AgentToolEffect.REVERSIBLE_WRITE
                        else AgentStepState.FAILED
                    )
                    failed = await self._repository.finish_agent_step(
                        replace(
                            step,
                            state=failure_state,
                            failure_reason=f"provider_failed:{type(error).__name__}",
                            completed_at=self._aware_now(),
                        )
                    )
                    accepted_steps.append(failed)
                    return await self._fail_run(
                        run,
                        AgentRunState.FAILED,
                        failed.failure_reason or "provider_failed",
                        accepted_steps,
                    )

                result_bytes = execution.result_content.encode("utf-8")
                failure_reason = None
                state = AgentStepState.COMPLETED
                if (
                    max(0.0, self._monotonic() - started)
                    >= self._limits.max_duration_seconds
                ):
                    state = AgentStepState.FAILED
                    failure_reason = "duration_exceeded"
                if (
                    descriptor.effect is AgentToolEffect.REVERSIBLE_WRITE
                    and execution.restore_token is None
                ):
                    state = AgentStepState.RESTORE_FAILED
                    failure_reason = "restore_token_missing"
                finished = await self._repository.finish_agent_step(
                    replace(
                        step,
                        state=state,
                        result_size_bytes=len(result_bytes),
                        result_sha256=hashlib.sha256(result_bytes).hexdigest(),
                        restore_token=execution.restore_token,
                        failure_reason=failure_reason,
                        completed_at=self._aware_now(),
                    )
                )
                accepted_steps.append(finished)
                cost_units_used += descriptor.cost_units
                previous_result = execution
                if failure_reason == "duration_exceeded":
                    return await self._fail_run(
                        run,
                        AgentRunState.FAILED,
                        "duration_exceeded",
                        accepted_steps,
                    )
                if state is AgentStepState.RESTORE_FAILED:
                    return await self._fail_run(
                        run,
                        AgentRunState.FAILED,
                        "restore_token_missing",
                        accepted_steps,
                    )
        except asyncio.CancelledError:
            await self._fail_run(
                run,
                AgentRunState.CANCELLED,
                "cancelled",
                accepted_steps,
            )
            raise
        except Exception as error:
            return await self._fail_run(
                run,
                AgentRunState.FAILED,
                f"source_failed:{type(error).__name__}",
                accepted_steps,
            )

    async def _deny_unknown_request(
        self,
        run: AgentRun,
        request: AgentStepRequest,
        ordinal: int,
        reason: str,
        accepted_steps: list[AgentStep],
        *,
        action_hash: str = "",
    ) -> AgentRun:
        return await self._deny_known_request(
            run,
            request,
            None,
            None,
            ordinal,
            action_hash,
            reason,
            accepted_steps,
        )

    async def _deny_known_request(
        self,
        run: AgentRun,
        request: AgentStepRequest,
        descriptor: AgentToolDescriptor | None,
        classification: DataClassification | None,
        ordinal: int,
        action_hash: str,
        reason: str,
        accepted_steps: list[AgentStep],
    ) -> AgentRun:
        now = self._aware_now()
        denied = AgentStep(
            id=str(uuid4()),
            run_id=run.id,
            ordinal=ordinal,
            tool_name=request.tool_name,
            arguments=dict(request.arguments),
            action_hash=action_hash,
            data_classification=classification,
            effect=descriptor.effect if descriptor is not None else None,
            destination=(descriptor.destination if descriptor is not None else None),
            cost_class=(descriptor.cost_class if descriptor is not None else None),
            cost_units=descriptor.cost_units if descriptor is not None else None,
            state=AgentStepState.DENIED,
            result_size_bytes=None,
            result_sha256=None,
            restore_token=None,
            failure_reason=reason,
            created_at=now,
            completed_at=now,
        )
        await self._repository.create_agent_step(denied)
        restore_failed = await self._restore(accepted_steps)
        final_reason = f"{reason};restore_failed" if restore_failed else reason
        return await self._finish_run(run, AgentRunState.DENIED, final_reason)

    async def _fail_run(
        self,
        run: AgentRun,
        state: AgentRunState,
        reason: str,
        accepted_steps: list[AgentStep],
    ) -> AgentRun:
        restore_failed = await self._restore(accepted_steps)
        final_reason = f"{reason};restore_failed" if restore_failed else reason
        return await self._finish_run(run, state, final_reason)

    async def _restore(self, steps: list[AgentStep]) -> bool:
        restore_failed = any(
            step.effect is AgentToolEffect.REVERSIBLE_WRITE
            and step.state is AgentStepState.RESTORE_FAILED
            for step in steps
        )
        for step in reversed(steps):
            if step.effect is not AgentToolEffect.REVERSIBLE_WRITE:
                continue
            if step.state not in {
                AgentStepState.COMPLETED,
                AgentStepState.FAILED,
            }:
                continue
            if step.restore_token is None:
                restore_failed = True
                await self._repository.finish_agent_step(
                    replace(
                        step,
                        state=AgentStepState.RESTORE_FAILED,
                        failure_reason="restore_token_missing_after_failure",
                    )
                )
                continue
            provider = self._providers.get(step.tool_name)
            try:
                if provider is None:
                    raise RuntimeError("provider missing")
                await provider.restore(step.restore_token)
                state = AgentStepState.RESTORED
                reason = None
            except Exception as error:
                state = AgentStepState.RESTORE_FAILED
                reason = f"restore_failed:{type(error).__name__}"
                restore_failed = True
            await self._repository.finish_agent_step(
                replace(step, state=state, failure_reason=reason)
            )
        return restore_failed

    async def _finish_run(
        self,
        run: AgentRun,
        state: AgentRunState,
        reason: str | None = None,
    ) -> AgentRun:
        return await self._repository.finish_agent_run(
            replace(
                run,
                state=state,
                failure_reason=reason,
                completed_at=self._aware_now(),
            )
        )

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    def _approval_is_fresh(self, approval: AgentActionApproval) -> bool:
        approved_at = approval.approved_at
        if approved_at.tzinfo is None or approved_at.utcoffset() is None:
            return False
        age_seconds = (
            self._aware_now() - approved_at.astimezone(UTC)
        ).total_seconds()
        return 0 <= age_seconds <= 30
