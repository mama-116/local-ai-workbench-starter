from __future__ import annotations

from dataclasses import replace

import pytest

from local_llm_chat.domain.models import (
    ComputerActionRequest,
    ComputerPlan,
    ComputerUseLimits,
    DesktopSecurityContext,
)
from local_llm_chat.domain.policies.computer_use import ComputerUsePolicy
from local_llm_chat.domain.states import (
    ComputerActionType,
    ComputerPolicyDecision,
    DesktopIntegrityLevel,
)


def safe_context() -> DesktopSecurityContext:
    return DesktopSecurityContext(
        controller_user_sid="same-user",
        target_user_sid="same-user",
        controller_session_id=1,
        target_session_id=1,
        controller_integrity=DesktopIntegrityLevel.MEDIUM,
        target_integrity=DesktopIntegrityLevel.MEDIUM,
        controller_elevated=False,
        target_elevated=False,
        ui_access=False,
        secure_desktop=False,
    )


def type_action(text: str, target: str = "notepad_edit") -> ComputerActionRequest:
    return ComputerActionRequest(
        "type", ComputerActionType.TYPE_PLAIN_TEXT, target, text=text
    )


@pytest.mark.parametrize(
    ("context", "reason"),
    [
        (replace(safe_context(), target_elevated=True), "target_elevated"),
        (replace(safe_context(), secure_desktop=True), "secure_desktop_denied"),
        (
            replace(
                safe_context(),
                controller_integrity=DesktopIntegrityLevel.HIGH,
            ),
            "controller_integrity_denied",
        ),
        (
            replace(
                safe_context(), target_integrity=DesktopIntegrityLevel.SYSTEM
            ),
            "target_integrity_denied",
        ),
        (
            replace(
                safe_context(),
                controller_session_id=-1,
                target_session_id=-1,
            ),
            "session_invalid",
        ),
    ],
)
def test_unsafe_windows_security_context_is_denied(
    context: DesktopSecurityContext, reason: str
) -> None:
    result = ComputerUsePolicy().evaluate_action(
        type_action("安全確認"), context
    )

    assert result.decision is ComputerPolicyDecision.DENY
    assert result.reason == reason


@pytest.mark.parametrize("text", ["line1\nline2", "zero\u200bwidth", "x" * 201])
def test_control_or_oversized_text_is_denied(text: str) -> None:
    result = ComputerUsePolicy().evaluate_action(type_action(text), safe_context())

    assert result.decision is ComputerPolicyDecision.DENY


def test_japanese_and_emoji_plain_text_is_allowed() -> None:
    result = ComputerUsePolicy().evaluate_action(
        type_action("安全確認🙂"), safe_context()
    )

    assert result.decision is ComputerPolicyDecision.ALLOW


@pytest.mark.parametrize(
    "action",
    [
        ComputerActionRequest(
            "launch",
            ComputerActionType.LAUNCH_ALLOWED_APP,
            "powershell",
        ),
        ComputerActionRequest(
            "click",
            ComputerActionType.CLICK_UIA_ELEMENT,
            "windows_settings",
        ),
        type_action("safe", "password_field"),
    ],
)
def test_arbitrary_app_or_target_profile_is_denied(
    action: ComputerActionRequest,
) -> None:
    result = ComputerUsePolicy().evaluate_action(action, safe_context())

    assert result.decision is ComputerPolicyDecision.DENY


def test_plan_must_have_exact_fixed_action_sequence() -> None:
    plan = ComputerPlan(
        "conversation",
        "objective",
        "observation",
        (type_action("AI says this is optimal"),),
    )

    result = ComputerUsePolicy().evaluate_plan(plan, safe_context())

    assert result.decision is ComputerPolicyDecision.DENY
    assert result.reason == "fixed_action_sequence_required"


def test_computer_use_limits_cannot_relax_the_safety_ceiling() -> None:
    with pytest.raises(ValueError):
        ComputerUseLimits(max_actions=6)
    with pytest.raises(ValueError):
        ComputerUseLimits(max_duration_seconds=60.001)
    with pytest.raises(ValueError):
        ComputerUseLimits(approval_timeout_seconds=30.001)
