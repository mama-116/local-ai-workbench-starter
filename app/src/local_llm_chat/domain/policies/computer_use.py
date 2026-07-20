from __future__ import annotations

import unicodedata

from local_llm_chat.domain.models import (
    ComputerActionRequest,
    ComputerPlan,
    ComputerPolicyResult,
    DesktopSecurityContext,
)
from local_llm_chat.domain.states import (
    ComputerActionType,
    ComputerPolicyDecision,
    DesktopIntegrityLevel,
)


class ComputerUsePolicy:
    _EXPECTED_SEQUENCE = (
        ComputerActionType.LAUNCH_ALLOWED_APP,
        ComputerActionType.CLICK_UIA_ELEMENT,
        ComputerActionType.TYPE_PLAIN_TEXT,
    )

    def evaluate_plan(
        self,
        plan: ComputerPlan,
        context: DesktopSecurityContext,
    ) -> ComputerPolicyResult:
        if not plan.conversation_id or not plan.objective or not plan.observation_id:
            return self._deny("plan_identity_required")
        if len(plan.actions) > plan.limits.max_actions:
            return self._deny("max_actions_exceeded")
        if tuple(action.action_type for action in plan.actions) != self._EXPECTED_SEQUENCE:
            return self._deny("fixed_action_sequence_required")
        action_ids = tuple(action.id for action in plan.actions)
        if any(not action_id for action_id in action_ids) or len(set(action_ids)) != len(
            action_ids
        ):
            return self._deny("action_id_invalid")
        for action in plan.actions:
            result = self.evaluate_action(action, context)
            if result.decision is ComputerPolicyDecision.DENY:
                return result
        return ComputerPolicyResult(ComputerPolicyDecision.REQUIRE_APPROVAL)

    def evaluate_action(
        self,
        action: ComputerActionRequest,
        context: DesktopSecurityContext,
    ) -> ComputerPolicyResult:
        context_result = self._evaluate_context(context)
        if context_result is not None:
            return context_result
        if action.action_type is ComputerActionType.LAUNCH_ALLOWED_APP:
            if action.target_profile_id != "windows_notepad":
                return self._deny("app_profile_denied")
            if action.text is not None:
                return self._deny("launch_text_denied")
        elif action.action_type is ComputerActionType.CLICK_UIA_ELEMENT:
            if action.target_profile_id != "notepad_edit":
                return self._deny("target_profile_denied")
            if action.text is not None:
                return self._deny("click_text_denied")
        elif action.action_type is ComputerActionType.TYPE_PLAIN_TEXT:
            if action.target_profile_id != "notepad_edit":
                return self._deny("target_profile_denied")
            if action.text is None or not 1 <= len(action.text) <= 200:
                return self._deny("plain_text_length_denied")
            if any(unicodedata.category(char).startswith("C") for char in action.text):
                return self._deny("plain_text_control_character_denied")
        return ComputerPolicyResult(ComputerPolicyDecision.ALLOW)

    def _evaluate_context(
        self, context: DesktopSecurityContext
    ) -> ComputerPolicyResult | None:
        if context.ui_access:
            return self._deny("ui_access_denied")
        if context.controller_elevated:
            return self._deny("controller_elevated")
        if context.target_elevated:
            return self._deny("target_elevated")
        if context.secure_desktop:
            return self._deny("secure_desktop_denied")
        if context.controller_user_sid != context.target_user_sid:
            return self._deny("user_sid_mismatch")
        if context.controller_session_id != context.target_session_id:
            return self._deny("session_mismatch")
        if context.controller_session_id < 0:
            return self._deny("session_invalid")
        if context.controller_integrity is not DesktopIntegrityLevel.MEDIUM:
            return self._deny("controller_integrity_denied")
        if context.target_integrity is not DesktopIntegrityLevel.MEDIUM:
            return self._deny("target_integrity_denied")
        return None

    @staticmethod
    def _deny(reason: str) -> ComputerPolicyResult:
        return ComputerPolicyResult(ComputerPolicyDecision.DENY, reason)
