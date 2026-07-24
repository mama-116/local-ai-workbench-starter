from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from local_llm_chat.domain.canonical_memory import (
    CanonicalMemoryEvent,
    CanonicalMemoryFact,
    CanonicalMemoryReviewItem,
    MemoryApprovalDecision,
)
from local_llm_chat.domain.group_turns import (
    ConversationCast,
    ConversationGroupConfiguration,
    ConversationGroupSettings,
    TurnBatch,
    TurnBatchDraft,
)
from local_llm_chat.domain.explicit_memory import (
    ExplicitMemoryDecision,
    ExplicitMemoryEvent,
    ExplicitMemoryReviewItem,
)
from local_llm_chat.domain.models import (
    BranchInfo,
    CharacterVersion,
    Conversation,
    ContextSummary,
    ContextSummaryPreparation,
    Message,
    ModelProfile,
    LatestTelemetry,
    RunSession,
    Translation,
    TranslationPreparation,
    TelemetryMetric,
    ToolCallAudit,
    ToolCallRequest,
    ToolFolderGrant,
)
from local_llm_chat.domain.states import (
    ContextSummaryState,
    MemoryApprovalState,
    MessageState,
    ToolCallState,
    TurnMode,
    TranslationState,
)


class AppRepository(Protocol):
    async def initialize(self) -> None: ...

    async def recover_interrupted_runs(self) -> None: ...

    async def set_tool_folder_grant(
        self, conversation_id: str, root_path: Path
    ) -> ToolFolderGrant: ...

    async def get_tool_folder_grant(
        self, conversation_id: str
    ) -> ToolFolderGrant | None: ...

    async def revoke_tool_folder_grant(self, conversation_id: str) -> None: ...

    async def create_tool_call(
        self,
        conversation_id: str,
        run_id: str | None,
        provider: str,
        request: ToolCallRequest,
    ) -> ToolCallAudit: ...

    async def mark_tool_call_running(self, call_id: str) -> None: ...

    async def finish_tool_call(
        self,
        call_id: str,
        state: ToolCallState,
        result_content: str | None = None,
        result_item_count: int | None = None,
        failure_reason: str | None = None,
    ) -> None: ...

    async def list_tool_calls(
        self, conversation_id: str, limit: int = 100
    ) -> list[ToolCallAudit]: ...

    async def ensure_default_character(self) -> CharacterVersion: ...

    async def list_character_versions(self) -> list[CharacterVersion]: ...

    async def create_character_version(
        self,
        display_name: str,
        system_prompt: str,
        character_id: str | None = None,
    ) -> CharacterVersion: ...

    async def get_character_version(self, version_id: str) -> CharacterVersion: ...

    async def ensure_model_profile(
        self,
        provider: str,
        model_name: str,
        parameters: dict[str, Any],
    ) -> ModelProfile: ...

    async def get_model_profile(self, profile_id: str) -> ModelProfile: ...

    async def create_conversation(
        self,
        title: str,
        character_version_id: str,
        model_profile_id: str,
    ) -> Conversation: ...

    async def list_conversations(self) -> list[Conversation]: ...

    async def list_archived_conversations(self) -> list[Conversation]: ...

    async def get_conversation(self, conversation_id: str) -> Conversation: ...

    async def update_conversation_selection(
        self,
        conversation_id: str,
        character_version_id: str,
        model_profile_id: str,
    ) -> None: ...

    async def archive_conversation(self, conversation_id: str) -> None: ...

    async def restore_conversation(self, conversation_id: str) -> None: ...

    async def set_conversation_auto_translate(
        self, conversation_id: str, enabled: bool
    ) -> None: ...

    async def list_active_messages(self, conversation_id: str) -> list[Message]: ...

    async def get_message(self, message_id: str) -> Message: ...

    async def get_memory_source_message(
        self, conversation_id: str, branch_id: str, source_message_id: str
    ) -> Message: ...

    async def get_response_model(self, message_id: str) -> tuple[str, str]: ...

    async def prepare_translation(
        self,
        message_id: str,
        target_language: str,
        provider: str,
        model: str,
        force: bool,
    ) -> TranslationPreparation: ...

    async def get_translation(self, translation_id: str) -> Translation: ...

    async def get_current_translation(self, message_id: str) -> Translation | None: ...

    async def list_current_translations(
        self, message_ids: list[str]
    ) -> dict[str, Translation]: ...

    async def mark_translation_running(self, translation_id: str) -> Translation: ...

    async def finish_translation(
        self,
        translation_id: str,
        content: str,
        state: TranslationState,
        error_code: str | None = None,
    ) -> Translation: ...

    async def list_branches(self, conversation_id: str) -> list[BranchInfo]: ...

    async def list_all_branches(self, conversation_id: str) -> list[BranchInfo]: ...

    async def activate_branch(self, conversation_id: str, branch_id: str) -> None: ...

    async def hide_branch(self, conversation_id: str, branch_id: str) -> None: ...

    async def restore_branch(self, conversation_id: str, branch_id: str) -> None: ...

    async def append_canonical_memory_event(
        self, event: CanonicalMemoryEvent
    ) -> None: ...

    async def get_conversation_cast(
        self, conversation_id: str
    ) -> ConversationCast: ...

    async def set_conversation_cast(
        self, conversation_id: str, character_version_ids: tuple[str, ...]
    ) -> ConversationCast: ...

    async def get_conversation_group_settings(
        self, conversation_id: str
    ) -> ConversationGroupSettings: ...

    async def set_conversation_group_settings(
        self,
        conversation_id: str,
        enabled: bool,
        mode: TurnMode,
        spotlight_character_id: str | None,
    ) -> ConversationGroupSettings: ...

    async def get_conversation_group_configuration(
        self, conversation_id: str
    ) -> ConversationGroupConfiguration: ...

    async def set_conversation_group_configuration(
        self,
        conversation_id: str,
        character_version_ids: tuple[str, ...],
        enabled: bool,
        mode: TurnMode,
        spotlight_character_id: str | None,
    ) -> ConversationGroupConfiguration: ...

    async def append_canonical_memory_events(
        self, events: tuple[CanonicalMemoryEvent, ...]
    ) -> None: ...

    async def append_captured_memory_events(
        self, events: tuple[CanonicalMemoryEvent, ...]
    ) -> tuple[str, ...]: ...

    async def append_memory_approval_decision(
        self, decision: MemoryApprovalDecision
    ) -> None: ...

    async def decide_canonical_memory(
        self,
        conversation_id: str,
        branch_id: str,
        target_event_id: str,
        state: MemoryApprovalState,
        decision_id: str,
        recorded_at: datetime,
    ) -> MemoryApprovalDecision: ...

    async def project_canonical_memory(
        self,
        conversation_id: str,
        branch_id: str,
        speaker_character_id: str,
        current_source_message_id: str | None = None,
        include_historical: bool = False,
    ) -> tuple[CanonicalMemoryFact, ...]: ...

    async def list_canonical_memory_review_items(
        self, conversation_id: str, branch_id: str
    ) -> tuple[CanonicalMemoryReviewItem, ...]: ...

    async def remember_explicit_memory(
        self,
        request_id: str,
        conversation_id: str,
        branch_id: str,
        source_message_id: str,
        expected_character_id: str,
        value: str,
        recorded_at: datetime,
    ) -> ExplicitMemoryEvent: ...

    async def undo_explicit_memory(
        self,
        decision_id: str,
        conversation_id: str,
        branch_id: str,
        target_event_id: str,
        expected_character_id: str,
        recorded_at: datetime,
    ) -> ExplicitMemoryDecision: ...

    async def project_explicit_memory(
        self,
        conversation_id: str,
        branch_id: str,
        character_id: str,
    ) -> tuple[ExplicitMemoryReviewItem, ...]: ...

    async def start_send(self, conversation_id: str, content: str) -> RunSession: ...

    async def start_rewrite(
        self,
        conversation_id: str,
        source_message_id: str,
        content: str,
    ) -> RunSession: ...

    async def start_regenerate(
        self,
        conversation_id: str,
        source_message_id: str,
    ) -> RunSession: ...

    async def start_turn_batch_regenerate(
        self,
        conversation_id: str,
        source_response_message_id: str,
        expected_active_branch_id: str,
    ) -> RunSession: ...

    async def context_to_message(self, message_id: str) -> list[Message]: ...

    async def prepare_context_summary(
        self,
        conversation_id: str,
        branch_id: str,
        source_message_ids: tuple[str, ...],
        source_hash: str,
        settings_hash: str,
        model: str,
        prompt_version: str,
    ) -> ContextSummaryPreparation: ...

    async def mark_context_summary_running(self, summary_id: str) -> ContextSummary: ...

    async def finish_context_summary(
        self,
        summary_id: str,
        content: str,
        state: ContextSummaryState,
        error_code: str | None = None,
    ) -> ContextSummary: ...

    async def list_context_summaries(
        self, conversation_id: str
    ) -> list[ContextSummary]: ...

    async def checkpoint_response(self, message_id: str, content: str) -> None: ...

    async def finish_response(
        self,
        session: RunSession,
        content: str,
        state: MessageState,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
        total_duration_ns: int | None = None,
        generation_duration_ns: int | None = None,
        response_duration_ms: int | None = None,
        error_code: str | None = None,
    ) -> Message: ...

    async def finish_turn_batch(
        self, session: RunSession, draft: TurnBatchDraft
    ) -> TurnBatch: ...

    async def get_turn_batch_for_response(
        self, response_message_id: str
    ) -> TurnBatch: ...

    async def list_turn_batches_for_responses(
        self, response_message_ids: tuple[str, ...]
    ) -> tuple[TurnBatch, ...]: ...

    async def save_telemetry_metrics(
        self, run_id: str, metrics: tuple[TelemetryMetric, ...]
    ) -> None: ...

    async def get_latest_telemetry(
        self, conversation_id: str | None = None
    ) -> LatestTelemetry | None: ...

    async def log_event(
        self,
        level: str,
        event_type: str,
        details: dict[str, Any],
        run_id: str | None = None,
    ) -> None: ...
