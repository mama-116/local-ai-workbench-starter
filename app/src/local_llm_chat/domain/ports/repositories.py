from __future__ import annotations

from typing import Any, Protocol

from local_llm_chat.domain.models import (
    BranchInfo,
    CharacterVersion,
    Conversation,
    Message,
    ModelProfile,
    LatestTelemetry,
    RunSession,
    Translation,
    TranslationPreparation,
    TelemetryMetric,
)
from local_llm_chat.domain.states import MessageState, TranslationState


class AppRepository(Protocol):
    async def initialize(self) -> None: ...

    async def recover_interrupted_runs(self) -> None: ...

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

    async def list_active_messages(self, conversation_id: str) -> list[Message]: ...

    async def get_message(self, message_id: str) -> Message: ...

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

    async def activate_branch(self, conversation_id: str, branch_id: str) -> None: ...

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

    async def context_to_message(self, message_id: str) -> list[Message]: ...

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
