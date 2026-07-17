from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import math
from pathlib import Path
from typing import Any

from local_llm_chat.domain.states import (
    AgentPolicyDecision,
    AgentRunState,
    AgentStepState,
    AgentToolEffect,
    CostClass,
    DataClassification,
    Locality,
    MessageRole,
    MessageState,
    RunState,
    TranslationState,
    ToolCallState,
    ContextSummaryState,
    JobRunState,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ProviderMetadata:
    name: str
    locality: Locality
    cost_class: CostClass
    endpoint: str


@dataclass(frozen=True, slots=True)
class ProviderConnection:
    id: str
    provider_name: str
    display_name: str
    endpoint: str
    cloud_disabled_confirmed: bool
    is_builtin: bool = False


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    size_bytes: int
    format: str
    family: str
    parameter_size: str
    quantization: str
    license_text: str = ""


@dataclass(frozen=True, slots=True)
class CharacterVersion:
    id: str
    character_id: str
    display_name: str
    version: int
    system_prompt: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ModelProfile:
    id: str
    provider: str
    model_name: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    title: str
    active_branch_id: str
    character_version_id: str
    model_profile_id: str
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ConversationSelection:
    character: CharacterVersion
    model_profile: ModelProfile


@dataclass(frozen=True, slots=True)
class BranchInfo:
    id: str
    conversation_id: str
    parent_branch_id: str | None
    forked_from_message_id: str | None
    head_message_id: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Message:
    id: str
    conversation_id: str
    parent_message_id: str | None
    source_message_id: str | None
    role: MessageRole
    content: str
    state: MessageState
    created_at: datetime
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Translation:
    id: str
    message_id: str
    source_hash: str
    target_language: str
    provider: str
    model: str
    content: str
    state: TranslationState
    created_at: datetime
    completed_at: datetime | None = None
    error_code: str | None = None
    reused_from_id: str | None = None


@dataclass(frozen=True, slots=True)
class TranslationPreparation:
    translation: Translation
    should_enqueue: bool


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    conversation_id: str
    request_message_id: str
    response_message_id: str
    character_version_id: str
    provider: str
    model: str
    parameters: dict[str, Any]
    state: RunState
    started_at: datetime | None = None
    completed_at: datetime | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_duration_ns: int | None = None
    generation_duration_ns: int | None = None
    response_duration_ms: int | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class TelemetryMetric:
    name: str
    value: float | None
    unit: str
    source: str
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class LatestTelemetry:
    run: RunRecord
    metrics: tuple[TelemetryMetric, ...]

    @property
    def tokens_per_second(self) -> float | None:
        if (
            self.run.output_tokens is None
            or self.run.generation_duration_ns is None
            or self.run.generation_duration_ns <= 0
        ):
            return None
        return self.run.output_tokens / (self.run.generation_duration_ns / 1_000_000_000)

    def metric(self, name: str) -> TelemetryMetric:
        found = next((metric for metric in self.metrics if metric.name == name), None)
        if found is not None:
            return found
        return TelemetryMetric(name, None, "", "未取得", "収集結果がありません")


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    title: str
    media_type: str
    content: str
    content_hash: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    id: str
    document_id: str
    ordinal: int
    content: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class RagCitation:
    document_id: str
    document_title: str
    chunk_id: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class RagSearchResult:
    content: str
    citation: RagCitation
    score: float


@dataclass(frozen=True, slots=True)
class MessageCitation:
    message_id: str
    document_id: str
    document_title: str
    chunk_id: str
    start_offset: int
    end_offset: int
    content: str


@dataclass(frozen=True, slots=True)
class MessageRagUsage:
    message_id: str
    selected_document_count: int
    citations: tuple[MessageCitation, ...]


@dataclass(frozen=True, slots=True)
class RunSession:
    user_message: Message
    assistant_message: Message
    run: RunRecord
    branch_id: str


@dataclass(frozen=True, slots=True)
class ContextSummary:
    id: str
    conversation_id: str
    branch_id: str
    source_message_ids: tuple[str, ...]
    source_hash: str
    settings_hash: str
    model: str
    prompt_version: str
    content: str
    state: ContextSummaryState
    created_at: datetime
    completed_at: datetime | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class ContextSummaryPreparation:
    summary: ContextSummary
    should_generate: bool


@dataclass(frozen=True, slots=True)
class ChatMessageInput:
    role: MessageRole
    content: str
    tool_name: str | None = None
    tool_calls: tuple[ToolCallRequest, ...] = ()


@dataclass(frozen=True, slots=True)
class ChatRequest:
    model: str
    system_prompt: str
    messages: tuple[ChatMessageInput, ...]
    options: dict[str, Any] = field(default_factory=dict)
    tools: tuple[ToolDefinition, ...] = ()


@dataclass(frozen=True, slots=True)
class ChatChunk:
    content: str = ""
    done: bool = False
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_duration_ns: int | None = None
    generation_duration_ns: int | None = None
    tool_calls: tuple[ToolCallRequest, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolProviderResult:
    content: str
    item_count: int
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ToolFolderGrant:
    conversation_id: str
    root_path: Path
    granted_at: datetime


@dataclass(frozen=True, slots=True)
class ToolCallAudit:
    id: str
    conversation_id: str
    run_id: str | None
    provider: str
    tool_name: str
    input_arguments: dict[str, object]
    state: ToolCallState
    result_content: str | None
    result_item_count: int | None
    result_size_bytes: int | None
    result_sha256: str | None
    failure_reason: str | None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    id: str
    handler_name: str
    interval_seconds: int
    first_due_at: datetime
    payload: dict[str, object]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class JobRun:
    id: str
    job_id: str
    scheduled_for: datetime
    attempt: int
    state: JobRunState
    retry_of_run_id: str | None
    failure_reason: str | None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AgentExecutionLimits:
    max_cost_units: int = 5
    max_steps: int = 5
    max_duration_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_cost_units <= 5:
            raise ValueError("max_cost_units must be between 1 and 5")
        if not 1 <= self.max_steps <= 5:
            raise ValueError("max_steps must be between 1 and 5")
        if (
            not math.isfinite(self.max_duration_seconds)
            or not 0 < self.max_duration_seconds <= 60
        ):
            raise ValueError("max_duration_seconds must be between 0 and 60")


@dataclass(frozen=True, slots=True)
class AgentToolDescriptor:
    name: str
    effect: AgentToolEffect
    destination: Locality
    cost_units: int = 1
    cost_class: CostClass = CostClass.UNKNOWN

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if self.cost_units < 1:
            raise ValueError("cost_units must be positive")


@dataclass(frozen=True, slots=True)
class AgentStepRequest:
    id: str
    tool_name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class AgentActionApproval:
    id: str
    action_hash: str
    approved_at: datetime


@dataclass(frozen=True, slots=True)
class AgentToolExecution:
    result_content: str
    restore_token: str | None = None


@dataclass(frozen=True, slots=True)
class AgentPolicyResult:
    decision: AgentPolicyDecision
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRun:
    id: str
    conversation_id: str
    objective: str
    allowed_tools: tuple[str, ...]
    limits: AgentExecutionLimits
    state: AgentRunState
    failure_reason: str | None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AgentStep:
    id: str
    run_id: str
    ordinal: int
    tool_name: str
    arguments: dict[str, object]
    action_hash: str
    data_classification: DataClassification | None
    effect: AgentToolEffect | None
    destination: Locality | None
    cost_class: CostClass | None
    cost_units: int | None
    state: AgentStepState
    result_size_bytes: int | None
    result_sha256: str | None
    restore_token: str | None
    failure_reason: str | None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
