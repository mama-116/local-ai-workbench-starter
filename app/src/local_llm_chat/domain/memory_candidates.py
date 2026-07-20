from __future__ import annotations

from dataclasses import dataclass

from local_llm_chat.domain.states import (
    MemoryApprovalState,
    MemoryCandidateDisposition,
    MemoryCandidateReason,
    MemoryCardinality,
    MemoryEvidenceMode,
    MemoryKind,
)


MAX_MEMORY_CANDIDATES_PER_MESSAGE = 8


@dataclass(frozen=True, slots=True)
class MemoryCandidateRequest:
    conversation_id: str
    branch_id: str
    source_message_id: str
    model_name: str
    content: str
    author_subject_id: str
    allowed_subject_ids: frozenset[str]
    allowed_knowledge_character_ids: frozenset[str]
    known_by_character_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class MemoryCandidateDraft:
    subject_id: str | None
    kind: MemoryKind
    slot: str
    value: str
    evidence_mode: MemoryEvidenceMode
    evidence_start: int
    evidence_end: int


@dataclass(frozen=True, slots=True)
class MemoryTemplate:
    kind: MemoryKind
    slot: str
    cardinality: MemoryCardinality
    requires_confirmation: bool
    evidence_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    conversation_id: str
    branch_id: str
    source_message_id: str
    subject_id: str | None
    kind: MemoryKind
    slot: str
    value: str
    cardinality: MemoryCardinality | None
    evidence_text: str
    evidence_start: int
    evidence_end: int
    known_by_character_ids: frozenset[str]
    disposition: MemoryCandidateDisposition
    reason: MemoryCandidateReason
    initial_approval: MemoryApprovalState | None
