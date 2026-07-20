from enum import StrEnum


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageState(StrEnum):
    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RunState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TranslationState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolCallState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"


class ContextSummaryState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRunState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRunState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DENIED = "denied"


class AgentStepState(StrEnum):
    PROPOSED = "proposed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"
    RESTORED = "restored"
    RESTORE_FAILED = "restore_failed"


class AgentToolEffect(StrEnum):
    READ = "read"
    REVERSIBLE_WRITE = "reversible_write"
    EXTERNAL_POST = "external_post"
    EXTERNAL_DELETE = "external_delete"
    BILLING = "billing"


class DataClassification(StrEnum):
    PRIVATE = "private"
    LOCAL_OPERATIONAL = "local_operational"
    SEARCH_QUERY = "search_query"
    PUBLIC_RESULT = "public_result"


class AgentPolicyDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class Locality(StrEnum):
    LOCAL = "local"
    LAN = "lan"
    REMOTE = "remote"
    UNKNOWN = "unknown"


class CostClass(StrEnum):
    NO_CHARGE = "no_charge"
    FREE_TIER = "free_tier"
    PAID = "paid"
    UNKNOWN = "unknown"


class ComputerActionType(StrEnum):
    LAUNCH_ALLOWED_APP = "launch_allowed_app"
    CLICK_UIA_ELEMENT = "click_uia_element"
    TYPE_PLAIN_TEXT = "type_plain_text"


class ComputerUseRunState(StrEnum):
    PENDING = "pending"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DENIED = "denied"


class ComputerActionState(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DENIED = "denied"


class ComputerPolicyDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class DesktopIntegrityLevel(StrEnum):
    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SYSTEM = "system"
    PROTECTED = "protected"


class MemoryKind(StrEnum):
    PREFERENCE = "preference"
    SAFETY_CONSTRAINT = "safety_constraint"
    GOAL = "goal"


class MemoryCardinality(StrEnum):
    SINGLE = "single"
    MULTIPLE = "multiple"


class MemoryApprovalState(StrEnum):
    AUTO_SAVED = "auto_saved"
    CONFIRMED = "confirmed"
    PENDING_CONFIRMATION = "pending_confirmation"
    REJECTED = "rejected"
    UNDONE = "undone"


class MemoryFactState(StrEnum):
    ACTIVE = "active"
    HISTORICAL = "historical"


class TurnMode(StrEnum):
    STORY = "story"
    ROUND_TABLE = "round_table"
    SPOTLIGHT = "spotlight"


class TurnBatchState(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"


class TurnRepairState(StrEnum):
    NOT_NEEDED = "not_needed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TurnSpeakerKind(StrEnum):
    CHARACTER = "character"
    GUEST = "guest"
    NARRATOR = "narrator"
    UNRESOLVED = "unresolved"


class MemoryEvidenceMode(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    HYPOTHETICAL = "hypothetical"
    QUOTED = "quoted"
    NEGATED = "negated"


class MemoryCandidateDisposition(StrEnum):
    AUTO_SAVE = "auto_save"
    REQUIRE_CONFIRMATION = "require_confirmation"
    BLOCK = "block"


class MemoryCandidateReason(StrEnum):
    EXPLICIT_LOW_RISK = "explicit_low_risk"
    TEMPLATE_REQUIRES_CONFIRMATION = "template_requires_confirmation"
    RESTRICTED_KNOWLEDGE_SCOPE = "restricted_knowledge_scope"
    NON_EXPLICIT_EVIDENCE = "non_explicit_evidence"
    SUBJECT_UNKNOWN = "subject_unknown"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    TEMPLATE_UNKNOWN = "template_unknown"
    INVALID_VALUE = "invalid_value"
    UNSAFE_EVIDENCE = "unsafe_evidence"
    UNSUPPORTED_EXPLICIT_FORM = "unsupported_explicit_form"
    THIRD_PARTY_SUBJECT = "third_party_subject"
