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
