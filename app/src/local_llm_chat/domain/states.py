from enum import StrEnum


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


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
