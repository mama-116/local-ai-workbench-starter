class AppError(Exception):
    """Base error safe to translate at the presentation boundary."""


class ValidationError(AppError):
    pass


class FreeOperationBlocked(AppError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class OllamaUnavailable(AppError):
    pass


class ToolUseUnavailable(AppError):
    pass


class AgentToolFailure(AppError):
    def __init__(self, message: str, restore_token: str | None = None) -> None:
        super().__init__(message)
        self.restore_token = restore_token


class ComputerActionDenied(AppError):
    """A fail-closed rejection raised before any desktop action is performed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ModelUnavailable(AppError):
    pass


class PersistenceError(AppError):
    pass


class ConversationNotFound(AppError):
    pass
