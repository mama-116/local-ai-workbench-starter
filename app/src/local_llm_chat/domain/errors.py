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


class ModelUnavailable(AppError):
    pass


class PersistenceError(AppError):
    pass


class ConversationNotFound(AppError):
    pass
