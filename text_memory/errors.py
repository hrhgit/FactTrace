"""Errors are explicit; failures never turn into substitute memories."""


class MemoryErrorBase(Exception):
    pass


class ValidationError(MemoryErrorBase):
    pass


class NotFoundError(MemoryErrorBase):
    pass


class ConflictError(MemoryErrorBase):
    pass


class ConfigurationError(MemoryErrorBase):
    pass


class ModelError(MemoryErrorBase):
    pass


class InvalidOutputError(ModelError):
    def __init__(self, raw_content: str, message: str, *, assistant_message: dict | None = None):
        super().__init__(message)
        self.raw_content = raw_content
        self.message = message
        self.assistant_message = assistant_message


class SearchLoopError(MemoryErrorBase):
    pass
