"""Errors are explicit; failures never turn into substitute memories."""

from __future__ import annotations

from copy import deepcopy
import json


_MISSING = object()


def _json_safe(value):
    """Keep diagnostics serializable without changing the reported value."""
    if value is _MISSING:
        return _MISSING
    if value is None or isinstance(value, (str, int, float, bool)):
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            return repr(value)
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)
    return deepcopy(value)


class _DiagnosticMixin:
    def _set_diagnostic(self, *, path=None, actual=_MISSING, expected=None, source=None,
                        model_message=None, details=None):
        self.path = path
        self.actual = actual
        self.expected = expected
        self.source = source
        self.model_message = model_message
        self.details = dict(details or {})

    def diagnostic(self) -> dict:
        """Return only JSON-safe, concrete details suitable for model feedback."""
        result = {}
        for name, value in (("path", self.path), ("actual", self.actual),
                            ("expected", self.expected), ("source", self.source)):
            if value is not None and value is not _MISSING:
                result[name] = _json_safe(value)
        for name, value in self.details.items():
            if name not in result and value is not None:
                result[name] = _json_safe(value)
        return result


class MemoryErrorBase(Exception):
    pass


class ValidationError(_DiagnosticMixin, MemoryErrorBase):
    def __init__(self, message: str, *, path=None, actual=_MISSING, expected=None,
                 source=None, model_message=None, details=None):
        super().__init__(message)
        self.message = message
        self._set_diagnostic(path=path, actual=actual, expected=expected, source=source,
                             model_message=model_message, details=details)


class NotFoundError(MemoryErrorBase):
    pass


class ConflictError(MemoryErrorBase):
    pass


class ConfigurationError(MemoryErrorBase):
    pass


class ModelError(MemoryErrorBase):
    pass


class ModelOutputError(ModelError):
    """The model returned content that the caller could not validate."""
    pass


class InvalidOutputError(_DiagnosticMixin, ModelError):
    def __init__(self, raw_content: str, message: str, *, assistant_message: dict | None = None,
                 path=None, actual=_MISSING, expected=None, source=None,
                 model_message=None, details=None):
        super().__init__(message)
        self.raw_content = raw_content
        self.message = message
        self.assistant_message = assistant_message
        self._set_diagnostic(path=path, actual=actual, expected=expected, source=source,
                             model_message=model_message, details=details)


class SearchLoopError(ModelError):
    pass
