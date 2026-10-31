"""Model JSON correction with explicit validation feedback.

The caller owns task rules and persistence. This module never invents actions,
changes models, truncates context or turns a failed request into a success.
"""
from __future__ import annotations

from copy import deepcopy
import time

from .errors import InvalidOutputError, ModelError, SearchLoopError, ValidationError
from .model import encode
from .schema import digest


class AgentLoop:
    def __init__(self, model, messages, *, purpose, trace):
        self.model, self.messages = model, messages
        self.purpose, self.trace = purpose, trace
        self.rejected = set()

    def _turn(self, run):
        entry = {"purpose": self.purpose}
        self.trace.append(entry)
        started = time.monotonic()
        try:
            result = run()
            entry["response"] = deepcopy(result)
            return result, entry
        except InvalidOutputError as error:
            entry.update(raw_content=error.raw_content,
                         error={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            entry["elapsed_seconds"] = time.monotonic() - started

    def _reject(self, raw, error, instruction, *, entry, assistant=None):
        feedback = {"type": type(error).__name__, "message": str(error), "instruction": instruction}
        entry.setdefault("corrections", []).append({"raw_content": raw, "error": feedback})
        signature = digest([raw, str(error)])
        if signature in self.rejected:
            raise SearchLoopError("模型重复返回相同无效内容且未取得进展；未写入替代结果") from error
        self.rejected.add(signature)
        self.messages.append(deepcopy(assistant) if assistant is not None
                             else {"role": "assistant", "content": raw})
        self.messages.append({"role": "user", "content": encode({"tool_error": feedback})})

    def complete(self, *, validate=None):
        """Correct malformed/invalid JSON in the original conversation."""
        while True:
            try:
                response, entry = self._turn(lambda: self.model.complete(
                    deepcopy(self.messages), purpose=self.purpose))
            except InvalidOutputError as error:
                self._reject(error.raw_content, error,
                    "Correct the JSON using the original input and instructions above. "
                    "Return the full corrected object, not a patch.", entry=self.trace[-1],
                    assistant=getattr(error, "assistant_message", None))
                continue
            try:
                if validate is not None:
                    validate(response)
                return response
            except ValidationError as error:
                self._reject(encode(response), error,
                    "Correct the reported validation error using the original input above. "
                    "Preserve all useful facts and valid evidence; return the full corrected object.", entry=entry)

