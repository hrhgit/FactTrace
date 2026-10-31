"""Model JSON correction with explicit validation feedback.

The caller owns task rules and persistence. This module never invents actions,
changes models, truncates context or turns a failed request into a success.
"""
from __future__ import annotations

from copy import deepcopy
import time

from .errors import InvalidOutputError, ModelError, ModelOutputError, SearchLoopError, ValidationError
from .model import encode
from .schema import digest


class AgentLoop:
    MAX_ATTEMPTS = 3

    def __init__(self, model, messages, *, purpose, trace, max_attempts=MAX_ATTEMPTS):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.model, self.messages = model, messages
        self.purpose, self.trace = purpose, trace
        self.max_attempts = max_attempts
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
                         error={"type": type(error).__name__, "message": str(error),
                                **error.diagnostic()})
            raise
        except ModelError as error:
            entry["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            entry["elapsed_seconds"] = time.monotonic() - started

    def _reject(self, raw, error, instruction, *, entry, attempt, assistant=None):
        feedback = {"type": type(error).__name__, **error.diagnostic(),
                    "message": (error.model_message or
                                "The model output failed validation."),
                    "instruction": instruction}
        entry.setdefault("corrections", []).append({"raw_content": raw, "error": feedback})
        entry["error"] = {"type": type(error).__name__, "message": str(error),
                           "model_message": feedback["message"], **error.diagnostic()}
        signature = digest([raw, str(error)])
        repeated = signature in self.rejected
        entry["retry_scheduled"] = not repeated and attempt < self.max_attempts
        # The scoring runner persists model.calls, while writing also persists
        # this loop's trace. Keep content validation failures in both records.
        calls = getattr(self.model, "calls", [])
        last = calls[-1] if calls and isinstance(calls[-1], dict) else {}
        if last:
            last["output_validation"] = {**entry["error"], "attempt": attempt,
                                         "retry_scheduled": entry["retry_scheduled"]}
        channel = (f"；最后渠道 {last['channel_index']}（{last.get('requested_model', '')}）"
                   if "channel_index" in last else "")
        reason = f"{self.purpose} 阶段在 {attempt} 次输出尝试后仍失败{channel}；{type(error).__name__}：{error}"
        if repeated:
            raise SearchLoopError(reason + "；模型重复返回相同无效内容；未返回替代结果") from error
        if attempt >= self.max_attempts:
            raise ModelOutputError(reason) from error
        self.rejected.add(signature)
        self.messages.append(deepcopy(assistant) if assistant is not None
                             else {"role": "assistant", "content": raw})
        self.messages.append({"role": "user", "content": encode({"tool_error": feedback})})

    def complete(self, *, validate=None):
        """Correct malformed/invalid JSON in the original conversation."""
        return self._complete(
            lambda: self.model.complete(deepcopy(self.messages), purpose=self.purpose),
            validate=validate, serialize=encode,
            instruction="Correct the reported error using the original input above. "
                        "Preserve all useful facts and valid evidence; return the full corrected JSON object, not a patch.")

    def complete_text(self, *, validate=None, temperature=None):
        """Correct invalid text judgments using the same bounded feedback loop."""
        return self._complete(
            lambda: self.model.complete_text(deepcopy(self.messages), purpose=self.purpose, temperature=temperature),
            validate=validate, serialize=lambda value: value,
            instruction="Correct the reported error using the original input above. "
                        "Return the full corrected response in the required format.")

    def _complete(self, run, *, validate, serialize, instruction):
        for attempt in range(self.max_attempts):
            try:
                response, entry = self._turn(run)
            except InvalidOutputError as error:
                self._reject(error.raw_content, error, instruction,
                    entry=self.trace[-1], attempt=attempt + 1,
                    assistant=getattr(error, "assistant_message", None))
                continue
            try:
                if validate is not None:
                    validate(response)
                return response
            except ValidationError as error:
                self._reject(serialize(response), error, instruction, entry=entry, attempt=attempt + 1)

