"""The memory model marks what a question needs; the program decides the material from these flags.

The model does not answer or reason about the question: it only says what the program should gather
(outputs/typed-actions-20261029; prompt chosen offline in outputs/question-flags-prompt-20261029).
"""
from copy import deepcopy
import time

from ..errors import InvalidOutputError, ModelError
from ..model import encode
from .prompts import QUESTION_FLAGS
from .contracts import question_flags

# combine: the answer is put together from several records; changed: the value may have been updated, the latest
# statement decides; ever: whether something has ever happened.
FLAGS = ("combine", "changed", "ever")
# Questions combining several records keep twice the ranked facts (32 at the default 16).
BROAD = ("combine",)
# Questions whose answer depends on when things were said get session blocks in chronological order.
CHRONOLOGICAL = ("combine", "changed")
MAX_ATTEMPTS = 3
# Seconds to wait before the next attempt after a model error (e.g. HTTP 429 that outlasted the client's own retries).
ERROR_DELAYS = (2.0, 5.0)


def check_question(model, question, *, sleep=time.sleep):
    """Return ``(flags, None)`` or raise after the judgment retries are used.

    A failed judgment must never become an all-false judgment.  The caller can
    record the raised error and retry the whole search without exporting
    material chosen under an unknown policy.
    """
    messages = [{"role": "system", "content": QUESTION_FLAGS},
                {"role": "user", "content": encode({"question": question,
                                                        "output_contract": question_flags()})}]
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            reply = model.complete(deepcopy(messages), purpose="question_flags")
            if not (isinstance(reply, dict) and set(reply) == set(FLAGS)
                    and all(isinstance(reply[k], bool) for k in FLAGS)):
                raise InvalidOutputError(encode(reply),
                    "invalid reply: question_flags 必须完整返回 combine、changed、ever 三个布尔值",
                    path="$", actual=reply,
                    expected="an object with exactly boolean combine, changed and ever fields",
                    source=question,
                    model_message=("Return exactly the three boolean fields combine, changed and ever "
                                   "and no other fields."))
            return {k: reply[k] for k in FLAGS}, None
        except ModelError as failure:
            last_error = failure
            if isinstance(failure, InvalidOutputError):
                feedback = {"type": type(failure).__name__, **failure.diagnostic(),
                            "message": (failure.model_message or
                                        "Return the complete JSON object required by the output contract."),
                            "instruction": ("Return the complete JSON object required by the output contract. "
                                             "Do not add explanations.")}
                calls = getattr(model, "calls", [])
                if calls and isinstance(calls[-1], dict):
                    calls[-1]["output_validation"] = {**feedback, "message": str(failure),
                                                      "model_message": feedback["message"],
                                                      "attempt": attempt + 1,
                                                      "retry_scheduled": attempt + 1 < MAX_ATTEMPTS}
                if attempt + 1 < MAX_ATTEMPTS:
                    messages.extend([
                        deepcopy(failure.assistant_message) if failure.assistant_message is not None
                        else {"role": "assistant", "content": failure.raw_content},
                        {"role": "user", "content": encode({"tool_error": feedback})},
                    ])
            if attempt + 1 < MAX_ATTEMPTS:
                sleep(ERROR_DELAYS[min(attempt, len(ERROR_DELAYS) - 1)])
    if last_error is None:
        raise ModelError(f"question_flags 在 {MAX_ATTEMPTS} 次尝试后没有返回结果")
    calls = getattr(model, "calls", [])
    last = calls[-1] if calls and isinstance(calls[-1], dict) else {}
    channel = (f"；最后渠道 {last['channel_index']}（{last.get('requested_model', '')}）"
               if "channel_index" in last else "")
    raise ModelError(f"question_flags 在 {MAX_ATTEMPTS} 次尝试后仍失败{channel}：{last_error}") from last_error


def mark_question(model, question):
    """Return the validated flags; a failed judgment propagates to the caller."""
    return check_question(model, question)[0]


def material_notes(flags):
    """Lines placed before the facts; they describe the material, not the answer."""
    notes = []
    if any(flags.get(k) for k in CHRONOLOGICAL):
        notes.append("Sessions below are in chronological order (earliest first).")
    if flags.get("changed"):
        notes.append("When statements about the same thing differ, the later one is the current one.")
    if flags.get("ever"):
        notes.append("If some records below affirm it and others deny it, point out that they conflict.")
    return notes
