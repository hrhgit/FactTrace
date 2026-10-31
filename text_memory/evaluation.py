"""Shared JSON/JSONL helpers, query validation and model identity for experiment runners.

Labels are read only by separate grading steps; nothing here opens them.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from .model import HttpJsonModel, JsonModel, encode
from .schema import digest, require, text
from .store import Scope


def jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, value: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encode(value) + "\n")
        handle.flush()


def write_json(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encode(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_query(query: dict) -> dict:
    require(isinstance(query, dict), "查询样本必须是对象")
    allowed = {"id", "user_id", "scope_id", "question", "options", "as_of", "category", "question_date"}
    require(not (set(query) - allowed), "查询输入不得包含标准答案或其他未知字段")
    for key in ("id", "user_id", "question"):
        text(query.get(key), key)
    Scope(query["user_id"], query.get("scope_id", "default"), query.get("as_of"))
    if "options" in query:
        require(isinstance(query["options"], list) and all(isinstance(v, str) for v in query["options"]), "options 必须是文字数组")
    return query


def identity(model: JsonModel | None) -> dict:
    if model is None:
        return {"not_used": True}
    config = getattr(model, "config", None)
    return {"model": config.model, "endpoint_hash": digest(config.base_url)} if config else {"injected_test_model": type(model).__name__}


class TimedModel(HttpJsonModel):
    def _post(self, *args, **kwargs):
        start, index = time.monotonic(), len(self.calls)
        try:
            return super()._post(*args, **kwargs)
        finally:
            if len(self.calls) > index:
                self.calls[index]["elapsed_seconds"] = time.monotonic() - start


def usage(calls):
    """Measure every attempt, including an invalid tool turn later corrected.

    Completion is checked by the caller's actual write/answer outcome. Accounting
    must not turn a recovered, answered case into an execution failure. A missing
    measurement stays unknown; the raw status and usage remain in each call.
    """
    fields = {"prompt_tokens", "completion_tokens", "total_tokens"}
    fields.update(key for call in calls for key, value in (call.get("usage") or {}).items()
                  if type(value) in (int, float))
    result = {}
    for key in sorted(fields):
        values = [(call.get("usage") or {}).get(key) for call in calls]
        result[key] = sum(values) if all(type(v) in (int, float) for v in values) else None
    return result

