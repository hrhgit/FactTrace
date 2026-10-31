"""Minimal persisted records and structural/provenance checks.

Sources are split into paragraph units at ingest; evidence cites unit references.
These checks prove the cited units exist. They cannot prove a model's
interpretation of a unit is semantically correct.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re

from .errors import ValidationError
from .model import encode

STATES = {"active", "disputed", "corrected", "superseded"}
POINT_KEYS = {"id", "text", "evidence", "details", "state", "change"}
# A unit reference is a source id plus a 1-based paragraph number: s_<hash>.3
UNIT_REF = re.compile(r"(s_[0-9a-f]{64})\.([1-9][0-9]*)")
# Paragraphs longer than this are split at sentence boundaries. Splitting only
# numbers parts of the original text; nothing is dropped or truncated.
UNIT_SPLIT_CHARS = 600
# A single sentence longer than this (a table or data block without punctuation) is cut at spaces.
UNIT_MAX_CHARS = 24000
_ITEM = re.compile(r"\s*(?:[-*•+]\s|\d+[.)]\s|#{1,6}\s|\*\*[^*]+\*\*)")
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def text(value: object, name: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{name} 必须是非空文字")
    return value


def digest(value: object) -> str:
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def normalize_add(payload: dict) -> dict:
    require(isinstance(payload, dict), "写入请求必须是对象")
    allowed = {"user_id", "scope_id", "session_id", "request_id", "messages"}
    require(not (set(payload) - allowed), "写入请求包含未知字段")
    result = deepcopy(payload)
    for key in ("user_id", "session_id", "request_id"):
        text(result.get(key), key)
    result.setdefault("scope_id", "default")
    text(result["scope_id"], "scope_id")
    require(isinstance(result.get("messages"), list) and bool(result["messages"]), "messages 必须是非空数组")
    for message in result["messages"]:
        require(isinstance(message, dict), "每条消息必须是对象")
        text(message.get("role"), "消息 role")
        text(message.get("content"), "消息 content")
        if "timestamp" in message:
            require(isinstance(message["timestamp"], (str, int, float)) and not isinstance(message["timestamp"], bool),
                    "timestamp 必须是原有时间文字或数字")
    # Reject non-JSON values before any model call or database mutation.
    try:
        encode(result)
    except (TypeError, ValueError) as error:
        raise ValidationError("消息必须可以保存为 JSON") from error
    return result


def _sentences(content: str, start: int, end: int) -> list[list[int]]:
    """Greedy sentence groups within one long paragraph; a single long sentence stays whole."""
    cuts = [start] + [start + match.end() for match in _SENTENCE_END.finditer(content[start:end])] + [end]
    spans, begin = [], cuts[0]
    for previous, cut in zip(cuts[1:-1], cuts[2:]):
        if cut - begin > UNIT_SPLIT_CHARS:
            spans.append([begin, previous])
            begin = previous
    spans.append([begin, end])
    return [[a, a + len(content[a:b].rstrip())] for a, b in spans if content[a:b].strip()]


def _bounded(content: str, start: int, end: int) -> list[list[int]]:
    """Cut an oversized span at its last space within the limit, or at the limit when it has none."""
    spans = []
    while end - start > UNIT_MAX_CHARS:
        cut = content.rfind(" ", start + 1, start + UNIT_MAX_CHARS + 1)
        cut = cut + 1 if cut > start else start + UNIT_MAX_CHARS
        spans.append([start, cut])
        start = cut
    return spans + [[start, end]]


def split_units(content: str) -> list[list[int]]:
    """Character spans of paragraphs; list items and headings start their own unit."""
    spans: list[list[int]] = []
    current: list[int] | None = None
    offset = 0
    for line in content.split("\n"):
        line_start, offset = offset, offset + len(line) + 1
        if not line.strip():
            current = None
            continue
        start = line_start + len(line) - len(line.lstrip())
        end = line_start + len(line.rstrip())
        if current is None or _ITEM.match(line):
            current = [start, end]
            spans.append(current)
        else:
            current[1] = end
    result = []
    for start, end in spans:
        for a, b in (_sentences(content, start, end) if end - start > UNIT_SPLIT_CHARS else [[start, end]]):
            result.extend(_bounded(content, a, b))
    return result


def source_records(payload: dict) -> list[dict]:
    return [{"id": "s_" + digest([payload["user_id"], payload["scope_id"], payload["request_id"], i]),
             "kind": "source", "session_id": payload["session_id"],
             "request_id": payload["request_id"], "position": i, "message": deepcopy(message),
             "units": split_units(message["content"])}
            for i, message in enumerate(payload["messages"])]


def unit_text(source: dict, number: int) -> str:
    start, end = source["units"][number - 1]
    return source["message"]["content"][start:end]


def unit_refs(value: object) -> set[str]:
    """Every unit reference string anywhere inside a value."""
    if isinstance(value, str):
        return {value} if UNIT_REF.fullmatch(value) else set()
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return set().union(*(unit_refs(item) for item in value)) if value else set()
    return set()


def validate_evidence(value: object, sources: dict[str, dict]) -> None:
    require(isinstance(value, list) and bool(value), "依据 evidence 必须是非空的段落编号数组")
    for ref in value:
        match = UNIT_REF.fullmatch(ref) if isinstance(ref, str) else None
        require(match is not None, "依据必须是段落编号，例如 m2.3：" + str(ref))
        source_id, number = match.group(1), int(match.group(2))
        require(source_id in sources, "依据不在允许的历史范围内：" + ref)
        require(number <= len(sources[source_id]["units"]), "原文没有这个段落：" + ref)


def validate_change(change: object, sources: dict[str, dict]) -> None:
    require(isinstance(change, dict) and set(change) == {"reason", "evidence"},
            "修改说明 change 必须包含 reason 和 evidence")
    text(change["reason"], "修改原因")
    validate_evidence(change["evidence"], sources)


def common(record: dict, allowed: set[str], sources: dict[str, dict]) -> None:
    require(not (set(record) - allowed), "记录包含未知字段：" + ", ".join(sorted(set(record) - allowed)))
    text(record.get("id"), "记录 id")
    require(isinstance(record.get("state", "active"), str) and record.get("state", "active") in STATES, "无效的记录状态")
    if "details" in record:
        require(isinstance(record["details"], dict), "details 必须是对象")
        require(all(isinstance(k, str) and isinstance(v, str) and v.strip()
                    for k, v in record["details"].items()), "details 只保存有用的文字条件")
    if "change" in record:
        validate_change(record["change"], sources)


def validate_point(record: dict, sources: dict[str, dict]) -> None:
    require(isinstance(record, dict), "点必须是对象")
    common(record, POINT_KEYS, sources)
    text(record.get("text"), "点的 text")
    validate_evidence(record.get("evidence"), sources)


