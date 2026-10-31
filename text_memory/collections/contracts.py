"""Small JSON contracts shared by the formal model stages.

The model client still requests a JSON object for compatibility with the
project's OpenAI-compatible endpoints.  These contracts make the exact shape
and the request-local identifiers explicit in the prompt; the existing Python
validators remain authoritative after decoding.
"""
from __future__ import annotations

import json


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def extraction(refs: list[str] | set[str]) -> str:
    allowed = sorted(set(refs))
    return (
        "Output contract: return exactly an object with a facts array; each item has only "
        "text (non-empty string) and evidence (a non-empty array of allowed paragraph refs). "
        "Do not add ignored, answer, explanation, or other fields. "
        "The only allowed evidence refs for this request are " + _json(allowed) + "."
    )


def gap_fill(refs: list[str] | set[str], indexes: list[int] | range) -> str:
    allowed = sorted(set(refs))
    return (
        "Output contract: return exactly an object with revised, facts, covered and no_fact arrays. "
        "Existing fact indexes are the index values given with the facts; "
        "all paragraph refs must be from " + _json(allowed) + ". "
        "Do not add instructions, commentary, or other fields."
    )


def relation_pairs(ids: list[str]) -> str:
    return (
        "Output contract: return exactly an object with a pairs array. "
        "Each pair has only id and relation; every id must appear exactly once. "
        "The allowed pair ids are " + _json(ids) + "; allowed relations are "
        + _json(["same", "another", "replaces", "conflicts", "outcome", "none"]) + "."
    )


def relation_negation(ids: list[str]) -> str:
    return (
        "Output contract: return exactly an object with a conflicts array. "
        "Every value must be one of the allowed record ids " + _json(ids) + "; do not add fields."
    )


def question_flags() -> str:
    return (
        "Output contract: return exactly an object with the three boolean fields "
        "combine, changed and ever, and no other fields."
    )
