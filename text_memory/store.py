"""SQLite persistence with atomic, append-only revisions and scoped reads."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterator

from .errors import ConflictError, NotFoundError, ValidationError
from .model import encode
from .schema import (UNIT_REF, digest, normalize_add, require, source_records,
                     split_units, text, unit_refs, validate_change, validate_point)

OBJECT_KINDS = {"point"}


@dataclass(frozen=True)
class Scope:
    user_id: str
    scope_id: str = "default"
    as_of: int | None = None

    def __post_init__(self):
        text(self.user_id, "user_id")
        text(self.scope_id, "scope_id")
        require(self.as_of is None or (type(self.as_of) is int and self.as_of >= 0), "as_of 必须是非负修订号")


def terms(value: str) -> list[str]:
    """Deterministic bilingual lexical index, not a semantic model.

    Han unigrams/bigrams/trigrams support Chinese without a downloaded model;
    the same terms are used to compare full questions and fact text.
    """
    result: set[str] = set()
    for word in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", value.casefold()):
        if re.fullmatch(r"[a-z0-9]+", word):
            result.add("w" + word)
        else:
            for width in (1, 2, 3):
                for start in range(len(word) - width + 1):
                    result.add("h" + word[start:start + width].encode("utf-8").hex())
    return sorted(result)


class MemoryStore:
    def __init__(self, path: str | Path = "outputs/memory.sqlite"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Wait for another writer's transaction; this is only a database lock
        # wait, not a model deadline, retry or substitute result.
        self.db = sqlite3.connect(str(path), isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS scopes (
                user_id TEXT NOT NULL, scope_id TEXT NOT NULL, revision INTEGER NOT NULL,
                PRIMARY KEY(user_id, scope_id));
            CREATE TABLE IF NOT EXISTS requests (
                user_id TEXT NOT NULL, scope_id TEXT NOT NULL, request_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL, result TEXT NOT NULL,
                PRIMARY KEY(user_id, scope_id, request_id));
            CREATE TABLE IF NOT EXISTS sources (
                user_id TEXT NOT NULL, scope_id TEXT NOT NULL, id TEXT NOT NULL,
                revision INTEGER NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(user_id, scope_id, id));
            CREATE TABLE IF NOT EXISTS objects (
                user_id TEXT NOT NULL, scope_id TEXT NOT NULL, kind TEXT NOT NULL,
                id TEXT NOT NULL, revision INTEGER NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(user_id, scope_id, kind, id, revision));
        """)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def revision(self, scope: Scope) -> int:
        row = self.db.execute("SELECT revision FROM scopes WHERE user_id=? AND scope_id=?",
                              (scope.user_id, scope.scope_id)).fetchone()
        current = row[0] if row else 0
        if scope.as_of is not None:
            require(scope.as_of <= current, "as_of 超过已写入的历史")
            return scope.as_of
        return current

    def snapshot(self, scope: Scope) -> Scope:
        return Scope(scope.user_id, scope.scope_id, self.revision(scope))

    def receipt(self, payload: dict) -> dict | None:
        payload = normalize_add(payload)
        row = self.db.execute("SELECT payload_hash,result FROM requests WHERE user_id=? AND scope_id=? AND request_id=?",
                              (payload["user_id"], payload["scope_id"], payload["request_id"])).fetchone()
        if row:
            if row["payload_hash"] != digest(payload):
                raise ConflictError("同一 request_id 已用于不同的材料")
            return json.loads(row["result"])
        return None


    def get(self, scope: Scope, kind: str, ident: str) -> dict:
        revision = self.revision(scope)
        if kind == "source":
            row = self.db.execute("SELECT data,revision FROM sources WHERE user_id=? AND scope_id=? AND id=? AND revision<=?",
                                  (scope.user_id, scope.scope_id, ident, revision)).fetchone()
        elif kind in OBJECT_KINDS:
            row = self.db.execute("SELECT data,revision FROM objects WHERE user_id=? AND scope_id=? AND kind=? AND id=? AND revision<=? ORDER BY revision DESC LIMIT 1",
                                  (scope.user_id, scope.scope_id, kind, ident, revision)).fetchone()
        else:
            raise ValidationError("未知的记录类型")
        if row is None:
            raise NotFoundError("在允许的历史范围内没有找到记录：" + ident)
        record = json.loads(row["data"]) | {"kind": kind, "revision": row["revision"]}
        if kind == "source" and "units" not in record:
            # Sources stored before paragraph units are split the same way on read.
            record["units"] = split_units(record["message"]["content"])
        return record

    def maybe_get(self, scope: Scope, kind: str, ident: str) -> dict | None:
        try:
            return self.get(scope, kind, ident)
        except NotFoundError:
            return None

    def records(self, scope: Scope, kind: str) -> Iterator[dict]:
        revision = self.revision(scope)
        if kind == "source":
            rows = self.db.execute("SELECT id FROM sources WHERE user_id=? AND scope_id=? AND revision<=? ORDER BY revision,id",
                                   (scope.user_id, scope.scope_id, revision))
        else:
            require(kind in OBJECT_KINDS, "未知的记录类型")
            rows = self.db.execute("SELECT DISTINCT id FROM objects WHERE user_id=? AND scope_id=? AND kind=? AND revision<=? ORDER BY id",
                                   (scope.user_id, scope.scope_id, kind, revision))
        for row in rows:
            yield self.get(scope, kind, row["id"])


    @staticmethod
    def _body(record: dict) -> dict:
        return {key: value for key, value in record.items() if key not in {"kind", "revision"}}

    def commit(self, payload: dict, plan: dict, *, expected_revision: int) -> dict:
        payload = normalize_add(payload)
        require(type(expected_revision) is int and expected_revision >= 0, "expected_revision 必须是非负修订号")
        scope = Scope(payload["user_id"], payload["scope_id"])
        require(isinstance(plan, dict) and not (set(plan) - {"points"}), "写入只接受完整事实 points")
        points = plan.get("points", [])
        require(isinstance(points, list), "points 必须是数组")
        require(all(isinstance(p, dict) and isinstance(p.get("id"), str) for p in points), "事实需要文字 id")
        require(len({p["id"] for p in points}) == len(points), "写入包含重复事实 id")
        staged = source_records(payload)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self.receipt(payload)
            if prior:
                self.db.execute("COMMIT")
                return prior
            current = self.revision(scope)
            if current != expected_revision:
                raise ConflictError("历史在整理期间发生变化；本次未写入，请重新读取后再提交")
            snapshot = Scope(scope.user_id, scope.scope_id, current)
            sources = {record["id"]: record for record in staged}
            for ref in sorted(unit_refs(plan)):
                ident = UNIT_REF.fullmatch(ref).group(1)
                if ident not in sources:
                    sources[ident] = self.get(snapshot, "source", ident)
            for point in points:
                validate_point(point, sources)
                previous = self.maybe_get(snapshot, "point", point["id"])
                if previous and self._body(previous) != point:
                    validate_change(point.get("change"), sources)
            revision = current + 1
            for record in staged:
                self.db.execute("INSERT INTO sources VALUES(?,?,?,?,?)",
                    (scope.user_id, scope.scope_id, record["id"], revision, encode(record)))
            for point in points:
                self.db.execute("INSERT INTO objects VALUES(?,?,?,?,?,?)",
                    (scope.user_id, scope.scope_id, "point", point["id"], revision, encode(point)))
            self.db.execute("INSERT INTO scopes VALUES(?,?,?) ON CONFLICT(user_id,scope_id) DO UPDATE SET revision=excluded.revision",
                            (scope.user_id, scope.scope_id, revision))
            receipt = {"success": True, "user_id": scope.user_id, "scope_id": scope.scope_id,
                       "session_id": payload["session_id"], "request_id": payload["request_id"],
                       "revision": revision, "source_ids": [record["id"] for record in staged],
                       "point_ids": [point["id"] for point in points]}
            self.db.execute("INSERT INTO requests VALUES(?,?,?,?,?)",
                (scope.user_id, scope.scope_id, payload["request_id"], digest(payload), encode(receipt)))
            self.db.execute("COMMIT")
            return receipt
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
