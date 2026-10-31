"""Shared local usage, budgets and channel settings; never stores API keys or prompts."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from threading import RLock
import time
from urllib.parse import urlsplit
import uuid

from .errors import ConfigurationError, ModelError


PRICE_SOURCE = "https://developers.openai.com/api/docs/pricing"
PRICE_DATE = "2026-10-30"
# USD per million text tokens, Standard processing. These are reference prices,
# including for gateways/subscriptions; they are not the provider's actual bill.
# input, cached input, output
REFERENCE_PRICES = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-astra": (10.00, 1.00, 50.00),
}


class UsageLimitError(ModelError):
    pass


class ChannelConcurrencyError(ModelError):
    """The channel is enabled but its live concurrency limit is full."""
    pass


def usage_path_from_env() -> str:
    return os.environ.get("MODEL_USAGE_DB") or str(Path(__file__).resolve().parents[1] / "state/model-usage.sqlite")


def key_identity(base_url: str, api_key: str) -> str:
    # The same key at the same endpoint shares accounting across MEMORY/ANSWER/
    # JUDGE and models. A rotated key gets its own counters even with the same label.
    raw = base_url.rstrip("/") + "\0" + api_key
    return "key_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def default_source(base_url: str) -> str:
    return urlsplit(base_url).hostname or "model-service"


def _tokens(value):
    return value if type(value) is int and value >= 0 else None


def normalize_usage(usage: object) -> dict:
    """OpenAI Chat usage; missing measurements stay None, never invented zeros."""
    raw = usage if isinstance(usage, dict) else {}
    prompt = _tokens(raw.get("prompt_tokens", raw.get("input_tokens")))
    completion = _tokens(raw.get("completion_tokens", raw.get("output_tokens")))
    total = _tokens(raw.get("total_tokens"))
    if total is not None and total < (prompt or 0) + (completion or 0):
        # A compatible service can report a placeholder total of zero. It cannot
        # invalidate independently measured input/output tokens.
        total = None
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    details = raw.get("prompt_tokens_details", raw.get("input_tokens_details"))
    cached = _tokens(details.get("cached_tokens")) if isinstance(details, dict) else None
    details = raw.get("completion_tokens_details", raw.get("output_tokens_details"))
    reasoning = _tokens(details.get("reasoning_tokens")) if isinstance(details, dict) else None
    # Some compatible gateways omit details. Charging input at the uncached
    # reference rate is an explicitly labelled estimate in that case.
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total,
            "cached_input_tokens": cached, "reasoning_tokens": reasoning}


def reference_price(model: str) -> tuple[str, tuple] | None:
    if not isinstance(model, str):
        return None
    name = model
    for prefix in ("openai/", "codex/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    rates = REFERENCE_PRICES.get(name)
    return (name, rates) if rates else None


def estimate_cost_nusd(usage: dict, model: str) -> int | None:
    """Integer nano-dollars avoid accumulated rounding at budget boundaries."""
    price = reference_price(model)
    prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
    if price is None or prompt is None or completion is None:
        return None
    cached = usage["cached_input_tokens"] or 0
    if cached > prompt:
        return None
    input_rate, cache_rate, output_rate = (Decimal(str(v)) for v in price[1])
    # Official long-context multiplier for these GPT-6 text models.
    if price[0].startswith("gpt-6") and prompt > 272_000:
        input_rate *= 2
        cache_rate *= 2
        output_rate *= Decimal("1.5")
    cost = ((prompt - cached) * input_rate + cached * cache_rate + completion * output_rate) * 1000
    return int(cost.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _now():
    return datetime.now(timezone.utc).isoformat()


class UsageLedger:
    """SQLite transactions coordinate both threads and separate local processes.

    Attempts are reserved before sending. Counters and the result are committed
    together after each attempt, including invalid output and transport errors.
    A process that exits mid-request leaves an explicit unfinished attempt.
    """
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser().resolve() if path else None
        self._lock = RLock()
        try:
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(self.path) if self.path else ":memory:",
                                      timeout=30, check_same_thread=False)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS key_usage (
                    key_id TEXT PRIMARY KEY, source TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_nusd INTEGER NOT NULL DEFAULT 0,
                    max_requests INTEGER, max_total_tokens INTEGER, max_cost_nusd INTEGER,
                    reference_model TEXT,
                    max_concurrency INTEGER,
                    active_requests INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS model_attempts (
                    id TEXT PRIMARY KEY, key_id TEXT NOT NULL, channel_group TEXT NOT NULL,
                    model TEXT NOT NULL, purpose TEXT NOT NULL, started_at TEXT NOT NULL,
                    finished_at TEXT, status TEXT, usage_json TEXT, estimated_cost_nusd INTEGER,
                    price_model TEXT, exhausted_fallback INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS attempts_key ON model_attempts(key_id);
                CREATE TABLE IF NOT EXISTS channel_settings (
                    channel_group TEXT PRIMARY KEY, strategy TEXT NOT NULL, priority TEXT NOT NULL,
                    cursor INTEGER NOT NULL DEFAULT 0, generation INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS channel_changes (
                    id INTEGER PRIMARY KEY, changed_at TEXT NOT NULL, target TEXT NOT NULL,
                    kind TEXT NOT NULL, before_json TEXT NOT NULL, after_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_settings (
                    setting_group TEXT PRIMARY KEY,
                    max_concurrency INTEGER,
                    changed_at TEXT NOT NULL
                );
            """)
            with self.transaction() as db:
                columns = {row["name"] for row in db.execute("PRAGMA table_info(model_attempts)")}
                if "exhausted_fallback" not in columns:
                    db.execute("ALTER TABLE model_attempts ADD COLUMN exhausted_fallback INTEGER NOT NULL DEFAULT 0")
                columns = {row["name"] for row in db.execute("PRAGMA table_info(key_usage)")}
                if "max_concurrency" not in columns:
                    db.execute("ALTER TABLE key_usage ADD COLUMN max_concurrency INTEGER")
                if "active_requests" not in columns:
                    db.execute("ALTER TABLE key_usage ADD COLUMN active_requests INTEGER NOT NULL DEFAULT 0")
            if self.path:
                self.path.chmod(0o600)
        except (OSError, sqlite3.Error) as error:
            raise ConfigurationError("模型用量数据库无法打开") from error

    @contextmanager
    def transaction(self):
        with self._lock:
            try:
                self.db.execute("BEGIN IMMEDIATE")
                yield self.db
                self.db.commit()
            except sqlite3.Error as error:
                self.db.rollback()
                raise ConfigurationError("模型用量数据库读写失败") from error
            except BaseException:
                self.db.rollback()
                raise

    def register(self, key_id, source, *, explicit=True):
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO key_usage (key_id, source) VALUES (?,?)", (key_id, source))
            if explicit:
                db.execute("UPDATE key_usage SET source=? WHERE key_id=?", (source, key_id))
            return db.execute("SELECT source FROM key_usage WHERE key_id=?", (key_id,)).fetchone()["source"]

    def settings(self, group, default_strategy, count, *, advance=False):
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO channel_settings(channel_group,strategy,priority) VALUES (?,?,?)",
                       (group, "", "[]"))
            row = dict(db.execute("SELECT * FROM channel_settings WHERE channel_group=?", (group,)).fetchone())
            row["strategy"] = row["strategy"] or default_strategy
            priority = json.loads(row["priority"])
            priority = [i for i in priority if type(i) is int and 1 <= i <= count]
            priority += [i for i in range(1, count + 1) if i not in priority]
            row["priority"] = priority
            if advance and row["strategy"] == "round_robin":
                start = row["cursor"] % count
                row["priority"] = priority[start:] + priority[:start]
                db.execute("UPDATE channel_settings SET cursor=? WHERE channel_group=?", ((start + 1) % count, group))
            return {k: row[k] for k in ("strategy", "priority", "generation")}

    def update_settings(self, group, *, priority=None, strategy=None):
        if strategy is not None and strategy not in {"failover", "round_robin"}:
            raise ConfigurationError("模型渠道策略必须是 failover 或 round_robin")
        with self.transaction() as db:
            row = db.execute("SELECT * FROM channel_settings WHERE channel_group=?", (group,)).fetchone()
            if row is None:
                raise ConfigurationError("请先加载模型渠道配置")
            before = {"strategy": row["strategy"], "priority": json.loads(row["priority"])}
            after = {"strategy": strategy or row["strategy"],
                     "priority": priority if priority is not None else before["priority"]}
            db.execute("""UPDATE channel_settings SET strategy=?,priority=?,cursor=0,generation=generation+1
                          WHERE channel_group=?""",
                       (strategy or row["strategy"], json.dumps(priority) if priority is not None else row["priority"], group))
            db.execute("INSERT INTO channel_changes(changed_at,target,kind,before_json,after_json) VALUES (?,?,?,?,?)",
                       (_now(), group, "priority", json.dumps(before), json.dumps(after)))

    def update_limits(self, key_id, **changes):
        columns = {"max_requests": "max_requests", "max_total_tokens": "max_total_tokens",
                   "max_cost_usd": "max_cost_nusd", "reference_model": "reference_model",
                   "max_concurrency": "max_concurrency"}
        if not changes or set(changes) - columns.keys():
            raise ConfigurationError("未指定有效的用量上限")
        values = []
        for name, value in changes.items():
            if value is not None:
                if name == "reference_model":
                    if reference_price(value) is None:
                        raise ConfigurationError("未收录这个 OpenAI 参考模型的单价")
                elif name == "max_cost_usd":
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ConfigurationError("金额上限必须是非负数字")
                elif name == "max_concurrency":
                    if type(value) is not int or value < 0:
                        raise ConfigurationError("渠道并发上限必须是非负整数")
                elif type(value) is not int or value < 0:
                    raise ConfigurationError("请求数和 token 上限必须是非负整数")
            values.append(None if value is None else round(value * 1e9) if name == "max_cost_usd" else value)
        with self.transaction() as db:
            current = db.execute("SELECT * FROM key_usage WHERE key_id=?", (key_id,)).fetchone()
            if current is None:
                raise ConfigurationError("没有找到这个 key 来源")
            statement = ",".join(columns[name] + "=?" for name in changes)
            changed = db.execute("UPDATE key_usage SET " + statement + " WHERE key_id=?", (*values, key_id))
            if changed.rowcount != 1:
                raise ConfigurationError("没有找到这个 key 来源")
            before = {name: current[columns[name]] for name in changes}
            after = dict(zip(changes, values))
            db.execute("INSERT INTO channel_changes(changed_at,target,kind,before_json,after_json) VALUES (?,?,?,?,?)",
                       (_now(), key_id, "limits", json.dumps(before), json.dumps(after)))

    @staticmethod
    def _reached_limit(state):
        for field, cap in (("requests", "max_requests"), ("total_tokens", "max_total_tokens"),
                           ("estimated_cost_nusd", "max_cost_nusd")):
            if state[cap] is not None and state[field] >= state[cap]:
                return cap
        return None

    def _all_exhausted(self, db, key_ids):
        if not key_ids:
            return False
        for key_id in set(key_ids):
            state = db.execute("SELECT * FROM key_usage WHERE key_id=?", (key_id,)).fetchone()
            if state is None:
                raise ConfigurationError("请先登记模型 key 来源")
            if self._reached_limit(state) is None:
                return False
        return True

    def all_exhausted(self, key_ids):
        """Read all channel limits in one transaction, including live changes."""
        with self.transaction() as db:
            return self._all_exhausted(db, key_ids)

    def begin(self, key_id, group, model, purpose, *, exhausted_fallback_keys=None):
        with self.transaction() as db:
            state = db.execute("SELECT * FROM key_usage WHERE key_id=?", (key_id,)).fetchone()
            if state is None:
                raise ConfigurationError("请先登记模型 key 来源")
            terminal_fallback = exhausted_fallback_keys is not None
            if terminal_fallback:
                if key_id not in exhausted_fallback_keys or not self._all_exhausted(db, exhausted_fallback_keys):
                    raise UsageLimitError("尚有渠道未达到上限，不能启用超额兜底")
            elif cap := self._reached_limit(state):
                raise UsageLimitError("该 key 来源已达到用量上限：" + cap.replace("_nusd", "_usd"))
            concurrency = state["max_concurrency"]
            if concurrency == 0:
                raise UsageLimitError("该模型渠道已禁用")
            if concurrency is not None and state["active_requests"] >= concurrency:
                raise ChannelConcurrencyError("该模型渠道当前并发已满")
            attempt_id = uuid.uuid4().hex
            # Freeze the reference used by this attempt while the request is running.
            price_model = state["reference_model"] or model
            if state["max_cost_nusd"] is not None and reference_price(price_model) is None:
                raise UsageLimitError("该 key 来源已设置美元上限，但当前模型没有参考单价；请设置 reference_model")
            db.execute("""INSERT INTO model_attempts
                (id,key_id,channel_group,model,purpose,started_at,price_model,exhausted_fallback)
                VALUES (?,?,?,?,?,?,?,?)""",
                (attempt_id, key_id, group, model, purpose, _now(), price_model, int(terminal_fallback)))
            db.execute("UPDATE key_usage SET requests=requests+1 WHERE key_id=?", (key_id,))
            db.execute("UPDATE key_usage SET active_requests=active_requests+1 WHERE key_id=?", (key_id,))
            return attempt_id

    def finish(self, attempt_id, usage, status):
        normalized = normalize_usage(usage)
        with self.transaction() as db:
            attempt = db.execute("SELECT * FROM model_attempts WHERE id=?", (attempt_id,)).fetchone()
            if attempt is None or attempt["finished_at"] is not None:
                raise ConfigurationError("模型用量请求记录不存在或已结算")
            if attempt["purpose"] == "embedding" and normalized["prompt_tokens"] is not None:
                normalized["completion_tokens"] = 0
                if normalized["total_tokens"] is None:
                    normalized["total_tokens"] = normalized["prompt_tokens"]
            cost = estimate_cost_nusd(normalized, attempt["price_model"])
            db.execute("""UPDATE model_attempts SET finished_at=?,status=?,usage_json=?,estimated_cost_nusd=? WHERE id=?""",
                       (_now(), status, json.dumps(normalized), cost, attempt_id))
            fields = list(normalized)
            updates = ",".join(field + "=" + field + "+?" for field in fields)
            measured = {field: normalized[field] or 0 for field in fields}
            # Count the known part even when a response lacks a complete total;
            # usage_json retains None so reports still mark the request unmeasured.
            measured["total_tokens"] = max(measured["total_tokens"],
                                             measured["prompt_tokens"] + measured["completion_tokens"])
            db.execute("UPDATE key_usage SET " + updates + ",estimated_cost_nusd=estimated_cost_nusd+? WHERE key_id=?",
                       (*[measured[field] for field in fields], cost or 0, attempt["key_id"]))
            db.execute("UPDATE key_usage SET active_requests=MAX(active_requests-1, 0) WHERE key_id=?",
                       (attempt["key_id"],))

    def run_concurrency(self, group, default=None):
        with self.transaction() as db:
            row = db.execute("SELECT max_concurrency FROM run_settings WHERE setting_group=?", (group,)).fetchone()
            if row is None:
                db.execute("INSERT INTO run_settings(setting_group,max_concurrency,changed_at) VALUES (?,?,?)",
                           (group, default, _now()))
                return default
            return row["max_concurrency"] if row["max_concurrency"] is not None else default

    def update_run_concurrency(self, group, value):
        if value is not None and (type(value) is not int or value < 1):
            raise ConfigurationError("总并发上限必须是正整数")
        with self.transaction() as db:
            db.execute("INSERT INTO run_settings(setting_group,max_concurrency,changed_at) VALUES (?,?,?) "
                       "ON CONFLICT(setting_group) DO UPDATE SET max_concurrency=excluded.max_concurrency,changed_at=excluded.changed_at",
                       (group, value, _now()))

    def concurrency_snapshot(self, key_id):
        row = self.db.execute("SELECT max_concurrency,active_requests FROM key_usage WHERE key_id=?", (key_id,)).fetchone()
        if row is None:
            raise ConfigurationError("没有找到这个 key 来源")
        return dict(row)


    def report(self, key_id):
        with self.transaction() as db:
            state = dict(db.execute("SELECT * FROM key_usage WHERE key_id=?", (key_id,)).fetchone())
            counts = dict(db.execute("""SELECT
                SUM(finished_at IS NULL) AS unfinished_requests,
                SUM(status='completed') AS completed,
                SUM(finished_at IS NOT NULL AND status!='completed') AS errors,
                SUM(exhausted_fallback) AS exhausted_fallback_requests,
                SUM(finished_at IS NOT NULL AND json_extract(usage_json,'$.total_tokens') IS NULL) AS unmeasured_requests,
                SUM(finished_at IS NOT NULL AND estimated_cost_nusd IS NULL) AS unpriced_requests
                FROM model_attempts WHERE key_id=?""", (key_id,)).fetchone())
        usage = {field: state[field] for field in ("requests", "prompt_tokens", "completion_tokens", "total_tokens",
                                                   "cached_input_tokens", "reasoning_tokens")}
        usage["active_requests"] = state["active_requests"]
        usage.update({key: value or 0 for key, value in counts.items()})
        usage["estimated_cost_usd"] = state["estimated_cost_nusd"] / 1e9
        usage["tokens_complete"] = usage["unmeasured_requests"] == 0 and usage["unfinished_requests"] == 0
        usage["cost_complete"] = usage["unpriced_requests"] == 0 and usage["unfinished_requests"] == 0
        limits = {"max_requests": state["max_requests"], "max_total_tokens": state["max_total_tokens"],
                  "max_cost_usd": None if state["max_cost_nusd"] is None else state["max_cost_nusd"] / 1e9,
                  "max_concurrency": state["max_concurrency"]}
        exhausted = self._reached_limit(state) is not None
        return {"key_id": key_id, "source": state["source"], "usage": usage,
                "limits": limits, "exhausted": exhausted, "reference_model": state["reference_model"]}

    def close(self):
        with self._lock:
            self.db.close()


class RuntimeConcurrency:
    """A task limiter whose ceiling can be changed through the usage database."""

    def __init__(self, ledger: UsageLedger, group: str, default: int):
        if type(default) is not int or default < 1:
            raise ConfigurationError("总并发上限必须是正整数")
        self.ledger, self.group, self.default = ledger, group, default
        self._lock = RLock()
        self._active = 0

    def acquire(self):
        while True:
            with self._lock:
                limit = self.ledger.run_concurrency(self.group, self.default)
                if self._active < limit:
                    self._active += 1
                    return
            time.sleep(0.1)

    def release(self):
        with self._lock:
            self._active = max(0, self._active - 1)
