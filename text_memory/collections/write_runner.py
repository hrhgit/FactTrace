"""Write-only checkpoints: committed receipts and valid extractions are reused."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

from ..model import HttpJsonModel, encode
from ..schema import digest, normalize_add, require
from ..store import Scope
from . import gap_fill
from .engine import CollectionEngine
from .store import CollectionStore


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def token(history):
    return digest([history["user_id"], history["scope_id"], history["request_id"]])


def usage(calls):
    fields = {"prompt_tokens", "completion_tokens", "total_tokens"}
    fields.update(k for c in calls for k, v in (c.get("usage") or {}).items() if type(v) in (int, float))
    return {k: (sum(values) if all(type(v) in (int, float) for v in values) else None)
            for k in sorted(fields) for values in [[(c.get("usage") or {}).get(k) for c in calls]]}


def recorded_calls(output):
    """Count persisted new attempts only; inherited calls live elsewhere."""
    return [json.loads(path.read_text()) for path in sorted((output / "attempts").glob("*/*.json"))]


class JournaledWriteModel(HttpJsonModel):
    """Save each response, including a rejected output, before the next call."""
    def __init__(self, config, directory, *, transport=None, channel_pool=None):
        super().__init__(config, transport=transport, channel_pool=channel_pool)
        self.directory = directory
        self.next_index = max((int(p.stem) for p in directory.glob("*.json")), default=-1) + 1

    def _post(self, body, messages, purpose, finished, parse):
        if purpose not in {"collection_extract", "collection_gap_fill",
                           "collection_relation_pairs", "collection_relation_negation"}:
            raise RuntimeError("写入仅允许事实提取、补漏与关系标注")
        start, index = time.monotonic(), len(self.calls)
        try:
            return super()._post(body, messages, purpose, finished, parse)
        finally:
            for call in self.calls[index:]:
                call.update(elapsed_seconds=time.monotonic() - start, recorded_at=now(), new_attempt=True)
                save(self.directory / f"{self.next_index:06d}.json", call)
                self.next_index += 1


def recover_extraction(output, history):
    """Close the response/checkpoint gap without another extraction request."""
    directory = output / "attempts" / token(history)
    if len(CollectionEngine.extraction_parts(history)) > 1:
        # A multi-part extraction is only complete after every part; it is
        # checkpointed as a whole, not reassembled from single responses.
        return None
    for path in sorted(directory.glob("*.json"), reverse=True):
        call = json.loads(path.read_text())
        if call.get("purpose") != "collection_extract" or call.get("status") != "completed":
            continue
        if call.get("messages", [])[:2] != CollectionEngine.extraction_messages(history):
            continue
        # Responses are data; only the existing JSON decoder and task validator
        # can make an extraction checkpoint eligible for reuse.
        from ..errors import InvalidOutputError, ValidationError
        from ..model import decode_json_object
        try:
            content = call["response"]["choices"][0]["message"]["content"]
            extracted = decode_json_object(content)
            CollectionEngine.extraction_points(history, extracted)
        except (KeyError, IndexError, TypeError, InvalidOutputError, ValidationError):
            continue
        return {"payload_sha256": digest(history), "extraction": extracted,
                "recovered_from_attempt": str(path), "calls": [call], "steps": []}
    return None


def recover_gap_fill(output, history, extracted):
    """Reuse a journaled gap-fill reply that validates against this extraction."""
    from ..errors import InvalidOutputError, ValidationError
    from ..model import decode_json_object
    aliases, sources = CollectionEngine.source_view(history)
    gaps = gap_fill.find_gaps(sources, extracted)
    if not gaps or len(gaps) > gap_fill.BATCH:
        # Several batches are checkpointed as a whole, not reassembled from single responses.
        return None
    prefix = CollectionEngine.gap_request(history, extracted, extracted["facts"], gaps)
    for path in sorted((output / "attempts" / token(history)).glob("*.json"), reverse=True):
        call = json.loads(path.read_text())
        if call.get("purpose") != gap_fill.PURPOSE or call.get("status") != "completed":
            continue
        if call.get("messages", [])[:len(prefix)] != prefix:
            continue
        try:
            reply = decode_json_object(call["response"]["choices"][0]["message"]["content"])
            facts, errors = gap_fill.apply_fill(extracted["facts"], reply, gaps, aliases,
                                                {s["ref"]: s["text"] for s in sources})
            if errors:
                continue
            filled = {"facts": facts, "conversation": [*prefix, {"role": "assistant", "content": encode(reply)}]}
            CollectionEngine.extraction_points(history, filled)
        except (KeyError, IndexError, TypeError, InvalidOutputError, ValidationError):
            continue
        return {"payload_sha256": digest(history), "base_sha256": digest(extracted), "gaps": gaps,
                "extraction": filled, "recovered_from_attempt": str(path), "calls": [call], "steps": []}
    return None


def run(histories, output, model_factory, *, embedder=None, floor=None, log=None):
    """Serial, resumable write operation; the caller pins inputs and code."""
    import fcntl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    histories = [normalize_add(h) for h in histories]
    identities = [(h["user_id"], h["scope_id"], h["request_id"]) for h in histories]
    require(len(set(identities)) == len(identities), "写入请求编号重复")
    # A live process must never compete with a second resume in this directory.
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(histories, output, model_factory, embedder=embedder, floor=floor, log=log)


def _run_locked(histories, output, model_factory, *, embedder, floor, log=None):
    started = time.monotonic()
    state = {"status": "running", "pid": os.getpid(), "started_at": now(),
             "planned_writes": len(histories), "completed_writes": 0, "answer_calls": 0, "judge_calls": 0}
    current, model, engine = None, None, None

    def event(value):
        record = {"time": now(), **value}
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), file=log, flush=True)

    with CollectionStore(output / "memory.sqlite") as store:
        try:
            completed = sum(store.receipt(h) is not None for h in histories)
            state.update(completed_writes=completed, remaining_writes=len(histories)-completed)
            save(output / "_state.json", state)
            for index, history in enumerate(histories, 1):
                current = history["request_id"]
                receipt = store.receipt(history)
                model, engine = None, None
                if receipt is not None:
                    engine = CollectionEngine(store, None, embedder=embedder, floor=floor)
                    if engine.relations_pending(history):
                        # Committed before its relation stage finished: run only that stage.
                        model = model_factory(history)
                        engine = CollectionEngine(store, model, embedder=embedder, floor=floor)
                        state.update(updated_at=now(), request_id=current, current_index=index, phase="relations")
                        save(output / "_state.json", state)
                        engine.ensure_relations(history)
                        save(output / "relations" / (token(history) + ".json"), {"status": "ok", "calls": deepcopy(model.calls),
                             "steps": deepcopy(engine.trace), "usage_new": usage(model.calls)})
                    event({"phase": "write", "status": "reused_commit", "request_id": current})
                    continue
                model = model_factory(history)
                engine = CollectionEngine(store, model, embedder=embedder, floor=floor)
                checkpoint = output / "extractions" / (token(history) + ".json")
                cached = json.loads(checkpoint.read_text()) if checkpoint.exists() else recover_extraction(output, history)
                state.update(updated_at=now(), request_id=current, current_index=index, phase="extract")
                save(output / "_state.json", state)
                if cached is not None:
                    require(cached.get("payload_sha256") == digest(history), "已保存的抽取结果与原会话不一致")
                    extracted = cached["extraction"]
                    engine.extraction_points(history, extracted)
                    if not checkpoint.exists():
                        save(checkpoint, cached)
                    extraction_steps = cached.get("steps", [])
                    event({"phase": "extract", "status": "reused", "request_id": current, "facts": len(extracted["facts"])})
                else:
                    event({"phase": "extract", "status": "started", "request_id": current})
                    extracted = engine.extract(history)
                    extraction_steps = deepcopy(engine.trace)
                    save(checkpoint, {"payload_sha256": digest(history), "extraction": extracted,
                        "calls": deepcopy(model.calls), "steps": extraction_steps, "usage": usage(model.calls)})
                    event({"phase": "extract", "status": "ok", "request_id": current, "facts": len(extracted["facts"])})
                gap_checkpoint = output / "gap_fills" / (token(history) + ".json")
                gap_cached = (json.loads(gap_checkpoint.read_text()) if gap_checkpoint.exists()
                              else recover_gap_fill(output, history, extracted))
                state.update(updated_at=now(), phase="gap_fill")
                save(output / "_state.json", state)
                if gap_cached is not None:
                    require(gap_cached.get("payload_sha256") == digest(history)
                            and gap_cached.get("base_sha256") == digest(extracted), "已保存的补漏结果与抽取结果不一致")
                    filled = gap_cached["extraction"]
                    engine.extraction_points(history, filled)
                    if not gap_checkpoint.exists():
                        save(gap_checkpoint, gap_cached)
                    gap_steps = gap_cached.get("steps", [])
                    event({"phase": "gap_fill", "status": "reused", "request_id": current, "facts": len(filled["facts"])})
                else:
                    first = len(model.calls)
                    engine.trace = []
                    filled = engine.fill_gaps(history, extracted)
                    gap_steps = deepcopy(engine.trace)
                    gaps = gap_steps[0]["result"]["flagged"]
                    save(gap_checkpoint, {"payload_sha256": digest(history), "base_sha256": digest(extracted),
                        "gaps": gaps, "extraction": filled, "calls": deepcopy(model.calls[first:]),
                        "steps": gap_steps, "usage": usage(model.calls[first:])})
                    event({"phase": "gap_fill", "status": "ok", "request_id": current, "flagged": len(gaps),
                           "facts_before": len(extracted["facts"]), "facts": len(filled["facts"])})
                extraction_steps, extracted = extraction_steps + gap_steps, filled
                points = engine.extraction_points(history, extracted)
                state.update(updated_at=now(), phase="commit_and_relations")
                save(output / "_state.json", state)
                receipt = engine.add(history, extraction=extracted)
                save(output / "writes" / (token(history) + ".json"),
                    {"status": "ok", "receipt": receipt, "extraction_reused": cached is not None,
                     "calls": deepcopy(model.calls), "steps": extraction_steps + engine.trace,
                     "usage_new": usage(model.calls)})
                completed += 1
                state.update(updated_at=now(), completed_writes=completed, remaining_writes=len(histories)-completed,
                             usage_new=usage(recorded_calls(output)))
                save(output / "_state.json", state)
                event({"phase": "write", "status": "ok", "request_id": current,
                       "completed_writes": completed, "facts": len(receipt["point_ids"])})
            state.update(status="complete", phase="finished", counts=dict(store.db.execute(
                "SELECT kind, COUNT(DISTINCT id) FROM objects GROUP BY kind").fetchall()))
            save(output / "summary.json", {**state, "elapsed_seconds": time.monotonic()-started,
                                           "usage_new": usage(recorded_calls(output))})
            event({"phase": "finished", "status": "complete", "completed_writes": completed})
        except BaseException as error:
            completed = sum(store.receipt(h) is not None for h in histories)
            failure = {"time": now(), "request_id": current, "error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc(), "calls": model.calls if model else [],
                       "steps": engine.trace if engine else []}
            failures = output / "failures"
            save(failures / (str(time.time_ns()) + ".json"), failure)
            save(output / "failure.json", failure)
            state.update(status="failed", completed_writes=completed, remaining_writes=len(histories)-completed,
                         error_type=type(error).__name__, error=str(error))
            event({"phase": state.get("phase"), "status": "failed", "request_id": current,
                   "completed_writes": completed, "error_type": type(error).__name__, "error": str(error)})
            raise
        finally:
            state.update(updated_at=now(), elapsed_seconds=time.monotonic()-started, usage_new=usage(recorded_calls(output)))
            save(output / "_state.json", state)
    return state


