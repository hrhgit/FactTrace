"""Write a mixed local set (data/local-eval) through formal Add, then export formal Search contexts.

write   every history goes through text_memory.collections.write_runner, one
        memory database per user_id. Model calls (extraction and gap fill) first
        run in parallel and are journaled; write_runner then
        commits each user's sessions in order and reuses those journals, so it
        makes no new call for them; after each commit it runs the write-time
        relation stage (live calls, one session at a time). Reruns resume;
        --reuse-writes copies finished user databases of an earlier run whose
        histories are identical (e.g. mixed-v1 inside mixed-v1-extended), and a
        copied or older database gets only its missing relation stage.
search  every query goes through CollectionApplication /search (the AML Search
        adapter; ranking makes no model call) on its user's
        database; the concatenated data[*].content is written to
        search-top<N>/contexts.jsonl for scripts/run_mixed_answers.py, and each
        query's model calls to search-top<N>/search/. --fact-limit sets N (the
        facts kept after ranking); several limits share one write.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from threading import Lock, local
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from text_memory.collections import write_runner
from text_memory.collections.answer_context import POLICY as ANSWER_CONTEXT_POLICY
from text_memory.collections.engine import CollectionEngine
from text_memory.collections.query import DEFAULT_POLICY, DIRECT_LIMIT
from text_memory.collections.relations import PROMPT_HASH as RELATION_PROMPT_HASH
from text_memory.collections.service import CollectionApplication
from text_memory.collections.store import CollectionStore
from text_memory.evaluation import append_jsonl, jsonl, write_json
from text_memory.mixed_eval import PIPELINES
from text_memory.model import HttpJsonModel, ModelChannelPool, ModelConfig
from text_memory.usage import RuntimeConcurrency
from text_memory.retrieval import embed_floor_from_env, embedder_from_env
from text_memory.schema import digest, normalize_add, require
from text_memory.store import Scope

# Every stage records the same formal source tree; archives cannot enter this fingerprint.
CORE_CODE = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "text_memory").rglob("*.py"))]
WRITE_CODE = SEARCH_CODE = CORE_CODE + [str(Path(__file__).relative_to(ROOT))]
# A search whose question judgment failed is repeated (after 10 s, then 20 s) before the run gives up.
SEARCH_ATTEMPTS, SEARCH_RETRY_DELAY = 3, 10.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_name(user_id: str) -> str:
    # Readable and collision-free: the hash suffix keeps distinct ids apart.
    return re.sub(r"[^A-Za-z0-9._-]", "_", user_id)[:60] + "-" + digest(user_id)[:10]


def selected(args):
    """Histories grouped by user, and queries, for the chosen benchmarks."""
    tasks = {t["id"]: t for t in jsonl(args.set / "inputs/tasks.jsonl")}
    queries = [q for q in jsonl(args.set / "inputs/queries.jsonl")
               if args.datasets is None or tasks[q["id"]]["dataset"] in args.datasets]
    users = {q["user_id"] for q in queries}
    groups = defaultdict(list)
    for history in jsonl(args.set / "inputs/histories.jsonl"):
        if history["user_id"] in users:
            groups[history["user_id"]].append(normalize_add(history))
    # A user with no history (single-turn CL-Bench: the material is in the question) is never written;
    # Search for it returns no data, as AML Search does for an empty user.
    require(bool(queries) and set(groups) <= users, "没有选中题目，或历史不属于任何题目")
    return groups, queries


def check_manifest(path: Path, manifest: dict, message: str):
    if path.exists():
        require(json.loads(path.read_text()) == manifest, message)
    else:
        write_json(path, manifest)


def relations_missing(user_dir: Path, histories: list) -> bool:
    """A committed user whose write-time relation stage has not run for every session."""
    with CollectionStore(user_dir / "memory.sqlite") as store:
        return any(not store.relations_done(Scope(h["user_id"], h["scope_id"]), h["request_id"],
                                            prompt_hash=RELATION_PROMPT_HASH) for h in histories)


def prepare_calls(user_dir: Path, history: dict, config: ModelConfig, *, channel_pool=None):
    """Make and journal this session's model calls; write_runner reuses them on commit."""
    model = write_runner.JournaledWriteModel(config, user_dir / "attempts" / write_runner.token(history),
                                             channel_pool=channel_pool)
    with CollectionStore() as scratch:
        engine = CollectionEngine(scratch, model)
        path = user_dir / "extractions" / (write_runner.token(history) + ".json")
        if not path.exists():
            cached = write_runner.recover_extraction(user_dir, history)
            if cached is None:
                extracted = engine.extract(history)
                cached = {"payload_sha256": digest(history), "extraction": extracted, "calls": deepcopy(model.calls),
                          "steps": deepcopy(engine.trace), "usage": write_runner.usage(model.calls)}
            write_runner.save(path, cached)
        extracted = json.loads(path.read_text())["extraction"]
        path = user_dir / "gap_fills" / (write_runner.token(history) + ".json")
        if not path.exists():
            cached = write_runner.recover_gap_fill(user_dir, history, extracted)
            if cached is None:
                first = len(model.calls)
                engine.trace = []
                filled = engine.fill_gaps(history, extracted)
                cached = {"payload_sha256": digest(history), "base_sha256": digest(extracted),
                          "gaps": engine.trace[0]["result"]["flagged"], "extraction": filled,
                          "calls": deepcopy(model.calls[first:]), "steps": deepcopy(engine.trace),
                          "usage": write_runner.usage(model.calls[first:])}
            write_runner.save(path, cached)
        filled = json.loads(path.read_text())["extraction"]


def write(args):
    groups, _ = selected(args)
    config = ModelConfig.from_env("MEMORY")
    channel_pool = ModelChannelPool(config)
    runtime_limit = RuntimeConcurrency(channel_pool.usage, "MEMORY", args.workers)
    embedder = embedder_from_env()
    require(embedder is not None, "正式查找要求配置向量模型")
    floor = embed_floor_from_env()
    output = args.output
    try:
        check_manifest(output / "write-manifest.json", {
            "set": str(args.set), "datasets": args.datasets,
            "histories_sha256": {user: digest(hs) for user, hs in sorted(groups.items())},
            "memory": {"model": config.model, "endpoint_hash": digest(config.base_url),
                       "reasoning_effort": config.reasoning_effort},
            "embedding": embedder.identity, "floor": floor,
            "code_sha256": {path: sha256(ROOT / path) for path in WRITE_CODE}},
            "题集、写入模型、向量配置或写入代码已变化；请使用新的输出目录")
        reused = []
        for user, histories in groups.items():
            target = output / "write" / directory_name(user)
            source = args.reuse_writes / "write" / directory_name(user) if args.reuse_writes else None
            if not target.exists() and source and (source / "summary.json").exists() \
                    and (source / "histories.sha256").read_text().strip() == digest(histories):
                shutil.copytree(source, target, ignore=shutil.ignore_patterns("run.lock"))
                reused.append(user)
            target.mkdir(parents=True, exist_ok=True)
            (target / "histories.sha256").write_text(digest(histories) + "\n")
        lock, failures = Lock(), []

        def attempt(phase, user, request_id, work):
            try:
                runtime_limit.acquire()
                work()
            except Exception as error:  # noqa: BLE001 - logged; the rerun resumes from journals
                with lock:
                    failures.append({"phase": phase, "user_id": user, "request_id": request_id})
                    append_jsonl(output / "write-errors.jsonl", {"time": time.time(), "phase": phase, "user_id": user,
                                 "request_id": request_id, "error_type": type(error).__name__, "error": str(error)})
            finally:
                runtime_limit.release()

        def committed(user):
            return (output / "write" / directory_name(user) / "summary.json").exists()

        def finished(user):
            directory = output / "write" / directory_name(user)
            if not committed(user) or relations_missing(directory, groups[user]):
                return False
            return True

        calls = [(user, h) for user, hs in groups.items() if not committed(user) for h in hs]
        with ThreadPoolExecutor(max_workers=max(args.workers, 128)) as pool:
            list(pool.map(lambda item: attempt("model_calls", item[0], item[1]["request_id"], lambda: prepare_calls(
                output / "write" / directory_name(item[0]), item[1], config, channel_pool=channel_pool)), calls))
        # A user with a failed call is not committed now; its sessions must stay in order. A committed user
        # (an earlier run or --reuse-writes) still runs its missing relation stage, request by request.
        pending = [user for user in groups if not finished(user) and user not in {f["user_id"] for f in failures}]

        def commit(user):
            directory = output / "write" / directory_name(user)
            with (directory / "run.log").open("a") as log:
                write_runner.run(groups[user], directory,
                                 lambda h: write_runner.JournaledWriteModel(
                                     config, directory / "attempts" / write_runner.token(h), channel_pool=channel_pool),
                                 embedder=embedder, floor=floor, log=log)
        # Different users have separate databases and logs. Sessions within
        # one user still commit in order through the formal write runner.
        commit_workers = getattr(args, "commit_workers", None) or args.workers
        require(type(commit_workers) is int and commit_workers > 0, "写入并发数必须为正整数")
        with ThreadPoolExecutor(max_workers=max(commit_workers, 128)) as pool:
            list(pool.map(lambda user: attempt("commit", user, None, lambda: commit(user)), pending))
    finally:
        embedder.close()
    done = [user for user in groups if finished(user)]
    return {"users": len(groups), "users_complete": len(done), "reused_users": len(reused),
            "write_requests": sum(len(hs) for hs in groups.values()), "failed_this_run": failures}


def search(args):
    groups, queries = selected(args)
    written = args.output
    output = written / f"search-top{args.fact_limit}"
    for user, histories in groups.items():
        directory = written / "write" / directory_name(user)
        require((directory / "summary.json").exists(), "写入还没有完成：" + user)
        with CollectionStore(directory / "memory.sqlite") as store:
            require(all(store.receipt(h) for h in histories), "写入还没有完成：" + user)
        require(not relations_missing(directory, histories), "写入后的关系标注还没有完成：" + user)
    config = ModelConfig.from_env("MEMORY")
    channel_pool = ModelChannelPool(config)
    runtime_limit = RuntimeConcurrency(channel_pool.usage, "MEMORY", args.workers)
    embedder = embedder_from_env()
    require(embedder is not None, "正式查找要求配置向量模型")
    floor = embed_floor_from_env()
    current = local()

    def factory():
        # One model per request; the thread keeps it to save that request's calls.
        current.model = HttpJsonModel(config, channel_pool=channel_pool)
        return current.model
    try:
        check_manifest(output / "search-manifest.json", {
            "set": str(args.set), "datasets": args.datasets, "queries_sha256": digest(queries),
            "top_k": args.top_k, "query_policy": DEFAULT_POLICY, "fact_limit": args.fact_limit,
            "relation_notes": True, "relation_prompt_sha256": RELATION_PROMPT_HASH,
            "answer_context_policy": ANSWER_CONTEXT_POLICY,
            "memory": {"model": config.model, "endpoint_hash": digest(config.base_url),
                       "reasoning_effort": config.reasoning_effort},
            "embedding": embedder.identity, "floor": floor,
            "code_sha256": {path: sha256(ROOT / path) for path in SEARCH_CODE}},
            "题集、查找配置或查找代码已变化；请使用新的输出目录")
        applications = {user: CollectionApplication(written / "write" / directory_name(user) / "memory.sqlite",
                                                    factory, embedder=embedder, floor=floor,
                                                    fact_limit=args.fact_limit) for user in groups}

        def one(query):
            runtime_limit.acquire()
            body = {"query": query["question"], "user_id": query["user_id"],
                    "scope_id": query.get("scope_id", "default"), "top_k": args.top_k}
            if query.get("question_date"):
                body["question_date"] = query["question_date"]
            started = time.monotonic()
            calls, data, error = [], [], None
            # A failed model judgment raises.  Retry the same search request
            # after a delay; never export material selected under a default.
            for attempt in range(SEARCH_ATTEMPTS if query["user_id"] in applications else 0):
                current.model = None
                try:
                    result, data = applications[query["user_id"]].search(body)
                    error = None
                    break
                except Exception as failure:
                    error = f"{type(failure).__name__}: {failure}"
                    data = []
                finally:
                    calls += current.model.calls if current.model else []
                if attempt + 1 < SEARCH_ATTEMPTS:
                    time.sleep(SEARCH_RETRY_DELAY * (attempt + 1))
            try:
                write_json(output / "search" / (re.sub(r"[^A-Za-z0-9._-]", "_", query["id"]) + ".json"),
                           {"id": query["id"], "calls": calls, "error": error,
                            "question_flags_error": error})
                # AML answers from the ordered concatenation of data[*].content.
                return {"id": query["id"], "context": "\n\n".join(item["content"] for item in data),
                        "packets": len(data), "elapsed_seconds": time.monotonic() - started, "error": error}
            finally:
                runtime_limit.release()
        with ThreadPoolExecutor(max_workers=max(args.workers, 128)) as pool:
            rows = list(pool.map(one, queries))
    finally:
        embedder.close()
    failed = [r["id"] for r in rows if r["error"]]
    if failed:
        write_json(output / "search-failures.json", {r["id"]: r["error"] for r in rows if r["error"]})
        return {"questions": len(rows), "failed_this_run": failed, "failures": str(output / "search-failures.json")}
    temporary = output / "contexts.jsonl.tmp"
    temporary.write_text("".join(json.dumps({"id": r["id"], "context": r["context"]}, ensure_ascii=False) + "\n" for r in rows))
    temporary.replace(output / "contexts.jsonl")
    write_json(output / "search-stats.json", {"questions": len(rows), "empty_contexts": sum(not r["context"] for r in rows),
               "context_chars": {r["id"]: len(r["context"]) for r in rows},
               "elapsed_seconds": {r["id"]: r["elapsed_seconds"] for r in rows}})
    return {"questions": len(rows), "empty_contexts": sum(not r["context"] for r in rows),
            "mean_context_chars": sum(len(r["context"]) for r in rows) / len(rows),
            "contexts": str(output / "contexts.jsonl")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("write", "search"):
        command = commands.add_parser(name)
        command.add_argument("--set", type=Path, required=True, help="评测集目录，例如 data/local-eval/mixed-v1")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--datasets", nargs="+", choices=sorted(PIPELINES), help="只处理这些题库的题目及其历史；默认全部")
        command.add_argument("--workers", type=int, default=6)
        if name == "write":
            command.add_argument("--commit-workers", type=int,
                                 help="同时保存及标注关系的用户数；每个用户内部保持原顺序，默认与 workers 相同")
            command.add_argument("--reuse-writes", type=Path, help="较早的运行目录；历史相同且已写完的用户直接复制")
        else:
            command.add_argument("--top-k", type=int, default=10, help="AML Search 请求必填的 top_k")
            command.add_argument("--fact-limit", type=int, default=DIRECT_LIMIT,
                                 help=f"排序后保留的事实条数，默认 {DIRECT_LIMIT}；结果写入 search-top<N>/")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    args.output.mkdir(parents=True, exist_ok=True)
    result = (write if args.command == "write" else search)(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result.get("failed_this_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
