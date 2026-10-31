"""Answer, grade and summarize a mixed local set (data/local-eval) from frozen Search contexts.

answer  one official answer per question; labels are not opened.
grade   one official grade per answer, only after every answer exists.
summary scores per benchmark and category; --subset reports a contained set
        (e.g. mixed-v1 inside mixed-v1-extended) and the remaining questions apart.

Contexts are a JSONL file of {"id", "context"} rows, the ordered Search output
for each question. Failed calls
are logged and left unanswered/ungraded; rerunning the same command retries only
those.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
from threading import Lock
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from text_memory.evaluation import TimedModel, usage
from text_memory.evaluation import append_jsonl, jsonl, write_json
from text_memory.mixed_eval import PIPELINES, answer_messages, grade
from text_memory.model import ModelChannelPool, ModelConfig
from text_memory.usage import RuntimeConcurrency
from text_memory.schema import digest, require


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filename(ident: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", ident) + ".json"


def model_identity(config: ModelConfig) -> dict:
    return {"model": config.model, "endpoint_hash": digest(config.base_url), "reasoning_effort": config.reasoning_effort}


def load_set(directory: Path, datasets: list[str] | None) -> list[dict]:
    queries = jsonl(directory / "inputs/queries.jsonl")
    tasks = {t["id"]: t for t in jsonl(directory / "inputs/tasks.jsonl")}
    require([q["id"] for q in queries] == list(tasks), "queries 与 tasks 不对应")
    selected = [q | {"task": tasks[q["id"]]} for q in queries if datasets is None or tasks[q["id"]]["dataset"] in datasets]
    require(bool(selected), "没有选中任何题目")
    require(len({filename(q["id"]) for q in selected}) == len(selected), "题号转成文件名后重复")
    return selected


def run_calls(items, work, workers, output, phase, runtime_limit):
    lock, failed = Lock(), []

    def one(item):
        try:
            runtime_limit.acquire()
            work(item)
        except Exception as error:  # noqa: BLE001 - every failure is logged and retried on rerun
            with lock:
                failed.append(item["id"])
                append_jsonl(output / "errors.jsonl", {"id": item["id"], "phase": phase, "time": time.time(),
                                                       "error_type": type(error).__name__, "error": str(error)})
        finally:
            runtime_limit.release()
    with ThreadPoolExecutor(max_workers=max(workers, 128)) as pool:
        list(pool.map(one, items))
    return failed


def answer(args):
    output = args.output
    selected = load_set(args.set, args.datasets)
    rows = jsonl(args.contexts)
    contexts = {r["id"]: r["context"] for r in rows}
    require(len(contexts) == len(rows) and all(isinstance(v, str) for v in contexts.values()), "contexts 题号重复或内容不是文字")
    require(set(contexts) >= {q["id"] for q in selected}, "contexts 缺少部分题目")
    contexts_identity = {"path": str(args.contexts), "sha256": sha256(args.contexts)}
    config = ModelConfig.from_env("ANSWER")
    channel_pool = ModelChannelPool(config)
    runtime_limit = RuntimeConcurrency(channel_pool.usage, "ANSWER", args.workers)
    # The answer phase does not touch labels at all; grade records their hash.
    manifest = {"set": str(args.set), "set_sha256": {name: sha256(args.set / name) for name in
                                                     ("inputs/queries.jsonl", "inputs/tasks.jsonl")},
                "datasets": args.datasets, "questions": len(selected), "contexts": contexts_identity,
                "answer": model_identity(config), "answer_temperature": 0,
                "pipelines_sha256": {name: sha256(ROOT / path) for name, path in PIPELINES.items()},
                "code_sha256": {path: sha256(ROOT / path) for path in ("text_memory/mixed_eval.py", "scripts/run_mixed_answers.py")}}
    if (output / "manifest.json").exists():
        saved = json.loads((output / "manifest.json").read_text())
        require({k: v for k, v in saved.items() if k != "judge"} == manifest, "题集、材料、模型或代码已变化；请使用新的输出目录")
    else:
        write_json(output / "manifest.json", manifest)
    pending = [q for q in selected if not (output / "answers" / filename(q["id"])).exists()]

    def work(q):
        model = TimedModel(config, channel_pool=channel_pool)
        messages = answer_messages(q["task"], q["question"], contexts[q["id"]], config.model)
        started = time.monotonic()
        try:
            response = model.complete_text(messages, purpose="answer", temperature=0)
        finally:
            write_json(output / "traces" / ("answer-" + filename(q["id"])), {"messages": messages, "calls": model.calls})
        write_json(output / "answers" / filename(q["id"]), {
            "id": q["id"], "dataset": q["task"]["dataset"], "category": q["task"]["category"], "question": q["question"],
            "generated_answer": response, "context_chars": len(contexts[q["id"]]),
            "elapsed_seconds": time.monotonic() - started, "usage": usage(model.calls)})
    failed = run_calls(pending, work, args.workers, output, "answer", runtime_limit)
    done = sum((output / "answers" / filename(q["id"])).exists() for q in selected)
    return {"planned": len(selected), "answered": done, "failed_this_run": failed}


def grade_all(args):
    output = args.output
    manifest = json.loads((output / "manifest.json").read_text())
    directory = Path(manifest["set"])
    for name, expected in manifest["set_sha256"].items():
        require(sha256(directory / name) == expected, "题集文件已变化：" + name)
    selected = load_set(directory, manifest["datasets"])
    missing = [q["id"] for q in selected if not (output / "answers" / filename(q["id"])).exists()]
    require(not missing, f"还有 {len(missing)} 题没有答案；先完成 answer")
    config = ModelConfig.from_env("JUDGE")
    channel_pool = ModelChannelPool(config)
    runtime_limit = RuntimeConcurrency(channel_pool.usage, "JUDGE", args.workers)
    judge = {"judge": model_identity(config), "judge_temperature": 0,
             "labels_sha256": sha256(directory / "scoring/labels.jsonl")}
    if "judge" in manifest:
        require(manifest["judge"] == judge, "评分模型或标签已变化；请另存评分")
    else:
        write_json(output / "manifest.json", manifest | {"judge": judge})
    # Labels are read only now, after every answer is fixed on disk.
    labels = {l["id"]: l for l in jsonl(directory / "scoring/labels.jsonl")}
    pending = [q for q in selected if not (output / "grades" / filename(q["id"])).exists()]

    def work(q):
        stored = json.loads((output / "answers" / filename(q["id"])).read_text())
        model = TimedModel(config, channel_pool=channel_pool)
        try:
            result = grade(q["task"], q["question"], labels[q["id"]], stored["generated_answer"], model)
        finally:
            write_json(output / "traces" / ("grade-" + filename(q["id"])), {"calls": model.calls})
        write_json(output / "grades" / filename(q["id"]), {"id": q["id"], "dataset": q["task"]["dataset"],
                   "category": q["task"]["category"], **result, "usage": usage(model.calls)})
    failed = run_calls(pending, work, args.workers, output, "grade", runtime_limit)
    done = sum((output / "grades" / filename(q["id"])).exists() for q in selected)
    return {"planned": len(selected), "graded": done, "failed_this_run": failed}


def summarize(rows):
    by_dataset, by_category = defaultdict(list), defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row["score"])
        by_category[(row["dataset"], row["category"])].append(row["score"])
    mean = lambda values: sum(values) / len(values)
    return {"questions": len(rows), "micro_score": mean([r["score"] for r in rows]) if rows else None,
            # Every benchmark counts equally, however many questions it has.
            "macro_score": mean([mean(v) for v in by_dataset.values()]) if rows else None,
            "datasets": {d: {"questions": len(v), "score": mean(v), "sum": sum(v)} for d, v in sorted(by_dataset.items())},
            "categories": {f"{d}/{c}": {"questions": len(v), "score": mean(v)} for (d, c), v in sorted(by_category.items())}}


def summary(args):
    output = args.output
    manifest = json.loads((output / "manifest.json").read_text())
    selected = load_set(Path(manifest["set"]), manifest["datasets"])
    grades = [json.loads((output / "grades" / filename(q["id"])).read_text())
              for q in selected if (output / "grades" / filename(q["id"])).exists()]
    result = {"planned": len(selected), "graded": len(grades), "all": summarize(grades)}
    if args.subset:
        inside = {q["id"] for q in jsonl(args.subset / "inputs/queries.jsonl")}
        result["subset"] = {"set": str(args.subset), "inside": summarize([g for g in grades if g["id"] in inside]),
                            "outside": summarize([g for g in grades if g["id"] not in inside])}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("answer")
    run.add_argument("--set", type=Path, required=True, help="评测集目录，例如 data/local-eval/mixed-v1")
    run.add_argument("--contexts", type=Path, required=True, help="每行 {id, context} 的 JSONL，是每题的 Search 输出")
    run.add_argument("--datasets", nargs="+", choices=sorted(PIPELINES), help="只跑这些题库；默认全部")
    for command in (run, commands.add_parser("grade"), commands.add_parser("summary")):
        command.add_argument("--output", type=Path, required=True)
        if command.prog.endswith("summary"):
            command.add_argument("--subset", type=Path, help="被包含的较小评测集目录，分开报告其内外的分数")
        else:
            command.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    result = {"answer": answer, "grade": grade_all, "summary": summary}[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result.get("failed_this_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
