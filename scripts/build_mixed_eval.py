"""Build a mixed local evaluation set from every public text benchmark with history.

Selection uses only ids, question types and source sizes, never answers or
scores. Items already used by earlier runs under outputs/ are excluded where the
benchmark has enough unused material. Labels stay in their own file; the
search-side queries carry only fields the memory runner accepts.

Every pool and its order are fixed by the seed alone, and larger amounts only
take further items from the same order, so a larger set built with the same
seed contains every smaller one (checked with --must-include).
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq

from scripts.prepare_longmemeval import preserve_empty_messages
from text_memory.datasets import prepare_locomo, prepare_longmemeval
from text_memory.evaluation import jsonl, validate_query, write_json
from text_memory.schema import normalize_add, require

PUBLIC = ROOT / "data/public"
LOCOMO_CATEGORIES = {"1": "multi_hop", "2": "temporal", "3": "open_domain", "4": "single_hop"}
# Conversations of the 2026-10-23 split: conv-26/47/48 were tuned on, the other
# seven were run once as holdout (outputs/holdout-locomo-routes-20261028).
LOCOMO_TUNED = {"conv-26", "conv-47", "conv-48"}
# Pool rules stay fixed whatever amounts are requested, so sets nest.
LOCOMO_MIN_OPEN_DOMAIN = 6
PERSONAMEM_V2_MIN_PER_TYPE = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(seed: int, *parts) -> str:
    """Stable order independent of Python's per-process string hashing."""
    return hashlib.sha256(json.dumps([seed, *parts]).encode()).hexdigest()


def amount(value: str) -> int | None:
    """A count, or None for every available item."""
    return None if value == "all" else int(value)


def stratified(items: list, key, quota: dict, seed: int, tag: str) -> list:
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    chosen = []
    for group, count in quota.items():
        candidates = sorted(groups[group], key=lambda item: rank(seed, tag, group, item["_id"]))
        require(count is None or len(candidates) >= count, f"{tag} 的 {group} 数量不足")
        chosen.extend(candidates if count is None else candidates[:count])
    return chosen


def used_ids(outputs: Path) -> dict[str, set[str]]:
    """Ids prepared by earlier experiments, so the mixed set favours unseen items."""
    used = {"longmemeval": set(), "personamem-v1": set()}
    for path in outputs.rglob("preparation.json"):
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        name = str(report.get("dataset", ""))
        if name.startswith("LongMemEval"):
            used["longmemeval"].update(map(str, report.get("selected_ids", [])))
        elif name.startswith("personamem-v1"):
            used["personamem-v1"].update(p["shared_context_id"] for p in report.get("prefixes", []))
    for path in outputs.glob("*/sets/longmemeval/ids-*.txt"):
        used["longmemeval"].update(path.read_text().split())
    for path in outputs.glob("layered-*/ids.txt"):
        used["longmemeval"].update(path.read_text().split())
    return used


def read_prepared(directory: Path) -> tuple[list, list, list]:
    labels = directory / "labels.jsonl"
    return jsonl(directory / "histories.jsonl"), jsonl(directory / "queries.jsonl"), jsonl(labels) if labels.exists() else []


def locomo(seed: int, per_category: dict) -> dict:
    source = PUBLIC / "locomo-refined/data/raw/locomo_refined.json"
    rows = json.loads(source.read_text())
    counts = {str(r["sample_id"]): Counter(str(q["category"]) for q in r["qa"]) for r in rows}
    eligible = sorted(c for c, n in counts.items() if c not in LOCOMO_TUNED and n["3"] >= LOCOMO_MIN_OPEN_DOMAIN)
    conversation = min(eligible, key=lambda c: rank(seed, "locomo", c))
    # Some published evidence ids point at missing messages (conv-42#q0172,
    # conv-49#q0030), which prepare_locomo's label step rejects. Prepare inputs
    # only and take the answer and raw evidence annotation from qa directly.
    with tempfile.TemporaryDirectory() as temporary:
        prepare_locomo(rows, Path(temporary) / "p", ids={conversation}, media_text="captions", include_labels=False)
        histories, queries, _ = read_prepared(Path(temporary) / "p")
    qa = next(r for r in rows if str(r["sample_id"]) == conversation)["qa"]
    labels = [{"id": f"{conversation}#q{i:04d}", "gold_answer": item["answer"], "evidence": item["evidence"]}
              for i, item in enumerate(qa)]
    for query in queries:
        query["_id"] = query["id"]
    chosen = stratified(queries, lambda q: q["category"], per_category, seed, "locomo")
    keep = {q["id"] for q in chosen}
    queries = [{k: v for k, v in q.items() if k != "_id"} | {"category": LOCOMO_CATEGORIES[q["category"]]}
               for q in queries if q["id"] in keep]
    tasks = [{"id": q["id"], "dataset": "locomo-refined", "category": q["category"],
              "answer_format": "open_ended", "official_pipeline": "vendor/aml/data/locomo-refined/pipeline.py"}
             for q in queries]
    labels = [{"id": l["id"], "dataset": "locomo-refined", "scoring": "llm_judge_correct",
               "gold_answer": l["gold_answer"], "evidence": l["evidence"]} for l in labels if l["id"] in keep]
    return {"histories": histories, "queries": queries, "labels": labels, "tasks": tasks,
            "report": {"source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
                       "conversation": conversation, "sessions": len(histories), "questions": len(queries),
                       "category_counts": dict(Counter(t["category"] for t in tasks)),
                       "note": "从只运行过一次、开放题不少于 6 道的留出对话中选一段，完整保留全部会话；按题型抽题。"
                               "该对话已有 holdout-locomo-routes-20261028 的写入结果，可作写入对照。"}}


def longmemeval(seed: int, per_type: int, unanswerable: int, filler: int, used: set[str]) -> dict:
    source = PUBLIC / "longmemeval/longmemeval_s_cleaned.json"
    rows = json.loads(source.read_text())
    base = lambda ident: ident.removesuffix("_abs")
    blocked = {base(i) for i in used}
    items = [{"_id": str(r["question_id"]), "type": r["question_type"]} for r in rows
             if base(str(r["question_id"])) not in blocked]
    # Answerable questions whose _abs twin exists stay out of the answerable
    # pool, so the two pools never compete and both grow independently.
    twins = {base(i["_id"]) for i in items if i["_id"].endswith("_abs")}
    answered = [i for i in items if not i["_id"].endswith("_abs") and i["_id"] not in twins]
    types = sorted({r["question_type"] for r in rows})
    chosen = stratified(answered, lambda i: i["type"], {t: per_type for t in types}, seed, "longmemeval")
    abstain = sorted((i for i in items if i["_id"].endswith("_abs")),
                     key=lambda i: rank(seed, "longmemeval-abs", i["_id"]))[:unanswerable]
    require(len(abstain) == unanswerable, "LongMemEval 无答案题数量不足")
    ids = {i["_id"] for i in chosen + abstain}
    adapted, empty = preserve_empty_messages(rows, ids, policy="preserve")
    with tempfile.TemporaryDirectory() as temporary:
        prepare_longmemeval(adapted, Path(temporary) / "p", ids=ids, filler=filler)
        histories, queries, labels = read_prepared(Path(temporary) / "p")
    tasks = [{"id": q["id"], "dataset": "longmemeval", "category": q["category"] + ("_abstention" if q["id"].endswith("_abs") else ""),
              "answer_format": "open_ended", "official_pipeline": "vendor/aml/data/longmemeval-s/pipeline.py"}
             for q in queries]
    labels = [l | {"dataset": "longmemeval", "scoring": "llm_judge_correct"} for l in labels]
    return {"histories": histories, "queries": queries, "labels": labels, "tasks": tasks,
            "report": {"source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
                       "questions": len(queries), "sessions": len(histories), "reduced_history_filler": filler,
                       "excluded_previously_used": len(blocked), "empty_message_locations": empty,
                       "category_counts": dict(Counter(t["category"] for t in tasks)),
                       "note": f"每题历史独立且原本约 48 段会话；为控制写入量，只保留证据会话加 {filler} 段按固定哈希选出的其他会话，"
                               "保持原顺序、不截断内容。这一缩减用到了证据位置，查找难度低于完整历史。"}}


def beam(seed: int, conversations: int) -> dict:
    source = PUBLIC / "beam/data/100K-00000-of-00001.parquet"
    rows = pq.read_table(source).to_pylist()
    size = {r["conversation_id"]: sum(len(m["content"]) for batch in r["chat"] for m in batch) for r in rows}
    # Personal-life topics exercise user memory more than coding/math tutoring;
    # take at most one conversation per topic for variety.
    order = sorted((r for r in rows if r["conversation_seed"]["category"] not in {"Coding", "Math"}),
                   key=lambda r: rank(seed, "beam", r["conversation_id"]))
    chosen, topics = [], set()
    for row in order:
        if row["conversation_seed"]["category"] not in topics and len(chosen) < conversations:
            chosen.append(row)
            topics.add(row["conversation_seed"]["category"])
    require(len(chosen) == conversations, "BEAM 不同主题的对话数量不足")
    parts = [beam_conversation(row) for row in chosen]
    return {name: [x for part in parts for x in part[name]] for name in ("histories", "queries", "labels", "tasks")} | {
        "report": {"source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
                   "conversations": [{"conversation": r["conversation_id"], "topic": r["conversation_seed"]["category"],
                                      "source_chars": size[r["conversation_id"]], "batches": len(r["chat"]),
                                      "sessions": len(p["histories"])} for r, p in zip(chosen, parts)],
                   "questions": sum(len(p["queries"]) for p in parts),
                   "category_counts": dict(Counter(t["category"] for p in parts for t in p["tasks"])),
                   "note": "每段对话完整保留并取全部 20 道探测题（10 种能力各 2 道），每个主题至多一段。批次按编号问题切成写入请求，"
                           "日期使用批次时间锚点；消息原文（含 ->-> 编号标记）不改。"}}


def beam_conversation(row: dict) -> dict:
    conversation = row["conversation_id"]
    user = "beam-100k:" + conversation
    histories = []
    for b, batch in enumerate(row["chat"], 1):
        anchor = batch[0].get("time_anchor")
        require(bool(anchor), "BEAM 批次缺少时间锚点")
        # Each published batch holds ~27 numbered user questions ("index"
        # b,q) with answers and follow-ups. Split at those numbered questions
        # so one write request is one exchange; the batch time anchor is the
        # only date the source gives and applies to the whole batch.
        exchanges, current = [], []
        for message in batch:
            if message.get("index") and message["role"] == "user" and current:
                exchanges.append(current)
                current = []
            current.append(message)
        exchanges.append(current)
        for e, messages in enumerate(exchanges, 1):
            index = messages[0].get("index") or f"{b},0"
            histories.append(normalize_add({
                "user_id": user, "scope_id": "history", "session_id": f"batch{b}-q{index.split(',')[1]}",
                "request_id": f"{conversation}:batch{b}:{e}",
                "messages": [{"role": m["role"], "content": m["content"], "session_date": anchor} for m in messages]}))
    probing = ast.literal_eval(row["probing_questions"])
    queries, labels, tasks = [], [], []
    for category in sorted(probing):
        for n, item in enumerate(probing[category]):
            ident = f"beam-100k:{conversation}:{category}:{n}"
            queries.append({"id": ident, "user_id": user, "scope_id": "history", "question": item["question"],
                            "category": category})
            tasks.append({"id": ident, "dataset": "beam", "category": category, "answer_format": "open_ended_rubric",
                          "official_pipeline": "vendor/aml/data/beam/pipeline.py"})
            reference = next((item[k] for k in ("ideal_response", "ideal_answer", "answer", "ideal_summary") if k in item), None)
            labels.append({"id": ident, "dataset": "beam", "scoring": "rubric_llm_judge", "question_type": category,
                           "rubric": item["rubric"], "reference_answer": reference,
                           "annotation": {k: v for k, v in item.items() if k not in {"question", "rubric"}}})
    return {"histories": histories, "queries": queries, "labels": labels, "tasks": tasks}


def personamem_v1(seed: int, minimum: int, used: set[str]) -> dict:
    source = PUBLIC / "personamem-v1"
    with (source / "questions_32k.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["shared_context_id"]].append(row)
    order = sorted((k for k in grouped if k not in used), key=lambda k: rank(seed, "personamem-v1", k))
    chosen, total = [], 0
    for key in order:
        # Same rule as prepare_personamem: one history per user, all questions
        # at its latest published boundary, history = context[:end].
        end = max(int(r["end_index_in_shared_context"]) for r in grouped[key])
        chosen.append((key, end, [r for r in grouped[key] if int(r["end_index_in_shared_context"]) == end]))
        total += len(chosen[-1][2])
        if total >= minimum:
            break
    contexts = {}
    with (source / "shared_contexts_32k.jsonl").open() as stream:
        for line in stream:
            for key, messages in json.loads(line).items():
                if key in {c[0] for c in chosen}:
                    contexts[key] = messages
    histories, queries, labels, tasks, prefixes = [], [], [], [], []
    for key, end, selected in chosen:
        uid = "personamem-v1:" + key
        histories.append(normalize_add({"user_id": uid, "scope_id": "history", "session_id": "published-prefix",
                                        "request_id": f"{key}:{end}", "messages": contexts[key][:end]}))
        prefixes.append({"shared_context_id": key, "end_index_exclusive": end, "questions": len(selected)})
        for row in sorted(selected, key=lambda r: r["question_id"]):
            options = ast.literal_eval(row["all_options"])
            require(isinstance(options, list) and all(isinstance(v, str) for v in options), "选项格式无效")
            queries.append({"id": row["question_id"], "user_id": uid, "scope_id": "history",
                            "question": row["user_question_or_message"], "category": row["question_type"]})
            tasks.append({"id": row["question_id"], "dataset": "personamem-v1", "category": row["question_type"],
                          "answer_format": "multiple_choice", "all_options": row["all_options"],
                          "official_pipeline": "vendor/aml/data/personamem/pipeline_v1.py"})
            labels.append({"id": row["question_id"], "dataset": "personamem-v1", "scoring": "option_match",
                           "correct_answer": row["correct_answer"]})
    return {"histories": histories, "queries": queries, "labels": labels, "tasks": tasks,
            "report": {"source": "data/public/personamem-v1/questions_32k.csv",
                       "source_sha256": {n: sha256(source / n) for n in ("questions_32k.csv", "shared_contexts_32k.jsonl")},
                       "prefixes": prefixes, "questions": len(queries), "excluded_previously_used": len(used),
                       "category_counts": dict(Counter(t["category"] for t in tasks)),
                       "note": "沿用 prepare_personamem 的规则：每份共享历史只取最晚的发布边界及该边界上的全部问题，"
                               "历史为 context[:end] 一整段；跳过此前试测用过的历史。"}}


def personamem_v2(seed: int, personas: int, per_type: int | None) -> dict:
    source = PUBLIC / "personamem-v2"
    with (source / "benchmark/text/benchmark.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_persona = defaultdict(list)
    for row in rows:
        by_persona[row["persona_id"]].append(row)
    types = sorted({r["pref_type"] for r in rows})
    # A persona whose questions cover every preference type with enough items.
    eligible = sorted((p for p, items in by_persona.items()
                       if all(sum(r["pref_type"] == t for r in items) >= PERSONAMEM_V2_MIN_PER_TYPE for t in types)
                       and len({r["chat_history_32k_link"] for r in items}) == 1),
                      key=lambda p: rank(seed, "personamem-v2", p))
    require(len(eligible) >= personas, "PersonaMem v2 符合条件的用户不足")
    parts = [personamem_v2_persona(source, persona, by_persona[persona], types, per_type, seed)
             for persona in eligible[:personas]]
    return {name: [x for part in parts for x in part[name]] for name in ("histories", "queries", "labels", "tasks")} | {
        "report": {"source": "data/public/personamem-v2/benchmark/text/benchmark.csv",
                   "source_sha256": sha256(source / "benchmark/text/benchmark.csv"),
                   "personas": [p["report"] for p in parts], "questions": sum(len(p["queries"]) for p in parts),
                   "category_counts": dict(Counter(t["category"] for p in parts for t in p["tasks"])),
                   "note": "每位用户的 32k 历史整段写入，按偏好类型分层抽题（all 为该用户全部题）；历史首条是发布方给出的 system 人设说明，"
                           "原样保留。选项顺序用固定种子打乱，正确字母只在 labels。"}}


def personamem_v2_persona(source: Path, persona: str, rows: list, types: list, per_type: int | None, seed: int) -> dict:
    items = [r | {"_id": str(i)} for i, r in enumerate(rows)]
    chosen = stratified(items, lambda r: r["pref_type"], {t: per_type for t in types}, seed, "personamem-v2")
    link = chosen[0]["chat_history_32k_link"]
    history_file = source / link
    messages = json.loads(history_file.read_text())["chat_history"]
    uid = f"personamem-v2:persona{persona}"
    histories = [normalize_add({"user_id": uid, "scope_id": "history", "session_id": "chat-history-32k",
                                "request_id": f"persona{persona}:32k", "messages": messages})]
    queries, labels, tasks = [], [], []
    for row in chosen:
        ident = f"personamem-v2:persona{persona}:{row['_id']}"
        query = ast.literal_eval(row["user_query"]) if row["user_query"].strip().startswith("{") else row["user_query"]
        question = query.get("content", "") if isinstance(query, dict) else str(query)
        incorrect = json.loads(row["incorrect_answers"]) if row["incorrect_answers"].strip().startswith("[") \
            else ast.literal_eval(row["incorrect_answers"])
        # The official pipeline shuffles with Python hash(), which changes per
        # process; fix one reproducible order here instead.
        options = [row["correct_answer"], *map(str, incorrect)]
        random.Random(rank(seed, "personamem-v2-options", ident)).shuffle(options)
        letter = chr(65 + options.index(row["correct_answer"]))
        queries.append({"id": ident, "user_id": uid, "scope_id": "history", "question": question,
                        "category": row["pref_type"]})
        tasks.append({"id": ident, "dataset": "personamem-v2", "category": row["pref_type"],
                      "answer_format": "multiple_choice", "options": options, "persona_id": persona,
                      "official_pipeline": "vendor/aml/data/personamem/pipeline_v2.py"})
        labels.append({"id": ident, "dataset": "personamem-v2", "scoring": "option_letter",
                       "correct_letter": letter, "correct_answer": row["correct_answer"],
                       "preference": row["preference"], "updated": row["updated"], "prev_pref": row["prev_pref"],
                       "who": row["who"], "related_conversation_snippet": row["related_conversation_snippet"]})
    return {"histories": histories, "queries": queries, "labels": labels, "tasks": tasks,
            "report": {"persona": persona, "history_file": link, "history_sha256": sha256(history_file),
                       "history_messages": len(messages), "questions": len(queries)}}


def clbench(seed: int, per_category: int, max_chars: int, single_per_category: int = 0) -> dict:
    source = PUBLIC / "clbench/CL-bench.jsonl"
    rows = jsonl(source)
    items, seen = [], set()
    for row in sorted(rows, key=lambda r: rank(seed, "clbench-context", r["metadata"]["task_id"])):
        meta, messages = row["metadata"], row["messages"]
        size = sum(len(m["content"]) for m in messages)
        # Only multi-turn tasks have earlier turns to remember; one task per
        # shared context keeps the histories independent.
        if len(messages) < 4 or size > max_chars or meta["context_id"] in seen:
            continue
        require(messages[0]["role"] == "system" and messages[-1]["role"] == "user", "CL-Bench 消息结构不符")
        seen.add(meta["context_id"])
        items.append(row | {"_id": meta["task_id"]})
    categories = sorted({r["metadata"]["context_category"] for r in rows})
    chosen = stratified(items, lambda r: r["metadata"]["context_category"], {c: per_category for c in categories},
                        seed, "clbench")
    histories, queries, labels, tasks = [], [], [], []
    for row in chosen:
        meta, messages = row["metadata"], row["messages"]
        ident = "clbench:" + meta["task_id"]
        uid = "clbench:" + meta["context_id"]
        histories.append(normalize_add({"user_id": uid, "scope_id": "history", "session_id": "earlier-turns",
                                        "request_id": meta["task_id"], "messages": messages[1:-1]}))
        queries.append({"id": ident, "user_id": uid, "scope_id": "history", "question": messages[-1]["content"],
                        "category": meta["context_category"]})
        tasks.append({"id": ident, "dataset": "clbench", "category": meta["context_category"],
                      "sub_category": meta["sub_category"], "answer_format": "open_ended_rubric",
                      "system_prompt": messages[0]["content"], "official_pipeline": "vendor/aml/data/clbench/pipeline.py"})
        labels.append({"id": ident, "dataset": "clbench", "scoring": "rubric_llm_judge", "rubrics": row["rubrics"]})
    # Single-turn tasks (two thirds of CL-Bench): the reference document is part of the question, as the
    # official answer template says, so there is no history to write. Their own ranked pool keeps sets nested.
    singles, seen = [], set()
    for row in sorted(rows, key=lambda r: rank(seed, "clbench-single", r["metadata"]["task_id"])):
        meta, messages = row["metadata"], row["messages"]
        if len(messages) != 2 or sum(len(m["content"]) for m in messages) > max_chars or meta["context_id"] in seen:
            continue
        require(messages[0]["role"] == "system" and messages[1]["role"] == "user", "CL-Bench 消息结构不符")
        seen.add(meta["context_id"])
        singles.append(row | {"_id": meta["task_id"]})
    for row in stratified(singles, lambda r: r["metadata"]["context_category"],
                          {c: single_per_category for c in categories}, seed, "clbench-single") if single_per_category else []:
        meta, messages = row["metadata"], row["messages"]
        ident = "clbench-single:" + meta["task_id"]
        category = "single-turn: " + meta["context_category"]
        queries.append({"id": ident, "user_id": ident, "scope_id": "history", "question": messages[1]["content"],
                        "category": category})
        tasks.append({"id": ident, "dataset": "clbench", "category": category, "sub_category": meta["sub_category"],
                      "answer_format": "open_ended_rubric", "system_prompt": messages[0]["content"],
                      "official_pipeline": "vendor/aml/data/clbench/pipeline.py"})
        labels.append({"id": ident, "dataset": "clbench", "scoring": "rubric_llm_judge", "rubrics": row["rubrics"]})
    return {"histories": histories, "queries": queries, "labels": labels, "tasks": tasks,
            "report": {"source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
                       "questions": len(queries), "max_task_chars": max_chars,
                       "multi_turn_questions": len(chosen), "single_turn_questions": len(queries) - len(chosen),
                       "category_counts": dict(Counter(t["category"] for t in tasks)),
                       "note": "多轮任务：system 提示留给答题（官方模板含 system_prompt），最后一条用户消息作问题，"
                               "其余轮次（含参考文档）作为历史写入；AML 对多轮任务的划分未公开，这是本地约定。"
                               "单轮任务（类别前缀 single-turn:）：参考文档就在问题里，与官方模板“问题包含参考文档”一致，"
                               "没有需要写入的历史，Search 对该用户返回空材料。"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--locomo-per-category", default="1=8,2=8,3=6,4=8",
                        help="LoCoMo 原题型编号=题数（all 为全部）；1 多跳、2 时间、3 开放、4 单跳")
    parser.add_argument("--longmemeval-per-type", type=int, default=3)
    parser.add_argument("--longmemeval-unanswerable", type=int, default=3)
    parser.add_argument("--longmemeval-filler", type=int, default=4, help="证据会话之外保留的其他会话数")
    parser.add_argument("--beam-conversations", type=int, default=1)
    parser.add_argument("--personamem-v1-min-questions", type=int, default=12)
    parser.add_argument("--personamem-v2-personas", type=int, default=1)
    parser.add_argument("--personamem-v2-per-type", type=amount, default=2, help="每种偏好类型的题数，all 为该用户全部题")
    parser.add_argument("--clbench-per-category", type=int, default=3)
    parser.add_argument("--clbench-max-chars", type=int, default=40000)
    parser.add_argument("--clbench-single-per-category", type=int, default=0,
                        help="单轮 CL-Bench 题（文档在问题里）每个大类的题数；mixed-v1 为 0")
    parser.add_argument("--must-include", type=Path, help="较小的评测集目录；要求其题目与写入请求原样包含在新集中")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录已有内容；请使用空目录")
    used = used_ids(ROOT / "outputs")
    quota = {k: amount(v) for k, v in (part.split("=") for part in args.locomo_per_category.split(","))}
    parts = {
        "locomo-refined": locomo(args.seed, quota),
        "longmemeval": longmemeval(args.seed, args.longmemeval_per_type, args.longmemeval_unanswerable,
                                   args.longmemeval_filler, used["longmemeval"]),
        "beam": beam(args.seed, args.beam_conversations),
        "personamem-v1": personamem_v1(args.seed, args.personamem_v1_min_questions, used["personamem-v1"]),
        "personamem-v2": personamem_v2(args.seed, args.personamem_v2_personas, args.personamem_v2_per_type),
        "clbench": clbench(args.seed, args.clbench_per_category, args.clbench_max_chars,
                           args.clbench_single_per_category),
    }
    combined = {name: [row for part in parts.values() for row in part[name]]
                for name in ("histories", "queries", "labels", "tasks")}
    for query in combined["queries"]:
        validate_query(query)
    for history in combined["histories"]:
        normalize_add(history)
    ids = [q["id"] for q in combined["queries"]]
    require(len(set(ids)) == len(ids), "题目编号重复")
    require(ids == [t["id"] for t in combined["tasks"]] and set(ids) == {l["id"] for l in combined["labels"]},
            "queries / tasks / labels 没有一一对应")
    require(len({(h["user_id"], h["scope_id"], h["request_id"]) for h in combined["histories"]}) == len(combined["histories"]),
            "写入请求编号重复")
    # Only single-turn CL-Bench questions carry their material in the question and have no history.
    require({q["user_id"] for q in combined["queries"] if not q["id"].startswith("clbench-single:")}
            <= {h["user_id"] for h in combined["histories"]}, "问题没有对应历史")
    if args.must_include:
        for name, key in (("inputs/queries", "id"), ("inputs/tasks", "id"), ("scoring/labels", "id"),
                          ("inputs/histories", "request_id")):
            new = {r[key] if key == "id" else (r["user_id"], r["request_id"]): r
                   for r in combined[name.split("/")[1]]}
            for row in jsonl(args.must_include / f"{name}.jsonl"):
                ident = row[key] if key == "id" else (row["user_id"], row["request_id"])
                require(new.get(ident) == row, f"新集没有原样包含 {args.must_include} 的 {name}：{ident}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in combined.items():
        directory = args.output / ("scoring" if name == "labels" else "inputs")
        directory.mkdir(exist_ok=True)
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    summary = {name: {"questions": len(part["queries"]), "write_requests": len(part["histories"]),
                      "memory_scopes": len({h["user_id"] for h in part["histories"]}),
                      "history_chars": sum(len(m["content"]) for h in part["histories"] for m in h["messages"])}
               for name, part in parts.items()}
    manifest = {"name": args.output.name, "seed": args.seed, "arguments": {k: str(v) for k, v in vars(args).items()},
                "totals": {"questions": len(ids), "write_requests": len(combined["histories"]),
                           "memory_scopes": len({h["user_id"] for h in combined["histories"]}),
                           "history_chars": sum(s["history_chars"] for s in summary.values())},
                "summary": summary, "datasets": {name: part["report"] for name, part in parts.items()},
                "excluded": {"scriptmem": "公开版不含剧本对话原文（版权原因），没有可写入的历史，无法本地评测。"},
                "selection_fields": "只用题号、题型、对话/历史编号和原文长度；不看答案或已有分数",
                "file_sha256": {str(p.relative_to(args.output)): sha256(p) for p in sorted(args.output.rglob("*.jsonl"))},
                "builder_sha256": sha256(Path(__file__)),
                "contains": str(args.must_include) if args.must_include else None}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"totals": manifest["totals"], "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
